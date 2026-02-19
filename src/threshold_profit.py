import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # Drop ID column if present
    if "Customer_ID" in df.columns:
        df = df.drop(columns=["Customer_ID"])

    # Target
    if "churn" not in df.columns:
        raise ValueError("Expected a 'churn' column in the dataset.")

    df = df.dropna(subset=["churn"]).copy()

    # Ensure churn is 0/1 integer
    if df["churn"].dtype == "object":
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0, "True": 1, "False": 0})
    df["churn"] = df["churn"].astype(int)

    X = df.drop(columns=["churn"])
    y = df["churn"]
    return X, y


def make_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    # returns TP, FP, FN, TN
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def expected_profit(tp: int, fp: int, fn: int, tn: int,
                    churn_loss: float, contact_cost: float, save_rate: float) -> float:
    """
    Simple decision model:
      - If we contact a customer (pred=1), we pay contact_cost.
      - If they're a true churner (TP), contacting prevents churn with probability save_rate,
        so expected benefit is save_rate * churn_loss.
      - If we do NOT contact a true churner (FN), we incur churn_loss.
      - TN has 0 value in this simplified setup.
    """
    return (
        tp * (save_rate * churn_loss - contact_cost) +
        fp * (-contact_cost) +
        fn * (-churn_loss) +
        tn * 0.0
    )


def main():
    parser = argparse.ArgumentParser(description="Profit-based threshold selection for churn decision support.")
    parser.add_argument("--data", default="data/churn.csv", help="Path to CSV dataset.")
    parser.add_argument("--contact_cost", type=float, default=5.0, help="Cost to contact 1 customer.")
    parser.add_argument("--churn_loss", type=float, default=200.0, help="Cost of losing 1 customer (LTV).")
    parser.add_argument("--save_rate", type=float, default=0.25,
                        help="Probability that contacting a true churner prevents churn (0..1).")
    parser.add_argument("--steps", type=int, default=19,
                        help="Number of thresholds from 0.05..0.95 to evaluate.")
    args = parser.parse_args()

    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = make_preprocess(X)

    # Your tuned-ish RF (from your search). Feel free to adjust later.
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        max_features=0.5,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    clf = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", rf),
    ])

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    thresholds = np.linspace(0.05, 0.95, args.steps)

    rows = []
    best = None

    y_true = y_test.to_numpy()

    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tp, fp, fn, tn = confusion_counts(y_true, y_pred)

        profit = expected_profit(
            tp, fp, fn, tn,
            churn_loss=args.churn_loss,
            contact_cost=args.contact_cost,
            save_rate=args.save_rate,
        )

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        contacted = int((y_pred == 1).sum())

        row = {
            "threshold": float(t),
            "profit": float(profit),
            "contacted": contacted,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        rows.append(row)

        if best is None or profit > best["profit"]:
            best = row

    results = pd.DataFrame(rows).sort_values("profit", ascending=False)

    print("\n=== Churn Decision Support: Profit-Based Threshold Sweep ===")
    print(f"Model: RandomForest | Test ROC-AUC: {auc:.3f}")
    print(f"Assumptions: contact_cost=${args.contact_cost:.2f}, churn_loss=${args.churn_loss:.2f}, save_rate={args.save_rate:.2f}")
    print("\nTop thresholds by expected profit (test set):")
    print(results.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Recommended Operating Point (max profit) ===")
    print(
        f"threshold={best['threshold']:.2f} | profit={best['profit']:.2f} | "
        f"contacted={best['contacted']} | precision={best['precision']:.3f} | "
        f"recall={best['recall']:.3f} | f1={best['f1']:.3f} | "
        f"TP={best['tp']} FP={best['fp']} FN={best['fn']} TN={best['tn']}"
    )
    print("\nInterpretation: contact the customers with churn probability >= threshold.\n")


if __name__ == "__main__":
    main()
