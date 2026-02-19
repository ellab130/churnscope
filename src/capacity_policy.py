import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str):
    df = pd.read_csv(path)

    # keep an ID column if present so we can export a contact list
    id_col = "Customer_ID" if "Customer_ID" in df.columns else None

    if "churn" not in df.columns:
        raise ValueError("Expected a 'churn' column in the dataset.")

    df = df.dropna(subset=["churn"]).copy()

    # ensure churn is 0/1 ints
    if df["churn"].dtype == "object":
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0, "True": 1, "False": 0})
    df["churn"] = df["churn"].astype(int)

    # Separate X/y, but keep id separately
    ids = df[id_col].copy() if id_col else pd.Series(df.index, name="row_index")
    X = df.drop(columns=["churn"] + ([id_col] if id_col else []))
    y = df["churn"]

    return X, y, ids


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


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def expected_profit(tp, contacts, churn_loss, contact_cost, save_rate):
    # Incremental profit relative to "do nothing"
    # Expected saved value from contacted true churners minus cost to contact everyone contacted.
    return tp * (save_rate * churn_loss) - contacts * contact_cost



def main():
    parser = argparse.ArgumentParser(description="Capacity-based churn targeting policy (decision support).")
    parser.add_argument("--data", default="data/churn.csv", help="Path to dataset CSV.")
    parser.add_argument("--max_contacts", type=int, default=5000, help="Max customers you can contact this period.")
    parser.add_argument("--contact_cost", type=float, default=5.0, help="Cost to contact 1 customer.")
    parser.add_argument("--churn_loss", type=float, default=200.0, help="Cost of losing 1 customer (LTV).")
    parser.add_argument("--save_rate", type=float, default=0.25, help="Probability contact saves a churner (0..1).")
    parser.add_argument("--export", default="", help="Optional path to export contact list CSV (e.g. outputs/targets.csv).")
    args = parser.parse_args()

    X, y, ids = load_data(args.data)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = make_preprocess(X_train)

    # RandomForest with your tuned-ish params
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

    # Capacity policy: contact top N by churn probability
    N = min(args.max_contacts, len(proba))
    order = np.argsort(-proba)  # descending
    contact_idx = order[:N]

    y_true = y_test.to_numpy()
    y_pred = np.zeros_like(y_true)
    y_pred[contact_idx] = 1  # contacted = predicted positive

    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    contacts = int(y_pred.sum())
    profit = expected_profit(tp, contacts, args.churn_loss, args.contact_cost, args.save_rate)

    contacted = int(y_pred.sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print("\n=== Churn Decision Support: Capacity-Based Targeting ===")
    print(f"Model: RandomForest | Test ROC-AUC: {auc:.3f}")
    print(f"Policy: contact top {contacted} customers by churn probability")
    print(f"Assumptions: contact_cost=${args.contact_cost:.2f}, churn_loss=${args.churn_loss:.2f}, save_rate={args.save_rate:.2f}")
    print("\nOutcomes on test set (as a proxy):")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Precision={precision:.3f} Recall={recall:.3f}")
    print(f"Expected profit (test set): {profit:.2f}")

    # Optional export
    if args.export:
        out = pd.DataFrame({
            "customer": ids_test.to_numpy(),
            "churn_probability": proba,
            "contact": y_pred,
            "actual_churn": y_true,
        })
        out_contacts = out[out["contact"] == 1].sort_values("churn_probability", ascending=False)
        out_contacts.to_csv(args.export, index=False)
        print(f"\nExported contact list to: {args.export}")

if __name__ == "__main__":
    main()
