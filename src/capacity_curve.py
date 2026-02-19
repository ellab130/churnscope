import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str):
    df = pd.read_csv(path)

    id_col = "Customer_ID" if "Customer_ID" in df.columns else None

    if "churn" not in df.columns:
        raise ValueError("Expected a 'churn' column in the dataset.")

    df = df.dropna(subset=["churn"]).copy()

    if df["churn"].dtype == "object":
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0, "True": 1, "False": 0})
    df["churn"] = df["churn"].astype(int)

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


def confusion_counts(y_true: np.ndarray, contacted_mask: np.ndarray):
    # contacted_mask: True means we contact (predict positive)
    y_pred = contacted_mask.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def expected_profit(tp, contacts, churn_loss, contact_cost, save_rate):
    return tp * (save_rate * churn_loss) - contacts * contact_cost


def main():
    parser = argparse.ArgumentParser(description="Profit vs outreach capacity curve for churn decision support.")
    parser.add_argument("--data", default="data/churn.csv", help="Path to dataset CSV.")
    parser.add_argument("--contact_cost", type=float, default=20.0, help="Cost to contact 1 customer.")
    parser.add_argument("--churn_loss", type=float, default=200.0, help="Cost of losing 1 customer (LTV).")
    parser.add_argument("--save_rate", type=float, default=0.25, help="Probability contact saves a churner (0..1).")
    parser.add_argument("--min_contacts", type=int, default=0, help="Minimum contacts in sweep.")
    parser.add_argument("--max_contacts", type=int, default=20000, help="Maximum contacts in sweep.")
    parser.add_argument("--step", type=int, default=500, help="Step size for contacts sweep.")
    parser.add_argument("--plot_path", default="outputs/capacity_profit_curve.png",
                        help="Where to save the plot image.")
    parser.add_argument("--csv_path", default="outputs/capacity_profit_curve.csv",
                        help="Where to save the results CSV.")
    args = parser.parse_args()

    X, y, ids = load_data(args.data)
# ---- Simulate realistic churn rate by downsampling positives ----
    target_rate = 0.20  # change to 0.15–0.25 if you want to experiment

    if y.mean() > target_rate:
        df_temp = X.copy()
        df_temp["churn"] = y
        df_temp["id"] = ids

        positives = df_temp[df_temp["churn"] == 1]
        negatives = df_temp[df_temp["churn"] == 0]

        desired_pos = int(len(negatives) * target_rate / (1 - target_rate))
        positives_sampled = positives.sample(n=min(desired_pos, len(positives)), random_state=42)

        df_balanced = pd.concat([positives_sampled, negatives]).sample(frac=1, random_state=42)

        y = df_balanced["churn"]
        ids = df_balanced["id"]
        X = df_balanced.drop(columns=["churn", "id"])

        print(f"\nSimulated churn rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = make_preprocess(X_train)

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
    y_true = y_test.to_numpy()

    # Order test customers by predicted churn risk
    order = np.argsort(-proba)  # descending

    caps = list(range(args.min_contacts, min(args.max_contacts, len(order)) + 1, args.step))
    if caps[-1] != min(args.max_contacts, len(order)):
        caps.append(min(args.max_contacts, len(order)))

    rows = []
    break_even = None

    for N in caps:
        contacted_mask = np.zeros_like(y_true, dtype=bool)
        contacted_mask[order[:N]] = True

        tp, fp, fn, tn = confusion_counts(y_true, contacted_mask)
        contacts = int(contacted_mask.sum())

        profit = expected_profit(tp, fp, fn, tn, args.churn_loss, args.contact_cost, args.save_rate)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        rows.append({
            "max_contacts": int(N),
            "profit": float(profit),
            "precision": float(precision),
            "recall": float(recall),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })

        if break_even is None and profit >= 0:
            break_even = N

    results = pd.DataFrame(rows)

    # Print summary
    print("\n=== Capacity → Profit Curve (Decision Support) ===")
    print(f"Model: RandomForest | Test ROC-AUC: {auc:.3f}")
    print(f"Assumptions: contact_cost=${args.contact_cost:.2f}, churn_loss=${args.churn_loss:.2f}, save_rate={args.save_rate:.2f}")
    print(f"Sweep: {caps[0]}..{caps[-1]} step={args.step} (evaluated on held-out test set as proxy)")

    best_row = results.iloc[results["profit"].idxmax()]
    print("\nBest capacity in sweep:")
    print(
        f"max_contacts={int(best_row['max_contacts'])} | profit={best_row['profit']:.2f} | "
        f"precision={best_row['precision']:.3f} | recall={best_row['recall']:.3f} | "
        f"TP={int(best_row['tp'])} FP={int(best_row['fp'])} FN={int(best_row['fn'])} TN={int(best_row['tn'])}"
    )

    if break_even is None:
        print("\nBreak-even: not reached in this sweep (profit stayed negative).")
    else:
        print(f"\nBreak-even capacity (first non-negative profit): {break_even} contacts")

    # Save CSV
    results.to_csv(args.csv_path, index=False)
    print(f"\nSaved CSV: {args.csv_path}")

    # Plot
    plt.figure()
    plt.plot(results["max_contacts"], results["profit"])
    plt.axhline(0, linewidth=1)
    plt.title("Expected Profit vs Outreach Capacity")
    plt.xlabel("Max contacts")
    plt.ylabel("Expected profit (test set proxy)")
    plt.tight_layout()

    # Save plot
    import os
    os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)
    plt.savefig(args.plot_path, dpi=160)
    print(f"Saved plot: {args.plot_path}")


if __name__ == "__main__":
    main()
