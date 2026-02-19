import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def load_data(path: str):
    df = pd.read_csv(path)

    if "churn" not in df.columns:
        raise ValueError("Expected a 'churn' column.")

    df = df.dropna(subset=["churn"]).copy()
    df["churn"] = df["churn"].astype(int)

    X = df.drop(columns=["churn"])
    y = df["churn"]
    return X, y


def make_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])


def expected_profit(tp: int, contacts: int, contact_cost: float, churn_loss: float, save_rate: float) -> float:
    """
    Incremental ROI vs "do nothing":
      Profit = (expected saved value from contacted true churners) - (cost to contact everyone contacted)
            = tp * (save_rate * churn_loss) - contacts * contact_cost
    """
    return tp * (save_rate * churn_loss) - contacts * contact_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    parser.add_argument("--top_n", type=int, default=5000)
    parser.add_argument("--churn_loss", type=float, default=200.0)
    args = parser.parse_args()

    # Helps catch "wrong file / not saved" issues
    print(f"Running: {os.path.abspath(__file__)}")

    X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ("prep", make_preprocess(X_train)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            max_features=0.5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"\nModel AUC: {auc:.3f}")

    # Sweep contact_cost and save_rate
    contact_costs = [5, 10, 20, 30]
    save_rates = [0.10, 0.20, 0.30, 0.40]

    print(f"\n=== Incremental Profit at Top {args.top_n} Customers ===")

    order = np.argsort(-proba)
    N = min(args.top_n, len(order))
    mask = np.zeros(len(order), dtype=bool)
    mask[order[:N]] = True

    y_true = y_test.to_numpy()
    tp = int(((y_true == 1) & mask).sum())
    fp = int(((y_true == 0) & mask).sum())
    contacts = int(mask.sum())

    print(f"Selected contacts: {contacts} | TP={tp} | FP={fp} | Precision={tp/(tp+fp):.3f}")

    for cost in contact_costs:
        for sr in save_rates:
            profit = expected_profit(
                tp=tp,
                contacts=contacts,
                contact_cost=float(cost),
                churn_loss=float(args.churn_loss),
                save_rate=float(sr),
            )
            print(f"contact_cost={cost:>2} | save_rate={sr:.2f} | profit={profit:,.0f}")


if __name__ == "__main__":
    main()
