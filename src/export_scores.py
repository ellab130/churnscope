import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str):
    df = pd.read_csv(path)

    id_col = "Customer_ID" if "Customer_ID" in df.columns else None
    if "churn" not in df.columns:
        raise ValueError("Expected a 'churn' column.")

    df = df.dropna(subset=["churn"]).copy()
    df["churn"] = df["churn"].astype(int)

    ids = df[id_col].copy() if id_col else pd.Series(df.index, name="row_index")
    X = df.drop(columns=["churn"] + ([id_col] if id_col else []))
    y = df["churn"]
    return X, y, ids


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/churn.csv")
    parser.add_argument("--out", default="site/data/churn_scores.csv")
    args = parser.parse_args()

    X, y, ids = load_data(args.data)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, stratify=y, random_state=42
    )

    model = Pipeline([
        ("prep", make_preprocess(X_train)),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=16,
            max_features=0.5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    out_df = pd.DataFrame({
        "customer_id": ids_test.to_numpy(),
        "churn_probability": proba,
        "actual_churn": y_test.to_numpy(),  # keep for demo evaluation; you can remove later
    }).sort_values("churn_probability", ascending=False)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
