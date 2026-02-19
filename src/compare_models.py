import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def load_data(path="data/churn.csv"):
    df = pd.read_csv(path)
    if "Customer_ID" in df.columns:
        df = df.drop(columns=["Customer_ID"])
    df = df.dropna(subset=["churn"]).copy()

    # Ensure churn is int 0/1
    if df["churn"].dtype == "object":
        df["churn"] = df["churn"].map({"Yes": 1, "No": 0, "True": 1, "False": 0})
    df["churn"] = df["churn"].astype(int)

    X = df.drop(columns=["churn"])
    y = df["churn"]
    return X, y

def make_preprocess(X):
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

def evaluate(model_name, clf, X_test, y_test):
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "model": model_name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }

def main():
    X, y = load_data("data/churn.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = make_preprocess(X)

    models = {
        "LogReg": LogisticRegression(max_iter=4000),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=16,
            min_samples_leaf=1,
            max_features= 0.5
        ),
        "GradBoost": GradientBoostingClassifier(random_state=42),
    }

    rows = []
    for name, model in models.items():
        clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        clf.fit(X_train, y_train)
        rows.append(evaluate(name, clf, X_test, y_test))

    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    # Pretty print
    print("\n=== Model Comparison (threshold=0.5) ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

if __name__ == "__main__":
    main()
