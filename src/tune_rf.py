import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def load_data(path="data/churn.csv"):
    df = pd.read_csv(path)
    if "Customer_ID" in df.columns:
        df = df.drop(columns=["Customer_ID"])
    df = df.dropna(subset=["churn"]).copy()
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

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = make_preprocess(X)

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # Hyperparameters for the model inside the pipeline use prefix: "model__"
    param_dist = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 6, 10, 16],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", 0.5, None],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=6,                 # increase to 40 if you want more thorough
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print("\nBest ROC-AUC (CV):", search.best_score_)
    print("Best params:\n", search.best_params_)

    # Optional: evaluate tuned model on held-out test
    tuned = search.best_estimator_
    test_auc = __import__("sklearn.metrics").metrics.roc_auc_score(
        y_test, tuned.predict_proba(X_test)[:, 1]
    )
    print("Test ROC-AUC:", test_auc)

if __name__ == "__main__":
    main()
