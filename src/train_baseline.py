import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/churn.csv")

# Target + ID column
TARGET = "churn"
ID_COL = "Customer_ID"

# Drop ID if present
if ID_COL in df.columns:
    df = df.drop(columns=[ID_COL])

# Drop rows missing target
df = df.dropna(subset=[TARGET]).copy()

# Ensure churn is 0/1 ints
# (Some churn datasets use True/False or Yes/No. This handles common cases.)
if df[TARGET].dtype == "object":
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0, "True": 1, "False": 0})
df[TARGET] = df[TARGET].astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split (stratify keeps churn ratio similar in train/test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Columns by type
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# Pipelines
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ]
)

# Model (balanced helps with churn imbalance)
model = LogisticRegression(max_iter=4000, class_weight="balanced")

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model),
])

# Train
clf.fit(X_train, y_train)

# Evaluate
proba = clf.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("\n=== Baseline: Logistic Regression (threshold=0.5) ===")
print(classification_report(y_test, pred, digits=3))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

print("\nChurn rate in dataset:", y.mean().round(3))
