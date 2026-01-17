import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

print("Loading dataset...")

# -------------------------------
# LOAD DATASET
# -------------------------------
df = pd.read_csv("Breast_Cancer_METABRIC_Epic_Hospital.csv")

# -------------------------------
# DROP ID COLUMN
# -------------------------------
if "Patient ID" in df.columns:
    df.drop("Patient ID", axis=1, inplace=True)

# -------------------------------
# TARGET COLUMN (REAL SURVIVAL LABEL)
# -------------------------------
target_col = "Overall Survival Status"
print("Using target column:", target_col)

# -------------------------------
# CLEAN + ENCODE TARGET
# -------------------------------
def encode_survival(val):
    val = str(val).strip().lower()
    if val in ["living", "alive", "1", "yes", "true"]:
        return 1
    if val in ["deceased", "dead", "died", "0", "no", "false"]:
        return 0
    return np.nan

df[target_col] = df[target_col].apply(encode_survival)

# Show class balance
print("\nSurvival label distribution:")
print(df[target_col].value_counts())

# Drop invalid rows
df = df.dropna(subset=[target_col])

# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("Unknown")
    else:
        df[col] = df[col].fillna(df[col].median())

# -------------------------------
# ENCODE CATEGORICAL FEATURES
# -------------------------------
encoder = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# -------------------------------
# SPLIT FEATURES & TARGET
# -------------------------------
X = df.drop(target_col, axis=1)
y = df[target_col]

# -------------------------------
# SCALE FEATURES
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# MODEL
# -------------------------------
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

print("\nTraining model...")
model.fit(X_train, y_train)

# -------------------------------
# EVALUATION
# -------------------------------
y_pred = model.predict(X_test)

print("\nMODEL PERFORMANCE\n")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# SAVE FILES
# -------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("\nFILES SAVED:")
print("model.pkl")
print("scaler.pkl")
print("features.pkl")
print("\nTraining complete!")
