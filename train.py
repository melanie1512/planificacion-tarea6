import kagglehub

path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")

print("Path to dataset files:", path)

import pandas as pd
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, recall_score, precision_score
import joblib

import pandas as pd 

df = pd.read_csv(path + "/heart.csv")

X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_cols = ["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"]
num_cols = [c for c in X.columns if c not in cat_cols]

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# 5) Modelos
logreg = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=2000))
])

rf = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    ))
])

scoring = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
    "accuracy": "accuracy",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_log = cross_validate(logreg, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
cv_rf  = cross_validate(rf,    X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

def summarize(name, cvres):
    import numpy as np
    print(f"\n{name}")
    for k,v in cvres.items():
        if k.startswith("test_"):
            print(f"  {k[5:]}: {np.mean(v):.3f} ± {np.std(v):.3f}")

summarize("LogReg (CV)", cv_log)
summarize("RandomForest (CV)", cv_rf)

best = logreg
best.fit(X_train, y_train)

from sklearn.metrics import classification_report, roc_auc_score
proba = best.predict_proba(X_test)[:,1]
pred  = (proba >= 0.5).astype(int)
print("\nHoldout:")
print("  ROC-AUC:", roc_auc_score(y_test, proba).round(3))
print(classification_report(y_test, pred, digits=3))

joblib.dump(best, "heart_model.pkl")
print("\nModelo guardado en heart_model.pkl")
