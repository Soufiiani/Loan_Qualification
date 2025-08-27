
"""
train_and_export.py — Regenerate loan model artifacts compatible with your current environment.

Usage (Windows, from your venv):
  python train_and_export.py --csv "D:\ML Projects\Loan Qualification\notebook\loan.csv" --out "D:\ML Projects\Loan Qualification\notebook"

Optional args:
  --target Loan_Status         # if your target has a non-standard name, set it explicitly
  --model rf                   # 'rf' (default) or 'logreg' if you prefer linear model
  --test_size 0.2              # holdout fraction (default 0.2)
"""

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def infer_target(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit and explicit in df.columns:
        return explicit
    candidates = [
        "Loan_Status", "loan_status", "Status", "status", "Approved", "approved",
        "LoanApproved", "loan_approved", "Decision", "decision", "Outcome", "outcome",
        "label", "target"
    ]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    # heuristic: near-binary column with status-like name
    for c in df.columns:
        nunq = df[c].dropna().nunique()
        if nunq in (2, 3, 4) and any(k in c.lower() for k in ["status","approved","decision","outcome","label","target"]):
            return c
    # fallback: last column
    return df.columns[-1]


def train_pipeline(df: pd.DataFrame, target_col: str, model_type: str = "rf", test_size: float = 0.2):
    # Drop ID-like fields
    id_like = [c for c in df.columns if c.lower() in {"loan_id","id","customer_id","applicant_id"}]
    y = df[target_col]
    X = df.drop(columns=[target_col] + id_like, errors="ignore")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([("num", numeric, num_cols),
                                    ("cat", categorical, cat_cols)], remainder="drop")

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=500, n_jobs=None)
    else:
        clf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)

    pipe = Pipeline([("prep", preprocess), ("clf", clf)])

    strat = y if y.nunique() <= 20 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat)
    pipe.fit(X_tr, y_tr)

    # Metrics
    y_pred = pipe.predict(X_te)
    report = classification_report(y_te, y_pred, digits=3)
    auc_text = ""
    try:
        if hasattr(pipe, "predict_proba") and len(np.unique(y_te)) == 2:
            classes_ = pipe.classes_
            proba = pipe.predict_proba(X_te)
            pos = np.argmax(classes_)
            y_true_bin = (y_te == classes_[pos]).astype(int)
            auc = roc_auc_score(y_true_bin, proba[:, pos])
            auc_text = f"ROC-AUC (positive='{classes_[pos]}'): {auc:.3f}"
    except Exception as e:
        auc_text = f"ROC-AUC: not computed ({e})"

    return pipe, report, auc_text, num_cols, cat_cols


def build_schema(X: pd.DataFrame, target_col: str, num_cols: list[str], cat_cols: list[str]):
    schema = {"target_col": target_col, "numeric_features": [], "categorical_features": [],
              "defaults": {}, "categories": {}, "ranges": {}}

    # Numeric defaults and ranges
    for c in num_cols:
        series = X[c]
        med = float(series.median()) if series.size else 0.0
        schema["numeric_features"].append(c)
        schema["defaults"][c] = med
        try:
            cmin = float(np.nanpercentile(series, 1)); cmax = float(np.nanpercentile(series, 99))
        except Exception:
            cmin, cmax = float(series.min()), float(series.max())
        if not np.isfinite(cmin): cmin = 0.0
        if not np.isfinite(cmax): cmax = cmin + 1.0
        schema["ranges"][c] = {"min": cmin, "max": cmax}

    # Categorical defaults and choices
    for c in cat_cols:
        series = X[c].astype("object")
        vals = sorted([str(v) for v in series.dropna().unique().tolist()])[:1000]
        default = series.mode().iloc[0] if not series.mode().empty else (vals[0] if vals else "")
        schema["categorical_features"].append(c)
        schema["defaults"][c] = str(default)
        schema["categories"][c] = vals

    return schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to loan.csv")
    ap.add_argument("--out", required=True, help="Output folder for loan_pipeline.pkl & schema.json")
    ap.add_argument("--target", default=None, help="Target column name (optional)")
    ap.add_argument("--model", default="rf", choices=["rf","logreg"], help="Model type: rf or logreg")
    ap.add_argument("--test_size", default=0.2, type=float, help="Holdout fraction (default 0.2)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    target_col = infer_target(df, args.target)

    # Build feature matrix for schema after ID drop
    id_like = [c for c in df.columns if c.lower() in {"loan_id","id","customer_id","applicant_id"}]
    X = df.drop(columns=[target_col] + id_like, errors="ignore")

    pipe, report, auc_text, num_cols, cat_cols = train_pipeline(df, target_col, args.model, args.test_size)

    # Save artifacts
    model_path = out_dir / "loan_pipeline.pkl"
    schema_path = out_dir / "schema.json"
    joblib.dump(pipe, str(model_path))

    schema = build_schema(X, target_col, num_cols, cat_cols)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    print("\n=== Training Summary ===")
    print(f"Target column: {target_col}")
    print(f"Numeric features: {len(num_cols)} | Categorical features: {len(cat_cols)}")
    print("\nClassification report:\n", report)
    if auc_text:
        print(auc_text)
    print(f"\nSaved:\n - {model_path}\n - {schema_path}")


if __name__ == "__main__":
    main()
