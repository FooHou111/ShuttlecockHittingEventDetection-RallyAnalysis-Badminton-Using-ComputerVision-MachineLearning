"""evaluate_rally_models_groupkfold.py

Evaluate Winner prediction on rally-level datasets (1 row per VideoName)
using grouped folds (groups=VideoName by default). Since each rally row is
already per-Video, grouping is mainly for API consistency and to avoid
accidental leakage if you later add multiple rallies per match.

Models:
  - Logistic Regression
  - Random Forest
  - XGBoost (optional, if installed)

Feature sets:
  - F0
  - F1
  - F0+F1

Metrics:
  - Accuracy
  - Macro-F1
  - ROC-AUC (binary)

Outputs:
  - model_results_rally_groupkfold.csv (mean +/- std over folds)

Run:
  python evaluate_rally_models_groupkfold.py --f0 winner_F0_rally.csv --f1 winner_F1_rally.csv --f0f1 winner_F0F1_rally.csv --out model_results_rally_groupkfold.csv
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False


POS_LABEL = "B"  # treat B as positive class for AUC


def _encode_y_ab(y: np.ndarray) -> np.ndarray:
    # A->0, B->1
    y = np.asarray(y)
    return (y == POS_LABEL).astype(int)


def _build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
        ],
        remainder="drop",
    )


def _split_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # avoid leaking label / ids
    drop_cols = {"Winner", "VideoName", "group"}

    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if df[c].dtype == object:
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return cat_cols, num_cols


@dataclass
class ModelSpec:
    name: str
    estimator: object


def evaluate_df(df: pd.DataFrame, model: ModelSpec, n_splits: int = 5, seed: int = 42) -> Dict[str, float]:
    # Label column: prefer 'Winner' (shot-level), fall back to 'Winner_video' (rally-level)
    label_col = "Winner" if "Winner" in df.columns else ("Winner_video" if "Winner_video" in df.columns else None)
    if label_col is None:
        raise KeyError("Expected a label column 'Winner' or 'Winner_video' in the dataset")

    # keep only A/B labels
    df = df[df[label_col].isin(["A", "B"])].copy()
    df = df.reset_index(drop=True)

    y = df[label_col].astype(str).values
    X = df.drop(columns=[label_col])

    # Split feature columns from X (not from the label-containing df)
    cat_cols, num_cols = _split_cols(X)
    pre = _build_preprocessor(cat_cols, num_cols)

    # Build pipeline. XGBoost needs numeric y.
    if model.name == "XGBoost":
        clf = model.estimator
    else:
        clf = model.estimator

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accs, mf1s, aucs = [], [], []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_tr = y[tr_idx]
        y_te = y[te_idx]

        if model.name == "XGBoost":
            y_tr_fit = _encode_y_ab(y_tr)
        else:
            y_tr_fit = y_tr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_tr, y_tr_fit)

        # predictions
        pred = pipe.predict(X_te)
        if model.name == "XGBoost":
            pred = np.where(pred == 1, "B", "A")

        accs.append(accuracy_score(y_te, pred))
        mf1s.append(f1_score(y_te, pred, average="macro"))

        # auc: need probabilities
        try:
            proba = pipe.predict_proba(X_te)[:, 1]
            y_bin = (y_te == POS_LABEL).astype(int)
            aucs.append(roc_auc_score(y_bin, proba))
        except Exception:
            pass

    res = {
        "Acc_mean": float(np.mean(accs)),
        "Acc_std": float(np.std(accs)),
        "MacroF1_mean": float(np.mean(mf1s)),
        "MacroF1_std": float(np.std(mf1s)),
        "AUC_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
        "AUC_std": float(np.std(aucs)) if len(aucs) else np.nan,
        "n_samples": int(len(df)),
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0", required=True, help="winner_F0_rally.csv")
    ap.add_argument("--f1", required=True, help="winner_F1_rally.csv")
    ap.add_argument("--f0f1", required=True, help="winner_F0F1_rally.csv")
    ap.add_argument("--out", default="model_results_rally_groupkfold.csv")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    datasets = {
        "F0": pd.read_csv(args.f0),
        "F1": pd.read_csv(args.f1),
        "F0+F1": pd.read_csv(args.f0f1),
    }

    models: List[ModelSpec] = [
        ModelSpec(
            "LogReg",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        ModelSpec(
            "RandomForest",
            RandomForestClassifier(
                n_estimators=400,
                random_state=42,
                class_weight="balanced",
                max_depth=None,
            ),
        ),
    ]

    if XGB_OK:
        models.append(
            ModelSpec(
                "XGBoost",
                XGBClassifier(
                    n_estimators=600,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    eval_metric="logloss",
                ),
            )
        )

    rows = []
    for dname, df in datasets.items():
        for m in models:
            res = evaluate_df(df, m, n_splits=args.splits)
            rows.append({"FeatureSet": dname, "Model": m.name, **res})
            print(f"[DONE] {dname} + {m.name} -> Acc {res['Acc_mean']:.3f}, MacroF1 {res['MacroF1_mean']:.3f}, AUC {res['AUC_mean']:.3f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
