# evaluate_models_groupkfold.py
# Evaluate F0 vs F1 vs F0+F1 using Grouped K-Fold by match (VideoName).
# Models: Logistic Regression, Random Forest, XGBoost (if installed)
# Metrics: Accuracy, Macro-F1, ROC-AUC (B as positive class)

import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except Exception:
    HAS_SGK = False
    from sklearn.model_selection import GroupKFold

# XGBoost optional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# =========================
# EDIT PATHS
# =========================
DATA_DIR = r"C:\Users\user\Badminton\src\csv\match_level"
F0_PATH   = os.path.join(DATA_DIR, "match_F0.csv")
F1_PATH   = os.path.join(DATA_DIR, "match_F1.csv")
F0F1_PATH = os.path.join(DATA_DIR, "match_F0F1.csv")

K = 5
RANDOM_SEED = 42

LABEL = "Winner_video"
GROUP = "VideoName"
POS_LABEL = "B"   # ROC-AUC positive class


# =========================
# Helpers
# =========================
def encode_y_AB_to01(y):
    """Map A->0, B->1"""
    y = np.array(y)
    return (y == "B").astype(int)

def split_folds(df):
    y = df[LABEL].values
    groups = df[GROUP].values

    if HAS_SGK:
        cv = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)
        return list(cv.split(df, y, groups=groups))
    else:
        cv = GroupKFold(n_splits=K)
        return list(cv.split(df, y, groups=groups))

def build_preprocessor(df, feature_cols):
    X = df[feature_cols]
    cat_cols = [c for c in feature_cols if X[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), num_cols),
        ],
        remainder="drop"
    )
    return pre, cat_cols, num_cols

def get_models():
    models = {}

    models["LogReg"] = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=RANDOM_SEED
    )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_SEED,
        class_weight="balanced_subsample"
    )

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_SEED
        )

    return models


def evaluate_one(df, feature_set_name, model_name, model):
    # Keep only A/B labeled matches
    df = df[df[LABEL].isin(["A", "B"])].copy()
    if len(df) < K:
        raise ValueError(f"Too few A/B samples ({len(df)}) for K={K} folds.")

    # Features (drop group + label)
    feature_cols = [c for c in df.columns if c not in [GROUP, LABEL]]

    pre, cat_cols, num_cols = build_preprocessor(df, feature_cols)
    pipe = Pipeline([("pre", pre), ("clf", model)])

    folds = split_folds(df)

    accs, mf1s, aucs = [], [], []

    for tr, te in folds:
        X_tr = df.iloc[tr][feature_cols]
        y_tr = df.iloc[tr][LABEL].values

        X_te = df.iloc[te][feature_cols]
        y_te = df.iloc[te][LABEL].values  # always keep as "A"/"B" for reporting

        # ---- Fit with correct label format ----
        if model_name == "XGBoost":
            y_tr_fit = encode_y_AB_to01(y_tr)   # 0/1
        else:
            y_tr_fit = y_tr                      # "A"/"B"

        pipe.fit(X_tr, y_tr_fit)

        # ---- Predict ----
        pred = pipe.predict(X_te)

        # XGB gives 0/1 -> convert back to "A"/"B" so metrics align
        if model_name == "XGBoost":
            pred = np.where(pred == 1, "B", "A")

        # ---- Metrics: Accuracy + MacroF1 ----
        accs.append(accuracy_score(y_te, pred))
        mf1s.append(f1_score(y_te, pred, average="macro"))

        # ---- ROC-AUC ----
        auc = np.nan
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_te)

                y_true_bin = (y_te == POS_LABEL).astype(int)

                if model_name == "XGBoost":
                    # classes are [0,1], positive prob is column 1
                    p_pos = proba[:, 1]
                else:
                    classes = list(pipe.classes_)
                    if POS_LABEL not in classes:
                        p_pos = None
                    else:
                        p_pos = proba[:, classes.index(POS_LABEL)]

                if p_pos is not None and len(np.unique(y_true_bin)) == 2:
                    auc = roc_auc_score(y_true_bin, p_pos)
            except Exception:
                auc = np.nan

        aucs.append(auc)

    return {
        "FeatureSet": feature_set_name,
        "Model": model_name,
        "Accuracy_mean": float(np.mean(accs)),
        "Accuracy_std": float(np.std(accs)),
        "MacroF1_mean": float(np.mean(mf1s)),
        "MacroF1_std": float(np.std(mf1s)),
        "ROCAUC_mean": float(np.nanmean(aucs)),
        "ROCAUC_std": float(np.nanstd(aucs)),
        "n_videos_AB": int(df.shape[0]),
        "n_features": int(len(feature_cols)),
        "n_cat_cols": int(len(cat_cols)),
        "n_num_cols": int(len(num_cols)),
    }


def run_all():
    datasets = {
        "F0": pd.read_csv(F0_PATH),
        "F1": pd.read_csv(F1_PATH),
        "F0+F1": pd.read_csv(F0F1_PATH),
    }

    models = get_models()

    all_rows = []
    for fs_name, df in datasets.items():
        for mname, model in models.items():
            row = evaluate_one(df, fs_name, mname, model)
            all_rows.append(row)
            print(
                f"[DONE] {fs_name} + {mname} -> "
                f"Acc {row['Accuracy_mean']:.3f}, MacroF1 {row['MacroF1_mean']:.3f}, AUC {row['ROCAUC_mean']:.3f}"
            )

    res = pd.DataFrame(all_rows)
    out_csv = os.path.join(DATA_DIR, "model_results_groupkfold.csv")
    res.to_csv(out_csv, index=False)

    print("\nSaved results:", out_csv)
    print(res.sort_values(["FeatureSet", "Model"]))


if __name__ == "__main__":
    if not HAS_XGB:
        print("[INFO] xgboost not installed. XGBoost will be skipped.")
        print("       Install: pip install xgboost")
    run_all()
