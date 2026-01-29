import argparse, ast
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

POS_LABEL = "B"

def build_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

def infer_cols(df, label_col="Winner"):
    group_col = "VideoName" if "VideoName" in df.columns else None
    if group_col is None:
        df["_GROUP_"] = np.arange(len(df))
        group_col = "_GROUP_"

    drop_cols = {label_col, group_col}
    cols = [c for c in df.columns if c not in drop_cols]

    cat_cols, num_cols = [], []
    for c in cols:
        if df[c].dtype == "object":
            cat_cols.append(c)
        else:
            num_cols.append(c)
    return group_col, cat_cols, num_cols

def make_model(model_name):
    if model_name.endswith("+LogReg"):
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if model_name.endswith("+RF"):
        return RandomForestClassifier(class_weight="balanced", random_state=42)
    if model_name.endswith("+SVM"):
        return SVC(class_weight="balanced", probability=True, random_state=42)
    if model_name.endswith("+XGB"):
        if not XGB_OK:
            raise RuntimeError("XGBoost not installed in this env.")
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    raise ValueError(f"Unknown model: {model_name}")

def apply_bestparams(clf, bestparams_dict):
    # bestparams in CSV look like: {"clf__n_estimators": 500, ...}
    # remove "clf__" prefix for estimator.set_params
    cleaned = {}
    for k, v in bestparams_dict.items():
        kk = k.replace("clf__", "")
        cleaned[kk] = v
    clf.set_params(**cleaned)

def cv_predict_and_report(df, label_col="Winner", n_splits=5):
    df = df.copy()
    df[label_col] = df[label_col].astype(str)
    df = df[df[label_col].isin(["A","B"])].copy()

    group_col, cat_cols, num_cols = infer_cols(df, label_col=label_col)
    groups = df[group_col].astype(str).values

    X = df.drop(columns=[label_col])
    y = df[label_col].values

    gkf = GroupKFold(n_splits=n_splits)

    y_all, p_all, proba_all = [], [], []

    for tr, te in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        yield X_tr, y_tr, X_te, y_te

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="winner_399_f0.csv / f1 / f0_f1")
    ap.add_argument("--results", required=True, help="winner_results_2.csv (tuned table)")
    ap.add_argument("--feature_prefix", required=True, help="F0 or F1 or F0+F1")
    ap.add_argument("--label", default="Winner")
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df[args.label] = df[args.label].astype(str)
    df = df[df[args.label].isin(["A","B"])].copy()

    # pick the best tuned row for this feature set (highest MacroF1_mean)
    res = pd.read_csv(args.results)
    sub = res[res["Model"].astype(str).str.startswith(args.feature_prefix + "+")].copy()
    if len(sub) == 0:
        raise ValueError("No rows in results match feature_prefix. Check your prefix text.")

    best_row = sub.sort_values("MacroF1_mean", ascending=False).iloc[0]
    model_name = best_row["Model"]
    best_params = ast.literal_eval(str(best_row["BestParams"]))

    print("=== Using tuned best row ===")
    print("Model:", model_name)
    print("BestParams:", best_params)
    print("Reported (from table): Acc=", round(best_row["Acc_mean"],4),
          "MacroF1=", round(best_row["MacroF1_mean"],4),
          "AUC=", round(best_row["AUC_mean"],4))
    print()

    group_col, cat_cols, num_cols = infer_cols(df, label_col=args.label)
    groups = df[group_col].astype(str).values

    pre = build_preprocessor(cat_cols, num_cols)
    clf = make_model(model_name)
    apply_bestparams(clf, best_params)

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X = df.drop(columns=[args.label])
    y = df[args.label].values

    gkf = GroupKFold(n_splits=args.splits)

    y_true_all, y_pred_all, y_score_all = [], [], []

    for tr, te in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        y_true_all.extend(list(y_te))
        y_pred_all.extend(list(pred))

        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_te)
            cls = list(pipe.named_steps["clf"].classes_)
            if POS_LABEL in cls:
                pos_idx = cls.index(POS_LABEL)
                y_score_all.extend(list(proba[:, pos_idx]))
            elif set(cls) == {0,1}:
                y_score_all.extend(list(proba[:, 1]))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    acc = accuracy_score(y_true_all, y_pred_all)
    mf1 = f1_score(y_true_all, y_pred_all, average="macro")

    auc = np.nan
    if len(y_score_all) == len(y_true_all):
        y_bin = (y_true_all == POS_LABEL).astype(int)
        if len(np.unique(y_bin)) == 2:
            auc = roc_auc_score(y_bin, np.array(y_score_all))

    print("=== CV (pooled predictions across folds) ===")
    print("Acc:", round(acc,4), "MacroF1:", round(mf1,4), "AUC:", (round(auc,4) if not np.isnan(auc) else "nan"))
    print()
    print("Confusion Matrix [rows=true A,B ; cols=pred A,B]:")
    print(confusion_matrix(y_true_all, y_pred_all, labels=["A","B"]))
    print()
    print("Classification Report:")
    print(classification_report(y_true_all, y_pred_all, labels=["A","B"], digits=4))

if __name__ == "__main__":
    main()
