import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception as e:
    XGB_OK = False
    print("[WARN] xgboost not available:", e)


POS_LABEL = "B"  # for AUC when labels are A/B


def infer_feature_cols(df: pd.DataFrame):
    """Infer label column, group column, and categorical/numerical feature columns."""

    label_candidates = [
        "Winner", "winner", "Winner_video",
        "MatchWinner", "match_winner",
        "RallyWinner", "rally_winner",
        "Hitter",
        "Label", "label", "y",
    ]

    label_col = None
    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        raise ValueError(
            f"Cannot find label column. Tried: {label_candidates}. "
            f"Your columns: {list(df.columns)[:50]} ..."
        )

    group_candidates = [
        "VideoName", "video", "video_name",
        "MatchID", "match_id", "match",
        "GameID", "game_id",
    ]

    group_col = None
    for c in group_candidates:
        if c in df.columns:
            group_col = c
            break

    if group_col is None:
        # fallback: each row becomes its own group (still works, but weaker than true grouping)
        df["_GROUP_"] = np.arange(len(df))
        group_col = "_GROUP_"
        print("[WARN] No group column found, using one-row-per-sample groups (_GROUP_).")

    drop_cols = {label_col, group_col}
    drop_cols |= {"MatchID", "RallyID", "Rally", "StartFrame", "EndFrame"}

    cols = [c for c in df.columns if c not in drop_cols]

    cat_cols, num_cols = [], []
    for c in cols:
        if df[c].dtype == "object":
            cat_cols.append(c)
        else:
            num_cols.append(c)

    return label_col, group_col, cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    """Same preprocessing logic as your tuning script (safe to use inside CV)."""

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def eval_groupkfold(df, label_col, group_col, pipe, n_splits=5):
    """Evaluate a fixed (non-tuned) pipeline using GroupKFold."""

    groups = df[group_col].astype(str).values
    y = df[label_col].values
    X = df.drop(columns=[label_col])

    gkf = GroupKFold(n_splits=n_splits)

    accs, mf1s, aucs = [], [], []

    for tr, te in gkf.split(X, y, groups):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)

        accs.append(accuracy_score(y_te, pred))
        mf1s.append(f1_score(y_te, pred, average="macro"))

        # ---- AUC (binary) ----
        try:
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_te)
                cls = list(pipe.classes_)
                y_te_arr = np.array(y_te)

                # numeric labels 0/1
                if set(cls) == {0, 1} or set(cls) == {0.0, 1.0}:
                    pos_idx = cls.index(1) if 1 in cls else cls.index(1.0)
                    y_true_bin = y_te_arr.astype(int)

                # string labels A/B
                elif POS_LABEL in cls:
                    pos_idx = cls.index(POS_LABEL)
                    y_true_bin = (y_te_arr == POS_LABEL).astype(int)

                else:
                    aucs.append(np.nan)
                    continue

                if len(np.unique(y_true_bin)) < 2:
                    aucs.append(np.nan)
                else:
                    aucs.append(roc_auc_score(y_true_bin, proba[:, pos_idx]))
            else:
                aucs.append(np.nan)
        except Exception:
            aucs.append(np.nan)

    return (
        float(np.mean(accs)), float(np.std(accs)),
        float(np.mean(mf1s)), float(np.std(mf1s)),
        float(np.nanmean(aucs)), float(np.nanstd(aucs)),
    )


def load_and_filter(path, force_label=None, ablation=True):
    df = pd.read_csv(path)

    # Same ablation drop as tuning script
    if ablation:
        ablation_drop = [
            "LandingX", "LandingY",
            "HitterLocationX", "HitterLocationY",
            "DefenderLocationX", "DefenderLocationY",
        ]
        df = df.drop(columns=[c for c in ablation_drop if c in df.columns], errors="ignore")

    label_candidates = [
        "Hitter", "Winner_video", "Winner", "winner",
        "MatchWinner", "match_winner", "RallyWinner", "rally_winner",
        "Label", "label", "y",
    ]

    label_col = force_label if (force_label and force_label in df.columns) else None
    if label_col is None:
        for c in label_candidates:
            if c in df.columns:
                label_col = c
                break
    if label_col is None:
        raise ValueError(f"[ERROR] No label column found in {path}.")

    print("[INFO] Detected label_col:", label_col)

    # keep only A/B
    df[label_col] = df[label_col].astype(str)
    df = df[df[label_col].isin(["A", "B"])].copy()

    return df


def baseline_one(df, model_name, base_model, n_splits=5, y_is_int=False):
    # infer cols
    label_col, group_col, cat_cols, num_cols = infer_feature_cols(df)

    # preprocessing
    pre = build_preprocessor(cat_cols, num_cols)

    # pipeline
    pipe = Pipeline([
        ("pre", pre),
        ("clf", base_model),
    ])

    # evaluate
    acc_m, acc_s, f1_m, f1_s, auc_m, auc_s = eval_groupkfold(
        df, label_col, group_col, pipe, n_splits=n_splits
    )

    out = {
        "Model": model_name,
        "Mode": "Baseline",
        "Acc_mean": acc_m,
        "Acc_std": acc_s,
        "MacroF1_mean": f1_m,
        "MacroF1_std": f1_s,
        "AUC_mean": auc_m,
        "AUC_std": auc_s,
        "n_samples": len(df),
        "n_cat_cols": len(cat_cols),
        "n_num_cols": len(num_cols),
        "Params": str(base_model.get_params()),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0", required=True)
    ap.add_argument("--f1", required=True)
    ap.add_argument("--f0f1", required=True)
    ap.add_argument("--out", default="baseline_results_ablation.csv")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--no_ablation", action="store_true", help="Disable ablation drop of landing/location columns")
    ap.add_argument(
        "--models",
        default="logreg,rf",
        help=(
            "Comma-separated models to run: logreg,rf,svm,xgb. "
            "Default is 'logreg,rf' for speed."
        ),
    )
    args = ap.parse_args()

    ablation = not args.no_ablation

    datasets = {
        "F0": load_and_filter(args.f0, force_label="Winner", ablation=ablation),
        "F1": load_and_filter(args.f1, force_label="Winner", ablation=ablation),
        "F0+F1": load_and_filter(args.f0f1, force_label="Winner", ablation=ablation),
    }

    results = []

    for fs_name, df in datasets.items():
        print("\n====================")
        print("FeatureSet:", fs_name, "rows:", len(df))

        want = {m.strip().lower() for m in args.models.split(",") if m.strip()}

        # Logistic Regression baseline
        if "logreg" in want:
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            r = baseline_one(df, f"{fs_name}+LogReg", lr, n_splits=args.splits)
            results.append(r)
            print("[BASE]", r["Model"], "MacroF1:", round(r["MacroF1_mean"], 3), "Acc:", round(r["Acc_mean"], 3))

        # Random Forest baseline
        if "rf" in want:
            rf = RandomForestClassifier(class_weight="balanced", random_state=42)
            r = baseline_one(df, f"{fs_name}+RF", rf, n_splits=args.splits)
            results.append(r)
            print("[BASE]", r["Model"], "MacroF1:", round(r["MacroF1_mean"], 3), "Acc:", round(r["Acc_mean"], 3))

        # SVM baseline (can be slow)
        if "svm" in want:
            svm = SVC(class_weight="balanced", probability=True, random_state=42)
            r = baseline_one(df, f"{fs_name}+SVM", svm, n_splits=args.splits)
            results.append(r)
            print("[BASE]", r["Model"], "MacroF1:", round(r["MacroF1_mean"], 3), "Acc:", round(r["Acc_mean"], 3))

        # XGBoost baseline (optional)
        if "xgb" in want:
            if XGB_OK:
                label_col, _, _, _ = infer_feature_cols(df)
                df2 = df.copy()
                df2[label_col] = (df2[label_col].astype(str) == POS_LABEL).astype(int)

                xgb = XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                    tree_method="hist",
                    device="cpu",
                )

                r = baseline_one(df2, f"{fs_name}+XGB", xgb, n_splits=args.splits)
                results.append(r)
                print("[BASE]", r["Model"], "MacroF1:", round(r["MacroF1_mean"], 3), "Acc:", round(r["Acc_mean"], 3))
            else:
                print("[SKIP] XGBoost not installed.")

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)

    print("\nSaved baseline results ->", args.out)
    print(out_df[["Model", "Acc_mean", "MacroF1_mean", "AUC_mean"]].sort_values("MacroF1_mean", ascending=False))


if __name__ == "__main__":
    main()
