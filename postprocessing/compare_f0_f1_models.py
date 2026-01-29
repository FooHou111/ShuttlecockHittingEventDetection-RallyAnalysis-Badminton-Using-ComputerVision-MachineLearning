# compare_f0_f1_models.py
# Compare Winner prediction performance: F0 vs F1 vs F0+F1
# - F0: author/original golfdb features
# - F1: engineered features from your mvp.py (ball_zone, dt_seconds, speeds, distances, etc.)
# - Splits by VideoName using GroupShuffleSplit (avoids leakage)

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# =========================
# EDIT PATHS
# =========================
F0_CSV = r"C:\Users\user\Badminton\src\csv\golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case2.csv"

# Use YOUR F1 output (clean or presentation both ok)
F1_CSV = r"C:\Users\user\Badminton\src\csv\golfdb_F1_ballzone_presentation.csv"
# or:
# F1_CSV = r"C:\Users\user\Badminton\src\csv\golfdb_F1_ballzone_clean.csv"

TEST_SIZE = 0.2
RANDOM_SEED = 42

# =========================
# Load
# =========================
f0 = pd.read_csv(F0_CSV)
f1 = pd.read_csv(F1_CSV)

# Ensure merge keys exist
for k in ["VideoName", "ShotSeq", "HitFrame"]:
    if k not in f0.columns:
        raise ValueError(f"F0 missing key: {k}")
    if k not in f1.columns:
        raise ValueError(f"F1 missing key: {k}")

# Force numeric keys for safe merge
for df in (f0, f1):
    df["ShotSeq"] = pd.to_numeric(df["ShotSeq"], errors="coerce")
    df["HitFrame"] = pd.to_numeric(df["HitFrame"], errors="coerce")

f0 = f0.dropna(subset=["ShotSeq", "HitFrame"]).copy()
f1 = f1.dropna(subset=["ShotSeq", "HitFrame"]).copy()

f0["ShotSeq"] = f0["ShotSeq"].astype(int)
f0["HitFrame"] = f0["HitFrame"].astype(int)
f1["ShotSeq"] = f1["ShotSeq"].astype(int)
f1["HitFrame"] = f1["HitFrame"].astype(int)

# Merge (keep Winner from F0)
# If your F1 file also has Winner, we drop it to avoid confusion.
if "Winner" in f1.columns:
    f1 = f1.drop(columns=["Winner"])

df = f0.merge(f1, on=["VideoName", "ShotSeq", "HitFrame"], how="inner")

print("Merged rows:", len(df))
print("Videos:", df["VideoName"].nunique())

# Keep only A/B for modelling
df = df[df["Winner"].isin(["A", "B"])].copy()
print("A/B rows:", len(df))

if len(df) < 20:
    print("[WARN] Very small A/B sample size. Results will be unstable (but ok for demo).")

# =========================
# Feature sets
# =========================
label_col = "Winner"
group_col = "VideoName"

# F0 (baseline/original)
F0_cat = ["Hitter", "RoundHead", "Backhand", "BallHeight", "BallType"]
F0_num = [
    "ShotSeq", "LandingX", "LandingY",
    "HitterLocationX", "HitterLocationY",
    "DefenderLocationX", "DefenderLocationY"
]

# F1 (engineered)
# NOTE: treat ball_zone as categorical (it's an id 0..8)
F1_cat = ["ball_zone"]
F1_num = [
    "dt_seconds", "ball_speed_px_s",
    "dist_hitter_ball", "dist_AB",
    "ball_x_norm", "ball_y_norm"
]

def keep_exist(cols):
    return [c for c in cols if c in df.columns]

F0_cat, F0_num, F1_cat, F1_num = map(keep_exist, [F0_cat, F0_num, F1_cat, F1_num])

print("\nUsing columns:")
print("F0_cat:", F0_cat)
print("F0_num:", F0_num)
print("F1_cat:", F1_cat)
print("F1_num:", F1_num)

# =========================
# Train/Eval helper
# =========================
def run_lr(title, cat_cols, num_cols):
    X = df[cat_cols + num_cols].copy()
    y = df[label_col].astype(str).values
    groups = df[group_col].astype(str).values

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), num_cols)
        ],
        remainder="drop"
    )

    model = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=700, class_weight="balanced"))
    ])

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    tr, te = next(splitter.split(X, y, groups=groups))

    model.fit(X.iloc[tr], y[tr])
    pred = model.predict(X.iloc[te])

    acc = accuracy_score(y[te], pred)
    mf1 = f1_score(y[te], pred, average="macro")
    cm = confusion_matrix(y[te], pred, labels=["A", "B"])

    print("\n==============================")
    print(title)
    print("n_train:", len(tr), "n_test:", len(te), "n_total:", len(df))
    print("Accuracy:", round(acc, 4))
    print("Macro-F1:", round(mf1, 4))
    print("Confusion matrix (rows=true, cols=pred) [A,B]:\n", cm)
    print("\nReport:\n", classification_report(y[te], pred, labels=["A", "B"]))

    return acc, mf1, cm

# =========================
# Run 3 comparisons
# =========================
run_lr("F0 only (baseline)", F0_cat, F0_num)
run_lr("F1 only (engineered)", F1_cat, F1_num)
run_lr("F0 + F1 (combined)", F0_cat + F1_cat, F0_num + F1_num)
