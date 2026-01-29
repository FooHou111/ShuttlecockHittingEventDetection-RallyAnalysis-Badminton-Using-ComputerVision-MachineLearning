import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = r"C:\Users\user\Badminton\src\csv\winner_399_f0_f1.csv"   # change to f0 / f1 / f0_f1 if needed
LABEL_COL = "Winner"
GROUP_COL = "VideoName"
OUT_DIR = r"C:\Users\user\Badminton\src\csv\eda_winner_399"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ---------- Basic info ----------
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Unique videos:", df[GROUP_COL].nunique() if GROUP_COL in df.columns else "N/A")

# ---------- Label distribution ----------
if LABEL_COL in df.columns:
    vc = df[LABEL_COL].astype(str).value_counts()
    print("\nWinner counts:", vc.to_dict())

    plt.figure()
    vc.plot(kind="bar")
    plt.title("Winner distribution (A vs B)")
    plt.xlabel("Winner")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "winner_distribution.png"), dpi=200)
    plt.close()

# ---------- Rows per video ----------
if GROUP_COL in df.columns:
    per_vid = df.groupby(GROUP_COL).size()
    print("\nShots per video min/mean/max:",
          int(per_vid.min()), float(per_vid.mean()), int(per_vid.max()))

    plt.figure()
    plt.hist(per_vid.values, bins=30)
    plt.title("Samples per video (histogram)")
    plt.xlabel("Rows per video")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rows_per_video_hist.png"), dpi=200)
    plt.close()

# ---------- Missing values ----------
na = df.isna().mean().sort_values(ascending=False)
top_na = na.head(25)
print("\nTop missing-rate columns (top 25):")
print(top_na)

plt.figure(figsize=(8, 6))
top_na[::-1].plot(kind="barh")
plt.title("Top-25 missing-rate columns")
plt.xlabel("Missing rate")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "missing_rate_top25.png"), dpi=200)
plt.close()

# ---------- Categorical distributions (if exist) ----------
for c in ["BallType", "ball_zone_name", "Hitter"]:
    if c in df.columns:
        vc = df[c].astype(str).value_counts().head(15)
        plt.figure(figsize=(8, 4))
        vc.plot(kind="bar")
        plt.title(f"Top categories: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"cat_{c}.png"), dpi=200)
        plt.close()

# ---------- Numeric histograms (choose common ones if present) ----------
num_candidates = [
    "dt_seconds","ball_speed_px_s","dist_hitter_ball","dist_HD","dist_AB",
    "LandingX","LandingY","ball_x_norm","ball_y_norm"
]
for c in num_candidates:
    if c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) == 0:
            continue
        plt.figure()
        plt.hist(x.values, bins=40)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"hist_{c}.png"), dpi=200)
        plt.close()

# ---------- Correlation heatmap (numeric only, limited) ----------
num_df = df.select_dtypes(include=[np.number]).copy()
if LABEL_COL in num_df.columns:
    num_df = num_df.drop(columns=[LABEL_COL], errors="ignore")
if num_df.shape[1] > 1:
    # keep at most 25 numeric cols for readable plot
    cols = list(num_df.columns)[:25]
    corr = num_df[cols].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation heatmap (first 25 numeric cols)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "corr_heatmap_25.png"), dpi=200)
    plt.close()

# ---------- Leakage check ----------
leak_cols = [c for c in df.columns if "Winner" in c and c != LABEL_COL]
print("\n[Leakage check] Columns containing 'Winner' besides label:", leak_cols)

print("\nSaved EDA figures to:", OUT_DIR)
