# mvp.py (CLEAN VERSION)
# Goal: Add BALL position zone per shot (ball_zone_name + ball_zone) using TrackNet near HitFrame
# - Keeps ALL shots (Winner can be X)
# - Minimizes extra columns to avoid confusion
# - Tries to fill ball_zone_name for most rows (clamp y + optional forward fill within a video)

import os
import json
from glob import glob

import cv2
import numpy as np
import pandas as pd


# =========================
# (A) EDIT THESE PATHS
# =========================
ORI_CSV = r"C:\Users\user\Badminton\src\csv\golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case2.csv"

TRACKNET_DIR = r"C:\Users\user\Badminton\src\TrackNetV2-pytorch-option\trajectorylatest_csv\denoise"
YOLO_DIR     = r"C:\Users\user\Badminton\src\yolov5\runs\detect\exp11\labels"
VIDEO_DIR    = r"C:\Users\user\Badminton\data\part1\val"

ZONE_JSON = r"C:\Users\user\Badminton\src\zone_lines_00001.json"  # set None if you don't have
H_FRAME_DEFAULT = 720

FPS = 30.0

OUT_CSV  = r"C:\Users\user\Badminton\src\csv\golfdb_F1_ballzone_clean.csv"


# =========================
# (B) FILENAME PATTERNS
# =========================
def tracknet_path_from_videoname(videoname: str) -> str:
    stem = os.path.splitext(videoname)[0]
    return os.path.join(TRACKNET_DIR, f"{stem}_xgg_predict_denoise.csv")

def video_path(videoname: str) -> str:
    stem = os.path.splitext(videoname)[0]
    return os.path.join(VIDEO_DIR, stem, f"{stem}.mp4")


# =========================
# Helpers
# =========================
def get_video_wh(vpath: str, default=(1280, 720)):
    if not os.path.exists(vpath):
        return default
    cap = cv2.VideoCapture(vpath)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or default[0]
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or default[1]
    cap.release()
    return (w, h)

def safe_norm(x, denom):
    if pd.isna(x) or denom <= 0:
        return np.nan
    return float(x) / float(denom)

def dist(x1, y1, x2, y2):
    if np.any(pd.isna([x1, y1, x2, y2])):
        return np.nan
    return float(np.hypot(float(x1) - float(x2), float(y1) - float(y2)))


# =========================
# YOLO (optional for dist_AB etc.)
# =========================
yolo_index_cache = {}

def get_available_yolo_frames(video_name: str):
    stem = os.path.splitext(video_name)[0].replace("_xgg", "")
    if stem in yolo_index_cache:
        return yolo_index_cache[stem]

    files = glob(os.path.join(YOLO_DIR, f"{stem}_*.txt"))
    frames = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]  # 00001_00466
        try:
            fr = int(base.split("_")[1])
            frames.append(fr)
        except:
            pass
    frames = sorted(frames)
    yolo_index_cache[stem] = frames
    return frames

def yolo_txt_nearest(video_name: str, hitframe: int, max_diff=5):
    stem = os.path.splitext(video_name)[0].replace("_xgg", "")
    frames = get_available_yolo_frames(video_name)
    if not frames:
        return None, None
    best = min(frames, key=lambda f: abs(f - hitframe))
    if abs(best - hitframe) > max_diff:
        return None, None
    path = os.path.join(YOLO_DIR, f"{stem}_{best:05d}.txt")
    return path, best

def parse_yolo_centers(txt_path: str, W: int, H: int):
    if (txt_path is None) or (not os.path.exists(txt_path)):
        return []
    centers = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, bw, bh = map(float, parts[1:5])
            cx = x * W
            cy = y * H
            centers.append((cls, cx, cy, bw * W, bh * H))
    return centers

def assign_A_B(centers):
    if len(centers) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    centers = sorted(centers, key=lambda t: t[1])  # by cx
    _, ax, ay, _, _ = centers[0]
    _, bx, by, _, _ = centers[1]
    return (ax, ay, bx, by)


# =========================
# TrackNet
# =========================
def normalize_tracknet_columns(tdf: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in tdf.columns:
        cl = c.lower()
        if cl == "frame": rename_map[c] = "Frame"
        if cl == "x": rename_map[c] = "X"
        if cl == "y": rename_map[c] = "Y"
        if cl in ("vis", "visibility"): rename_map[c] = "Visibility"
    if rename_map:
        tdf = tdf.rename(columns=rename_map)
    return tdf

def get_ball_at_hit(tdf: pd.DataFrame, hit_frame: int, window=12):
    """
    Larger window = higher chance to find ball near hit
    """
    if tdf is None or len(tdf) == 0:
        return np.nan, np.nan
    if "Frame" not in tdf.columns or "X" not in tdf.columns or "Y" not in tdf.columns:
        return np.nan, np.nan

    lo = hit_frame - window
    hi = hit_frame + window
    cand = tdf[(tdf["Frame"] >= lo) & (tdf["Frame"] <= hi)].copy()
    if len(cand) == 0:
        return np.nan, np.nan

    cand = cand.dropna(subset=["X", "Y"])
    cand = cand[~((cand["X"] == 0) & (cand["Y"] == 0))]
    if "Visibility" in cand.columns:
        vis = cand[cand["Visibility"] == 1]
        if len(vis) > 0:
            cand = vis
    if len(cand) == 0:
        return np.nan, np.nan

    cand["d"] = (cand["Frame"] - hit_frame).abs()
    best = cand.sort_values("d").iloc[0]
    return float(best["X"]), float(best["Y"])


# =========================
# Zones (ball position)
# =========================
ZONE_LINES_FALLBACK = {
    "TOP_BACK": 345,
    "TOP_MID": 389,
    "TOP_FRONT": 455,
    "BOT_FRONT": 475,
    "BOT_MID": 579,
    "BOT_BACK": 671
}

def load_zone_lines(path):
    if (path is None) or (not os.path.exists(path)):
        print("[WARN] ZONE_JSON not found. Using fallback ZONE_LINES.")
        return ZONE_LINES_FALLBACK
    with open(path, "r") as f:
        z = json.load(f)
    return z

ZLINES = load_zone_lines(ZONE_JSON)

zone_to_id = {
    "front_left":0, "front_center":1, "front_right":2,
    "mid_left":3,   "mid_center":4,   "mid_right":5,
    "back_left":6,  "back_center":7,  "back_right":8,
}

def zone_from_6lines(xn, yn, zlines, H):
    """
    Clamp y into [TOP_BACK, BOT_BACK] so zone is almost always produced if xn/yn exist.
    """
    if pd.isna(xn) or pd.isna(yn):
        return np.nan

    xi = int(float(xn) * 3)
    xi = max(0, min(2, xi))
    x_name = ["left", "center", "right"][xi]

    y = float(yn) * float(H)

    TOP_BACK  = zlines["TOP_BACK"]
    TOP_MID   = zlines["TOP_MID"]
    TOP_FRONT = zlines["TOP_FRONT"]
    BOT_FRONT = zlines["BOT_FRONT"]
    BOT_MID   = zlines["BOT_MID"]
    BOT_BACK  = zlines["BOT_BACK"]

    NET_MID = 0.5 * (TOP_FRONT + BOT_FRONT)

    # CLAMP instead of drop
    y = max(TOP_BACK, min(BOT_BACK, y))

    if y < NET_MID:
        if y <= TOP_MID:
            depth = "back"
        elif y <= TOP_FRONT:
            depth = "mid"
        else:
            depth = "front"
    else:
        if y < BOT_FRONT:
            depth = "front"
        elif y < BOT_MID:
            depth = "mid"
        else:
            depth = "back"

    return f"{depth}_{x_name}"


# =========================
# Step 1: Load base CSV
# =========================
df = pd.read_csv(ORI_CSV)

df["HitFrame"] = pd.to_numeric(df["HitFrame"], errors="coerce")
df["ShotSeq"]  = pd.to_numeric(df["ShotSeq"],  errors="coerce")
df = df.dropna(subset=["HitFrame", "ShotSeq"]).copy()
df["HitFrame"] = df["HitFrame"].astype(int)
df["ShotSeq"]  = df["ShotSeq"].astype(int)
df = df.sort_values(["VideoName", "ShotSeq"]).copy()


# =========================
# Step 2: Add BallX/BallY (TrackNet) + Ax/Ay/Bx/By (YOLO optional)
# =========================
ballx = []
bally = []
Ax = []; Ay = []; Bx = []; By = []

track_cache = {}
wh_cache = {}

for _, r in df.iterrows():
    vn = r["VideoName"]
    hf = int(r["HitFrame"])

    if vn not in wh_cache:
        wh_cache[vn] = get_video_wh(video_path(vn), default=(1280, H_FRAME_DEFAULT))
    W, H = wh_cache[vn]

    # TrackNet
    tpath = tracknet_path_from_videoname(vn)
    if tpath not in track_cache:
        if os.path.exists(tpath):
            tdf = pd.read_csv(tpath)
            tdf = normalize_tracknet_columns(tdf)
            track_cache[tpath] = tdf
        else:
            track_cache[tpath] = None

    tdf = track_cache[tpath]
    if tdf is None:
        bx, by = np.nan, np.nan
    else:
        bx, by = get_ball_at_hit(tdf, hf, window=12)

    # YOLO (optional)
    ypath, _ = yolo_txt_nearest(vn, hf, max_diff=5)
    centers = parse_yolo_centers(ypath, W, H) if ypath else []
    ax, ay, bx_p, by_p = assign_A_B(centers)

    ballx.append(bx); bally.append(by)
    Ax.append(ax); Ay.append(ay); Bx.append(bx_p); By.append(by_p)

df["BallX"] = ballx
df["BallY"] = bally
df["Ax"] = Ax; df["Ay"] = Ay; df["Bx"] = Bx; df["By"] = By


# =========================
# Step 3: Normalize + Ball zone (BALL POSITION)
# =========================
WZ, HZ = 1280.0, float(H_FRAME_DEFAULT)

df["ball_x_norm"] = df["BallX"].apply(lambda v: safe_norm(v, WZ)).clip(0, 1)
df["ball_y_norm"] = df["BallY"].apply(lambda v: safe_norm(v, HZ)).clip(0, 1)

df["ball_zone_name"] = df.apply(
    lambda r: zone_from_6lines(r["ball_x_norm"], r["ball_y_norm"], ZLINES, H_FRAME_DEFAULT),
    axis=1
)

# OPTIONAL: fill missing ball_zone_name within each video (coaching timeline)
# This only fills when TrackNet missed at some shots.
# df["ball_zone_name"] = df.groupby("VideoName")["ball_zone_name"].ffill().bfill()

df["ball_zone"] = df["ball_zone_name"].map(zone_to_id)


# =========================
# Step 4: Speed / tempo features
# =========================
df["next_HitFrame"] = df.groupby("VideoName")["HitFrame"].shift(-1)
df["next_BallX"] = df.groupby("VideoName")["BallX"].shift(-1)
df["next_BallY"] = df.groupby("VideoName")["BallY"].shift(-1)

df["dt_next"] = df["next_HitFrame"] - df["HitFrame"]
df.loc[df["dt_next"] <= 0, "dt_next"] = np.nan
df["dt_seconds"] = df["dt_next"] / float(FPS)

df["ball_speed_px_s"] = df.apply(
    lambda r: (dist(r["BallX"], r["BallY"], r["next_BallX"], r["next_BallY"]) / r["dt_seconds"])
    if pd.notna(r["dt_seconds"]) and r["dt_seconds"] > 0 else np.nan,
    axis=1
)

def hitter_ball_dist(row):
    h = str(row.get("Hitter", "")).upper()
    if h == "A":
        return dist(row["Ax"], row["Ay"], row["BallX"], row["BallY"])
    if h == "B":
        return dist(row["Bx"], row["By"], row["BallX"], row["BallY"])
    return np.nan

df["dist_hitter_ball"] = df.apply(hitter_ball_dist, axis=1)
df["dist_AB"] = df.apply(lambda r: dist(r["Ax"], r["Ay"], r["Bx"], r["By"]), axis=1)




# --- PREVIOUS-shot features (usable on last-shot Winner A/B rows) ---
df = df.sort_values(["VideoName","ShotSeq"]).copy()

df["prev_HitFrame"] = df.groupby("VideoName")["HitFrame"].shift(1)
df["prev_BallX"]    = df.groupby("VideoName")["BallX"].shift(1)
df["prev_BallY"]    = df.groupby("VideoName")["BallY"].shift(1)

df["dt_prev"] = df["HitFrame"] - df["prev_HitFrame"]
df.loc[df["dt_prev"] <= 0, "dt_prev"] = np.nan

df["dt_prev_seconds"] = df["dt_prev"] / float(FPS)

df["ball_speed_prev_px_s"] = df.apply(
    lambda r: dist(r["prev_BallX"], r["prev_BallY"], r["BallX"], r["BallY"]) / r["dt_prev"] * float(FPS)
    if pd.notna(r["dt_prev"]) else np.nan,
    axis=1
)



# =========================
# Step 5: Save a CLEAN output CSV
# =========================
keep_cols = [
    "VideoName","ShotSeq","HitFrame","Winner","Hitter",
    "BallX","BallY","ball_x_norm","ball_y_norm",
    "ball_zone_name","ball_zone",
    "next_HitFrame","dt_seconds","ball_speed_px_s",
    "Ax","Ay","Bx","By","dist_hitter_ball","dist_AB","dt_prev_seconds",
    "ball_speed_prev_px_s"
]
keep_cols = [c for c in keep_cols if c in df.columns]

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df[keep_cols].to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)
print("TrackNet missing rate (BallX):", df["BallX"].isna().mean())
print("ball_zone coverage:", df["ball_zone_name"].notna().mean())

OUT_SHOW = r"C:\Users\user\Badminton\src\csv\golfdb_F1_ballzone_presentation.csv"

df_show = df.copy()

# 1) Force integers for id / frame-like columns (keep NaN using Int64)
int_cols = [
    "ShotSeq","HitFrame","ball_zone",
    "next_HitFrame"
]
for c in int_cols:
    if c in df_show.columns:
        df_show[c] = pd.to_numeric(df_show[c], errors="coerce").round().astype("Int64")

# 2) Round ALL float columns to 5 decimals (xx.xxxxx)
float_cols = df_show.select_dtypes(include=["float", "float64"]).columns
df_show[float_cols] = df_show[float_cols].round(5)

# Optional: if you want dt_seconds / speeds only 2 decimals (more readable)
# for c in ["dt_seconds", "ball_speed_px_s", "dist_hitter_ball", "dist_AB"]:
#     if c in df_show.columns:
#         df_show[c] = df_show[c].round(2)

df_show.to_csv(OUT_SHOW, index=False)
print("Saved presentation:", OUT_SHOW)
