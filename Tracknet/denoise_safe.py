import os
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# EDIT PATHS (YOUR SETUP)
# =========================
IN_DIR  = Path(r"C:\Users\user\Badminton\src\TrackNetV2-pytorch-option\runs\detect")  # where *_predict.csv are
OUT_DIR = Path(r"C:\Users\user\Badminton\src\TrackNetV2-pytorch-option\trajectorylatest_csv\denoise")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# DENOISE TUNING
# =========================
MAX_GAP = 12       # fill missing gaps up to this many frames (try 8~20)
SMOOTH_WIN = 9     # smoothing window (odd number: 7,9,11)
MIN_CONF_VISIBLE = None  # set like 0.25 if your csv has conf and you want to drop low conf

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # support both formats:
    # (A) frame_num,visible,x,y
    # (B) Frame,Visibility,X,Y,Time
    colmap = {}
    if "frame_num" in df.columns: colmap["frame_num"] = "Frame"
    if "visible" in df.columns:   colmap["visible"] = "Visibility"
    if "x" in df.columns:         colmap["x"] = "X"
    if "y" in df.columns:         colmap["y"] = "Y"
    df = df.rename(columns=colmap)

    # basic requirements
    need = ["Frame", "Visibility", "X", "Y"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"CSV missing column: {c}. Found: {list(df.columns)}")

    # types
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce").fillna(0).astype(int)
    df["Visibility"] = pd.to_numeric(df["Visibility"], errors="coerce").fillna(0).astype(int)
    df["X"] = pd.to_numeric(df["X"], errors="coerce")
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")

    return df.sort_values("Frame").reset_index(drop=True)

def denoise_one(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    # Optional: if your CSV has "conf" and you want to treat low conf as invisible
    if MIN_CONF_VISIBLE is not None and "conf" in df.columns:
        conf = pd.to_numeric(df["conf"], errors="coerce").fillna(0.0)
        df.loc[conf < MIN_CONF_VISIBLE, "Visibility"] = 0
        df.loc[df["Visibility"] == 0, ["X", "Y"]] = 0

    # mark missing as NaN for interpolation
    miss = (df["Visibility"] == 0) | (df["X"] <= 0) | (df["Y"] <= 0)
    df.loc[miss, ["X", "Y"]] = np.nan

    # interpolate ONLY short gaps
    df["X"] = df["X"].interpolate(method="linear", limit=MAX_GAP, limit_area="inside")
    df["Y"] = df["Y"].interpolate(method="linear", limit=MAX_GAP, limit_area="inside")

    # smooth (ignores NaN)
    win = int(SMOOTH_WIN)
    if win >= 3:
        if win % 2 == 0:
            win += 1
        df["X"] = df["X"].rolling(win, center=True, min_periods=1).median()
        df["Y"] = df["Y"].rolling(win, center=True, min_periods=1).median()

    # update visibility after denoise
    good = df["X"].notna() & df["Y"].notna()
    df["Visibility"] = good.astype(int)

    # finalize output types (ints like original style)
    df["X"] = df["X"].round().fillna(0).astype(int)
    df["Y"] = df["Y"].round().fillna(0).astype(int)

    out_path = OUT_DIR / (csv_path.stem + "_denoise.csv")
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

def main():
    csvs = sorted(IN_DIR.glob("*_predict.csv"))
    print("Found", len(csvs), "predict csv in", IN_DIR)
    if not csvs:
        print("Nothing to denoise. Make sure your files are like 00001_xgg_predict.csv in runs\\detect\\")
        return

    for p in csvs:
        print("Denoising:", p.name)
        try:
            denoise_one(p)
        except Exception as e:
            print("[FAILED]", p.name, "->", e)

    print("\nDONE. Output in:", OUT_DIR)

if __name__ == "__main__":
    main()
