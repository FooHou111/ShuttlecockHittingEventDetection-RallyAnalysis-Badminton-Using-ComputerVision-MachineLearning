"""build_rally_level_winner_datasets.py

Creates rally-level datasets (1 row per VideoName) for Winner prediction.

Inputs (per-shot / per-hit, many rows per VideoName):
  1) F0 CSV    : baseline fields + Winner
  2) F1 CSV    : engineered fields + Winner
  3) F0+F1 CSV : merged fields + Winner

Outputs (rally-level, one row per VideoName):
  - winner_F0_rally.csv
  - winner_F1_rally.csv
  - winner_F0F1_rally.csv

Key idea
--------
Rally analysis should use ALL shots; winner prediction should be rally-level.
This script aggregates per-shot features into rally-level indicators, which:
  - avoids the dt_seconds/ball_speed all-NaN issue (last shot has no next)
  - is much more aligned with "rally analysis" in your project title

Run (Windows example)
---------------------
python build_rally_level_winner_datasets.py --f0 <path_to_F0.csv> --f1 <path_to_F1.csv> --f0f1 <path_to_F0F1.csv> --outdir <output_folder>

If you omit args, it will try to use the example filenames in the same folder.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _winner_per_video(g: pd.DataFrame) -> Optional[str]:
    """Winner label for a rally/video.

    Your per-shot table typically has Winner = 'X' for intermediate shots,
    and Winner = 'A'/'B' at the point-ending (last) shot.

    We take the last non-X in time order.
    """
    if "Winner" not in g.columns:
        return None

    gg = g.sort_values([c for c in ["ShotSeq", "HitFrame"] if c in g.columns])
    w = gg["Winner"].astype(str)
    w = w[w.isin(["A", "B"])]
    if len(w) == 0:
        return None
    return w.iloc[-1]


def _safe_stats(arr: np.ndarray) -> Dict[str, float]:
    """Return mean/median/std/min/max for a numeric array; NaN if empty."""
    arr = arr.astype(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(np.nanmean(arr)),
        "median": float(np.nanmedian(arr)),
        "std": float(np.nanstd(arr, ddof=0)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def _valuecount_features(
    series: pd.Series,
    prefix: str,
    normalize: bool = True,
    allowed_values: Optional[List] = None,
) -> Dict[str, float]:
    """Create frequency features from a categorical series."""
    s = series.dropna()
    if allowed_values is not None:
        s = s[s.isin(allowed_values)]

    if len(s) == 0:
        # still create zeros for allowed_values if provided
        out: Dict[str, float] = {}
        if allowed_values is not None:
            for v in allowed_values:
                out[f"{prefix}_{v}"] = 0.0
        return out

    vc = s.value_counts(normalize=normalize)
    out = {f"{prefix}_{k}": float(v) for k, v in vc.items()}

    # ensure all allowed_values appear
    if allowed_values is not None:
        for v in allowed_values:
            out.setdefault(f"{prefix}_{v}", 0.0)

    return out


def _tempo_from_hitframes(g: pd.DataFrame, fps: float) -> Dict[str, float]:
    """Compute tempo from consecutive HitFrame differences (independent of shift columns)."""
    if "HitFrame" not in g.columns:
        return {"tempo_mean_s": np.nan, "tempo_median_s": np.nan, "tempo_std_s": np.nan}

    gg = g.dropna(subset=["HitFrame"]).sort_values("HitFrame")
    hf = gg["HitFrame"].astype(float).values
    if hf.size < 2:
        return {"tempo_mean_s": np.nan, "tempo_median_s": np.nan, "tempo_std_s": np.nan}

    dt_frames = np.diff(hf)
    dt_frames = dt_frames[dt_frames > 0]
    dt_s = dt_frames / float(fps)
    st = _safe_stats(dt_s)
    return {
        "tempo_mean_s": st["mean"],
        "tempo_median_s": st["median"],
        "tempo_std_s": st["std"],
    }


def _speed_from_ballxy(g: pd.DataFrame, fps: float) -> Dict[str, float]:
    """Compute ball speed (px/s) from consecutive BallX/BallY (independent of shift columns)."""
    if not {"BallX", "BallY", "HitFrame"}.issubset(set(g.columns)):
        return {
            "speed_mean_px_s": np.nan,
            "speed_median_px_s": np.nan,
            "speed_std_px_s": np.nan,
            "speed_max_px_s": np.nan,
        }

    gg = g.dropna(subset=["BallX", "BallY", "HitFrame"]).sort_values("HitFrame")
    if len(gg) < 2:
        return {
            "speed_mean_px_s": np.nan,
            "speed_median_px_s": np.nan,
            "speed_std_px_s": np.nan,
            "speed_max_px_s": np.nan,
        }

    x = gg["BallX"].astype(float).values
    y = gg["BallY"].astype(float).values
    hf = gg["HitFrame"].astype(float).values

    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.hypot(dx, dy)

    dt_frames = np.diff(hf)
    valid = dt_frames > 0
    if valid.sum() == 0:
        return {
            "speed_mean_px_s": np.nan,
            "speed_median_px_s": np.nan,
            "speed_std_px_s": np.nan,
            "speed_max_px_s": np.nan,
        }

    speed_px_s = (dist[valid] / dt_frames[valid]) * float(fps)
    st = _safe_stats(speed_px_s)
    return {
        "speed_mean_px_s": st["mean"],
        "speed_median_px_s": st["median"],
        "speed_std_px_s": st["std"],
        "speed_max_px_s": st["max"],
    }


def build_rally_level(df: pd.DataFrame, fps: float, feature_set_name: str) -> pd.DataFrame:
    """Aggregate per-shot df into rally-level df (1 row per VideoName)."""

    if "VideoName" not in df.columns:
        raise ValueError("Input must contain 'VideoName' column")

    # ensure numeric where expected
    for c in ["ShotSeq", "HitFrame"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    rows: List[Dict] = []

    # Ball zone bins we want consistent columns for (0..8)
    zone_ids = list(range(9))

    for vid, g in df.groupby("VideoName"):
        out: Dict[str, float] = {"VideoName": vid}

        # label
        w = _winner_per_video(g)
        out["Winner_video"] = w

        # basic rally length
        out["rally_len"] = int(len(g))

        # hitter usage
        if "Hitter" in g.columns:
            out.update(_valuecount_features(g["Hitter"].astype(str), "hitter_prop", True, ["A", "B"]))
            # also counts (non-normalized) are sometimes useful
            out.update(_valuecount_features(g["Hitter"].astype(str), "hitter_cnt", False, ["A", "B"]))

        # rally duration (from hitframes)
        if "HitFrame" in g.columns:
            gg = g.dropna(subset=["HitFrame"]).sort_values("HitFrame")
            if len(gg) >= 2:
                dur_frames = float(gg["HitFrame"].iloc[-1] - gg["HitFrame"].iloc[0])
                out["rally_dur_frames"] = dur_frames
                out["rally_dur_s"] = dur_frames / float(fps)
            else:
                out["rally_dur_frames"] = np.nan
                out["rally_dur_s"] = np.nan

        # tempo + speed computed robustly
        out.update(_tempo_from_hitframes(g, fps=fps))
        out.update(_speed_from_ballxy(g, fps=fps))

        # zone distribution (if exists)
        if "ball_zone" in g.columns:
            bz = pd.to_numeric(g["ball_zone"], errors="coerce")
            out.update(_valuecount_features(bz, "ballzone_prop", True, zone_ids))
            out.update(_valuecount_features(bz, "ballzone_cnt", False, zone_ids))

        # zone name distribution (optional, for explainability)
        if "ball_zone_name" in g.columns:
            zn = g["ball_zone_name"].astype(str)
            # keep only known pattern with '_' to reduce garbage
            zn = zn[zn.str.contains("_")]
            vc = zn.value_counts(normalize=True)
            # only keep top 12 to avoid huge columns
            top = vc.head(12)
            for k, v in top.items():
                out[f"ballzoneName_prop_{k}"] = float(v)

        # numeric summaries for common columns (if present)
        num_cols = [
            "LandingX", "LandingY",
            "HitterLocationX", "HitterLocationY",
            "DefenderLocationX", "DefenderLocationY",
            "dist_hitter_ball", "dist_AB",
            "ball_x_norm", "ball_y_norm",
        ]
        for c in num_cols:
            if c in g.columns:
                arr = pd.to_numeric(g[c], errors="coerce").values.astype(float)
                st = _safe_stats(arr)
                out[f"{c}_mean"] = st["mean"]
                out[f"{c}_std"] = st["std"]

        # categorical distributions for baseline columns (if present)
        cat_cols = ["BallType", "RoundHead", "Backhand", "BallHeight"]
        for c in cat_cols:
            if c in g.columns:
                s = pd.to_numeric(g[c], errors="coerce")
                # GolfDB often uses small integer codes; keep top values
                vc = s.dropna().astype(int).value_counts(normalize=True)
                top = vc.head(12)
                for k, v in top.items():
                    out[f"{c}_prop_{k}"] = float(v)

        rows.append(out)

    rally = pd.DataFrame(rows)

    # Drop videos without A/B winner label (can keep if you want, but models need labels)
    rally = rally[rally["Winner_video"].isin(["A", "B"])].copy()

    # Make sure we have consistent column order
    first_cols = ["VideoName", "Winner_video", "rally_len", "rally_dur_s"]
    cols = first_cols + [c for c in rally.columns if c not in first_cols]
    rally = rally[cols]

    print(f"[{feature_set_name}] Rally-level samples (A/B only):", len(rally), "videos")
    print(f"[{feature_set_name}] Class balance:\n", rally["Winner_video"].value_counts())

    return rally


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0", default="golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case2.csv")
    ap.add_argument("--f1", default="golfdb_F1_.csv")
    ap.add_argument("--f0f1", default="golfdb_F0_F1_.csv")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load
    df_f0 = pd.read_csv(args.f0)
    df_f1 = pd.read_csv(args.f1)
    df_f0f1 = pd.read_csv(args.f0f1)

    rally_f0 = build_rally_level(df_f0, fps=args.fps, feature_set_name="F0")
    rally_f1 = build_rally_level(df_f1, fps=args.fps, feature_set_name="F1")
    rally_f0f1 = build_rally_level(df_f0f1, fps=args.fps, feature_set_name="F0F1")

    out_f0 = os.path.join(args.outdir, "winner_F0_rally.csv")
    out_f1 = os.path.join(args.outdir, "winner_F1_rally.csv")
    out_f0f1 = os.path.join(args.outdir, "winner_F0F1_rally.csv")

    rally_f0.to_csv(out_f0, index=False)
    rally_f1.to_csv(out_f1, index=False)
    rally_f0f1.to_csv(out_f0f1, index=False)

    print("[DONE] Wrote:")
    print(" -", out_f0)
    print(" -", out_f1)
    print(" -", out_f0f1)


if __name__ == "__main__":
    main()
