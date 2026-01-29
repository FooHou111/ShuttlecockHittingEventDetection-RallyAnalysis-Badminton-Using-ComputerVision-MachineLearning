import os
import re
import cv2
from pathlib import Path

# =======================
# EDIT THESE PATHS
# =======================
VIDEO_PATH = r"C:\Users\user\Badminton\src\preprocess\val_test_xgg_latest\00001_xgg.mp4"
HITFRAME_IMG_DIR = r"C:\Users\user\Badminton\src\yolo_hitframes\images"  # contains 00001_00056.jpg ...
OUT_DIR = r"C:\Users\user\Badminton\src\hit_eval_windows\00001"

# offsets to export (frames)
OFFSETS = [-5, -3, 0, 3, 5]

# how many hitframes to sample (set None to export all for this video)
MAX_HITS = 15

# =======================
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

vid_name = Path(VIDEO_PATH).stem  # 00001_xgg
m = re.match(r"(\d+)", vid_name)
if not m:
    raise ValueError("Could not infer video id from VIDEO_PATH name.")
video_id = m.group(1)  # "00001"

# collect hitframes from filenames like 00001_00056.jpg
hitframes = []
pat = re.compile(rf"^{video_id}_(\d+)\.(jpg|png)$", re.IGNORECASE)

for fn in os.listdir(HITFRAME_IMG_DIR):
    mm = pat.match(fn)
    if mm:
        hitframes.append(int(mm.group(1)))

hitframes = sorted(set(hitframes))
if not hitframes:
    raise FileNotFoundError(f"No hitframe images found for video {video_id} in {HITFRAME_IMG_DIR}")

if MAX_HITS is not None:
    hitframes = hitframes[:MAX_HITS]

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError("Cannot open video: " + VIDEO_PATH)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def read_frame(idx):
    idx = max(0, min(idx, frame_count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return ok, frame

for hf in hitframes:
    for off in OFFSETS:
        fidx = hf + off
        ok, frame = read_frame(fidx)
        if not ok:
            continue
        out = Path(OUT_DIR) / f"{video_id}_hf{hf:05d}_f{fidx:05d}_off{off:+d}.jpg"
        cv2.imwrite(str(out), frame)

cap.release()
print("Done. Saved windows to:", OUT_DIR)
print("HitFrames used:", hitframes)
