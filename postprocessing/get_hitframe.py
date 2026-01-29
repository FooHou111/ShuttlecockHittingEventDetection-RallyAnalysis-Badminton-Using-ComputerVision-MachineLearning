import os
import cv2
import pandas as pd

csvPath  = r"C:\Users\user\Badminton\src\preprocess\hitframe_xgg_latest.csv"
savePath = r"C:\Users\user\Badminton\src\postprocess\HitFrame\1"
valPath  = r"C:\Users\user\Badminton\data\part1\val"

os.makedirs(savePath, exist_ok=True)

data = pd.read_csv(csvPath)
vns  = data["VideoName"].tolist()
hits = data["HitFrame"].tolist()

print("rows:", len(vns), len(hits))

saved = 0
missing_videos = 0

for vn, hit in zip(vns, hits):
    folder = os.path.splitext(vn)[0]          # "00001"
    videoPath = os.path.join(valPath, folder, vn)

    if not os.path.exists(videoPath):
        missing_videos += 1
        # print("[missing video]", videoPath)
        continue

    cap = cv2.VideoCapture(videoPath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(hit))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        continue

    # crop (assumes 1280x720 input, like repo)
    frame = frame[:, 280:1280-280]

    out_name = f"{folder}_{int(hit):05d}.jpg"
    out_path = os.path.join(savePath, out_name)
    ok = cv2.imwrite(out_path, frame)
    if ok:
        saved += 1

print("saved:", saved)
print("missing videos:", missing_videos)
print("files in folder:", len([f for f in os.listdir(savePath) if f.lower().endswith('.jpg')]))
