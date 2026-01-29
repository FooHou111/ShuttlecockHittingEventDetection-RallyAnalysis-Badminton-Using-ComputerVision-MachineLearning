import os
import cv2
import pandas as pd

csvPath  = r"C:\Users\user\Badminton\src\postprocess\golfdb_G3_fold5_iter3000_val_test_X.csv"
savePath = r"C:\Users\user\Badminton\src\postprocess\HitFrame_yolo"
videoDir = r"C:\Users\user\Badminton\src\preprocess\val_test_xgg_latest"  # <-- your _xgg.mp4 folder

os.makedirs(savePath, exist_ok=True)

data = pd.read_csv(csvPath)
vns  = data["VideoName"].tolist()   # e.g. 00001.mp4
hits = data["HitFrame"].tolist()    # frame index

print("rows:", len(vns), len(hits))

saved = 0
missing_video = 0
failed_read = 0

for i in range(len(vns)):
    base = os.path.splitext(vns[i])[0]          # "00001"
    real_name = f"{base}_xgg.mp4"               # "00001_xgg.mp4"
    videoPath = os.path.join(videoDir, real_name)

    if not os.path.exists(videoPath):
        missing_video += 1
        continue

    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        failed_read += 1
        continue

    hf = int(hits[i])
    cap.set(cv2.CAP_PROP_POS_FRAMES, hf)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        failed_read += 1
        continue

    out_name = f"{base}_{hf:05d}.jpg"           # 00001_00123.jpg
    out_path = os.path.join(savePath, out_name)

    ok = cv2.imwrite(out_path, frame)
    if ok:
        saved += 1

print("saved:", saved, "missing_video:", missing_video, "failed_read:", failed_read)
print("output files now:", len(os.listdir(savePath)))
