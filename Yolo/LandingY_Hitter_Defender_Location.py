import os
import re
import cv2
import numpy as np
import pandas as pd


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2) ** 0.5
    pts = []
    if dist == 0:
        return img
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        pts.append((x, y))

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        for k, p in enumerate(pts):
            s = e
            e = p
            if k % 2 == 1:
                cv2.line(img, s, e, color, thickness)
    return img


# =========================
# CONFIG (EDIT THESE)
# =========================
WIDTH, HEIGHT = 1280, 720
CASE = 1

csvPath = r"C:\Users\user\Badminton\src\TrackNetV2-pytorch-option\golfdb_G3_fold5_iter3000_val_test_hitter_vote_roundhead_vote_backhand_vote_ballheight_vote_LXY.csv"
labelsDir = r"C:\Users\user\Badminton\src\yolov5\runs\detect\exp6\labels"     # txt labels live here
imagesDir = r"C:\Users\user\Badminton\src\yolov5\runs\detect\exp6"            # images live here (no "labels" inside)
outDir    = rf"C:\Users\user\Badminton\src\yolov5\runs\detect\case{CASE}"     # output

os.makedirs(outDir, exist_ok=True)


# =========================
# LOAD CSV
# =========================
df = pd.read_csv(csvPath)

# required columns
need_cols = ["VideoName", "HitFrame", "Hitter", "LandingX", "LandingY"]
for c in need_cols:
    if c not in df.columns:
        raise ValueError(f"CSV missing column: {c}. Found columns: {list(df.columns)}")

df_videoname = df["VideoName"]
df_hitframe  = df["HitFrame"]
df_hitter    = df["Hitter"]
df_landingx  = df["LandingX"]
df_landingy  = df["LandingY"]


# =========================
# GET IMAGE LIST (exp6 output)
# =========================
yoloImg = sorted([f for f in os.listdir(imagesDir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
if len(yoloImg) == 0:
    raise ValueError(f"No images found in imagesDir: {imagesDir}")

# IMPORTANT:
# This script assumes yoloImg[i] corresponds to row i of df.
# If your exp6 image naming/order doesn't match df rows, you'll need a mapping.
# (But this version will at least NOT crash and will fill zeros safely.)

N = len(df)

# =========================
# BUILD personList aligned to df rows
# =========================
# classes: A=0, B=1 (your YOLO labels)
personList = [{'A': [0, 0, 0, 0], 'B': [0, 0, 0, 0]} for _ in range(N)]

ct_missing_or_short = 0

for i in range(N):
    s1 = str(df_videoname.iloc[i]).split('.')[0]           # 00001
    hf = int(df_hitframe.iloc[i])
    s2 = f"{hf:05d}"                                       # 00056

    tp = os.path.join(labelsDir, f"{s1}_{s2}.txt")         # exp6/labels/00001_00056.txt

    lines = []
    try:
        with open(tp, "r") as f:
            lines = f.readlines()
    except Exception as e:
        # missing txt is common; keep boxes as zeros
        print("[WARN] missing label:", tp, "|", e)

    if len(lines) < 2:
        ct_missing_or_short += 1

    for l in lines:
        l = l.strip()
        if not l:
            continue

        cj = l.split()[0]  # class id as string
        nums = [float(a) for a in re.findall(r"\d+\.\d+", l)]
        if len(nums) < 4:
            continue

        cx, cy, w, h = nums[:4]
        x1 = round(WIDTH * cx) - round(WIDTH * w) // 2
        y1 = round(HEIGHT * cy) - round(HEIGHT * h) // 2
        x2 = round(WIDTH * cx) + round(WIDTH * w) // 2
        y2 = round(HEIGHT * cy) + round(HEIGHT * h) // 2

        area = max(0, x2 - x1) * max(0, y2 - y1)

        if cj == "0":  # A
            old = personList[i]["A"]
            old_area = max(0, old[2] - old[0]) * max(0, old[3] - old[1])
            if area > old_area:
                personList[i]["A"] = [x1, y1, x2, y2]
        elif cj == "1":  # B
            old = personList[i]["B"]
            old_area = max(0, old[2] - old[0]) * max(0, old[3] - old[1])
            if area > old_area:
                personList[i]["B"] = [x1, y1, x2, y2]

print("number of length of lines less than 2:", ct_missing_or_short)
print("length of person list:", len(personList))


# =========================
# OUTPUT ARRAYS (pre-allocate so length always matches df)
# =========================
radius = 5
thickness = 2

newY = [0] * N
hitterlocationX = [0] * N
hitterlocationY = [0] * N
defenderlocationX = [0] * N
defenderlocationY = [0] * N


# =========================
# MAIN LOOP (draw + compute)
# =========================
for i in range(N):
    # landing coords must be int for OpenCV
    try:
        ballx = int(round(float(df_landingx.iloc[i])))
        bally = int(round(float(df_landingy.iloc[i])))
    except:
        ballx, bally = 0, 0

    A = personList[i]["A"]
    B = personList[i]["B"]

    A_cx = (A[0] + A[2]) // 2
    A_cy = (A[1] + A[3]) // 2
    B_cx = (B[0] + B[2]) // 2
    B_cy = (B[1] + B[3]) // 2

    # read image safely
    img_name = yoloImg[i] if i < len(yoloImg) else None
    img_path = os.path.join(imagesDir, img_name) if img_name else None
    im = cv2.imread(img_path) if img_path else None

    if im is None:
        # keep zeros and continue
        if img_path:
            print("[WARN] can't read image:", img_path)
        newY[i] = 0
        hitterlocationX[i] = hitterlocationY[i] = 0
        defenderlocationX[i] = defenderlocationY[i] = 0
        continue

    # compute hitter/defender boxes
    h = str(df_hitter.iloc[i]).strip()

    if h == "A":
        h_box = A
        d_box = B
    else:
        h_box = B
        d_box = A

    h_x1, h_y1, h_x2, h_y2 = h_box
    d_x1, d_y1, d_x2, d_y2 = d_box

    # choose near x-edge; y is bottom of bbox
    if (h_x1, h_y1, h_x2, h_y2) == (0, 0, 0, 0) or ballx == 0:
        hitter_x = hitter_y = 0
    else:
        hitter_x = h_x1 if abs(ballx - h_x1) < abs(ballx - h_x2) else h_x2
        hitter_y = h_y2

    if (d_x1, d_y1, d_x2, d_y2) == (0, 0, 0, 0) or ballx == 0:
        defender_x = defender_y = 0
    else:
        defender_x = d_x1 if abs(ballx - d_x1) < abs(ballx - d_x2) else d_x2
        defender_y = d_y2

    hitterlocationX[i] = int(hitter_x)
    hitterlocationY[i] = int(hitter_y)
    defenderlocationX[i] = int(defender_x)
    defenderlocationY[i] = int(defender_y)

    # draw squares
    if hitter_x != 0:
        im = cv2.rectangle(im, (hitter_x-5, hitter_y-5), (hitter_x+5, hitter_y+5), (0,118,238), -1)
    if defender_x != 0:
        im = cv2.rectangle(im, (defender_x-5, defender_y-5), (defender_x+5, defender_y+5), (238,178,0), -1)

    # CASE logic -> compute projected Y and center_coordinates
    center_coordinates = None

    if CASE == 1:
        if A_cx + A_cy + B_cx + B_cy == 0 or ballx == 0:
            newY[i] = 0
        else:
            if h == "A":
                newY[i] = int(A[3])
                center_coordinates = (ballx, int(A[3]))
            else:
                newY[i] = int(B[3])
                center_coordinates = (ballx, int(B[3]))
    else:
        if A_cx + A_cy + B_cx + B_cy == 0 or ballx == 0:
            newY[i] = 0
        else:
            distA = np.sqrt((ballx - A_cx)**2 + (bally - A_cy)**2)
            distB = np.sqrt((ballx - B_cx)**2 + (bally - B_cy)**2)
            distA_x = abs(ballx - A_cx)
            distB_x = abs(ballx - B_cx)

            if distA + distA_x <= distB + distB_x:
                newY[i] = int(A[3])
                center_coordinates = (ballx, int(A[3]))
            else:
                newY[i] = int(B[3])
                center_coordinates = (ballx, int(B[3]))

    # draw ball
    im = cv2.circle(im, (ballx, bally), radius, (0, 255, 255), -1)

    # draw projection X + dotted line
    if ballx != 0 and center_coordinates is not None:
        im = cv2.line(im, (center_coordinates[0]-3, center_coordinates[1]-3),
                          (center_coordinates[0]+3, center_coordinates[1]+3), (0,0,0), thickness)
        im = cv2.line(im, (center_coordinates[0]-3, center_coordinates[1]+3),
                          (center_coordinates[0]+3, center_coordinates[1]-3), (0,0,0), thickness)
        im = drawline(im, (ballx, bally), center_coordinates, (0,0,0), thickness, style='dotted', gap=10)

        # optional: draw lines to A/B centers (like your old code)
        im = cv2.line(im, (ballx, bally), (A_cx, A_cy), (139,0,0), thickness)
        im = cv2.line(im, (ballx, bally), (B_cx, B_cy), (237,149,100), thickness)

    # save output
    out_name = img_name if img_name else f"{i:06d}.jpg"
    out_path = os.path.join(outDir, out_name)
    cv2.imwrite(out_path, im)


# =========================
# WRITE CSV
# =========================
df["LandingY"] = newY
df["HitterLocationX"] = hitterlocationX
df["HitterLocationY"] = hitterlocationY
df["DefenderLocationX"] = defenderlocationX
df["DefenderLocationY"] = defenderlocationY

out_csv = f"golfdb_G3_fold5_iter3000_val_test_hitter_vote_roundhead_vote_backhand_vote_ballheight_vote_LX_LY_case{CASE}_HD.csv"
df.to_csv(out_csv, index=False)

print("DONE. Saved CSV:", out_csv)
print("DONE. Saved images to:", outDir)
