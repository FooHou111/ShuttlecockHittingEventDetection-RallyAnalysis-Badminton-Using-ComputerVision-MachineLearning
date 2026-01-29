from pathlib import Path
import subprocess

REPO_DIR = Path(r"C:\Users\user\Badminton\src\TrackNetV2-pytorch-option")
VIDEO_DIR = Path(r"C:\Users\user\Badminton\src\preprocess\val_test_xgg_latest")  # <-- change if needed
WEIGHTS = REPO_DIR / "tf2torch" / "track.pt"
OUT_DIR = REPO_DIR / "trajectorylatest_csv"   # all CSVs + mp4 outputs will go here

OUT_DIR.mkdir(parents=True, exist_ok=True)

videos = sorted(VIDEO_DIR.glob("*.mp4"))
print("Found", len(videos), "videos")

for vp in videos:
    print("\n=== Running on:", vp.name)
    cmd = [
        "python", str(REPO_DIR / "detect.py"),
        "--source", str(vp),
        "--weights", str(WEIGHTS),
        "--project", str(OUT_DIR),
        "--save-txt",
    ]
    # run detect.py inside REPO_DIR so relative paths work
    subprocess.run(cmd, cwd=str(REPO_DIR), check=False)

print("\nDONE. Check:", OUT_DIR)
