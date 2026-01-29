import cv2

VIDEO = r"C:\Users\user\Badminton\data\part1\val\00001\00001.mp4"

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print("FPS:", fps)
print("Resolution:", w, "x", h)
print("Frames:", n)
print("Duration(s):", n / fps if fps else None)
