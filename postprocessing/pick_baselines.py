import cv2

VIDEO = r"C:\Users\user\Badminton\data\part1\val\00001\00001.mp4"
FRAME_TO_SHOW = 200

points = []  # store (x,y)

def on_mouse(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 1:
            print(f"Clicked TOP baseline y = {y}px (far side)")
            print("Now click BOTTOM baseline (near camera).")
        elif len(points) == 2:
            print(f"Clicked BOTTOM baseline y = {y}px (near side)")
            print("Press ESC to close.")

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_SHOW)
ok, frame = cap.read()
cap.release()
if not ok:
    raise RuntimeError("Failed to read frame")

H, W = frame.shape[:2]

cv2.namedWindow("Click TOP baseline then BOTTOM baseline", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Click TOP baseline then BOTTOM baseline", on_mouse)

while True:
    disp = frame.copy()

    # draw clicked points
    for i, (px, py) in enumerate(points):
        cv2.circle(disp, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(disp, f"{i+1}:{py}", (px+10, py-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.line(disp, (0, py), (W-1, py), (0, 0, 255), 2)

    cv2.imshow("Click TOP baseline then BOTTOM baseline", disp)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()

if len(points) >= 2:
    top_y = points[0][1]
    bot_y = points[1][1]
    print("\n=== COPY THESE INTO mvp.py ===")
    print(f"COURT_TOP_Y = {top_y}")
    print(f"COURT_BOT_Y = {bot_y}")
else:
    print("Not enough clicks.")
