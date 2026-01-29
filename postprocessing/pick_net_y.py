import cv2

VIDEO = r"C:\Users\user\Badminton\data\part1\val\00001\00001.mp4"
FRAME_TO_SHOW = 200  # any frame where net is visible

net_y = None

def on_mouse(event, x, y, flags, param):
    global net_y
    if event == cv2.EVENT_LBUTTONDOWN:
        net_y = y
        print(f"Clicked NET_Y = {net_y}px")
        print(f"NET_Y normalized = {net_y/param['H']:.4f}")
        print("Press ESC to close.")

cap = cv2.VideoCapture(VIDEO)
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# jump to a frame
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_SHOW)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Failed to read frame.")

cv2.namedWindow("Click on the NET line", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Click on the NET line", on_mouse, {"H": H, "W": W})

while True:
    disp = frame.copy()
    if net_y is not None:
        cv2.line(disp, (0, net_y), (W-1, net_y), (0, 0, 255), 2)
        cv2.putText(disp, f"NET_Y={net_y}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Click on the NET line", disp)
    if cv2.waitKey(30) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
