import torch
import torchvision

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from models.tracknet import TrackNet
from utils.general import get_shuttle_position


# ===== Anti-false-positive filters (tune these) =====
CONF_THRES = 0.35   # if peak < this, treat as "not visible"
BIN_THRES  = 0.50   # binary threshold for get_shuttle_position (keep same as before)

# ROI in resized heatmap coords (fraction of height/width to KEEP)
ROI_TOP    = 0.12   # ignore top scoreboard/banners
ROI_BOTTOM = 0.98
ROI_LEFT   = 0.06   # ignore extreme left (net post / corner artifacts)
ROI_RIGHT  = 0.94

# Max allowed jump in ORIGINAL pixels (auto-scaled from video size)
MAX_JUMP_FRAC = 0.18  # ~18% of max(w,h). Tune 0.12~0.25


# from yolov5 detect.py
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    parser = ArgumentParser()

    parser.add_argument('--source', type=str, default=ROOT / 'example_dataset/match/videos/1_10_12.mp4', help='Path to video.')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.csv')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[288, 512], help='image size h,w')
    parser.add_argument('--weights', type=str, default=ROOT / 'best.pt', help='Path to trained model weights.')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')

    opt = parser.parse_args()

    return opt


def main(opt):
    # imgsz = [288, 512]
    # imgsz = [360, 640]

    source_name = os.path.splitext(os.path.basename(opt.source))[0]
    b_save_txt = opt.save_txt
    b_view_img = opt.view_img
    d_save_dir = str(opt.project)
    f_weights = str(opt.weights)
    f_source = str(opt.source)
    imgsz = opt.imgsz

    # Build ROI mask in resized heatmap coordinates (imgsz is [h, w])
    roi_mask = np.zeros((imgsz[0], imgsz[1]), dtype=bool)
    y1 = int(imgsz[0] * ROI_TOP)
    y2 = int(imgsz[0] * ROI_BOTTOM)
    x1 = int(imgsz[1] * ROI_LEFT)
    x2 = int(imgsz[1] * ROI_RIGHT)
    roi_mask[y1:y2, x1:x2] = True

    # video_name ---> video_name_pred
    source_name = '{}_predict'.format(source_name)

    # runs/detect
    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)

    # runs/detect/video_name
    img_save_path = '{}/{}'.format(d_save_dir, source_name)
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = TrackNet().to(device)
    model.load_state_dict(torch.load(f_weights, map_location=device))
    model.eval()

    # import ncnn
    # net = ncnn.Net()
    # net.load_param("./pt_30_optimize.ncnn.param")
    # net.load_model("./pt_30_optimize.ncnn.bin")

    vid_cap = cv2.VideoCapture(f_source)
    video_end = False

    video_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('{}/{}.mp4'.format(d_save_dir, source_name), fourcc, fps, (w, h))

    if b_save_txt:
        f_save_txt = open('{}/{}.csv'.format(d_save_dir, source_name), 'w')
        f_save_txt.write('frame_num,visible,x,y\n')

    if b_view_img:
        cv2.namedWindow(source_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(source_name, (w, h))


    prev_visible = 0
    prev_cx, prev_cy = 0, 0
    MAX_JUMP = int(max(w, h) * MAX_JUMP_FRAC)

    count = 0
    while vid_cap.isOpened():
        imgs = []
        for _ in range(3):
            ret, img = vid_cap.read()
            if not ret:
                break
            imgs.append(img)

        # how many real frames we got this step (0~3)
        n_real = len(imgs)
        if n_real == 0:
            break

        # pad to 3 frames by repeating last frame (so model always gets 9 channels)
        while len(imgs) < 3:
            imgs.append(imgs[-1].copy())

        imgs_torch = []
        for img in imgs:
            # https://www.geeksforgeeks.org/converting-an-image-to-a-torch-tensor-in-python/
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_torch = torchvision.transforms.ToTensor()(img).to(device)   # already [0, 1]
            img_torch = torchvision.transforms.functional.resize(img_torch, imgsz, antialias=True)

            imgs_torch.append(img_torch)

        imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)

        preds = model(imgs_torch)
        preds = preds[0].detach().cpu().numpy()  # shape: (3, H, W)
        preds = preds[:n_real]   # only keep predictions for real frames

        for i in range(n_real):
            heat = preds[i].astype(np.float32).copy()  # float heatmap

            # 1) ROI mask: ignore borders/top area
            heat[~roi_mask] = 0.0

            # 2) confidence from peak
            cy_pred, cx_pred = np.unravel_index(np.argmax(heat), heat.shape)
            conf = float(heat[cy_pred, cx_pred])

            if conf < CONF_THRES:
                visible, cx, cy = 0, 0, 0
            else:
                visible = 1
                cx = int(cx_pred * w / imgsz[1])
                cy = int(cy_pred * h / imgsz[0])

                # 3) max-jump gate
                if prev_visible:
                    dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
                    if dist > MAX_JUMP:
                        visible, cx, cy = 0, 0, 0

            if visible:
                cv2.circle(imgs[i], (cx, cy), 8, (0, 0, 255), -1)
                prev_visible, prev_cx, prev_cy = 1, cx, cy
            else:
                prev_visible = 0


            if b_save_txt:
                f_save_txt.write('{},{},{},{}\n'.format(count, visible, cx, cy))

            if b_view_img:
                cv2.imwrite('{}/{}.png'.format(img_save_path, count), imgs[i])
                cv2.imshow(source_name, imgs[i])
                cv2.waitKey(1)

            out.write(imgs[i])
            print("{} ---- visible: {}  cx: {}  cy: {}  conf: {:.3f}".format(count, visible, cx, cy, conf))

            count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if b_save_txt:
        # 每次识别3张，最后可能有1-2张没有识别，补0
        while count < video_len:
            f_save_txt.write('{},0,0,0\n'.format(count))
            count += 1

        f_save_txt.close()

    out.release()
    vid_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
