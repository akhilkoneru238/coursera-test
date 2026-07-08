import argparse
import csv

import numpy as np
import torch
from torchvision.ops import nms
from PIL import Image
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None


def load_strip(path: str) -> np.ndarray:
    try:
        img = Image.open(path).convert("L")
        return np.array(img)
    except Exception:
        import tifffile
        arr = tifffile.imread(path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            lo, hi = np.percentile(arr, (1, 99))
            arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
            arr = arr.astype(np.uint8)
        return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--strip", required=True, help="path to the OHRC .tif strip")
    ap.add_argument("--out", default="detections.csv")
    ap.add_argument("--tile", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--conf", type=float, default=0.35,
                    help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--min-std", type=float, default=2.0,
                    help="skip tiles whose pixel std-dev is below this")
    args = ap.parse_args()

    model = YOLO(args.weights)
    strip = load_strip(args.strip)
    H, W = strip.shape
    step = args.tile - args.overlap
    print(f"Strip: {W} x {H} px | tile={args.tile} overlap={args.overlap} "
          f"conf={args.conf}")

    boxes, scores = [], []
    n_tiles = n_skipped = 0

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            tile = strip[y0:y0 + args.tile, x0:x0 + args.tile]
            if tile.shape[0] < 32 or tile.shape[1] < 32:
                continue
            n_tiles += 1
            if tile.std() < args.min_std:
                n_skipped += 1
                continue

            img = Image.fromarray(tile).convert("RGB")
            r = model.predict(img, conf=args.conf, iou=args.iou,
                              imgsz=args.imgsz, verbose=False)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                boxes.append([x0 + x1, y0 + y1, x0 + x2, y0 + y2])
                scores.append(float(b.conf[0]))

        print(f"  row y0={y0:>6}  cumulative raw detections: {len(boxes)}")

    print(f"Tiles processed: {n_tiles} (skipped {n_skipped} empty/shadow)")

    if not boxes:
        print("No detections. Try lowering --conf.")
        return

    t_boxes = torch.tensor(boxes, dtype=torch.float32)
    t_scores = torch.tensor(scores, dtype=torch.float32)
    keep = nms(t_boxes, t_scores, args.iou)
    t_boxes, t_scores = t_boxes[keep], t_scores[keep]
    print(f"Raw: {len(boxes)}  ->  after global NMS: {len(keep)}")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pixel", "scan", "width_px", "height_px", "conf"])
        for (x1, y1, x2, y2), s in zip(t_boxes.tolist(), t_scores.tolist()):
            w.writerow([
                round((x1 + x2) / 2, 1),
                round((y1 + y2) / 2, 1),
                round(x2 - x1, 1),
                round(y2 - y1, 1),
                round(s, 4)
            ])

    print(f"Wrote {len(keep)} craters -> {args.out}")


if __name__ == "__main__":
    main()
