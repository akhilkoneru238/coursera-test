"""
Crater/boulder detection on Chandrayaan-2 OHRC slices (fixed version of practice1.py).

LEARNING NOTES — the whole pipeline is 5 steps:
  1. Load trained weights into a YOLOv5 model
  2. Set thresholds (conf, iou) that filter raw predictions
  3. Run inference on each image
  4. Convert predictions to a table of boxes
  5. Draw the boxes and save results

Usage:
    python run_crater_detection.py --source slices --out detections --limit 20
    python run_crater_detection.py --source slices/slice_x2000_y4000.png

Weights: best.pt (Gurveer05/moon-crater-boulder-detection-yolov5, downloaded from HF)
"""
import argparse
import os
import sys
import pathlib

# Workaround for a common Windows crash: two copies of Intel's OpenMP runtime
# (one from torch, one from numpy/cv2) refuse to coexist unless this is set.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# torch >= 2.6 blocks pickled checkpoints by default; best.pt needs this.
# Only keep this for weights you trust.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2
import pandas as pd
import yolov5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt", help="path to model weights")
    ap.add_argument("--source", default="slices", help="image file or folder of .png slices")
    ap.add_argument("--out", default="detections", help="output folder")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--limit", type=int, default=0, help="max images to process (0 = all)")
    ap.add_argument("--skip-existing", action="store_true", help="skip images already processed")
    args = ap.parse_args()

    # STEP 1 — load the model.
    # best.pt is a checkpoint: the YOLOv5 architecture + weights learned by
    # training on labelled OHRC crater images. yolov5.load() rebuilds the
    # network and copies those weights in. device="cpu" because no CUDA GPU.
    #
    # FIX (Windows only): the checkpoint was saved on Linux, where any stored
    # path becomes a pathlib.PosixPath. Unpickling a PosixPath on Windows
    # raises "cannot instantiate 'PosixPath' on your system". Temporarily
    # aliasing PosixPath -> WindowsPath lets the unpickler build the object
    # (the resulting path string is unused by this script anyway).
    _posix_path_backup = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath

    try:
        model = yolov5.load(args.weights, device="cpu")
    finally:
        pathlib.PosixPath = _posix_path_backup

    # STEP 2 — thresholds. The raw network outputs THOUSANDS of candidate
    # boxes per image. Two filters clean them up:
    #   conf: keep only boxes the model is >= this sure about.
    #         Raise it -> fewer, more reliable boxes. Lower -> more, noisier.
    #   iou:  used by NMS (non-maximum suppression). When two boxes overlap
    #         more than this fraction, they're assumed to be the SAME crater
    #         and only the higher-confidence one survives.
    model.conf = args.conf
    model.iou = args.iou
    print(f"Model loaded. Classes: {model.names}", flush=True)

    # Build the list of images. A single OHRC strip is ~12000 x 90000 px —
    # far too big for the network (it resizes inputs to 640px). That's why
    # the strip was pre-cut into 1000x1000 tiles: small craters would vanish
    # if we shrank the whole strip at once.
    if os.path.isfile(args.source):
        files = [args.source]
    else:
        files = sorted(
            os.path.join(args.source, f)
            for f in os.listdir(args.source)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
    if args.limit:
        files = files[: args.limit]

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "detections.csv")
    all_rows = []

    for i, fpath in enumerate(files, 1):
        name = os.path.basename(fpath)

        image = cv2.imread(fpath)
        if image is None:
            print(f"[{i}/{len(files)}] SKIP unreadable: {name}", flush=True)
            continue

        h, w = image.shape[:2]

        # Run inference
        results = model(fpath, size=640)

        # Get detections
        df = results.pandas().xyxy[0]

        # -----------------------------
        # Save YOLO label file
        # -----------------------------
        label_dir = os.path.join(args.out, "labels")
        os.makedirs(label_dir, exist_ok=True)

        label_file = os.path.join(
            label_dir,
            os.path.splitext(name)[0] + ".txt"
        )

        with open(label_file, "w") as file:

            for _, row in df.iterrows():

                xmin = row["xmin"]
                ymin = row["ymin"]
                xmax = row["xmax"]
                ymax = row["ymax"]

                x_center = ((xmin + xmax) / 2) / w
                y_center = ((ymin + ymax) / 2) / h
                box_w = (xmax - xmin) / w
                box_h = (ymax - ymin) / h

                # Automatically assign class id
                class_id = int(row["class"])

                file.write(
                    f"{class_id} "
                    f"{x_center:.6f} "
                    f"{y_center:.6f} "
                    f"{box_w:.6f} "
                    f"{box_h:.6f}\n"
                )

        # -----------------------------
        # Extra statistics
        # -----------------------------
        df["width_px"] = df["xmax"] - df["xmin"]
        df["height_px"] = df["ymax"] - df["ymin"]
        df["diameter_px"] = (df["width_px"] + df["height_px"]) / 2

        df.insert(0, "slice", name)
        all_rows.append(df)

        counts = df["name"].value_counts().to_dict()
        print(f"[{i}/{len(files)}] {name}: {counts}", flush=True)

    # analysed at once (crater counts, size distribution, etc.).
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        if os.path.exists(csv_path):
            old = pd.read_csv(csv_path)
            combined = pd.concat([old, combined], ignore_index=True).drop_duplicates()
        combined.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} ({len(combined)} rows total)", flush=True)

        craters = combined[combined["name"] == "crater"]
        if len(craters):
            print(f"Total craters detected: {len(craters)}")
            print(f"Crater diameter (px) — min: {craters['diameter_px'].min():.1f}, "
                  f"max: {craters['diameter_px'].max():.1f}, "
                  f"mean: {craters['diameter_px'].mean():.1f}")


if __name__ == "__main__":
    main()