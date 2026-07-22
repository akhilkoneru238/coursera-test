"""
YOLO training for lunar CRATER detection — augmentations tuned for THIS dataset.

Context that drives the choices: tiles are near-nadir (top-down), essentially
grayscale, densely packed with craters across a wide size range (~80/image), at
416 or 640 px. That makes some augmentations very valuable and others useless
or harmful. Values below that DIFFER from Ultralytics defaults are marked [*].

  GEOMETRIC  — use aggressively; craters have no canonical orientation:
    degrees=180    [*] full rotation. Craters are ~rotationally symmetric and an
                       orbital tile has no "up", so every rotation is a real view.
    fliplr=0.5         horizontal flip — always safe.
    flipud=0.5     [*] vertical flip — also safe here (no inherent up); ~doubles data.
    scale=0.5          strong scale jitter — critical: crater diameters span a huge range.
    translate=0.1      positional robustness.
    mosaic=1.0         stitches 4 tiles → more context + scale variety per step;
                       excellent for dense small-object detection.
    close_mosaic=10    turn mosaic OFF for the final 10 epochs so boxes tighten
                       on the real layout (avoids leftover mosaic-edge artifacts).
    perspective=0.0    tiles are ~nadir → keep off (no realistic perspective warp).
    shear=0.0          little real shear in orbital tiles → keep off.

  PHOTOMETRIC — color is meaningless; brightness/contrast is the whole game:
    hsv_h=0.0      [*] grayscale imagery → hue jitter is pure noise (default 0.015).
    hsv_s=0.0      [*] almost no saturation to vary (default 0.7).
    hsv_v=0.4          BIG lever: crater appearance depends on sun angle / exposure;
                       value jitter simulates different illumination → better generalization.

  REGULARIZERS:
    erasing=0.4        random erasing — mild occlusion robustness, fine with dense labels.
    mixup=0.0          off: blending whole images muddies dense crater scenes.
                       Try 0.1 ONLY if you see overfitting.
    copy_paste=0.0     needs segmentation masks; this is a bbox dataset → not usable.

  CAVEAT (illumination): flips/rotations randomize the apparent sun/shadow
  direction. Shadows are a genuine crater cue, so this adds robustness but throws
  away that prior. If you specifically want the model to respect illumination
  geometry, lower flipud and degrees (e.g. flipud=0.0, degrees=15).

Requirements:  pip install ultralytics
Run:           python train_craters.py
"""

from ultralytics import YOLO

# ----------------------------- config -----------------------------
DATA   = "curated_crater_dataset/data.yaml"   # swap to large_crater_dataset/data.yaml
MODEL  = "yolo11s.pt"   # n=fastest · s=balanced · m=stronger on dense small craters
IMGSZ  = 640            # >=640 matters a lot for tiny craters; use 768/1024 if GPU allows
EPOCHS = 120
BATCH  = 16             # or -1 to auto-fit GPU memory
DEVICE = 0              # 0 = first GPU · "cpu" · [0,1] for multi-GPU

# Augmentations (the part you asked about) — tuned per the notes above.
AUG = dict(
    # geometric
    degrees=180.0, fliplr=0.5, flipud=0.5,
    scale=0.5, translate=0.1, shear=0.0, perspective=0.0,
    mosaic=1.0, close_mosaic=10,
    mixup=0.0, copy_paste=0.0,
    # photometric (brightness only)
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.4,
    # regularizer
    erasing=0.4,
)

def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, device=DEVICE,
        patience=25,            # early-stop if no val gain for 25 epochs
        cos_lr=True,            # cosine LR decay — smoother convergence
        optimizer="auto",
        # multi_scale=True,     # optional: trains across image sizes for extra scale robustness (slower)
        project="runs_crater", name="yolo11s_aug", seed=0,
        **AUG,
    )

    # Evaluate on the held-out test split.
    metrics = model.val(data=DATA, imgsz=IMGSZ, split="test")
    print(f"test mAP50-95: {metrics.box.map:.4f}   mAP50: {metrics.box.map50:.4f}")

    # TIP — see how well it does on LARGE craters specifically:
    # Ultralytics reports AP by area (small/medium/large) in the val output;
    # if large-crater recall is what you care about, watch the "large" row and
    # consider training/finetuning on large_crater_dataset/data.yaml.

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Inference with test-time augmentation (flips/scales) for a small accuracy bump:
#   model = YOLO("runs_crater/yolo11s_aug/weights/best.pt")
#   results = model.predict("path/to/tiles", imgsz=640, augment=True, conf=0.25)
#
# Quick augmentation priority if you only tweak a few things:
#   1) mosaic=1.0 + close_mosaic=10     (biggest win for dense craters)
#   2) degrees=180, flipud=0.5, fliplr  (free data, craters are orientation-free)
#   3) scale=0.5                        (handles the wide crater size range)
#   4) hsv_v=0.4 with hsv_h=hsv_s=0     (illumination, not color)
#   5) higher imgsz (768/1024)          (recovers tiny craters)
# ---------------------------------------------------------------------------
