from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():
    weights_path = "best.pt"
    data_yaml = "data.yaml"

    model = YOLO(weights_path)

    print("=" * 50)
    print("CRATER DETECTION MODEL — EVALUATION")
    print("=" * 50)
    print(f"Model      : {weights_path}")
    print(f"Dataset    : {data_yaml}")
    print()

    metrics = model.val(
        data=data_yaml,
        conf=0.001,       # standard eval setting — gives the full PR curve
        iou=0.6,
        max_det=3000,     # critical — matches the dense crater tiles
        plots=True,
        save_json=True,
    )

    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"{'mAP@0.5':<20}: {metrics.box.map50:.4f}")
    print(f"{'mAP@0.5:0.95':<20}: {metrics.box.map:.4f}")
    print(f"{'Precision':<20}: {metrics.box.mp:.4f}")
    print(f"{'Recall':<20}: {metrics.box.mr:.4f}")
    print("=" * 50)

    # Locate the val run folder (Ultralytics auto-names val, val2, val3...)
    save_dir = Path(metrics.save_dir)
    print(f"\nPlots saved to: {save_dir}")

    # Display the key visuals inline for the demo
    for fname, title in [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("PR_curve.png", "Precision-Recall Curve"),
        ("F1_curve.png", "F1 Confidence Curve"),
    ]:
        img_path = save_dir / fname
        if img_path.exists():
            plt.figure(figsize=(8, 8))
            plt.imshow(mpimg.imread(img_path))
            plt.axis("off")
            plt.title(title, fontsize=14)
            plt.tight_layout()
            plt.savefig(f"demo_{fname}", dpi=150, bbox_inches="tight")
            plt.show()

    # Sample predictions vs ground truth, side by side
    for i in range(2):
        labels_path = save_dir / f"val_batch{i}_labels.jpg"
        pred_path = save_dir / f"val_batch{i}_pred.jpg"
        if labels_path.exists() and pred_path.exists():
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            axes[0].imshow(mpimg.imread(labels_path))
            axes[0].set_title("Ground Truth", fontsize=14)
            axes[0].axis("off")
            axes[1].imshow(mpimg.imread(pred_path))
            axes[1].set_title("Model Predictions", fontsize=14)
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(f"demo_comparison_batch{i}.png", dpi=150, bbox_inches="tight")
            plt.show()


if __name__ == "__main__":
    main()




import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

weights_path = "/kaggle/input/datasets/akhilkoneru3/new-test-dataset/best.pt"
model = YOLO(weights_path)

metrics = model.val(
    data="/kaggle/working/data.yaml",
    split="val",          # matches the `val:` key in data.yaml
    conf=0.001,
    iou=0.6,
    max_det=3000,
    imgsz =  1024,
    plots=True,
    
    save_json=True,
    project="/kaggle/working/runs",
)

print(f"mAP@0.5      : {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
print(f"Precision    : {metrics.box.mp:.4f}")
print(f"Recall       : {metrics.box.mr:.4f}")