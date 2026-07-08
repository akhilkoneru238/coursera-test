from ultralytics import YOLO
import argparse
import cv2
from pathlib import Path                                 


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=str)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="inference_results")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=1024)
    args = ap.parse_args()

    model = YOLO(args.weights)                            
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images: {args.source} | Resolution {args.imgsz}")
    results = model.predict(source=args.source,            
                            conf=args.conf,
                            imgsz=args.imgsz,
                            save=False, device="cpu")

    total = 0
    for result in results:
        img_path = Path(result.path)
        boxes = result.boxes
        crater_count = len(boxes)
        total += crater_count

        print(f"Image {img_path.name} Detected craters: {crater_count}")
        annotated_frame = result.plot(labels=True, boxes=True)

        cv2.putText(annotated_frame, f"craters: {crater_count}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0, 255, 0),2)                                  
        output_file = out_dir / f"pred_{img_path.name}"  
        cv2.imwrite(str(output_file), annotated_frame)

    print(f"Done. {total} craters across {len(results)} images -> {out_dir}/")


if __name__ == "__main__":                                 
    main()
