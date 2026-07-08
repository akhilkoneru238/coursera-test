import argparse

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--x0", type=int, required=True)
    ap.add_argument("--y0", type=int, required=True)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--out", default="check.png")
    args = ap.parse_args()

    strip = Image.open(args.strip).convert("L")
    region = strip.crop((args.x0, args.y0,
                         args.x0 + args.size, args.y0 + args.size))
    canvas = region.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    det = pd.read_csv(args.detections)
    m = ((det["pixel"] >= args.x0) & (det["pixel"] < args.x0 + args.size) &
         (det["scan"] >= args.y0) & (det["scan"] < args.y0 + args.size) &
         (det["conf"] >= args.min_conf))
    sel = det[m]

    for _, r in sel.iterrows():
        cx, cy = r["pixel"] - args.x0, r["scan"] - args.y0
        w, h = r["width_px"], r["height_px"]
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        draw.rectangle(box, outline=(0, 255, 0), width=2)
        draw.text((box[0], max(box[1] - 12, 0)),
                  f"{r['conf']:.2f}", fill=(0, 255, 0))

    canvas.save(args.out)
    print(f"{len(sel)} detections in region -> {args.out}")


if __name__ == "__main__":
    main()
