import argparse

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True,
                    help="CSV from predict_strip.py")
    ap.add_argument("--grid", required=True,
                    help="OHRC *_g_grd_*.csv geolocation grid")
    ap.add_argument("--out", default="craters_georeferenced.csv")
    ap.add_argument("--gsd", type=float, default=0.25,
                    help="ground sampling distance in m/px")
    args = ap.parse_args()

    grid = pd.read_csv(args.grid)
    det = pd.read_csv(args.detections)
    print(f"{len(det)} detections, {len(grid)} grid points")

    pts = grid[["Pixel", "Scan"]].values
    q = det[["pixel", "scan"]].values

    for col, vals in (("lon", grid["Longitude"].values),
                      ("lat", grid["Latitude"].values)):
        lin = griddata(pts, vals, q, method="linear")
        nan = np.isnan(lin)
        if nan.any():
            lin[nan] = griddata(pts, vals, q[nan], method="nearest")
        det[col] = lin

    det["diameter_m"] = det[["width_px", "height_px"]].mean(axis=1) * args.gsd

    det = det.sort_values("conf", ascending=False)
    det.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(det.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
