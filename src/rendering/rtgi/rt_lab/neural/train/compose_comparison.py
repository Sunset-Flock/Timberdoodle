"""Lays the npz files of render_comparison.py out as comparison sheets (one PNG per view, no GPU needed).

  python -P compose_comparison.py --out sheets --runs "S 3x3 stats=cmp/S" "L lite=cmp/L" --views cam1_s235,cam4_s99 --levels 4,16,64 --training cam1_s235

Per view one sheet: for every history length two rows. Top: ground truth, unfiltered temporal, geometry only filter, variance guide, then one column
per network. Bottom: the matching absolute perceptual log error |log1p(8 e) - log1p(8 e_ref)| (mean over rgb, same color scale in every panel,
white = 0.35 and above) with the mean squared error of the interior pixels in the label. The baselines are taken from the first run.
"""

import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw
import matplotlib

matplotlib.use("Agg")
from matplotlib import cm

RADIANCE_SCALE = 1e4
ERROR_MAX = 0.35
SCALE = 2  # nearest neighbor upscale of the 192 px crops


def display(rad, exposure):
    e = np.maximum(rad, 0.0) * (math.pi * float(exposure) / RADIANCE_SCALE)
    v = 1.0 - np.exp(-e)
    return (np.clip(v, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


def error_map(rad, ref, exposure):
    f = math.pi * float(exposure) / RADIANCE_SCALE
    p = np.log1p(8.0 * np.maximum(rad * f, 0.0))
    q = np.log1p(8.0 * np.maximum(ref * f, 0.0))
    err = np.abs(p - q).mean(axis=-1)
    return err


def colorize(err):
    return (cm.magma(1.0 - np.clip(err / ERROR_MAX, 0, 1))[..., :3] * 255).astype(np.uint8)


def panel(img, label, bar=18):
    im = Image.fromarray(img).resize((img.shape[1] * SCALE, img.shape[0] * SCALE), Image.NEAREST)
    canvas = Image.new("RGB", (im.width, im.height + bar), (20, 20, 20))
    canvas.paste(im, (0, bar))
    ImageDraw.Draw(canvas).text((4, 3), label, fill=(255, 255, 255))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="label=folder of render_comparison.py output")
    ap.add_argument("--views", required=True)
    ap.add_argument("--levels", default="4,16,64")
    ap.add_argument("--training", default="", help="comma separated views that are TRAINING views (labelled in the sheet, the others are held out)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = [r.split("=", 1) for r in args.runs]
    training = set(args.training.split(","))
    for view in args.views.split(","):
        rows = []
        for level in [int(x) for x in args.levels.split(",")]:
            data = [np.load(os.path.join(folder, f"{view}_f{level}.npz")) for _, folder in runs]
            base = data[0]
            exposure = base["exposure"]
            mask = base["mask"][..., 0] > 0

            def mse(rad):
                f = math.pi * float(exposure) / RADIANCE_SCALE
                d = np.log1p(8.0 * np.maximum(rad * f, 0)) - np.log1p(8.0 * np.maximum(base["reference"] * f, 0))
                return float((d ** 2)[mask].mean())

            entries = [("ground truth", base["reference"]), ("temporal only", base["temporal"]), ("geometry only", base["geometry"]), ("variance guide", base["guide"])]
            entries += [(label, d["neural"]) for (label, _), d in zip(runs, data)]
            top = [panel(display(rad, exposure), f"{name}  ({level} frames)" if name == "ground truth" else name) for name, rad in entries]
            bottom = []
            for name, rad in entries:
                if name == "ground truth":
                    bottom.append(panel(np.full(rad.shape, 255, np.uint8), f"error scale: white 0, dark {ERROR_MAX}"))
                else:
                    bottom.append(panel(colorize(error_map(rad, base["reference"], exposure)), f"{name} mse {mse(rad):.4f}"))
            for strip in (top, bottom):
                row = Image.new("RGB", (sum(p.width for p in strip), strip[0].height))
                x = 0
                for p in strip:
                    row.paste(p, (x, 0))
                    x += p.width
                rows.append(row)
        sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + 24), (20, 20, 20))
        ImageDraw.Draw(sheet).text((4, 6), f"{view}: " + ("TRAINING view" if view in training else "held-out view"), fill=(255, 255, 255))
        y = 24
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        path = os.path.join(args.out, f"comparison_{view}.png")
        sheet.save(path)
        print("saved", path, sheet.size)


if __name__ == "__main__":
    main()
