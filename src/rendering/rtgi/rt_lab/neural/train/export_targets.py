"""Writes PNG screenshots of the training targets (the ground truth of every converted full sequence).

  python -P export_targets.py perf_tests/full --out perf_tests/neural_targets

Per sequence: <name>_target.png (the cleaned 4096 spp ground truth the network is trained against), <name>_raw4096.png (the raw 4096 spp image before the
cleanup), <name>_target_old_atrous.png (the previous cleanup, if reference_atrous_old.npy exists) and <name>_compare.png (raw | target | old side by side).
Tone mapping: exposed = radiance x pi x exposure / RADIANCE_SCALE (the reference exposure of the capture), ACES filmic (Narkowicz), gamma 2.2, times --brightness (default 0.3).
This is a plain visualization, not the engine's tonemapper. The sky (no geometry) is drawn dark blue gray.
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

RADIANCE_SCALE = 10000.0


def tonemap(rgb, exposure, sky, brightness):
    exposed = np.maximum(rgb, 0.0) * (np.pi * exposure / RADIANCE_SCALE) * brightness
    mapped = exposed * (2.51 * exposed + 0.03) / (exposed * (2.43 * exposed + 0.59) + 0.14)  # ACES filmic (Narkowicz)
    srgb = np.clip(mapped, 0.0, 1.0) ** (1.0 / 2.2)
    image = (srgb * 255.0 + 0.5).astype(np.uint8)
    image[sky] = (40, 44, 58)
    return image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="perf_tests/neural_targets")
    ap.add_argument("--only", default="")
    ap.add_argument("--brightness", type=float, default=0.3, help="extra factor on the exposed irradiance (the targets are irradiance, before the albedo: at the capture exposure they are mostly above display white)")
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    os.makedirs(args.out, exist_ok=True)
    for meta_path in sorted(glob.glob(os.path.join(args.folder, "*", "meta.json"))):
        folder = os.path.dirname(meta_path)
        name = os.path.basename(folder)
        if name.endswith(".tmp") or (only and not any(s in name for s in only)):
            continue
        meta = json.load(open(meta_path))
        exposure = max(meta["reference_exposure"], 1e-3)
        sky = np.load(os.path.join(folder, "depth.npy"))[..., 0] <= 0
        images = [("target", np.load(os.path.join(folder, "reference.npy"))), ("raw4096", np.load(os.path.join(folder, "reference_raw.npy")))]
        old = os.path.join(folder, "reference_atrous_old.npy")
        if os.path.exists(old):
            images.append(("target_old_atrous", np.load(old)))
        tonemapped = {}
        for suffix, rgb in images:
            tonemapped[suffix] = tonemap(rgb[..., :3], exposure, sky, args.brightness)
            Image.fromarray(tonemapped[suffix]).save(os.path.join(args.out, f"{name}_{suffix}.png"))
        order = [s for s in ("raw4096", "target", "target_old_atrous") if s in tonemapped]
        Image.fromarray(np.concatenate([tonemapped[s] for s in order], axis=1)).save(os.path.join(args.out, f"{name}_compare.png"))
        print(name, "exposure", round(exposure, 3), "->", ", ".join(f"{name}_{s}.png" for s, _ in images), flush=True)


if __name__ == "__main__":
    main()
