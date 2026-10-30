"""Contact sheet of a snapshot sequence (.pt from convert_dumps.py --snapshots) to LOOK at the training data, plus sanity numbers.

  python -P preview_sequence.py perf_tests/snap/*.pt           -> perf_tests/snap/preview_<name>.png

Top row: the snapshots after n = 1, 4, 8, 16, 32, 64 accumulated frames. Bottom row: the raw 4096 spp reference, the cleaned reference
(one a-trous pass, strict variance guide) and the difference (x 8). Tonemap = a plain display mapping of the exposed radiance (not the
engine's tonemapper): v = exposed / (1 + exposed), gamma 2.2. Printed: error of every snapshot against the cleaned reference
(MSE of the perceptual log error, the training metric) and how much the cleanup changed the reference.
"""

import math
import os
import sys

import numpy as np
import torch
from PIL import Image

RADIANCE_SCALE = 1.0e4
TILE = 256


GAIN = 1.0


def tonemap(rgb, exposure):
    exposed = np.clip(rgb, 0, None) * (math.pi * exposure / RADIANCE_SCALE) * GAIN
    mapped = exposed / (1.0 + exposed)
    return (np.clip(mapped, 0, 1) ** (1 / 2.2) * 255.0).astype(np.uint8)


def perceptual(rgb, exposure):
    return np.log1p(8.0 * np.clip(rgb, 0, None) * (math.pi * exposure / RADIANCE_SCALE))


def tile(img):
    return np.asarray(Image.fromarray(img).resize((TILE, TILE), Image.LANCZOS))


def main():
    for path in sys.argv[1:]:
        seq = torch.load(path, map_location="cpu")
        name = os.path.splitext(os.path.basename(path))[0]
        exposure = seq["exposure"]
        valid = (seq["depth"][..., 0] > 0).numpy()
        ref = seq["reference"].numpy()
        global GAIN
        mean_luma = float(((ref * np.array([0.2126, 0.7152, 0.0722])).sum(-1)[valid]).mean() * math.pi * exposure / RADIANCE_SCALE)
        GAIN = 0.5 / max(mean_luma, 1e-4)  # auto gain: the mean exposed luminance shows at 0.5 / (1 + 0.5)
        raw = seq["reference_raw"].numpy() if "reference_raw" in seq else ref
        top = [tile(tonemap(s["acc"].numpy()[..., :3], exposure)) for s in seq["snapshots"]]
        diff = np.abs(ref - raw)
        bottom = [tile(tonemap(raw, exposure)), tile(tonemap(ref, exposure)), tile(tonemap(diff * 8.0, exposure))]
        blank = np.zeros_like(bottom[0])
        sheet = np.concatenate([np.concatenate(top, axis=1), np.concatenate(bottom + [blank] * (len(top) - len(bottom)), axis=1)], axis=0)
        out = os.path.join(os.path.dirname(path), f"preview_{name}.png")
        Image.fromarray(sheet).save(out)

        pr = perceptual(ref, exposure)
        errs = []
        for s in seq["snapshots"]:
            e = ((perceptual(s["acc"].numpy()[..., :3], exposure) - pr) ** 2)[valid].mean()
            errs.append(f"n={s['n']}:{e:.4f}")
        change = ((perceptual(raw, exposure) - pr) ** 2)[valid].mean()
        lum = (ref * np.array([0.2126, 0.7152, 0.0722])).sum(-1)[valid]
        print(f"{name:12s} exposure {exposure:7.2f}  valid px {valid.mean() * 100:5.1f}%  mean exposed luma {lum.mean() * math.pi * exposure / RADIANCE_SCALE:6.3f}  "
              f"ref cleanup mse {change:.5f} | snapshot mse vs cleaned ref: " + " ".join(errs) + f"  -> {out}")


if __name__ == "__main__":
    main()
