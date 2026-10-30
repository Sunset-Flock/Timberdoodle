"""Converts the full image dumps (<name>_rec.rtlndump + <name>_ref.rtlndump, see make_snapshot_plan.py --crop 720) to streamable sequence
folders <folder>/<name>/ (fullseq.py) and cleans the ground truth.

  python -P convert_full.py perf_tests/full --only s99 --delete-raw

The ground truth = the 4096 spp image cleaned like the PROGRESSIVE preset's spatial filter: the separable gaussian (radius 8, plane tolerance 1.5,
normal power 8) with the variance guide at 0.4 sigmas (--radius / --sigma, see train.filter_reference); the raw 4096 spp image is kept as reference_raw.npy. --delete-raw removes the custom dump after a successful conversion (about 1 GB each); the reference dump stays.
"""

import argparse
import gc
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -P does not add the script directory

import dataset
import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--only", default="", help="comma separated substrings of the sequence names")
    ap.add_argument("--sigma", type=float, default=0.4, help="variance guide sigmas of the ground truth cleanup (the progressive preset: 0.4)")
    ap.add_argument("--radius", type=int, default=8, help="gaussian radius of the ground truth cleanup (the progressive preset: 8)")
    ap.add_argument("--delete-raw", action="store_true")
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    pipe = train.Pipeline()
    for custom in sorted(glob.glob(os.path.join(args.folder, "*_rec.rtlndump"))):
        name = os.path.basename(custom)[: -len("_rec.rtlndump")]
        if only and not any(s in name for s in only):
            continue
        reference = os.path.join(args.folder, f"{name}_ref.rtlndump")
        if not os.path.exists(reference):
            print("skip", name, "(no reference dump)")
            continue
        out = os.path.join(args.folder, name)
        if os.path.exists(os.path.join(out, "meta.json")):
            print("skip", name, "(already converted)")
            continue
        c = dataset.load_dump(custom)
        r = dataset.load_dump(reference)
        assert c["meta"]["crop"] == r["meta"]["crop"] and c["meta"]["crop_xy"] == r["meta"]["crop_xy"], "custom and reference dumps must cover the same crop"
        depth_error = float(np.abs(r["depth"] - c["depth"]).max())
        assert depth_error < 1e-2 * max(float(c["depth"].max()), 1.0), f"reference and custom depth differ by {depth_error}: not the same view"
        stats = dataset.accumulate_ray_stats(c["frames"])
        tmp = out + ".tmp"
        os.makedirs(tmp, exist_ok=True)
        for k in fullseq_static():
            np.save(os.path.join(tmp, f"{k}.npy"), c[k].astype(np.float32))
        frames = len(c["frames"])
        H, W = c["depth"].shape[:2]
        channels = {"acc": 4, "raw": 4, "variance": 3, "ray": 2, "ray_stats": 2}
        sh = c.get("sh_frames") or []
        if sh:
            channels.update({"sh_y": 4, "cocg": 2, "sq": 3})  # dump v3: the SH-Y feature set data (sq in f32: Y^2 overflows f16)
        mm = {k: np.lib.format.open_memmap(os.path.join(tmp, f"{k}.npy"), mode="w+", dtype=np.float32 if k == "sq" else np.float16, shape=(frames, H, W, n)) for k, n in channels.items()}
        for i, ((acc, rawf, variance, ray, _), st) in enumerate(zip(c["frames"], stats)):
            mm["acc"][i] = acc
            mm["raw"][i] = rawf
            mm["variance"][i] = variance[..., :3]
            mm["ray"][i] = ray
            mm["ray_stats"][i] = st
            if sh:
                mm["sh_y"][i], mm["cocg"][i], mm["sq"][i] = sh[i]
        for k in list(mm):
            mm[k].flush()
            del mm[k]  # the memory maps must be closed before the folder is renamed (Windows: access denied otherwise)
        del mm
        gc.collect()
        ref_acc, ref_variance = r["frames"][-1][0], r["frames"][-1][2]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda").contiguous()
        seq = {"position": t(c["position"]), "normal": t(c["normal"]), "rcp_width": t(c["rcp_width"]), "depth": t(c["depth"]), "reference_variance": t(ref_variance),
               "reference_acc": t(ref_acc), "exposure": max(r["exposures"][-1], 1e-3)}
        cleaned = train.filter_reference(pipe, seq, args.sigma, args.radius)
        np.save(os.path.join(tmp, "reference.npy"), cleaned.cpu().numpy().astype(np.float32))
        np.save(os.path.join(tmp, "reference_raw.npy"), ref_acc[..., :3].astype(np.float32))
        meta = {"crop": c["meta"]["crop"], "crop_xy": list(c["meta"]["crop_xy"]), "frames": frames, "exposures": [float(e) for e in c["exposures"]],
                "frame_indices": [int(f) for f in c["frame_indices"]], "reference_exposure": float(r["exposures"][-1]), "reference_sigma": args.sigma,
                "reference_filter": "gaussian", "reference_radius": args.radius, "sh_features": bool(sh)}
        json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"))
        os.replace(tmp, out)
        if args.delete_raw:
            os.remove(custom)
        print("saved", out, frames, "frames", (H, W), "exposure", round(meta["reference_exposure"], 3), flush=True)


def fullseq_static():
    return ("position", "normal", "rcp_width", "depth", "detail_normal", "albedo")


if __name__ == "__main__":
    main()
