"""Re-cleans the ground truth of already converted full sequences (convert_full.py) with the progressive preset's spatial filter.

  python -P reclean_targets.py perf_tests/full                 (every converted sequence folder)
  python -P reclean_targets.py perf_tests/full --only cam0     (substring of the sequence names)

The 4096 spp reference dump (<name>_ref.rtlndump, next to the sequence folders) is cleaned with train.filter_reference (separable gaussian, radius 8,
plane tolerance 1.5, normal power 8, variance guide 0.4 sigmas = the progressive preset), reference.npy is overwritten. The previous cleanup is kept once as
reference_atrous_old.npy. The per frame data of the sequences is not touched.
"""

import argparse
import glob
import json
import os
import shutil
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
    ap.add_argument("--sigma", type=float, default=0.4)
    ap.add_argument("--radius", type=int, default=8)
    args = ap.parse_args()
    only = [s for s in args.only.split(",") if s]
    train.configure_features(lite=True)  # the same compile cache as the training runs
    pipe = train.Pipeline()
    for meta_path in sorted(glob.glob(os.path.join(args.folder, "*", "meta.json"))):
        folder = os.path.dirname(meta_path)
        name = os.path.basename(folder)
        if name.endswith(".tmp") or (only and not any(s in name for s in only)):
            continue
        reference_dump = os.path.join(args.folder, f"{name}_ref.rtlndump")
        if not os.path.exists(reference_dump):
            print("skip", name, "(no reference dump)")
            continue
        r = dataset.load_dump(reference_dump)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to("cuda").contiguous()
        meta = json.load(open(meta_path))
        seq = {k: t(np.load(os.path.join(folder, f"{k}.npy"))) for k in ("position", "normal", "rcp_width", "depth")}
        seq["reference_variance"] = t(r["frames"][-1][2])
        seq["reference_acc"] = t(r["frames"][-1][0])
        seq["exposure"] = max(meta["reference_exposure"], 1e-3)
        cleaned = train.filter_reference(pipe, seq, args.sigma, args.radius).cpu().numpy().astype(np.float32)
        old = os.path.join(folder, "reference_atrous_old.npy")
        if not os.path.exists(old):
            shutil.copyfile(os.path.join(folder, "reference.npy"), old)
        previous = np.load(old)
        np.save(os.path.join(folder, "reference.npy"), cleaned)
        meta.pop("reference_steps", None)
        meta.update({"reference_sigma": args.sigma, "reference_filter": "gaussian", "reference_radius": args.radius})
        json.dump(meta, open(meta_path, "w"))
        mask = (seq["depth"][..., 0] > 0).cpu().numpy()
        change = float(np.abs(cleaned - previous)[mask].mean() / max(float(np.abs(previous)[mask].mean()), 1e-9))
        print(f"{name}: re-cleaned (gaussian r{args.radius}, guide {args.sigma} sigmas); mean |new - old a-trous| / old = {change * 100:.2f} %", flush=True)


if __name__ == "__main__":
    main()
