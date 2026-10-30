"""Renders the filtered images of one trained network (plus the baselines) for chosen views and history lengths, as float npz files.
compose_comparison.py then lays them out as comparison sheets. One process per feature set (the feature set is a Slang compile time switch).

  python -P render_comparison.py --features stats3 --weights S.bin --out cmp/S --views cam1_s235,cam4_s99 --levels 4,16,64
  python -P render_comparison.py --features lite   --weights L.bin --out cmp/L --views cam1_s235,cam4_s99 --levels 4,16,64

Every view is a converted recurrent sequence <data>/<view>_rec.pt. Recurrent networks run the latent chain from frame 1 up to the level, like
evaluate() in train.py. Output per view and level: <view>_f<level>.npz with float32 [H, W, 3] radiance images (reference = the cleaned ground
truth the network was trained against, temporal = the unfiltered accumulated image, geometry, guide, neural) and scalars exposure + interior mask.
"""

import argparse
import os

import numpy as np
import torch

import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["stats3", "window5", "raw3", "lite"], default="stats3")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="D:/repos/Timberdoodle/perf_tests/snap")
    ap.add_argument("--views", required=True, help="comma separated sequence names, e.g. cam1_s235,cam4_s99")
    ap.add_argument("--levels", default="4,16,64")
    ap.add_argument("--recurrent", type=int, default=1)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--target-samples", type=float, default=320000.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    train.configure_features(window=args.features == "window5", raw3=args.features == "raw3", lite=args.features == "lite")
    train.TARGET_SAMPLES = args.target_samples
    os.makedirs(args.out, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",")]
    pipe = train.Pipeline()
    params = train.load_params(args.weights, "cuda")
    for view in args.views.split(","):
        seq = train.prepare_sequence(torch.load(os.path.join(args.data, f"{view}_rec.pt"), map_location="cuda"))
        H, W = seq["depth"].shape[:2]
        win = (slice(0, H), slice(0, W))
        static = train.static_crop(seq, win)
        latent_prev = train.make_input(torch.zeros(H, W, train.LATENT, device="cuda"))
        wanted = {i for i in seq["snapshot_idx"] if seq["frames"][i]["frame"] in levels}
        for i in range(max(wanted) + 1):
            if i not in wanted and not args.recurrent:
                continue
            s = train.frame_sample(static, seq, i, win)
            with torch.no_grad():
                raw = pipe.network(params, s, latent_prev)
                if args.recurrent:
                    latent_prev = train.make_input(raw[..., : train.LATENT])
                if i not in wanted:
                    continue
                mask = train.interior_mask(s, args.passes)
                images = {
                    "reference": s["reference"],
                    "temporal": s["acc"][..., :3],
                    "geometry": train.filter_with(pipe, s, None, args.passes, "geometry"),
                    "guide": train.filter_with(pipe, s, None, args.passes, "guide"),
                    "neural": train.filter_with(pipe, s, raw, args.passes, "neural"),
                }
                out = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in images.items()}
                out["exposure"] = np.float32(s["loss_exposure"])
                out["mask"] = mask.detach().cpu().numpy().astype(np.float32)
                path = os.path.join(args.out, f"{view}_f{s['n']}.npz")
                np.savez_compressed(path, **out)
                errors = {k: train.sample_loss(v, s, mask).item() for k, v in images.items()}
                print(f"{view} frame {s['n']:>3d}: " + "  ".join(f"{k} {e:.5f}" for k, e in errors.items()), flush=True)


if __name__ == "__main__":
    main()
