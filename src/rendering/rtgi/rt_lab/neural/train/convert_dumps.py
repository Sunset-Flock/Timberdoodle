"""Pairs <view>_custom.rtlndump with <view>_ref.rtlndump of a folder into <view>.pt training sequences."""

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch

import dataset


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Pairs <name>_custom.rtlndump with <name>_ref.rtlndump into <name>.pt sequences")
    ap.add_argument("folder")
    ap.add_argument("--snapshots", action="store_true", help="snapshot sequences (1/4/8/16/32/64 frames) instead of per frame sequences")
    ap.add_argument("--recurrent", action="store_true", help="consecutive frame sequences (<name>_rec.rtlndump, 64 frames, dump v2) -> <name>_rec.pt: recurrent / raw 3x3 training")
    ap.add_argument("--reference-sigma", type=float, default=0.5, help="variance guide sigma of the one pass reference cleanup (0 = leave the reference raw)")
    ap.add_argument("--only", default="", help="comma separated substrings: convert only the sequences whose name contains one of them (e.g. s99,s94)")
    args = ap.parse_args()
    folder = args.folder
    only = [s for s in args.only.split(",") if s]
    pipe = None
    if (args.snapshots or args.recurrent) and args.reference_sigma > 0:
        import train

        pipe = train.Pipeline()
    suffix = "_rec" if args.recurrent else "_custom"
    for custom in sorted(glob.glob(os.path.join(folder, f"*{suffix}.rtlndump"))):
        name = os.path.basename(custom)[: -len(f"{suffix}.rtlndump")]
        if only and not any(s in name for s in only):
            continue
        reference = os.path.join(folder, f"{name}_ref.rtlndump")
        if not os.path.exists(reference):
            print("skip", name, "(no reference dump)")
            continue
        if args.recurrent:
            seq = dataset.build_recurrent_sequence(custom, reference)
        else:
            seq = dataset.build_snapshot_sequence(custom, reference) if args.snapshots else dataset.build_sequence(custom, reference)
        if pipe is not None:
            seq["reference_raw"] = seq["reference"]
            seq["reference"] = train.filter_reference(pipe, seq, args.reference_sigma)
            seq["reference_sigma"] = args.reference_sigma
        out = os.path.join(folder, f"{name}{suffix}.pt" if args.recurrent else f"{name}.pt")
        torch.save(seq, out)
        kind = "frames" if "frames" in seq and args.recurrent else ("snapshots" if "snapshots" in seq else "frames")
        count = len(seq["frames"]) if kind == "frames" and "frames" in seq else (len(seq["snapshots"]) if "snapshots" in seq else len(seq["acc"]))
        print("saved", out, kind, count, "crop", tuple(seq["depth"].shape[:2]), "exposure", round(seq["exposure"], 4))


if __name__ == "__main__":
    main()
