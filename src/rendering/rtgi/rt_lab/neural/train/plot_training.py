"""Plots the training logs of train.py: the training loss per iteration and the validation error at every evaluation.

  python -P plot_training.py --out loss.png A=path/to/A.log B=path/to/B.log C=path/to/C.log

Left: the training loss of a step (thin) and its running average (thick). The loss is normalized per sample by the error of the sample's
temporal-only image when --normalize-levels 1 (default): 1.0 = no better than the unfiltered temporal output, lower = better.
Right: the validation error (squared perceptual log error vs the ground truth, mean over the snapshot levels) of the network at every
evaluation, with the geometry-only filter and the lab's variance guide filter as dashed reference lines (they do not train).
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IT = re.compile(r"^it\s+(\d+)\s+loss\s+(\S+)\s+\(avg\s+(\S+)\)")
EVAL = re.compile(r"^eval @ (\w+)")
ROW = re.compile(r"^\s{3}(temporal only|geometry only|variance guide|neural \+ guide|neural)\s+((?:\d+\.\d+\s*)+)$")


def parse(path):
    its, loss, avg = [], [], []
    evals = []  # (iteration, {method: mean})
    current = None
    for line in open(path, errors="replace"):
        m = IT.match(line)
        if m:
            try:
                l, a = float(m.group(2)), float(m.group(3))
            except ValueError:
                continue
            its.append(int(m.group(1)))
            loss.append(l)
            avg.append(a)
            continue
        m = EVAL.match(line)
        if m:
            current = {"it": 0 if m.group(1) == "init" else int(m.group(1)), "rows": {}}
            evals.append(current)
            continue
        m = ROW.match(line)
        if m and current is not None:
            values = [float(v) for v in m.group(2).split()]
            current["rows"][m.group(1)] = values[-1]  # last column = mean over the levels
    return its, loss, avg, evals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training.png")
    ap.add_argument("logs", nargs="+", help="name=path")
    args = ap.parse_args()
    fig, (ax_l, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2a6fdb", "#e0642a", "#2a9d5c", "#8e44ad"]
    reference_drawn = False
    for k, spec in enumerate(args.logs):
        name, path = spec.split("=", 1)
        its, loss, avg, evals = parse(path)
        color = colors[k % len(colors)]
        if its:
            ax_l.plot(its, loss, color=color, alpha=0.25, linewidth=1)
            ax_l.plot(its, avg, color=color, linewidth=2.2, label=f"{name} (avg, {its[-1]} it)")
        pts = [(e["it"], e["rows"]["neural"]) for e in evals if "neural" in e["rows"]]
        if pts:
            ax_v.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker="o", linewidth=2.2, label=f"{name} neural")
        if evals and not reference_drawn:
            for method, style in (("variance guide", "--"), ("geometry only", ":")):
                if method in evals[0]["rows"]:
                    ax_v.axhline(evals[0]["rows"][method], color="#555555", linestyle=style, linewidth=1.5, label=method)
            reference_drawn = True
    ax_l.set_title("training loss (relative to the temporal-only error, lower = better)")
    ax_l.set_xlabel("iteration")
    ax_l.set_ylabel("loss")
    ax_l.grid(alpha=0.3)
    ax_l.legend()
    ax_v.set_title("validation error (mean over snapshot levels, held-out view)")
    ax_v.set_xlabel("iteration")
    ax_v.set_ylabel("squared perceptual log error")
    ax_v.grid(alpha=0.3)
    ax_v.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
