"""Plots the validation error per history length from the logs of train.py.

  python -P plot_levels.py --out levels.png --level 64 A=path/to/A.log B=path/to/B.log

Left: the validation error at one history length (--level, default 64 accumulated frames) of every run's network at each evaluation, against
the lab's variance guide filter and the geometry only filter (these do not train, so they are horizontal).
Right: the error of every method over all history lengths at the LAST evaluation of each run.
"""

import argparse
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL = re.compile(r"^eval @ (\w+)")
HEADER = re.compile(r"^\s{3}snapshot frames\s+((?:\d+\s+)+)mean")
ROW = re.compile(r"^\s{3}(temporal only|geometry only|variance guide|neural \+ guide|neural)\s+((?:\d+\.\d+\s*)+)$")


def parse(path):
    evals = []
    current = None
    for line in open(path, errors="replace"):
        m = EVAL.match(line)
        if m:
            current = {"it": 0 if m.group(1) == "init" else int(m.group(1)), "levels": [], "rows": {}}
            evals.append(current)
            continue
        if current is None:
            continue
        m = HEADER.match(line)
        if m:
            current["levels"] = [int(v) for v in m.group(1).split()]
            continue
        m = ROW.match(line)
        if m and current["levels"]:
            values = [float(v) for v in m.group(2).split()]
            current["rows"][m.group(1)] = dict(zip(current["levels"], values[: len(current["levels"])]))
    return [e for e in evals if "neural" in e["rows"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="levels.png")
    ap.add_argument("--level", type=int, default=64)
    ap.add_argument("logs", nargs="+", help="name=path")
    args = ap.parse_args()
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#2a6fdb", "#e0642a", "#2a9d5c", "#8e44ad"]
    last_evals = []
    reference_done = False
    for k, spec in enumerate(args.logs):
        name, path = spec.split("=", 1)
        evals = parse(path)
        if not evals:
            continue
        last_evals.append((name, evals[-1], colors[k % len(colors)]))
        ax_l.plot([e["it"] for e in evals], [e["rows"]["neural"][args.level] for e in evals], color=colors[k % len(colors)], marker="o", linewidth=2.2,
                  label=f"{name} neural")
        if not reference_done:
            first = evals[0]["rows"]
            ax_l.axhline(first["variance guide"][args.level], color="#c0392b", linestyle="--", linewidth=1.8, label="variance guide")
            ax_l.axhline(first["geometry only"][args.level], color="#555555", linestyle=":", linewidth=1.8, label="geometry only")
            reference_done = True
    ax_l.set_title(f"validation error at {args.level} accumulated frames (held-out views)")
    ax_l.set_xlabel("iteration")
    ax_l.set_ylabel("squared perceptual log error")
    ax_l.grid(alpha=0.3)
    ax_l.legend()
    if last_evals:
        rows = last_evals[0][1]["rows"]
        levels = last_evals[0][1]["levels"]
        ax_r.plot(levels, [rows["variance guide"][n] for n in levels], color="#c0392b", linestyle="--", marker="s", label="variance guide")
        ax_r.plot(levels, [rows["geometry only"][n] for n in levels], color="#555555", linestyle=":", marker="s", label="geometry only")
        for name, ev, color in last_evals:
            ax_r.plot(ev["levels"], [ev["rows"]["neural"][n] for n in ev["levels"]], color=color, marker="o", linewidth=2.2, label=f"{name} neural @ {ev['it']} it")
    ax_r.set_xscale("log", base=2)
    ax_r.set_yscale("log")
    ax_r.set_xticks(levels if last_evals else [])
    ax_r.set_xticklabels([str(n) for n in levels] if last_evals else [])
    ax_r.set_title("validation error per history length (last evaluation)")
    ax_r.set_xlabel("accumulated frames")
    ax_r.set_ylabel("squared perceptual log error")
    ax_r.grid(alpha=0.3, which="both")
    ax_r.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
