"""Appends one training run to ../TRAINING_LOG.md as a detailed entry, from the json summary train.py writes with --log-json.

  python -P log_run.py <run.json> --id 001 --title "lite recurrent, progressive-preset targets" --command "<the train.py command line>" \
      --notes "free text: what changed, what to look at" [--engine-json <engine results json>]

The entry has: what was run (arguments, data, targets, architecture), how long it took (first iteration = kernel compile, steady state s/it, phase split,
evaluation time, GPU memory), the validation tables at init and at every evaluation (perceptual log error MSE per snapshot frame count, per method) and the
change against the init and against the geometry-only / variance guide baselines. Nothing here is rewritten later: new runs are appended.
"""

import argparse
import datetime
import json
import os

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TRAINING_LOG.md")


def table(levels, rows, label_width=22):
    out = ["| method | " + " | ".join(f"n={n}" for n in levels) + " | mean |", "|---|" + "---:|" * (len(levels) + 1)]
    for method, row in rows.items():
        values = [row[str(n)] for n in levels]
        out.append(f"| {method} | " + " | ".join(f"{v:.5f}" for v in values) + f" | {sum(values) / len(values):.5f} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_json")
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--command", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--weights", default="", help="path of the saved weights (informational)")
    args = ap.parse_args()
    s = json.load(open(args.run_json))
    a = s["args"]
    lines = [f"", f"---", f"", f"## Run {args.id}: {args.title}", f"", f"Logged {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}.", ""]
    if args.command:
        lines += ["Command:", "", "```", args.command, "```", ""]
    lines += ["### Configuration", "",
              f"- features `{a['features']}` (input {s['input']}, {s['parameters']} parameters, weights feature mask {s['mask']}), recurrent {bool(a['recurrent'])}, bptt {a['bptt']}",
              f"- rotation tied first layer: {bool(a.get('tie_rotations', 0))}",
              f"- iterations {a['iterations']}, batch {a['batch']}, crop {a['crop']}, lr {a['lr']} (cosine to 5 %), w3 gain {a['w3_gain']}, seed {a['seed']}, loss power {a['loss_power']}, "
              f"normalize levels {bool(a['normalize_levels'])}",
              f"- spatial filter in the loss: {a['passes']} a-trous passes (geometry x learned affinity, target samples {a['target_samples']:.0f}); curriculum: first "
              f"{a['pretrain_fraction'] * 100:.0f} % of the iterations with {a['pretrain_passes']} passes",
              f"- frames cached on the GPU: {a.get('cache_frames', 'n/a')}",
              f"- training views: {', '.join(s['training_views'])}", f"- validation views: {', '.join(s['validation_views'])} (holdout `{a['holdout']}`)",
              f"- weights written to `{args.weights or a['out']}`", ""]
    lines += ["### Time and memory", "",
              f"- total wall time {s['total_seconds']:.0f} s ({s['total_seconds'] / 60:.1f} min), of which evaluation {s['evaluation_seconds']:.0f} s",
              f"- first iteration {s['first_iteration_seconds']:.1f} s (Slang kernel compile / cache load, once per module change)",
              f"- steady state {s['steady_seconds_per_iteration']:.3f} s per iteration over {s['iterations_timed']} iterations "
              f"(= {s['steady_seconds_per_iteration'] * a['iterations'] / 60:.1f} min for the iterations)",
              f"- peak GPU memory allocated by torch {s['peak_gpu_gb']:.2f} GB, skipped non finite steps {s['skipped_steps']}"]
    if s.get("profile_seconds"):
        p = s["profile_seconds"]
        total = sum(p.values())
        lines.append("- profiled split of a training step: " + ", ".join(f"{k} {v:.1f} s ({v / max(total, 1e-9) * 100:.0f} %)" for k, v in p.items()))
    lines.append("")
    evals = s["evals"]
    if evals:
        first, last = evals[0], evals[-1]
        lines += ["### Validation (perceptual log error, MSE vs the ground truth; lower is better)", ""]
        for e in evals:
            lines += [f"**after {e['tag']} iterations**" if e["tag"] != "init" else "**at init** (untrained network, starts as the geometry only filter)", ""]
            lines += table(e["levels"], e["table"]) + [""]
        levels = last["levels"]
        neural_first = [first["table"]["neural"][str(n)] for n in levels]
        neural_last = [last["table"]["neural"][str(n)] for n in levels]
        geometry = [last["table"]["geometry only"][str(n)] for n in levels]
        guide = [last["table"]["variance guide"][str(n)] for n in levels]
        temporal = [last["table"]["temporal only"][str(n)] for n in levels]
        lines += ["### Summary", "", "| | " + " | ".join(f"n={n}" for n in levels) + " | mean |", "|---|" + "---:|" * (len(levels) + 1)]

        def row(name, values):
            return f"| {name} | " + " | ".join(f"{v:.5f}" for v in values) + f" | {sum(values) / len(values):.5f} |"

        def ratio(name, num, den):
            r = [x / max(y, 1e-12) for x, y in zip(num, den)]
            return f"| {name} | " + " | ".join(f"{v:.3f}" for v in r) + f" | {sum(r) / len(r):.3f} |"

        lines += [row("temporal only", temporal), row("geometry only", geometry), row("variance guide (lab custom guide, 2 sigma)", guide), row("neural (final)", neural_last),
                  ratio("neural final / neural at init", neural_last, neural_first), ratio("neural final / geometry only", neural_last, geometry),
                  ratio("neural final / variance guide", neural_last, guide), ""]
    if args.notes:
        lines += ["### Notes", "", args.notes, ""]
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("appended run", args.id, "to", os.path.abspath(LOG))


if __name__ == "__main__":
    main()
