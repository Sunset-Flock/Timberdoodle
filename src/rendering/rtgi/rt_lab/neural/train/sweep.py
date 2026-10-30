"""Runs several train.py processes at once (the network is tiny, one process leaves the GPU mostly idle: the cost is Python
and kernel launch overhead). Every job gets its own output folder with weights and log.

  python -P sweep.py --jobs 4 --common "--data dumps/*.pt --iterations 3000" --vary seed=0,1,2,3
  python -P sweep.py --jobs 4 --common "--synthetic 6 --iterations 400" --vary lr=1e-3,2e-3 --vary window=4,8

--vary key=a,b,c may repeat, all combinations run (--jobs at a time). The shader caches (.cache) are shared, so only the
very first process pays the Slang compile.
"""

import argparse
import itertools
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def last_eval_block(text):
    """The last evaluation table of a train.py log (the rows of the snapshot error table), or why there is none."""
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("eval @ ")]
    if not starts:
        return "no eval yet: " + (lines[-1][:120] if lines else "empty log")
    block = [lines[starts[-1]]]
    for l in lines[starts[-1] + 1:]:
        if not l.startswith("   "):
            break
        block.append(l)
    return "\n".join(block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--common", type=str, default="")
    ap.add_argument("--vary", action="append", default=[])
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "..", "weights", "sweep"))
    args = ap.parse_args()

    axes = []
    for v in args.vary:
        key, values = v.split("=", 1)
        axes.append([(key, x) for x in values.split(",")])
    combos = list(itertools.product(*axes)) if axes else [()]
    os.makedirs(args.out, exist_ok=True)

    pending = []
    for combo in combos:
        name = "_".join(f"{k}{x}" for k, x in combo) or "run"
        folder = os.path.join(args.out, name)
        os.makedirs(folder, exist_ok=True)
        flags = " ".join(f"--{k.replace('_', '-')} {x}" for k, x in combo)
        cmd = [sys.executable, "-P", os.path.join(HERE, "train.py")] + args.common.split() + flags.split() + ["--out", os.path.join(folder, "weights.bin")]
        pending.append((name, folder, cmd))

    running = []
    results = {}
    # The first process compiles the kernels into the shared cache: let it finish its first iteration before the rest start.
    first = True
    while pending or running:
        while pending and len(running) < args.jobs and (not first or not running):
            name, folder, cmd = pending.pop(0)
            log = open(os.path.join(folder, "log.txt"), "w")
            running.append((name, folder, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"started {name}", flush=True)
            if first:
                # wait until the cache exists (first log line past compile)
                deadline = time.time() + 900
                while time.time() < deadline:
                    text = open(os.path.join(folder, "log.txt")).read()
                    if re.search(r"^it +1 ", text, re.M) or running[-1][2].poll() is not None:
                        break
                    time.sleep(2)
                first = False
        time.sleep(2)
        for item in list(running):
            name, folder, proc, log = item
            if proc.poll() is not None:
                log.close()
                text = open(os.path.join(folder, "log.txt")).read()
                results[name] = last_eval_block(text)
                print(f"finished {name}: {results[name]}", flush=True)
                running.remove(item)
    print("\n== summary")
    for name, r in results.items():
        print("##", name)
        print(r)


if __name__ == "__main__":
    main()
