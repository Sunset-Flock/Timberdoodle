"""Runs time_compile.py for every feature layout, cold (fresh cache) then warm (same cache), one process at a time, and writes a markdown table.

  python -P run_compile_timings.py --out perf_tests/neural_compile --layouts lite,stats3,window5,raylength3,raw3 --timeout 3600

Run it with nothing else on the GPU / CPU (the compile is CPU bound, python and slangc compete with anything else that runs): the numbers are wall clock.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(layout, cache, json_path, timeout):
    command = [sys.executable, "-P", os.path.join(HERE, "time_compile.py"), "--features", layout, "--cache-dir", cache, "--json", json_path]
    start = time.time()
    try:
        subprocess.run(command, check=True, timeout=timeout, cwd=HERE)
    except subprocess.TimeoutExpired:
        return None, time.time() - start
    return json.load(open(json_path)), time.time() - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf_tests/neural_compile")
    ap.add_argument("--layouts", default="lite,stats3,window5,raylength3,raw3")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per process")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for layout in args.layouts.split(","):
        cache = os.path.abspath(os.path.join(args.out, f"cache_{layout}"))
        shutil.rmtree(cache, ignore_errors=True)
        os.makedirs(cache)
        print(f"== {layout}: cold", flush=True)
        cold, cold_wall = run(layout, cache, os.path.join(args.out, f"{layout}_cold.json"), args.timeout)
        warm, warm_wall = (None, 0.0)
        if cold is not None:
            print(f"== {layout}: warm", flush=True)
            warm, warm_wall = run(layout, cache, os.path.join(args.out, f"{layout}_warm.json"), args.timeout)
        rows.append((layout, cold, cold_wall, warm, warm_wall))
    phases = ["device + module load", "network forward (first call)", "network backward (first call)", "geometry table (first call)", "a-trous forward (first call)",
              "a-trous backward (first call)", "progressive gaussian pass (first call)", "variance guide a-trous pass (first call)"]
    lines = ["| layout | input | parameters | cold: process total | warm: process total | " + " | ".join(f"cold: {p}" for p in phases) + " | cache size |", "|---|---:|---:|---:|---:|" + "---:|" * (len(phases) + 1)]
    for layout, cold, cold_wall, warm, warm_wall in rows:
        if cold is None:
            lines.append(f"| {layout} | | | timeout (> {args.timeout} s) | | " + " | ".join("" for _ in phases) + " | |")
            continue
        size = sum(cold["cache bytes"].values()) / 1e6
        lines.append(f"| {layout} | {cold['input']} | {cold['parameters']} | {cold['process total seconds']:.0f} s | " +
                     (f"{warm['process total seconds']:.0f} s" if warm else "") + " | " + " | ".join(f"{cold['phases'][p]:.1f} s" for p in phases) + f" | {size:.0f} MB |")
    text = "\n".join(lines)
    open(os.path.join(args.out, "compile_times.md"), "w").write(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    main()
