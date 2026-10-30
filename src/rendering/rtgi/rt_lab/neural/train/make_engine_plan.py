"""Engine check of trained weights: for the user's views the SAME RT Lab test (error vs the converged progressive reference, instability,
timers) with the neural passes off and on, in every execution variant (plain f32 loops, tensor cores, f16), alternated A B C D A so GPU clock
drift shows up as the difference between the two "off" runs.

  python -P make_engine_plan.py --weights src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural.bin --views 1,4 --suns sdef > plan.json

Compare in the results json (<out>/<name>_results.json): rt_lab_convergence (rmse, bias vs the reference), rt_lab_measure (instability),
groups -> RT_LAB timers NEURAL_LATENT / ATROUS0.. . Run it with run_isolated.sh when the user's app is open.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -P does not add the script directory
import make_snapshot_plan as base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="perf_tests/neural_engine")
    ap.add_argument("--name", default="neural_engine")
    ap.add_argument("--views", default="1,4", help="camera indices")
    ap.add_argument("--suns", default="sdef")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--reference-frames", type=int, default=1024)
    args = ap.parse_args()
    suns = [s for s in base.SUNS if s[0] in args.suns.split(",")]
    modes = [
        ("off", {"enabled": False}),
        ("plain", {"enabled": True, "coopvec": False, "half": False}),
        ("coopvec", {"enabled": True, "coopvec": True, "half": False}),
        ("half", {"enabled": True, "coopvec": True, "half": True}),
        ("off_again", {"enabled": False}),
    ]
    tests = []
    for c in [int(x) for x in args.views.split(",")]:
        for sun_name, sun in suns:
            common = {"camera": base.CAMERAS[c], "sun": sun, "settle_frames": 400, "force_full_pgi_update": True, "debug_draw": 0, "test": "measure",
                      "metrics": ["convergence", "instability", "reference"], "frames": args.frames, "reference_frames": args.reference_frames}
            for mode, neural in modes:
                tests.append({**common, "name": f"cam{c}_{sun_name}_{mode}", "neural": {"recurrent": False, "weights": args.weights, **neural}})
    print(json.dumps({"scene": base.SCENE, "resolution": [1280, 720], "out": args.out, "name": args.name, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
