"""Test plan for A/B-ing the neural affinity in the engine: per view the same RT Lab test (reference comparison + instability + lag)
with the neural passes off (A) and on (B). Compare rt_lab_convergence (rmse / bias vs the progressive reference), rt_lab_measure
(instability) and the timing groups (NEURAL_LATENT, ATROUS*) in the results json.

  python -P make_ab_plan.py --weights src/.../weights/rt_lab_neural.bin --out perf_tests/neural_ab > ab_plan.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -P does not add the script directory
import make_dump_plan as base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="perf_tests/neural_ab")
    ap.add_argument("--name", default="neural_ab")
    ap.add_argument("--frames", type=int, default=128)
    ap.add_argument("--reference-frames", type=int, default=1024)
    ap.add_argument("--views", type=str, default="street_a,rotation_b,wall_a,deep_b")
    args = ap.parse_args()
    tests = []
    for view_name, camera, sun in [v for v in base.VIEWS if v[0] in args.views.split(",")]:
        common = {"camera": camera, "sun": sun, "settle_frames": 400, "force_full_pgi_update": True, "debug_draw": 0, "test": "measure",
                  "metrics": ["convergence", "instability", "reference"], "frames": args.frames, "reference_frames": args.reference_frames}
        # A, B, A, B: the GPU clock drifts, alternate and repeat
        for rep, mode in enumerate(["off", "on", "off", "on"]):
            tests.append({**common, "name": f"{view_name}_{mode}{rep // 2}",
                          "neural": {"enabled": mode == "on", "weights": args.weights}})
    print(json.dumps({"scene": base.SCENE, "resolution": [1280, 720], "out": args.out, "name": args.name, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
