"""Writes a test plan (run with: TimberDoodle.exe --test-plan <plan.json>) that captures neural training data.

Per view: "<name>_custom" renders 110 frames of the custom preset from a reset and dumps frames 0..95 of a crop; "<name>_ref"
renders the progressive preset with no spatial filter for REF_FRAMES frames and dumps the last frame as the converged
reference (cap raised so the history is a plain mean). Both dumps of a view show the same crop, camera, scene and sun.
Afterwards: python -P convert_dumps.py <out dir>  (pairs them into .pt sequences for train.py --data).

  python -P make_dump_plan.py --out perf_tests/neural --name neural_data > plan.json
"""

import argparse
import json

SCENE = "assets/bistro_lights_compressed/bistro_lights_c.gltf"

# (name, camera, sun). Every view is captured with each CROPS entry (the crop does not change the render).
VIEWS = [
    ("street_a", "-3.7095866 -13.49784 4.966386 0.4746604 0.85808086 -0.19594593", "250 90"),
    ("street_b", "-3.7095866 -13.49784 4.966386 0.4746604 0.85808086 -0.19594593", "0.1 73"),
    ("rotation_a", "-7.992414 -7.186529 6.312605 -0.24827832 0.9531537 -0.17278868", "0.1 73"),
    ("rotation_b", "-7.992414 -7.186529 6.312605 -0.24827832 0.9531537 -0.17278868", "250 90"),
    ("wall_a", [-22, 4, 6, 90, 10], "0.1 73"),
    ("wall_b", [-22, 4, 6, 90, 10], "250 90"),
    ("deep_a", [-22, 4, 6, 180, 5], "250 90"),
    ("deep_b", [-22, 4, 6, 180, 5], "0.1 73"),
]
CROPS = [("c0", 544, 264), ("c1", 96, 400)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf_tests/neural")
    ap.add_argument("--name", default="neural_data")
    ap.add_argument("--ref-frames", type=int, default=4096, help="ground truth samples per pixel (progressive, no spatial filter)")
    ap.add_argument("--only", choices=["both", "custom", "ref"], default="both", help="capture only the custom runs or only the references")
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--views", type=str, default="", help="comma separated subset of view names")
    args = ap.parse_args()
    views = [v for v in VIEWS if not args.views or v[0] in args.views.split(",")]
    tests = []
    for view_name, camera, sun in views:
        for crop_name, cx, cy in CROPS:
            name = f"{view_name}_{crop_name}"
            common = {"camera": camera, "sun": sun, "settle_frames": 400, "force_full_pgi_update": True, "metrics": ["instability"], "debug_draw": 0}  # settle: auto exposure ramps up slowly
            if args.only != "ref":
              tests.append({**common, "name": f"{name}_custom", "preset": "custom", "frames": 110,
                          "neural": {"enabled": False, "dump": {"first": 0, "frames": 96, "crop": [cx, cy, args.crop], "path": f"{args.out}/{name}_custom.rtlndump"}}})
            if args.only != "custom":
              tests.append({**common, "name": f"{name}_ref", "preset": "progressive", "frames": args.ref_frames + 12,
                          "progressive": {"spatial_enabled": 0, "progressive_max_samples": 8192},
                          "neural": {"enabled": False, "dump": {"first": args.ref_frames, "frames": 1, "crop": [cx, cy, args.crop], "path": f"{args.out}/{name}_ref.rtlndump"}}})
    print(json.dumps({"scene": SCENE, "resolution": [1280, 720], "out": args.out, "name": args.name, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
