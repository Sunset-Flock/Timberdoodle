"""Test plan (TimberDoodle.exe --test-plan, or run_isolated.sh) that captures the neural training set: for every camera x sun
combination the snapshots of a reset history (after 1, 4, 8, 16, 32, 64 accumulated frames, 1 = the raw image without temporal)
and the ground truth (4096 spp, progressive, NO spatial filter in the capture; convert_dumps.py then applies one strict variance guide a-trous pass offline) incl. a screenshot of the raw 4k image.

  python -P make_snapshot_plan.py --out perf_tests/snap > plan.json
  python -P make_snapshot_plan.py --cameras 0,1,2 --name snap_a --out perf_tests/snap > plan_a.json     (subset, to run captures in parallel)

Then: python -P convert_dumps.py perf_tests/snap --snapshots   (pairs <name>_custom + <name>_ref -> <name>.pt)

The views and suns are the user's (memory: neural-affinity-training-setup). Camera strings are the View Camera clipboard format.
"""

import argparse
import json

SCENE = "assets/bistro_lights_compressed/bistro_lights_c.gltf"

CAMERAS = [
    "-15.986221 -7.424647 7.5500317 0.9813822 0.07637661 -0.176226",
    "-5.2041974 -12.085394 5.8547525 0.45447025 0.84758663 -0.2739594",
    "14.471761 -15.842095 2.6659694 -0.20784536 0.95727104 -0.20107809",
    "-52.49451 30.27136 11.3136425 0.93070936 0.29167232 -0.22069748",
    "-47.918396 36.651947 11.410064 -0.010557702 -0.93099564 -0.36487743",
]
# (name, "angle_x angle_y"). The default sun is settings/sky/default.json (94.854, 28.986), given explicitly so a previous test's sun never leaks.
SUNS = [("sdef", "94.854 28.986"), ("s94", "94.854 94"), ("s235", "235.4 86.5"), ("s99", "99.9 84.5")]
SNAPSHOT_FRAMES = [1, 4, 8, 16, 32, 64]  # RUN phase frames at the dump callback: the first frame after the reset is 1 = history length 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf_tests/snap")
    ap.add_argument("--name", default="snap")
    ap.add_argument("--cameras", default="0,1,2,3,4")
    ap.add_argument("--suns", default="", help="comma separated subset of sun names")
    ap.add_argument("--ref-frames", type=int, default=4096, help="ground truth samples per pixel (4k: the progressive history is a plain mean up to 2x this)")
    ap.add_argument("--crop", type=int, default=384, help="square crop edge (at crop-x / crop-y) dumped from each run")
    ap.add_argument("--crop-x", type=int, default=448)
    ap.add_argument("--crop-y", type=int, default=168)
    ap.add_argument("--only", choices=["both", "custom", "ref"], default="both")
    ap.add_argument("--consecutive", type=int, default=0, help="dump N CONSECUTIVE frames (1..N after the reset) instead of the snapshot frames: the recurrent training set (use --crop 192 --crop-x 544 --crop-y 264 --only custom, names get a _rec suffix)")
    args = ap.parse_args()
    suns = [s for s in SUNS if not args.suns or s[0] in args.suns.split(",")]
    tests = []
    for c in [int(x) for x in args.cameras.split(",")]:
        for sun_name, sun in suns:
            name = f"cam{c}_{sun_name}"
            common = {"camera": CAMERAS[c], "sun": sun, "settle_frames": 400, "force_full_pgi_update": True, "metrics": ["instability"], "debug_draw": 0}
            crop = [args.crop_x, args.crop_y, args.crop]
            if args.only != "ref":
                if args.consecutive:
                    dump = {"first": 1, "frames": args.consecutive, "crop": crop, "path": f"{args.out}/{name}_rec.rtlndump"}
                    tests.append({**common, "name": f"{name}_rec", "preset": "custom", "frames": args.consecutive + 10, "neural": {"enabled": False, "dump": dump}})
                else:
                    tests.append({**common, "name": f"{name}_custom", "preset": "custom", "frames": SNAPSHOT_FRAMES[-1] + 10,
                                  "neural": {"enabled": False, "dump": {"frame_list": SNAPSHOT_FRAMES, "crop": crop, "path": f"{args.out}/{name}_custom.rtlndump"}}})
            if args.only != "custom":
                tests.append({**common, "name": f"{name}_ref", "preset": "progressive", "frames": args.ref_frames + 12, "screenshot": True,
                              "progressive": {"spatial_enabled": 0, "progressive_max_samples": 2 * args.ref_frames},
                              "neural": {"enabled": False, "dump": {"first": args.ref_frames, "frames": 1, "crop": crop, "path": f"{args.out}/{name}_ref.rtlndump"}}})
    print(json.dumps({"scene": SCENE, "resolution": [1280, 720], "out": args.out, "name": args.name, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
