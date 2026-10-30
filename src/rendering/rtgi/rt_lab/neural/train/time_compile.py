"""Times the Slang compile of the trainer kernels for one feature layout (rt_lab_neural_core.slang compile time features).

  python -P time_compile.py --features lite --cache-dir <fresh dir> --json out.json     (cold: the cache dir is empty)
  python -P time_compile.py --features lite --cache-dir <same dir> --json out2.json     (warm: everything comes from the cache)

`run_compile_timings.py` runs every layout cold and warm, each in its own process (a process compiles each kernel once). The module and shader caches of
make_device() live in <cache-dir>/<layout name> (env TIDO_NEURAL_CACHE, default train/.cache). Phases are timed with a cuda synchronize on a tiny synthetic
64 px sequence: the module load (parse + semantic check), then the FIRST call of each kernel (that is where Slang generates code for the unrolled MLP and its
autodiff): network forward, network backward, a-trous forward, a-trous backward, the geometry table and the progressive gaussian pass; then the second call of the
network forward + backward as the plain run time.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -P does not add the script directory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["stats3", "window5", "raw3", "lite", "raylength3"], default="lite")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--size", type=int, default=64)
    args = ap.parse_args()
    os.environ["TIDO_NEURAL_CACHE"] = os.path.abspath(args.cache_dir)
    t_process = time.perf_counter()

    import torch
    import train

    t_import = time.perf_counter() - t_process
    train.configure_features(window=args.features == "window5", raw3=args.features == "raw3", lite=args.features == "lite", raylength3=args.features == "raylength3")
    timings = {}

    def timed(name, fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        timings[name] = time.perf_counter() - start
        print(f"{name}: {timings[name]:.2f} s", flush=True)
        return result

    pipe = timed("device + module load", train.Pipeline)
    n = args.size
    dev = "cuda"
    yy, xx = torch.meshgrid(torch.arange(n, device=dev, dtype=torch.float32), torch.arange(n, device=dev, dtype=torch.float32), indexing="ij")
    position = torch.stack([xx * 0.05, yy * 0.05, torch.zeros_like(xx)], dim=-1).contiguous()
    normal = torch.zeros(n, n, 3, device=dev)
    normal[..., 2] = 1.0
    static = {"position": position, "normal": normal, "rcp_width": torch.full((n, n, 1), 20.0, device=dev), "depth": torch.full((n, n, 1), 5.0, device=dev),
              "detail_normal": normal.clone(), "albedo": torch.full((n, n, 3), 0.5, device=dev), "geo": {}}
    g = torch.Generator(device=dev)
    g.manual_seed(1)
    frame = {"acc": torch.rand(n, n, 4, device=dev, generator=g) * 1000.0 + 1.0, "raw": torch.rand(n, n, 4, device=dev, generator=g) * 1000.0,
             "variance": torch.rand(n, n, 4, device=dev, generator=g) * 1000.0, "ray": torch.rand(n, n, 2, device=dev, generator=g),
             "ray_stats": torch.rand(n, n, 2, device=dev, generator=g), "exposure": 50.0}
    s = {**static, **frame}
    params = train.init_params(dev, 0, 0.5)
    latent_prev = train.make_input(torch.zeros(n, n, train.LATENT, device=dev))

    def network():
        return pipe.network(params, s, latent_prev)

    timed("network forward (first call)", lambda: network())
    raw = timed("network forward + autodiff setup", lambda: network())

    def backward_network():
        network().sum().backward()

    timed("network backward (first call)", backward_network)
    params.grad = None

    latent = train.make_input(raw[..., :train.LATENT].detach())
    affinity = train.make_input(raw[..., train.LATENT:].detach())
    color = train.make_input(s["acc"])
    timed("geometry table (first call)", lambda: pipe.geometry_table(s, 1))
    timed("a-trous forward (first call)", lambda: pipe.atrous(s, color, latent, affinity, 1, True))

    def backward_atrous():
        pipe.atrous(s, color, latent, affinity, 1, True)[..., :3].sum().backward()

    timed("a-trous backward (first call)", backward_atrous)
    timed("progressive gaussian pass (first call)", lambda: pipe.gauss_guide(s, color, (1, 0), 8, 0.4))
    timed("variance guide a-trous pass (first call)", lambda: pipe.guide(s, color, 1, 2.0))

    def second_call():
        network().sum().backward()

    timed("network forward + backward (second call, plain run time)", second_call)
    total = time.perf_counter() - t_process
    sizes = {}
    root = os.path.join(os.path.abspath(args.cache_dir))
    for dirpath, _, files in os.walk(root):
        for f in files:
            sizes[os.path.relpath(dirpath, root).split(os.sep)[0]] = sizes.get(os.path.relpath(dirpath, root).split(os.sep)[0], 0) + os.path.getsize(os.path.join(dirpath, f))
    summary = {"features": args.features, "input": train.IN, "parameters": train.PARAM_COUNT, "import seconds": t_import, "phases": timings, "process total seconds": total,
               "cache bytes": sizes}
    print(json.dumps(summary, indent=1), flush=True)
    if args.json:
        json.dump(summary, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
