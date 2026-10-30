"""Trains the RT Lab neural affinity (rt_lab_neural_core.slang) with Slang autodiff through SlangPy.

  python -P train.py --data "perf_tests/snap/cam*_rec.pt" --features raw3 --recurrent 1 --iterations 600
  python -P train.py --data "perf_tests/snap/cam*_rec.pt" --features stats3 --recurrent 0

Training set: sequences of a RESET lab run (convert_dumps.py --recurrent): all consecutive frames, frame 1 = the first frame after the
history reset. A sample = (view, snapshot frame n in 1/4/8/16/32/64, random crop): the network (and with --recurrent 1 the latent chain
through all frames before n, the last --bptt of them with gradients) produces latent + weighting function gates for frame n, the shared
core's 4 a-trous passes filter the frame's temporal output with them, the loss is the squared perceptual log error against the 4096 spp
(strictly cleaned) ground truth. The filter is geometry (plane distance, normal) x the learned affinity: none of the lab's guides (variance, ray length,
SH) is in the neural filter. The geometric weights are precomputed once per crop (geometry_table); --pretrain-passes trains the first part with fewer passes.
Evaluation: per snapshot level the error of the temporal output alone, the geometry only filter, the lab's variance guide filter and the
network, plus which weighting functions the network picks. The weights file header carries the feature mask (rt_lab_neural_state.hpp).
"""

import argparse
import glob
import json
import math
import os
import struct
import sys
import time

import slangpy as _s

_s.set_allow_torch_fallback(True)

import torch
import slangpy as spy
from slangpy import grid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # python -P does not add the script directory

import dataset
import fullseq

RADIANCE_SCALE = dataset.RADIANCE_SCALE
LATENT = 8
OUT = 16
GATES = 6
HID = 32
FEATURE_BITS = {"window": 1, "no_guide": 2, "raylength3": 4, "recurrent": 8, "raw3": 16, "lite": 32, "v2": 64, "w2": 128}  # rt_lab_neural_state.hpp RT_LAB_NEURAL_FEATURE_*
FEATURES = {k: False for k in FEATURE_BITS}
IN = 32
PARAM_COUNT = HID * IN + HID + HID * HID + HID + OUT * HID + OUT
STEPS = [1, 2, 4, 8, 16]
TARGET_SAMPLES = 1000000.0  # ring budget of the a-trous passes (effective samples)
LOSS_POWER = 2  # 2 = squared perceptual log error (default), 1 = absolute
AFFINITY_SCALE = 4.0
WEIGHTING_NAMES = ["gaussian", "exponential", "cauchy", "student-t", "compact", "soft step"]
WEIGHTING_NAMES_W2 = ["mean step", "AND step", "trim step", "mean compct", "AND compct", "mean gauss"]  # --weighting v2 (variance guide style on the latent vector)


V2_PIXEL_INPUTS = 12  # inputs of the shared pixel encoder (rt_lab_neural_core.slang RT_LAB_NEURAL_V2_PIXEL_INPUTS)
V2_ENC = 6             # its outputs per pixel


def param_count():
    """Parameters of the current feature set: the MLP, and for the SH feature set (v2) the shared pixel encoder at the END of the vector."""
    count = HID * IN + HID + HID * HID + HID + OUT * HID + OUT
    return count + (V2_ENC * V2_PIXEL_INPUTS + V2_ENC if FEATURES["v2"] else 0)


def configure_features(**flags):
    """Selects the compile time feature set (rt_lab_neural_core.slang defines): input size and parameter count follow."""
    global IN, PARAM_COUNT
    FEATURES.update({k: bool(v) for k, v in flags.items()})
    if FEATURES["v2"]:
        IN = 64
    elif FEATURES["lite"]:
        IN = 40
    elif FEATURES["raw3"]:
        IN = 192
    elif FEATURES["window"]:
        IN = 64
    elif FEATURES["raylength3"]:
        IN = 48
    else:
        IN = 32
    PARAM_COUNT = param_count()


def feature_mask():
    return sum(bit for k, bit in FEATURE_BITS.items() if FEATURES[k])


def shader_defines():
    d = {}
    if FEATURES["w2"]:
        d["RT_LAB_NEURAL_WEIGHTING_V2"] = "1"
    if FEATURES["v2"]:
        d["RT_LAB_NEURAL_V2"] = "1"
    if FEATURES["window"]:
        d["RT_LAB_NEURAL_WINDOW"] = "1"
    if FEATURES["no_guide"]:
        d["RT_LAB_NEURAL_NO_GUIDE_INPUTS"] = "1"
    if FEATURES["raylength3"]:
        d["RT_LAB_NEURAL_RAYLENGTH3"] = "1"
    if FEATURES["raw3"]:
        d["RT_LAB_NEURAL_RAW3"] = "1"
    if FEATURES["lite"]:
        d["RT_LAB_NEURAL_LITE"] = "1"
    return d


# ---------------------------------------------------------------------------------------------------------------------
# Rotation tied first layer (--tie-rotations, lite layout)
# ---------------------------------------------------------------------------------------------------------------------
# The lite inputs hold the plus shaped neighborhood as 5 pixels x 5 values: pixel 0 the center, 1 left, 2 right, 3 up, 4 down. A hidden unit h = f * 4 + r
# (filter f, rotation r) of the first layer reads the neighbor pixel at cycle position k (left, up, right, down) of the filter's weights at position (k + r) % 4:
# the same filter at four quarter turns of the screen space patch. The center pixel and the orientation independent inputs [25, 40) (detail normal, history
# length, depth, previous latent, bias) use the same weights in all four rotations, and so does the bias. The deeper layers stay dense. The tied parameters
# are expanded to the dense layout (expand_tied) before they reach the network: the weights file, the engine and the CoopVec path do not change.
TIE_FILTERS = 8
TIE_ROTATIONS = 4
LITE_NEIGHBOR_CYCLE = [1, 3, 2, 4]  # left, up, right, down: one quarter turn per step
V2_EDGE_CYCLE = [1, 3, 2, 4]    # SH feature set pixel order: 0 center, 1 left, 2 right, 3 up, 4 down, 5 top left, 6 top right, 7 bottom left, 8 bottom right
V2_CORNER_CYCLE = [5, 6, 8, 7]  # top left, top right, bottom right, bottom left: the same turn


def tie_maps():
    """(w1 map, bias map, tied size): tied parameter index of every dense W1 weight [HID * IN] and of every b1 entry [HID]."""
    assert (IN == 40 or (IN == 64 and FEATURES["v2"])) and HID == TIE_FILTERS * TIE_ROTATIONS
    w1 = torch.zeros(HID * IN, dtype=torch.long)
    for f in range(TIE_FILTERS):
        for r in range(TIE_ROTATIONS):
            h = f * TIE_ROTATIONS + r
            for i in range(IN):
                dest_index = i
                if FEATURES["v2"]:
                    # 9 pixels x 6 encoder channels (center, left, right, up, down, top left, top right, bottom left, bottom right), then the center only inputs.
                    p, v = divmod(i, V2_ENC)
                    if i < 9 * V2_ENC:
                        for cycle in (V2_EDGE_CYCLE, V2_CORNER_CYCLE):
                            if p in cycle:
                                dest_index = cycle[(cycle.index(p) + r) % 4] * V2_ENC + v
                else:
                    p, v = divmod(i, 5)
                    if 1 <= p <= 4 and i < 25:
                        k = LITE_NEIGHBOR_CYCLE.index(p)  # position of the source pixel in the filter's own orientation
                        dest_index = LITE_NEIGHBOR_CYCLE[(k + r) % 4] * 5 + v
                w1[h * IN + dest_index] = f * IN + i
    bias = torch.tensor([TIE_FILTERS * IN + f for f in range(TIE_FILTERS) for _ in range(TIE_ROTATIONS)], dtype=torch.long)
    rest = PARAM_COUNT - (HID * IN + HID)
    return w1, bias, TIE_FILTERS * IN + TIE_FILTERS + rest


def tied_count():
    return tie_maps()[2]


def expand_tied(tied, maps):
    """Dense parameter vector (the weights file layout) from the tied parameters: differentiable (a gather)."""
    w1, bias, _ = maps
    rest_start = TIE_FILTERS * IN + TIE_FILTERS
    return torch.cat([tied[w1], tied[bias], tied[rest_start:]])


def init_tied(device, seed=0, w3_gain=0.02):
    dense = init_params(device, seed, w3_gain).detach()
    g = torch.Generator(device=device)
    g.manual_seed(seed + 1)
    filters = torch.randn(TIE_FILTERS, IN, generator=g, device=device) * math.sqrt(2.0 / IN)
    tied = torch.cat([filters.flatten(), torch.zeros(TIE_FILTERS, device=device), dense[HID * IN + HID:]])
    return tied.clone().requires_grad_(True)


def init_params(device, seed=0, w3_gain=0.02):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def he(out_f, in_f, gain=1.0):
        return torch.randn(out_f, in_f, generator=g, device=device) * math.sqrt(2.0 / in_f) * gain

    parts = [he(HID, IN).flatten(), torch.zeros(HID, device=device), he(HID, HID).flatten(), torch.zeros(HID, device=device),
             he(OUT, HID, w3_gain).flatten(), torch.zeros(OUT, device=device)]
    if FEATURES["v2"]:
        parts += [he(V2_ENC, V2_PIXEL_INPUTS).flatten(), torch.zeros(V2_ENC, device=device)]  # the shared pixel encoder, after the MLP
    flat = torch.cat(parts)
    assert flat.numel() == PARAM_COUNT
    return flat.clone().requires_grad_(True)


def save_params(params, path):
    with open(path, "wb") as f:
        f.write(b"RTLN")
        f.write(struct.pack("<III", 1, PARAM_COUNT, feature_mask()))
        f.write(params.detach().cpu().numpy().astype("<f4").tobytes())


def load_params(path, device):
    raw = open(path, "rb").read()
    assert raw[:4] == b"RTLN"
    count = struct.unpack("<I", raw[8:12])[0]
    assert count == PARAM_COUNT, f"weights have {count} params, expected {PARAM_COUNT}"
    flat = torch.frombuffer(bytearray(raw[16:]), dtype=torch.float32).clone().to(device)
    return flat.requires_grad_(True)


def make_device():
    """create_torch_device with the module + shader caches on: the autodiff of the unrolled network costs minutes of Slang compile time per
    process, the caches (one per compile time feature set) make every later process start fast."""
    torch.cuda.init()
    torch.cuda.current_device()
    torch.cuda.current_stream()
    handles = spy.get_cuda_current_context_native_handles()
    name = "_".join(k for k in ("v2", "w2", "window", "raw3", "lite", "no_guide", "raylength3") if FEATURES[k]) or "stats3"
    cache = os.path.join(os.environ.get("TIDO_NEURAL_CACHE", os.path.join(HERE, ".cache")), name)  # TIDO_NEURAL_CACHE: another cache root (time_compile.py: a fresh one = cold compile)
    os.makedirs(cache, exist_ok=True)
    return spy.Device(
        type=spy.DeviceType.vulkan,
        compiler_options={"include_paths": [os.path.join(os.path.dirname(spy.__file__), "slang"), HERE], "defines": shader_defines()},
        enable_cuda_interop=True,
        existing_device_handles=handles,
        enable_hot_reload=False,  # editing the Slang sources while a run is going must not touch it (hot reload crashed a run: "Reflection object has been invalidated")
        module_cache_path=os.path.join(cache, "modules"),
        shader_cache_path=os.path.join(cache, "shaders"),
    )


class Pipeline:
    def __init__(self):
        self.device = make_device()
        self.module = spy.Module.load_from_file(self.device, "rt_lab_neural_train.slang")
        self.input_mask = torch.ones(IN, device="cuda")  # zeros = ablated inputs (--ablate)

    def network(self, params, s, latent_prev):
        H, W = s["depth"].shape[:2]
        return self.module.latent_step(pix=grid(shape=(H, W)), params=params, g_position=s["position"], g_normal=s["normal"],
                                       g_rcp_width=s["rcp_width"], g_depth=s["depth"], g_detail_normal=s["detail_normal"], g_albedo=s["albedo"],
                                       f_acc=s["acc"], f_raw=s["raw"], f_variance=s["variance"], f_ray=s["ray"], f_ray_stats=s["ray_stats"], f_sh_y=s["sh_y"], f_cocg=s["cocg"], f_sq=s["sq"],
                                       exposure=s["exposure"], latent_prev=latent_prev, input_mask=self.input_mask)

    def geometry_table(self, static, step):
        """[H, W, 25] plane distance x normal weights of the 5x5 window at `step` (parameter independent: cached per crop in static["geo"])."""
        cache = static.setdefault("geo", {})
        if step not in cache:
            H, W = static["depth"].shape[:2]
            cache[step] = self.module.geometry_table(pix=grid(shape=(H, W)), step=step, g_position=static["position"], g_normal=static["normal"],
                                                     g_rcp_width=static["rcp_width"], g_depth=static["depth"]).contiguous()
        return cache[step]

    def atrous(self, s, color, latent, affinity, step, use_latent):
        H, W = s["depth"].shape[:2]
        return self.module.atrous_pass(pix=grid(shape=(H, W)), step=step, target=TARGET_SAMPLES, use_latent=use_latent, g_depth=s["depth"],
                                       g_geo_table=self.geometry_table(s, step), color_in=color, latent=latent, affinity=affinity)

    def gauss_guide(self, s, color, direction, radius, sigma):
        """One pass of the progressive preset's separable gaussian (geometry x variance guide), direction (1, 0) or (0, 1)."""
        H, W = s["depth"].shape[:2]
        return self.module.gauss_guide_pass(pix=grid(shape=(H, W)), direction=spy.int2(direction[0], direction[1]), radius=radius, g_position=s["position"],
                                            g_normal=s["normal"], g_rcp_width=s["rcp_width"], g_depth=s["depth"], color_in=color, variance=s["variance"],
                                            exposure=s["exposure"], guide_sigma=sigma)

    def guide(self, s, color, step, sigma):
        """One pass of the lab's variance guide filter (evaluation baseline, ground truth cleanup; not part of the neural filter)."""
        H, W = s["depth"].shape[:2]
        return self.module.guide_pass(pix=grid(shape=(H, W)), step=step, target=TARGET_SAMPLES, g_position=s["position"], g_normal=s["normal"],
                                      g_rcp_width=s["rcp_width"], g_depth=s["depth"], color_in=color, variance=s["variance"], exposure=s["exposure"], guide_sigma=sigma)


def make_input(x):
    # DiffTensor inputs need a gradient buffer on this backend
    return x.detach().clone().requires_grad_(True)


# ---------------------------------------------------------------------------------------------------------------------
# Sequences and samples
# ---------------------------------------------------------------------------------------------------------------------

def prepare_sequence(seq):
    """Brings every supported sequence format to: frames (list of per frame dicts incl. nominal `frame`), snapshot_idx (the frames that
    are training / evaluation snapshots), static detail_normal / albedo and per frame ray_stats (zeros when the data has none)."""
    seq = dict(seq)
    H, W = seq["depth"].shape[:2]
    dev = seq["depth"].device
    if "frames" not in seq:
        if "snapshots" not in seq:
            seq = dataset.snapshots_from_frames(seq)
        seq["frames"] = [dict(sn, frame=sn.get("frame", sn["n"])) for sn in seq["snapshots"]]
        seq["snapshot_idx"] = list(range(len(seq["frames"])))
        seq["consecutive"] = False  # snapshots only: no latent chain possible
    else:
        seq["snapshot_idx"] = [i for i, f in enumerate(seq["frames"]) if f["frame"] in dataset.SNAPSHOT_COUNTS]
        seq["consecutive"] = True
    for k, channels in (("detail_normal", 3), ("albedo", 3)):
        if k not in seq:
            seq[k] = torch.zeros(H, W, channels, device=dev)
    for fr in seq["frames"]:
        if "ray_stats" not in fr:
            fr["ray_stats"] = torch.zeros(H, W, 2, device=dev)
        for k, channels in (("sh_y", 4), ("cocg", 2), ("sq", 3)):  # SH feature set data: zeros for sequences without it
            if k not in fr:
                fr[k] = torch.zeros(H, W, channels, device=dev)
    return seq


STATIC_KEYS = ("position", "normal", "rcp_width", "depth", "detail_normal", "albedo", "reference")
FRAME_KEYS = ("acc", "raw", "variance", "ray", "ray_stats", "sh_y", "cocg", "sq")


def static_crop(seq, win):
    static = {k: seq[k][win].contiguous() for k in STATIC_KEYS}
    static["geo"] = {}  # step -> precomputed geometric edge stopping of the 5x5 window (Pipeline.geometry_table), shared by every frame sample of the crop
    return static


def frame_sample(static, seq, i, win):
    fr = seq["frames"][i]
    s = dict(static)
    for k in FRAME_KEYS:
        s[k] = fr[k][win].contiguous()
    s["exposure"] = fr["exposure"]
    s["loss_exposure"] = seq["exposure"]
    s["n"] = fr["frame"]  # nominal accumulated frames (1, 4, 8, ...); the measured mean history length is lower
    return s


def random_window(seq, crop):
    H, W = seq["depth"].shape[:2]
    if crop and crop < min(H, W):
        y0 = int(torch.randint(0, H - crop + 1, (1,)).item())
        x0 = int(torch.randint(0, W - crop + 1, (1,)).item())
        return (slice(y0, y0 + crop), slice(x0, x0 + crop))
    return (slice(0, H), slice(0, W))


def perceptual(x, s):
    exposed = x * (math.pi * s["loss_exposure"] / RADIANCE_SCALE)
    return torch.log1p(8.0 * exposed.clamp(min=0))


def interior_mask(s, passes):
    H, W = s["depth"].shape[:2]
    margin = 2 * sum(STEPS[:passes]) + 2
    m = torch.zeros(H, W, 1, device=s["depth"].device)
    m[margin:H - margin, margin:W - margin] = 1.0
    return m * (s["depth"] > 0).float()


def sample_loss(out, s, mask):
    """Mean over the interior pixels and channels of |p(out) - p(ground truth)|^LOSS_POWER, p = log1p(8 x exposed radiance)."""
    diff = ((perceptual(out, s) - perceptual(s["reference"], s)).abs() ** LOSS_POWER) * mask
    return diff.sum() / (mask.sum() * 3).clamp(min=1)


def network_chain(pipe, params, seq, static, win, target_i, recurrent, bptt):
    """Raw network outputs [H, W, 16] of frame target_i. Recurrent: the latent chain starts at frame 0 (the first frame after the reset, previous
    latent 0) and runs through every frame before the target; only the last `bptt` frames before the target carry gradients."""
    H, W = static["depth"].shape[:2]
    latent_prev = make_input(torch.zeros(H, W, LATENT, device=static["depth"].device))
    if recurrent:
        assert seq["consecutive"], "recurrent training needs sequences of consecutive frames (convert_dumps.py --recurrent)"
        for i in range(target_i):
            s_i = frame_sample(static, seq, i, win)
            if i >= target_i - bptt:
                raw = pipe.network(params, s_i, latent_prev)
                latent_prev = raw[..., :LATENT].contiguous()
            else:
                with torch.no_grad():
                    raw = pipe.network(params, s_i, latent_prev)
                latent_prev = make_input(raw[..., :LATENT])
    s_t = frame_sample(static, seq, target_i, win)
    return pipe.network(params, s_t, latent_prev), s_t


def filter_with(pipe, s, raw, passes, mode):
    """mode: 'neural' (raw = the network outputs of the frame) | 'geometry' | 'guide' (the lab's variance guide, baseline only). Returns the filtered rgb [H, W, 3]."""
    H, W = s["depth"].shape[:2]
    color = make_input(s["acc"])
    if mode == "guide":
        for step in STEPS[:passes]:
            color = pipe.guide(s, color, step, 2.0)
        return color[..., :3]
    if mode == "neural":
        latent = raw[..., :LATENT].contiguous()
        affinity = raw[..., LATENT:].contiguous()
        use_latent = True
    else:
        latent = make_input(torch.zeros(H, W, LATENT, device=s["depth"].device))
        affinity = make_input(torch.zeros(H, W, OUT - LATENT, device=s["depth"].device))
        use_latent = False
    for step in STEPS[:passes]:
        color = pipe.atrous(s, color, latent, affinity, step, use_latent)
    return color[..., :3]


def filter_reference(pipe, seq, sigma=0.4, radius=8):
    """The ground truth cleanup = the PROGRESSIVE preset's spatial filter (rt_lab.cpp rt_lab_default_settings): the separable gaussian (horizontal,
    then vertical) of `radius` taps per side (sigma = radius / 2), plane tolerance 1.5 px, normal power 8 and the variance guide at `sigma` sigmas
    (default 0.4: strict acceptance, a tap only counts when it is statistically the same mean), applied to the 4096 spp image. Same likelihood as the
    lab's guide; both passes read the 4096 spp variance of the raw image."""
    H, W = seq["depth"].shape[:2]
    s = {"position": seq["position"], "normal": seq["normal"], "rcp_width": seq["rcp_width"], "depth": seq["depth"],
         "variance": seq["reference_variance"], "exposure": seq["exposure"]}
    with torch.no_grad():
        out = make_input(seq["reference_acc"])
        for direction in ((1, 0), (0, 1)):
            out = pipe.gauss_guide(s, out, direction, radius, sigma)
    return out[..., :3].contiguous()


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------------------------------

def gate_stats(raw, mask):
    """Mean softmax gates (per weighting function) and sharpness of the pixels in the mask."""
    logits = raw[..., LATENT:LATENT + GATES]
    gates = torch.softmax(logits, dim=-1)
    if FEATURES["w2"]:
        sharp = 2.0 * torch.exp(-raw[..., LATENT + GATES].clamp(-3, 3))  # kappa: the tolerance in sigmas
    else:
        sharp = AFFINITY_SCALE * torch.exp(raw[..., LATENT + GATES].clamp(-3, 3))
    m = mask[..., 0] > 0
    return gates[m].mean(dim=0), sharp[m].mean()


@torch.no_grad()
def evaluate(pipe, params, seqs, passes, recurrent):
    """Per snapshot level (nominal frame) mean error of every method over the validation views."""
    methods = ["temporal only", "geometry only", "variance guide", "neural"]
    err = {m: {} for m in methods}
    gates = {}
    for seq in seqs:
        H, W = seq["depth"].shape[:2]
        win = (slice(0, H), slice(0, W))
        static = static_crop(seq, win)
        latent_prev = make_input(torch.zeros(H, W, LATENT, device=seq["depth"].device))
        last = max(seq["snapshot_idx"])
        for i in range(last + 1):
            needed = i in seq["snapshot_idx"]
            if not needed and not recurrent:
                continue
            s = frame_sample(static, seq, i, win)
            raw = pipe.network(params, s, latent_prev)
            latent_prev = make_input(raw[..., :LATENT]) if recurrent else latent_prev
            if not needed:
                continue
            mask = interior_mask(s, passes)
            res = {"temporal only": sample_loss(s["acc"][..., :3], s, mask).item()}
            res["geometry only"] = sample_loss(filter_with(pipe, s, None, passes, "geometry"), s, mask).item()
            res["variance guide"] = sample_loss(filter_with(pipe, s, None, passes, "guide"), s, mask).item()
            res["neural"] = sample_loss(filter_with(pipe, s, raw, passes, "neural"), s, mask).item()
            for m in methods:
                err[m].setdefault(s["n"], []).append(res[m])
            g, sharp = gate_stats(raw, mask)
            gates.setdefault(s["n"], []).append(torch.cat([g, sharp[None]]))
    levels = sorted(next(iter(err.values())).keys())
    table = {m: {n: sum(v[n]) / len(v[n]) for n in levels} for m, v in err.items()}
    gate_table = {n: torch.stack(gates[n]).mean(dim=0).tolist() for n in levels}
    return levels, table, gate_table


def print_eval(tag, levels, table, gate_table):
    power = "mse" if LOSS_POWER == 2 else "l1"
    print(f"eval @ {tag}  ({power} of the perceptual log error vs ground truth, validation views)", flush=True)
    print("   " + "snapshot frames".ljust(18) + "".join(f"{n:>9d}" for n in levels) + f"{'mean':>10s}", flush=True)
    for m, row in table.items():
        vals = [row[n] for n in levels]
        mean = sum(vals) / len(vals)
        print("   " + m.ljust(18) + "".join(f"{v:9.5f}" for v in vals) + f"{mean:10.5f}", flush=True)
    base = table["variance guide"]
    neural = table["neural"]
    print("   " + "neural / guide".ljust(18) + "".join(f"{neural[n] / max(base[n], 1e-12):9.3f}" for n in levels), flush=True)
    print("   weighting function gates (mean) + sharpness chosen by the network, per snapshot:", flush=True)
    names = WEIGHTING_NAMES_W2 if FEATURES["w2"] else WEIGHTING_NAMES
    print("   " + "".join(f"{name[:9]:>10s}" for name in names) + f"{'kappa' if FEATURES['w2'] else 'sharpness':>10s}", flush=True)
    for n in levels:
        print("   n=" + f"{n:<3d}".ljust(2) + "".join(f"{v:10.3f}" for v in gate_table[n]), flush=True)


# Input groups of the lite layout (rt_lab_neural_core.slang RT_LAB_NEURAL_LITE): per pixel [0 center, 1 left, 2 right, 3 up, 4 down] x 5 values
# (relative depth, perceptual radiance rgb, capped ray length), then the center's detail normal [25, 28), history length [28], absolute depth [29],
# previous latent [30, 38), bias [38].
LITE_GROUPS = {
    "relative depth (4 neighbors)": [5, 10, 15, 20],
    "radiance (center)": [1, 2, 3],
    "radiance (4 neighbors)": [6, 7, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23],
    "ray length (center)": [4],
    "ray length (4 neighbors)": [9, 14, 19, 24],
    "detail normal (center)": [25, 26, 27],
    "history length": [28],
    "absolute depth": [29],
    "previous latent": list(range(30, 38)),
}


@torch.no_grad()
def run_ablation(pipe, params, val_seqs, passes, recurrent):
    """Permutation style importance without retraining: zero one input group at a time and measure the validation error of the network."""
    assert FEATURES["lite"], "the ablation groups are defined for --features lite"
    results = {}
    for name, idx in [("(none: full network)", [])] + list(LITE_GROUPS.items()):
        pipe.input_mask = torch.ones(IN, device="cuda")
        if idx:
            pipe.input_mask[idx] = 0.0
        levels, table, _ = evaluate(pipe, params, val_seqs, passes, recurrent)
        results[name] = [table["neural"][n] for n in levels]
    pipe.input_mask = torch.ones(IN, device="cuda")
    base = results["(none: full network)"]
    print(f"ablation (validation error of the network with one input group zeroed; levels {levels}):", flush=True)
    print("   " + "group".ljust(34) + "".join(f"{n:>9d}" for n in levels) + f"{'mean':>10s}{'increase':>10s}", flush=True)
    base_mean = sum(base) / len(base)
    for name, vals in sorted(results.items(), key=lambda kv: -sum(kv[1])):
        mean = sum(vals) / len(vals)
        print("   " + name.ljust(34) + "".join(f"{v:9.5f}" for v in vals) + f"{mean:10.5f}" + f"{(mean / base_mean - 1) * 100:+9.1f}%", flush=True)


# ---------------------------------------------------------------------------------------------------------------------

def main():
    global LOSS_POWER, TARGET_SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="glob of sequence .pt files (convert_dumps.py --recurrent) or of full image sequence folders (convert_full.py)")
    ap.add_argument("--eval-windows", type=int, default=2, help="full image sequences: evaluation windows per validation view (fixed, spread over the image)")
    ap.add_argument("--eval-size", type=int, default=256, help="edge of an evaluation window")
    ap.add_argument("--iterations", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8, help="samples (view, snapshot frame, crop) averaged per optimizer step")
    ap.add_argument("--weighting", choices=["v1", "v2"], default="v1", help="v1 = 6 weighting functions of one latent distance + a sharpness, v2 = variance guide style: latent dims 0..3 are means, 4..7 log standard deviations, the weighting functions test z^2 = d^2 / (sigma_c^2 + sigma_t^2) as a mean, a max (AND) and a trimmed mean")
    ap.add_argument("--features", choices=["stats3", "window5", "raw3", "lite", "v2"], default="stats3",
                    help="stats3 = pixel data + 3x3 neighborhood statistics (IN 32), window5 = raw 5x5 log luminance window (IN 64), raw3 = raw 3x3 data of 19 values per pixel (IN 192), lite = plus shaped neighborhood of 5 pixels x 5 values + center extras (IN 40), v2 = the SH feature set: 3x3 footprint through a shared pixel encoder (12 -> 6), SH-Y / CoCg / second moments, no depth (IN 64; needs a dump v3 dataset)")
    ap.add_argument("--no-guide-inputs", type=int, default=0, help="1 = no variance / ray length guide derived inputs (stats3 / window5)")
    ap.add_argument("--raylength3", type=int, default=0, help="1 = ray length guide of the 3x3 neighborhood as extra inputs (stats3)")
    ap.add_argument("--recurrent", type=int, default=0, help="1 = the network also sees its previous latent, trained through the latent chain of the frames before the snapshot")
    ap.add_argument("--bptt", type=int, default=4, help="frames before the snapshot frame that carry gradients (truncated backprop through time)")
    ap.add_argument("--crop", type=int, default=192, help="random crop edge of a training sample (0 = whole image)")
    ap.add_argument("--passes", type=int, default=4, help="a-trous passes in training (the engine runs 5: the 5th reuses the same latents). 4 = 30 px halo")
    ap.add_argument("--pretrain-passes", type=int, default=2, help="passes of the first --pretrain-fraction of the iterations (2 = 8 px halo, far cheaper); 0 or >= --passes = always --passes")
    ap.add_argument("--pretrain-fraction", type=float, default=0.6, help="fraction of the iterations trained with --pretrain-passes")
    ap.add_argument("--tie-rotations", type=int, default=0, help="1 = rotation tied first layer (lite layout): 8 filters x 4 quarter turns of the plus shaped patch, 4x fewer first layer parameters")
    ap.add_argument("--cache-frames", choices=["none", "gpu"], default="gpu", help="gpu = keep the per frame data of the full image sequences on the GPU (fp16, about 1 GB per sequence), none = stream from the memory mapped files")
    ap.add_argument("--profile", type=int, default=0, help="1 = time the phases of the training step (cuda synchronized) and print the split with every log line")
    ap.add_argument("--log-json", type=str, default="", help="write the run summary (args, timings, final evaluation) to this json file")
    ap.add_argument("--target-samples", type=float, default=1000000.0, help="ring budget of the spatial filter in effective samples")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w3-gain", type=float, default=0.02, help="init gain of the last layer (0.02 = latents ~0, gates uniform: starts at the geometry only filter)")
    ap.add_argument("--normalize-levels", type=int, default=1, help="1 = the training loss of a sample is divided by the error of its temporal-only image (relative improvement), so the 1 and 4 frame snapshots do not dominate; reports stay raw")
    ap.add_argument("--loss-power", type=int, default=2, help="2 = squared perceptual log error, 1 = absolute")
    ap.add_argument("--holdout", type=str, default="cam4", help="substrings of sequence file names used for validation only")
    ap.add_argument("--out", type=str, default=os.path.join(HERE, "..", "weights", "rt_lab_neural.bin"))
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", type=int, default=0, help="1 = no training: load --resume weights and report how much the validation error rises when each input group is zeroed")
    args = ap.parse_args()
    LOSS_POWER = args.loss_power
    TARGET_SAMPLES = args.target_samples
    configure_features(window=args.features == "window5", raw3=args.features == "raw3", lite=args.features == "lite", v2=args.features == "v2", w2=args.weighting == "v2", no_guide=args.no_guide_inputs, raylength3=args.raylength3,
                       recurrent=args.recurrent)
    print(f"features {args.features} (input {IN}, {PARAM_COUNT} parameters, mask {feature_mask()}), recurrent {bool(args.recurrent)}, "
          f"target samples {TARGET_SAMPLES:.0f}, filter = geometry x learned affinity", flush=True)

    pipe = Pipeline()
    torch.manual_seed(args.seed)
    train_seqs, val_seqs = [], []
    holdout = [h for h in args.holdout.split(",") if h]
    for f in sorted(glob.glob(args.data)):
        if os.path.isdir(f):
            is_holdout = any(h in os.path.basename(f) for h in holdout)
            skip = ("raw", "ray") if args.features in ("lite", "v2") else ()  # keys the feature set never reads: neither cached nor streamed
            seq = prepare_sequence(fullseq.load(f, "none" if is_holdout else args.cache_frames, skip))  # the holdout is only read in fixed windows: streamed
            if is_holdout:
                H, W = seq["depth"].shape[:2]
                size = min(args.eval_size, H, W)
                spots = [(0.12, 0.2), (0.62, 0.2), (0.12, 0.6), (0.62, 0.6)][: args.eval_windows]
                for fx, fy in spots:
                    val_seqs.append(prepare_sequence(fullseq.windowed(seq, int(fy * (H - size)), int(fx * (W - size)), size)))
                continue
            train_seqs.append(seq)
            continue
        seq = prepare_sequence(torch.load(f, map_location="cuda"))
        (val_seqs if any(h in os.path.basename(f) for h in holdout) else train_seqs).append(seq)
    if not val_seqs and len(train_seqs) > 1:
        val_seqs.append(train_seqs.pop())
    assert train_seqs, "no training data"
    print(f"{len(train_seqs)} training views, {len(val_seqs)} validation views, "
          f"{sum(len(s['snapshot_idx']) for s in train_seqs)} training snapshots", flush=True)

    maps = None
    if args.tie_rotations:
        assert args.features in ("lite", "v2") and not args.resume, "--tie-rotations needs --features lite or v2 and a fresh start (the tied parameters cannot be recovered from dense weights)"
        maps = tuple(m.to("cuda") if isinstance(m, torch.Tensor) else m for m in tie_maps())
        leaf = init_tied("cuda", args.seed, args.w3_gain)
        print(f"rotation tied first layer: {TIE_FILTERS} filters x {TIE_ROTATIONS} quarter turns, {leaf.numel()} tied parameters ({PARAM_COUNT} dense)", flush=True)
    else:
        leaf = load_params(args.resume, "cuda") if args.resume else init_params("cuda", args.seed, args.w3_gain)

    def dense():
        """The dense parameter vector (weights file layout) of the optimized parameters."""
        return expand_tied(leaf, maps) if maps is not None else leaf

    optimizer = torch.optim.Adam([leaf], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations, eta_min=args.lr * 0.05)
    if args.ablate:
        assert args.resume, "--ablate needs --resume <weights>"
        run_ablation(pipe, dense(), val_seqs, args.passes, bool(args.recurrent))
        return
    evals = []

    def run_eval(tag):
        t_eval = time.time()
        with torch.no_grad():
            res = evaluate(pipe, dense().detach(), val_seqs, args.passes, bool(args.recurrent))
        print_eval(tag, *res)
        evals.append({"tag": tag, "levels": list(res[0]), "table": {m: {str(n): v for n, v in row.items()} for m, row in res[1].items()}, "seconds": time.time() - t_eval})
        return res

    wall_start = time.time()
    if val_seqs:
        run_eval("init")
    start = time.time()
    last = start
    running = None
    skipped = 0
    pretrain = 0 < args.pretrain_passes < args.passes
    timers = {"crop": 0.0, "chain": 0.0, "filter": 0.0, "backward": 0.0}
    timer_total = dict(timers)
    tick_last = [time.perf_counter()]
    step_seconds = []

    def tick(name):
        if args.profile:
            torch.cuda.synchronize()
            now = time.perf_counter()
            timers[name] += now - tick_last[0]
            timer_total[name] += now - tick_last[0]
            tick_last[0] = now

    first_iteration_seconds = None
    for it in range(1, args.iterations + 1):
        iteration_start = time.time()
        passes = args.pretrain_passes if pretrain and it <= args.pretrain_fraction * args.iterations else args.passes
        if pretrain and it == int(args.pretrain_fraction * args.iterations) + 1:
            print(f"it {it:5d} pretrain done: {args.pretrain_passes} -> {args.passes} passes", flush=True)
        optimizer.zero_grad()
        batch_loss = 0.0
        tick_last[0] = time.perf_counter()
        for _ in range(args.batch):
            params = dense()
            seq = train_seqs[int(torch.randint(0, len(train_seqs), (1,)).item())]
            target_i = seq["snapshot_idx"][int(torch.randint(0, len(seq["snapshot_idx"]), (1,)).item())]
            for _try in range(8):  # sky / empty crops carry no loss: take windows with geometry
                win = random_window(seq, args.crop)
                if float((seq["depth"][win] > 0).float().mean()) > 0.3:
                    break
            static = static_crop(seq, win)
            tick("crop")
            raw, s = network_chain(pipe, params, seq, static, win, target_i, bool(args.recurrent), args.bptt)
            tick("chain")
            mask = interior_mask(s, passes)
            out = filter_with(pipe, s, raw, passes, "neural")
            loss = sample_loss(out, s, mask)
            if args.normalize_levels:
                loss = loss / sample_loss(s["acc"][..., :3], s, mask).detach().clamp(min=1e-4)
            loss = loss / args.batch
            tick("filter")
            loss.backward()
            tick("backward")
            batch_loss += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_([leaf], 1.0)
        if not (math.isfinite(batch_loss) and bool(torch.isfinite(grad_norm))):
            skipped += 1  # a non finite loss / gradient must never reach the weights
            optimizer.zero_grad()
            scheduler.step()
            if skipped <= 5 or skipped % 50 == 0:
                print(f"it {it:5d} skipped a non finite step (loss {batch_loss}, grad norm {float(grad_norm)}), {skipped} skipped so far", flush=True)
            continue
        optimizer.step()
        scheduler.step()
        running = batch_loss if running is None else 0.95 * running + 0.05 * batch_loss
        seconds = time.time() - iteration_start
        if first_iteration_seconds is None:
            first_iteration_seconds = seconds
        else:
            step_seconds.append(seconds)
        if it % 10 == 0 or it == 1:
            now = time.time()
            split = ""
            if args.profile:
                split = "  [" + " ".join(f"{k} {v / (1 if it == 1 else 10):.2f}" for k, v in timers.items()) + "]"
                timers = {k: 0.0 for k in timers}
            print(f"it {it:5d} loss {batch_loss:.5f} (avg {running:.5f}) lr {scheduler.get_last_lr()[0]:.2e} {(now - last) / (1 if it == 1 else 10):.2f}s/it{split}", flush=True)
            last = now
        if val_seqs and (it % args.eval_every == 0 or it == args.iterations):
            t_eval = time.time()
            run_eval(str(it))
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            save_params(dense().detach(), args.out)
            last += time.time() - t_eval  # evaluation time is not training time
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_params(dense().detach(), args.out)
    total_seconds = time.time() - wall_start
    steady = sum(step_seconds) / max(len(step_seconds), 1)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print("saved", os.path.abspath(args.out), f"(mask {feature_mask()}, {skipped} skipped steps)")
    print(f"timing: total {total_seconds:.0f} s (incl. {sum(e['seconds'] for e in evals):.0f} s evaluation), first iteration {first_iteration_seconds:.1f} s (kernel compile / cache load), "
          f"steady state {steady:.3f} s/it over {len(step_seconds)} iterations (batch {args.batch}, crop {args.crop}), peak GPU memory allocated by torch {peak_gb:.2f} GB", flush=True)
    if args.profile:
        total = sum(timer_total.values())
        print("profile (share of the profiled step time): " + ", ".join(f"{k} {v:.1f} s ({v / max(total, 1e-9) * 100:.0f} %)" for k, v in timer_total.items()), flush=True)
    if args.log_json:
        summary = {"args": vars(args), "features": args.features, "input": IN, "parameters": PARAM_COUNT, "mask": feature_mask(), "total_seconds": total_seconds,
                   "first_iteration_seconds": first_iteration_seconds, "steady_seconds_per_iteration": steady, "iterations_timed": len(step_seconds),
                   "evaluation_seconds": sum(e["seconds"] for e in evals), "peak_gpu_gb": peak_gb, "skipped_steps": skipped, "evals": evals,
                   "profile_seconds": timer_total if args.profile else None, "training_views": [s.get("name", "") for s in train_seqs], "validation_views": [s.get("name", "") for s in val_seqs]}
        json.dump(summary, open(args.log_json, "w"), indent=1)

if __name__ == "__main__":
    main()
