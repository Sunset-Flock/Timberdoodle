"""Training sequences for the RT Lab neural affinity.

A sequence is a dict of CUDA float32 tensors, the layout the engine dump (rt_lab_neural_dump) writes:
  geometry (constant, static camera): position [H, W, 3], normal [H, W, 3], rcp_width [H, W, 1] (1 / world width of a pixel),
                                      depth [H, W, 1] (linear, 0 = sky)
  frames (per frame, lists of T tensors): acc [H, W, 4] (rgb in RTGI_RADIANCE_SCALE units, history length),
                                          raw [H, W, 4] (frame mean rgb, rays), variance [H, W, 4] (per sample rgb variance),
                                          ray [H, W, 2] (ray length ao guide (< 0 none), shortness)
  exposure: float, reference: [H, W, 3] converged radiance in the same units.
`synthetic_sequence` fakes the lab's temporal pass on analytic scenes so the training loop can be validated without a dump.
"""

import math

import numpy as np
import torch

RADIANCE_SCALE = 1.0e4


def _blur(x, k):
    # box blur of [H, W, C] with a (2k+1)^2 window (replicate padding)
    t = x.permute(2, 0, 1)[None]
    t = torch.nn.functional.pad(t, (k, k, k, k), mode="replicate")
    t = torch.nn.functional.avg_pool2d(t, 2 * k + 1, 1)
    return t[0].permute(1, 2, 0)


def synthetic_sequence(seed, size=192, frames=96, light_probability=0.15, device="cuda"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    H = W = size
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=torch.float32), torch.arange(W, device=device, dtype=torch.float32), indexing="ij")
    pixel_width = 0.01
    xw = (xs - W / 2) * pixel_width
    yw = (ys - H / 2) * pixel_width
    # two planes: a fronto parallel one and a tilted one, split by a slanted edge
    split = (xs * 0.8 + ys * 0.35) > (0.55 * W + 0.1 * H * torch.rand(1, generator=g, device=device))
    slope = 0.4 + 0.4 * torch.rand(1, generator=g, device=device)
    z = torch.where(split, 6.0 + slope * xw, torch.full_like(xw, 4.0))
    position = torch.stack([xw, yw, z], dim=-1)
    n_tilted = torch.nn.functional.normalize(torch.stack([-slope.expand_as(xw), torch.zeros_like(xw), torch.ones_like(xw)], dim=-1), dim=-1)
    n_flat = torch.stack([torch.zeros_like(xw), torch.zeros_like(xw), torch.ones_like(xw)], dim=-1)
    normal = torch.where(split[..., None], n_tilted, n_flat)
    depth = z[..., None].clone()
    rcp_width = torch.full((H, W, 1), 1.0 / pixel_width, device=device)

    # irradiance: smooth field + a hard shadow edge that does not follow the geometry + small bright spots (scaled units)
    def smooth_noise(scale):
        low = torch.randn(1, 1, H // scale + 2, W // scale + 2, generator=g, device=device)
        return torch.nn.functional.interpolate(low, size=(H, W), mode="bicubic", align_corners=False)[0, 0]

    base = 0.5 + 0.25 * smooth_noise(48)
    angle = 6.2831853 * torch.rand(1, generator=g, device=device)
    edge_pos = (xs - W / 2) * torch.cos(angle) + (ys - H / 2) * torch.sin(angle) + 18.0 * smooth_noise(64)
    sharpness = 0.6 + 1.5 * torch.rand(1, generator=g, device=device)
    shadow = torch.sigmoid(edge_pos / sharpness)
    lit = 0.25 + 0.75 * shadow
    spots = torch.zeros(H, W, device=device)
    for _ in range(4):
        cx = W * torch.rand(1, generator=g, device=device)
        cy = H * torch.rand(1, generator=g, device=device)
        r = 3.0 + 6.0 * torch.rand(1, generator=g, device=device)
        spots = spots + 1.5 * torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * r * r))
    luma_target = 0.3  # display units
    irradiance = (base * lit + spots) * luma_target
    tint = torch.tensor([1.0, 0.92, 0.8], device=device) * (0.85 + 0.3 * shadow)[..., None]
    scale = RADIANCE_SCALE / math.pi
    clean = irradiance[..., None] * tint * scale  # radiance in RTGI_RADIANCE_SCALE units at exposure 1
    clean_ao = (0.25 + 0.75 * shadow * (0.6 + 0.4 * base)).clamp(0, 1)

    acc = clean.clone() * 0.0
    var_acc = torch.zeros(H, W, 3, device=device)
    seq = {"position": position, "normal": normal, "rcp_width": rcp_width, "depth": depth, "exposure": 1.0, "reference": clean, "acc": [], "raw": [], "variance": [], "ray": []}
    for t in range(frames):
        hit = (torch.rand(H, W, 1, generator=g, device=device) < light_probability).float()
        raw = clean * hit / light_probability  # unbiased, heavy tailed 1 spp
        length = float(min(t + 1, 64))
        a = 1.0 / length
        acc = acc + (raw - acc) * a
        local_mean = _blur(raw, 2)
        local_var = (_blur(raw * raw, 2) - local_mean ** 2).clamp(min=0)
        var_acc = var_acc + (local_var - var_acc) * a
        ao_noise = 0.15 * torch.randn(H, W, generator=g, device=device) / math.sqrt(length)
        ray = torch.stack([clean_ao + ao_noise, (1.0 - clean_ao) ** 2], dim=-1)
        seq["acc"].append(torch.cat([acc, torch.full((H, W, 1), length, device=device)], dim=-1).contiguous())
        seq["raw"].append(torch.cat([raw, torch.ones(H, W, 1, device=device)], dim=-1).contiguous())
        seq["variance"].append(torch.cat([var_acc, torch.zeros(H, W, 1, device=device)], dim=-1).contiguous())
        seq["ray"].append(ray.contiguous())
    for k in ("position", "normal", "rcp_width", "depth"):
        seq[k] = seq[k].contiguous()
    return seq


# ---------------------------------------------------------------------------------------------------------------------
# Engine dump (rt_lab_neural.inl RtLabNeuralDumpH / rt_lab_neural.cpp DumpFileHeader)
# ---------------------------------------------------------------------------------------------------------------------

def load_dump(path):
    """Parses an .rtlndump into numpy arrays: geometry + per frame lab data of the crop."""
    import numpy as np

    raw = np.fromfile(path, dtype=np.uint8)
    magic = bytes(raw[:8])
    assert magic == b"RTLNDUMP", f"{path} is not a neural dump"
    version, crop, frames, first_frame, lab_w, lab_h, crop_x, crop_y = np.frombuffer(raw[8:40].tobytes(), dtype="<u4")
    assert version in (1, 2, 3), f"unknown dump version {version}"
    data = np.frombuffer(raw[40:].tobytes(), dtype="<u4").reshape(-1, 4)
    pixels = int(crop) * int(crop)
    geometry_records = 3 if version >= 2 else 2  # v2: [2] detail normal.xyz, albedo.rgb (f16)
    geometry = np.ascontiguousarray(data[: pixels * geometry_records]).reshape(pixels, geometry_records, 4)
    geometry_f = geometry.view("<f4")
    position = geometry_f[:, 0, :3].reshape(crop, crop, 3)
    depth = geometry_f[:, 0, 3].reshape(crop, crop, 1)
    normal = geometry_f[:, 1, :3].reshape(crop, crop, 3)
    rcp_width = geometry_f[:, 1, 3].reshape(crop, crop, 1)
    detail_normal = np.zeros((crop, crop, 3), np.float32)
    albedo = np.zeros((crop, crop, 3), np.float32)
    if version >= 2:
        extra = np.ascontiguousarray(geometry[:, 2, :]).view("<f2").reshape(pixels, 8).astype(np.float32)
        detail_normal = extra[:, 0:3].reshape(crop, crop, 3)
        albedo = extra[:, 3:6].reshape(crop, crop, 3)
    frame_data = []
    sh_frames = []  # v3: per frame (SH-Y [H, W, 4], CoCg [H, W, 2], squared moments of Y, Co, Cg [H, W, 3])
    exposures = []
    frame_indices = []
    records = 4 if version == 3 else 2  # uint4 per pixel and frame
    for f in range(int(frames)):
        base = pixels * geometry_records + f * (1 + pixels * records)
        header = data[base]
        exposures.append(float(header[:1].view("<f4")[0]))
        frame_indices.append(int(header[1]))
        full_block = np.ascontiguousarray(data[base + 1: base + 1 + pixels * records]).reshape(pixels, records, 4)
        block = np.ascontiguousarray(full_block[:, 0:2, :])
        halves = block.view("<f2").reshape(pixels, 2, 8).astype(np.float32)
        acc = halves[:, 0, 0:4].reshape(crop, crop, 4)
        rawf = halves[:, 0, 4:8].reshape(crop, crop, 4)
        variance = np.concatenate([halves[:, 1, 0:3], np.zeros((pixels, 1), np.float32)], axis=1).reshape(crop, crop, 4)
        ray = halves[:, 1, 3:5].reshape(crop, crop, 2)
        # v2: this frame's per pixel ray statistics: mean log2(1 + hit distance), mean capped normalized hit distance, ray count
        ray_frame = halves[:, 1, 5:8].reshape(crop, crop, 3) if version >= 2 else np.zeros((crop, crop, 3), np.float32)
        frame_data.append((acc, rawf, variance, ray, ray_frame))
        if version == 3:
            # [2] SH-Y xyz, SH-Y w, Co, Cg (f16) + 2 spare   [3] mean Y^2, Co^2, Cg^2 (f32) + 1 spare
            sh_halves = np.ascontiguousarray(full_block[:, 2, :]).view("<f2").reshape(pixels, 8).astype(np.float32)
            sq = np.ascontiguousarray(full_block[:, 3, :]).view("<f4").reshape(pixels, 4)[:, :3]
            sh_frames.append((sh_halves[:, 0:4].reshape(crop, crop, 4), sh_halves[:, 4:6].reshape(crop, crop, 2), sq.reshape(crop, crop, 3).copy()))
    return {"position": position, "depth": depth, "normal": normal, "rcp_width": rcp_width, "detail_normal": detail_normal, "albedo": albedo, "frames": frame_data, "sh_frames": sh_frames, "exposures": exposures, "frame_indices": frame_indices,
            "meta": {"crop": int(crop), "frames": int(frames), "first_frame": int(first_frame), "lab": (int(lab_w), int(lab_h)), "crop_xy": (int(crop_x), int(crop_y))}}


def build_sequence(custom_path, reference_path, device="cuda"):
    """One training sequence: the frames of the custom run + the last accumulated frame of the (long, spatial off, progressive)
    reference run as the converged reference. Both dumps must show the same view (same camera, scene, sun, crop)."""
    c = load_dump(custom_path)
    r = load_dump(reference_path)
    assert c["meta"]["crop_xy"] == r["meta"]["crop_xy"] and c["meta"]["crop"] == r["meta"]["crop"], "dumps differ in crop"
    t = lambda a: torch.from_numpy(a).to(device).contiguous()
    seq = {"position": t(c["position"]), "normal": t(c["normal"]), "rcp_width": t(c["rcp_width"]), "depth": t(c["depth"]),
           "exposures": [max(e, 1e-3) for e in c["exposures"]],  # per frame (auto exposure ramps up in short runs; frame 0 is 0)
           "exposure": max(c["exposures"][-1], 1e-3),            # loss domain
           "reference": t(r["frames"][-1][0][..., :3]),
           "acc": [], "raw": [], "variance": [], "ray": []}
    for acc, rawf, variance, ray, _ in c["frames"]:
        seq["acc"].append(t(acc)); seq["raw"].append(t(rawf)); seq["variance"].append(t(variance)); seq["ray"].append(t(ray))
    return seq


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Combine a custom run dump and a reference dump into a training sequence (.pt)")
    ap.add_argument("--custom", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshots", action="store_true")
    a = ap.parse_args()
    s = build_snapshot_sequence(a.custom, a.reference) if a.snapshots else build_sequence(a.custom, a.reference)
    torch.save(s, a.out)
    print("saved", a.out, "frames", len(s["acc"]), "crop", tuple(s["depth"].shape[:2]), "exposure", s["exposure"])


# ---------------------------------------------------------------------------------------------------------------------
# Snapshot sequences (the training set of the neural affinity): one independent image per accumulation level.
# A snapshot = the lab's temporal output after n frames from a full history reset (n = 1: the raw 1 sample image, no temporal).
# ---------------------------------------------------------------------------------------------------------------------

SNAPSHOT_COUNTS = [1, 4, 8, 16, 32, 64]


def snapshots_from_frames(seq):
    """Old format (per frame lists, frame 0 = first frame after a reset) -> snapshot format."""
    out = dict(seq)
    snaps = []
    for n in SNAPSHOT_COUNTS:
        f = n - 1
        if f < len(seq["acc"]):
            exposure = seq["exposures"][f] if "exposures" in seq else seq["exposure"]
            snaps.append({"n": n, "acc": seq["acc"][f], "raw": seq["raw"][f], "variance": seq["variance"][f], "ray": seq["ray"][f], "exposure": exposure})
    out["snapshots"] = snaps
    return out


def build_snapshot_sequence(custom_path, reference_path, device="cuda"):
    """Snapshots of one view (dump of the custom runs frames) + its ground truth (last frame of the long progressive, no
    spatial filter run). n is read from the dumped history length, not assumed from the frame index."""
    import numpy as np

    c = load_dump(custom_path)
    r = load_dump(reference_path)
    assert c["meta"]["crop_xy"] == r["meta"]["crop_xy"] and c["meta"]["crop"] == r["meta"]["crop"], "dumps differ in crop"
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device).contiguous()
    valid = c["depth"][..., 0] > 0
    snaps = []
    for (acc, rawf, variance, ray, _), exposure, frame in zip(c["frames"], c["exposures"], c["frame_indices"]):
        n = int(round(float(acc[..., 3][valid].mean()))) if valid.any() else frame + 1
        snaps.append({"n": n, "frame": frame, "acc": t(acc), "raw": t(rawf), "variance": t(variance), "ray": t(ray), "exposure": max(exposure, 1e-3)})
    ref_acc = r["frames"][-1][0]
    ref_variance = r["frames"][-1][2]
    ref_exposure = max(r["exposures"][-1], 1e-3)
    return {"position": t(c["position"]), "normal": t(c["normal"]), "rcp_width": t(c["rcp_width"]), "depth": t(c["depth"]),
            "snapshots": snaps, "reference": t(ref_acc[..., :3]), "reference_acc": t(ref_acc), "reference_variance": t(ref_variance), "exposure": ref_exposure, "reference_samples": float(ref_acc[..., 3][valid].mean()) if valid.any() else 0.0}


# ---------------------------------------------------------------------------------------------------------------------
# Recurrent sequences (raw 3x3 feature set + recurrent training): ALL consecutive frames of a reset run (frame 1 = first frame after
# the reset), crops of 192x192, ground truth cropped out of the (larger) reference dump of the same view.
# ---------------------------------------------------------------------------------------------------------------------

def accumulate_ray_stats(frames):
    """Per pixel ray statistics (mean log distance, mean capped normalized distance) accumulated over the frames like the lab's temporal
    pass accumulates colors (static camera: no reprojection): new = lerp(old, this frame's mean, n / history length) with n = rays of the
    frame (raw[..., 3]) and the history length acc[..., 3]. Pixels without rays keep their value, pixels never hit by a ray stay 0."""
    out = []
    state = None
    for acc, rawf, _variance, _ray, ray_frame in frames:
        n = rawf[..., 3]
        length = np.maximum(acc[..., 3], 1.0)
        stats = ray_frame[..., 0:2]
        has = (ray_frame[..., 2] > 0) & (stats[..., 0] >= 0)
        blend = np.clip(n / length, 0.0, 1.0)[..., None]
        if state is None:
            state = np.zeros_like(stats)
            seen = np.zeros(stats.shape[:2], bool)
        first = has & ~seen
        state = np.where(first[..., None], stats, np.where(has[..., None], state + (stats - state) * blend, state))
        seen = seen | has
        out.append(state.copy())
    return out


def build_recurrent_sequence(custom_path, reference_path, device="cuda"):
    c = load_dump(custom_path)
    r = load_dump(reference_path)
    assert c["meta"]["lab"] == r["meta"]["lab"], "dumps differ in lab resolution"
    cs, cx, cy = c["meta"]["crop"], *c["meta"]["crop_xy"]
    rs, rx, ry = r["meta"]["crop"], *r["meta"]["crop_xy"]
    ox, oy = cx - rx, cy - ry
    assert 0 <= ox and 0 <= oy and ox + cs <= rs and oy + cs <= rs, f"custom crop {cs}@({cx},{cy}) is not inside the reference crop {rs}@({rx},{ry})"
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device).contiguous()
    window = (slice(oy, oy + cs), slice(ox, ox + cs))
    # the reference geometry must be the same surface: compare depths
    depth_error = float(np.abs(r["depth"][window] - c["depth"]).max())
    assert depth_error < 1e-2 * max(float(c["depth"].max()), 1.0), f"reference and custom depth differ by {depth_error}: not the same view"
    stats = accumulate_ray_stats(c["frames"])
    frames = []
    for (acc, rawf, variance, ray, _), st, exposure, frame in zip(c["frames"], stats, c["exposures"], c["frame_indices"]):
        frames.append({"frame": frame, "acc": t(acc), "raw": t(rawf), "variance": t(variance), "ray": t(ray), "ray_stats": t(st), "exposure": max(exposure, 1e-3)})
    ref_acc = r["frames"][-1][0][window]
    ref_variance = r["frames"][-1][2][window]
    valid = c["depth"][..., 0] > 0
    return {"position": t(c["position"]), "normal": t(c["normal"]), "rcp_width": t(c["rcp_width"]), "depth": t(c["depth"]),
            "detail_normal": t(c["detail_normal"]), "albedo": t(c["albedo"]),
            "frames": frames, "reference": t(ref_acc[..., :3]), "reference_acc": t(ref_acc), "reference_variance": t(ref_variance),
            "exposure": max(r["exposures"][-1], 1e-3), "reference_samples": float(ref_acc[..., 3][valid].mean()) if valid.any() else 0.0}
