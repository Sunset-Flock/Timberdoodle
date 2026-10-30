"""Full image training sequences, streamed from disk (written by convert_full.py).

A sequence folder holds the static geometry (float32 .npy [H, W, C], loaded onto the GPU: about 35 MB per sequence), the ground truth, and the
per frame data as float16 .npy files [frames, H, W, C] that stay on disk (memory mapped): only the window a training sample needs is read.
load() returns a dict in the format of dataset.build_recurrent_sequence, whose per frame entries are lazy: frame[key][window] reads that window.
windowed() cuts a fixed window out of a sequence (used for the evaluation: whole 720 px frames would make every evaluation minutes long).
"""

import json
import os

import numpy as np
import torch

STATIC = ("position", "normal", "rcp_width", "depth", "detail_normal", "albedo")
FRAME = ("acc", "raw", "variance", "ray", "ray_stats")
FRAME_SH = ("sh_y", "cocg", "sq")  # dump v3 sequences only (sh_y 4 channels, cocg 2, sq 3, f32)
CHANNELS = {"raw": 4, "ray": 2, "sh_y": 4, "cocg": 2, "sq": 3}


class _Window:
    """frame[key]: indexing with a (y slice, x slice) window reads exactly that window of the memory mapped frame."""

    def __init__(self, array, oy=0, ox=0):
        self.array = array
        self.oy = oy
        self.ox = ox

    def __getitem__(self, win):
        ys, xs = win
        view = self.array[ys.start + self.oy: ys.stop + self.oy, xs.start + self.ox: xs.stop + self.ox]
        if isinstance(view, torch.Tensor):  # frames cached on the GPU (fp16): a slice, no host transfer
            return view.float().contiguous()
        return torch.from_numpy(np.ascontiguousarray(view)).to("cuda").float()


class _ZeroWindow:
    """A key the sequence does not hold (older sequences without SH data, or keys the feature set does not read): zeros of the window."""

    def __init__(self, channels):
        self.channels = channels

    def __getitem__(self, win):
        ys, xs = win
        return torch.zeros(ys.stop - ys.start, xs.stop - xs.start, self.channels, device="cuda")


class LazyFrame:
    def __init__(self, arrays, index, exposure, frame, oy=0, ox=0):
        self.arrays = arrays
        self.index = index
        self.exposure = exposure
        self.frame = frame
        self.oy = oy
        self.ox = ox

    def __contains__(self, key):
        return key in FRAME or key in FRAME_SH or key in ("exposure", "frame")

    def __getitem__(self, key):
        if key == "exposure":
            return self.exposure
        if key == "frame":
            return self.frame
        if key not in self.arrays:
            return _ZeroWindow(CHANNELS[key])
        return _Window(self.arrays[key][self.index], self.oy, self.ox)


def load(folder, cache="none", skip=()):
    """cache = "gpu": the per frame data (fp16, about 1 GB per 64 frame 720 px sequence, 1.4 GB with the SH data) is read once and stays on the GPU: a training sample then
    only slices tensors instead of reading hundreds of small windows from the memory mapped files (the recurrent chain of a sample reads up to 64 frames).
    skip = keys the feature set never reads (served as zeros, neither cached nor streamed)."""
    meta = json.load(open(os.path.join(folder, "meta.json")))
    seq = {k: torch.from_numpy(np.load(os.path.join(folder, f"{k}.npy"))).to("cuda") for k in STATIC}
    seq["reference"] = torch.from_numpy(np.load(os.path.join(folder, "reference.npy"))).to("cuda")
    keys = [k for k in FRAME + FRAME_SH if k not in skip and os.path.exists(os.path.join(folder, f"{k}.npy"))]
    if cache == "gpu":
        arrays = {k: torch.from_numpy(np.load(os.path.join(folder, f"{k}.npy"))).to("cuda") for k in keys}
    else:
        arrays = {k: np.load(os.path.join(folder, f"{k}.npy"), mmap_mode="r") for k in keys}
    seq["frames"] = [LazyFrame(arrays, i, max(e, 1e-3), f) for i, (e, f) in enumerate(zip(meta["exposures"], meta["frame_indices"]))]
    seq["exposure"] = max(meta["reference_exposure"], 1e-3)
    seq["name"] = os.path.basename(os.path.normpath(folder))
    seq["arrays"] = arrays
    return seq


def windowed(seq, y0, x0, size):
    """The same sequence cut to one fixed window (static data copied, frame data still streamed)."""
    win = (slice(y0, y0 + size), slice(x0, x0 + size))
    out = {k: seq[k][win].contiguous() for k in STATIC + ("reference",)}
    out["frames"] = [LazyFrame(f.arrays, f.index, f.exposure, f.frame, f.oy + y0, f.ox + x0) for f in seq["frames"]]
    out["exposure"] = seq["exposure"]
    out["name"] = f"{seq['name']}@{x0},{y0}"
    return out
