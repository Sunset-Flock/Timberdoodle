# RT Lab neural affinity: training log

Every training run of the neural affinity network is appended here (newest at the bottom) with its exact configuration, timings and validation
tables, so runs can be compared over time. `train/log_run.py` appends an entry from the json summary `train.py --log-json` writes; the header
sections below describe what the numbers mean and how the data and the targets were made at each point in time.
Rule: entries are never rewritten, a correction goes in a new note under the entry.

## What is trained

`rt_lab_neural_core.slang` is the single source of truth shared by the engine and the trainer: a small MLP (hidden 32-32, leaky ReLU 0.05, 16 outputs:
an 8 value tanh latent, 6 weighting function gate logits, 1 log sharpness, 1 reserved) predicts per lab pixel a latent + a weighting function from the
temporally accumulated data. The a-trous passes (5x5 B3 spline, steps 1, 2, 4, 8, ...) weight every tap with

    kernel(offset) x plane distance x normal  (the geometric edge stopping, tolerance 1.5 px, normal power 8)  x  learned weighting(|f_center - f_tap|^2)

and nothing else. Since 2026-10-22 NO guide of the lab (variance, ray length, SH) is part of the neural filter or of the training (before, the
trainer could multiply the lab's variance guide with `--guide`; the engine did with "Keep Variance Guide").
The trainer differentiates exactly the shipped code with Slang autodiff through SlangPy. Weights file: header `RTLN`, version, parameter count, feature
mask, then float32 parameters.

Feature layouts (`--features`): `stats3` (input 32: pixel data + 3x3 neighborhood statistics), `window5` (64), `raw3` (192), `lite` (input 40, 39 used:
plus shaped neighborhood of 5 pixels x [relative log depth, perceptual radiance rgb, capped normalized ray length], detail normal, history length,
absolute depth, previous latent, constant 1). Weights feature mask: 1 window, 2 no guide inputs, 4 raylength3, 8 recurrent, 16 raw3, 32 lite.

## Loss and evaluation

- Perceptual transform `p(x) = log1p(8 * x * pi * exposure / RADIANCE_SCALE)` of the filtered rgb and of the ground truth; the loss of a sample is the mean
  over interior pixels (the margin is twice the sum of the pass steps + 2, sky excluded) and channels of `|p(out) - p(ground truth)|^2`.
- Training loss is normalized per sample by the error of the same sample's temporal-only image (`--normalize-levels 1`), so the 1 and 4 frame snapshots
  do not dominate; the reported validation numbers are raw (not normalized).
- A training sample = (sequence, snapshot frame n in {1, 4, 8, 16, 32, 64}, random crop). Recurrent training runs the latent chain over all frames
  before n, the last `--bptt` of them with gradients.
- Validation = the holdout sequence (`cam4`, two fixed 256 px windows), same table for: `temporal only` (the unfiltered accumulated image), `geometry only`
  (the a-trous passes with the geometric stopping only), `variance guide` (the lab's custom preset guide, 2 sigma, over the same passes), `neural`.
  Columns are the snapshot frame counts (history length of the temporal input).
- The network starts at the geometry only filter (last layer gain 0.02: latents ~ 0, gates uniform), so `neural` at init ~ `geometry only`.

## Data

Scene `assets/bistro_lights_compressed/bistro_lights_c.gltf`, 1280x720, the user's five cameras (`make_snapshot_plan.py` CAMERAS) x suns, lab custom preset,
400 settle frames (auto exposure), forced full PGI update. Full image sequences (`perf_tests/full/<cam>_<sun>/`, `fullseq.py`): central 720 px square crop at
(280, 0), 64 consecutive frames after a history reset (frame 1 = first frame), per frame fp16 temporal output / raw samples / variance / ray length guide /
ray stats, static geometry (position, normal, rcp pixel width, depth, detail normal, albedo), exposure per frame.

| sequence | sun (angle x, angle y) | frames | use |
|---|---|---:|---|
| cam0_s94 | 94.854, 94 | 64 | train |
| cam1_s94 | 94.854, 94 | 64 | train |
| cam2_s94 | 94.854, 94 | 64 | train |
| cam3_s94 | 94.854, 94 | 64 | train |
| cam4_s94 | 94.854, 94 | 64 | validation (holdout) |

Further reference dumps exist for suns s99 (99.9, 84.5) and s235 (235.4, 86.5) of cameras 0-3 (`perf_tests/full/*_ref.rtlndump`) but their custom runs were
not converted (two half converted `*.tmp` folders): capturing more suns / cameras is the main open lever for quality.

## Ground truth (targets)

- **v1 (until 2026-10-22):** 4096 spp progressive capture, no spatial filter in the capture, cleaned offline with 2 a-trous passes (steps 1, 2) of the lab's
  variance guide at sigma 1 (`reference_atrous_old.npy` keeps it).
- **v2 (from run 001, 2026-10-23): the progressive preset's spatial filter** (`rt_lab_default_settings(PROGRESSIVE)`: separable gaussian radius 8, plane tolerance 1.5,
  normal power 8, variance guide 0.4 sigmas, fine target samples 1M) applied to the same 4096 spp images (`train.filter_reference`, Slang
  `gauss_guide_pass`, mirrors `entry_rt_lab_blur`: taps d = 1..8 both sides, weight exp(-d^2 / (2 * 4^2)) x geometry x guide, count = sum of weight x tap count,
  horizontal then vertical, both passes read the 4096 spp per sample variance). Changes the targets by 1.6-4.1 % (mean abs relative) against v1.
  Screenshots of the raw 4096 spp image, the v2 target and the v1 target side by side: `perf_tests/neural_targets/<name>_compare.png`
  (`train/export_targets.py`; ACES filmic x 0.3, the targets are irradiance before albedo).

## Training speed measures (trainer changes)

- 2026-10-22: the geometric edge stopping of the 25 window taps is precomputed once per crop and step (`geometry_table`, cached per crop) instead of in every
  forward and every autodiff recompute; the variance guide is no longer part of the neural pass (nothing to precompute there).
- 2026-10-22: curriculum `--pretrain-passes 2 --pretrain-fraction 0.6` (2 a-trous passes = 8 px margin for the first 60 % of the iterations, then the full 4).
- 2026-10-23: `--cache-frames gpu` keeps the per frame data of all sequences on the GPU (fp16, ~1 GB per sequence) instead of reading hundreds of small
  windows from memory mapped files per sample (a recurrent sample reads up to 64 frames).
- 2026-10-23: `--tie-rotations 1` (lite layout): rotation tied first layer (8 filters x 4 quarter turns of the plus shaped patch), 328 instead of 1312 first layer
  parameters; expanded to the dense layout on export, the engine does not change.
- First iteration of a process = Slang autodiff kernel compile (~5 min, cached in `train/.cache/<features>` until the Slang module changes).

## Before this log (carried over, not measured by this log)

From the project notes (`CLAUDE.md`), trained with the older trainer (v1 targets, 4 passes always, target samples 320000, geometry + optional variance guide):

| weights file | features | params | mask | notes |
|---|---|---:|---:|---|
| `rt_lab_neural_stats3_stateless_it400.bin` | stats3 | 2640 | 0 | 400 iterations, stateless |
| `rt_lab_neural_stats3_recurrent_it200.bin` / `_it600.bin` | stats3 | 2640 | 8 | recurrent |
| `rt_lab_neural_lite_recurrent_it600.bin` = deployed `rt_lab_neural.bin` | lite | 2896 | 40 | recurrent, 600 iterations, written 2026-10-20 |

- Real data, the older recurrent setup: the network barely moved from its baseline (validation 0.1161 vs 0.1164), geometry only beat the variance guide (0.103 vs 0.116).
  "No claim that the network helps yet."
- Engine check of the lite network on the default sun (a sun it was not trained on): RMSE vs the converged reference, cam1: lite 0.2167, off 0.2132, stats3 0.2075;
  cam4: lite 0.5585, off 0.5807, stats3 0.5381. Latent pass 95-107 us + ray stats 50-57 us at 720p (coopvec). Trainer / engine parity was not checked.
- Synthetic data: the variance guide beat the network.

---

## Run 001: lite recurrent, progressive-preset targets (v2), geometry x affinity only, curriculum 2->4 passes, GPU frame cache

Logged 2026-10-23 03:10.

Command:

```
python -P src/rendering/rtgi/rt_lab/neural/train/train.py --data "perf_tests/full/cam*_s94" --holdout cam4 --features lite --recurrent 1 --iterations 400 --eval-every 100 --profile 1 --cache-frames gpu --out src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run001.bin --log-json perf_tests/neural_runs/run001.json
```

### Configuration

- features `lite` (input 40, 2896 parameters, weights feature mask 40), recurrent True, bptt 4
- rotation tied first layer: False
- iterations 400, batch 8, crop 192, lr 0.001 (cosine to 5 %), w3 gain 0.02, seed 0, loss power 2, normalize levels True
- spatial filter in the loss: 4 a-trous passes (geometry x learned affinity, target samples 1000000); curriculum: first 60 % of the iterations with 2 passes
- frames cached on the GPU: gpu
- training views: cam0_s94, cam1_s94, cam2_s94, cam3_s94
- validation views: cam4_s94@55,92, cam4_s94@287,92 (holdout `cam4`)
- weights written to `src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run001.bin`

### Time and memory

- total wall time 902 s (15.0 min), of which evaluation 8 s
- first iteration 2.6 s (Slang kernel compile / cache load, once per module change)
- steady state 2.233 s per iteration over 399 iterations (= 14.9 min for the iterations)
- peak GPU memory allocated by torch 5.22 GB, skipped non finite steps 0
- profiled split of a training step: crop 1.0 s (0 %), chain 521.7 s (58 %), filter 60.3 s (7 %), backward 310.2 s (35 %)

### Validation (perceptual log error, MSE vs the ground truth; lower is better)

**at init** (untrained network, starts as the geometry only filter)

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02566 | 0.05630 | 0.01739 | 0.01020 | 0.01009 | 0.01094 | 0.02176 |

**after 100 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02580 | 0.05265 | 0.01665 | 0.01013 | 0.01007 | 0.01092 | 0.02104 |

**after 200 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02576 | 0.05319 | 0.01683 | 0.01014 | 0.01004 | 0.01091 | 0.02114 |

**after 300 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02574 | 0.05354 | 0.01665 | 0.00957 | 0.00954 | 0.01062 | 0.02094 |

**after 400 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02577 | 0.05328 | 0.01660 | 0.00940 | 0.00924 | 0.01040 | 0.02078 |

### Summary

| | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide (lab custom guide, 2 sigma) | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural (final) | 0.02577 | 0.05328 | 0.01660 | 0.00940 | 0.00924 | 0.01040 | 0.02078 |
| neural final / neural at init | 1.004 | 0.946 | 0.955 | 0.922 | 0.916 | 0.950 | 0.949 |
| neural final / geometry only | 1.006 | 0.946 | 0.954 | 0.921 | 0.915 | 0.950 | 0.949 |
| neural final / variance guide | 1.006 | 0.997 | 1.069 | 1.196 | 1.230 | 1.247 | 1.124 |

### Notes

Raw console log: perf_tests/neural_runs/run001.log. The queued rotation tied A/B (run 002), the engine A/B and the compile timings were postponed behind the SH layout work.


---

## Run 003: SH feature set v2: 3x3 footprint through a shared pixel encoder, SH-Y / CoCg / second moments, no depth; same targets and trainer as 001

Logged 2026-10-23 03:44.

Command:

```
python -P src/rendering/rtgi/rt_lab/neural/train/train.py --data "perf_tests/full3/cam*_s94" --holdout cam4 --features v2 --recurrent 1 --iterations 400 --eval-every 100 --profile 1 --cache-frames gpu --out src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run003_v2.bin --log-json perf_tests/neural_runs/run003.json
```

### Configuration

- features `v2` (input 64, 3742 parameters, weights feature mask 72), recurrent True, bptt 4
- rotation tied first layer: False
- iterations 400, batch 8, crop 192, lr 0.001 (cosine to 5 %), w3 gain 0.02, seed 0, loss power 2, normalize levels True
- spatial filter in the loss: 4 a-trous passes (geometry x learned affinity, target samples 1000000); curriculum: first 60 % of the iterations with 2 passes
- frames cached on the GPU: gpu
- training views: cam0_s94, cam1_s94, cam2_s94, cam3_s94
- validation views: cam4_s94@55,92, cam4_s94@287,92 (holdout `cam4`)
- weights written to `src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run003_v2.bin`

### Time and memory

- total wall time 1142 s (19.0 min), of which evaluation 10 s
- first iteration 3.2 s (Slang kernel compile / cache load, once per module change)
- steady state 2.830 s per iteration over 399 iterations (= 18.9 min for the iterations)
- peak GPU memory allocated by torch 5.82 GB, skipped non finite steps 0
- profiled split of a training step: crop 1.0 s (0 %), chain 627.8 s (55 %), filter 60.5 s (5 %), backward 442.9 s (39 %)

### Validation (perceptual log error, MSE vs the ground truth; lower is better)

**at init** (untrained network, starts as the geometry only filter)

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02413 | 0.01287 | 0.01111 | 0.00998 | 0.00991 | 0.01076 | 0.01313 |

**after 100 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02418 | 0.01250 | 0.01024 | 0.00915 | 0.00901 | 0.00974 | 0.01247 |

**after 200 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02414 | 0.01282 | 0.01072 | 0.00936 | 0.00916 | 0.00989 | 0.01268 |

**after 300 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02420 | 0.01269 | 0.00921 | 0.00735 | 0.00700 | 0.00782 | 0.01138 |

**after 400 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02420 | 0.01246 | 0.00910 | 0.00739 | 0.00707 | 0.00788 | 0.01135 |

### Summary

| | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide (lab custom guide, 2 sigma) | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural (final) | 0.02420 | 0.01246 | 0.00910 | 0.00739 | 0.00707 | 0.00788 | 0.01135 |
| neural final / neural at init | 1.003 | 0.968 | 0.819 | 0.740 | 0.714 | 0.733 | 0.829 |
| neural final / geometry only | 1.003 | 0.968 | 0.819 | 0.740 | 0.713 | 0.732 | 0.829 |
| neural final / variance guide | 1.003 | 1.007 | 0.980 | 1.000 | 1.003 | 0.972 | 0.994 |

### Notes

Raw console log: perf_tests/neural_runs/run003.log. The engine does not run the v2 network yet (stage B); this run is evaluated in the trainer only. Data: perf_tests/full3 (dump v3, captured 2026-10-23).


---

## Run 005: lite STATELESS (no recurrent latent), otherwise as run 001

Logged 2026-10-23 03:53.

Command:

```
python -P src/rendering/rtgi/rt_lab/neural/train/train.py --data "perf_tests/full/cam*_s94" --holdout cam4 --features lite --recurrent 0 --iterations 400 --eval-every 100 --profile 1 --cache-frames gpu --out src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run005.bin --log-json perf_tests/neural_runs/run005.json
```

### Configuration

- features `lite` (input 40, 2896 parameters, weights feature mask 32), recurrent False, bptt 4
- rotation tied first layer: False
- iterations 400, batch 8, crop 192, lr 0.001 (cosine to 5 %), w3 gain 0.02, seed 0, loss power 2, normalize levels True
- spatial filter in the loss: 4 a-trous passes (geometry x learned affinity, target samples 1000000); curriculum: first 60 % of the iterations with 2 passes
- frames cached on the GPU: gpu
- training views: cam0_s94, cam1_s94, cam2_s94, cam3_s94
- validation views: cam4_s94@55,92, cam4_s94@287,92 (holdout `cam4`)
- weights written to `src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run005.bin`

### Time and memory

- total wall time 543 s (9.0 min), of which evaluation 13 s
- first iteration 309.1 s (Slang kernel compile / cache load, once per module change)
- steady state 0.553 s per iteration over 399 iterations (= 3.7 min for the iterations)
- peak GPU memory allocated by torch 2.63 GB, skipped non finite steps 0
- profiled split of a training step: crop 0.9 s (0 %), chain 30.0 s (6 %), filter 61.8 s (12 %), backward 436.9 s (83 %)

### Validation (perceptual log error, MSE vs the ground truth; lower is better)

**at init** (untrained network, starts as the geometry only filter)

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02566 | 0.05630 | 0.01739 | 0.01020 | 0.01009 | 0.01094 | 0.02176 |

**after 100 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02577 | 0.05271 | 0.01669 | 0.01015 | 0.01007 | 0.01093 | 0.02105 |

**after 200 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02573 | 0.05314 | 0.01684 | 0.01010 | 0.00997 | 0.01086 | 0.02111 |

**after 300 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02573 | 0.05337 | 0.01651 | 0.00930 | 0.00926 | 0.01044 | 0.02077 |

**after 400 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural | 0.02578 | 0.05291 | 0.01636 | 0.00903 | 0.00894 | 0.01022 | 0.02054 |

### Summary

| | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45843 | 0.11767 | 0.03767 | 0.01396 | 0.00822 | 0.00782 | 0.10729 |
| geometry only | 0.02562 | 0.05634 | 0.01740 | 0.01021 | 0.01010 | 0.01095 | 0.02177 |
| variance guide (lab custom guide, 2 sigma) | 0.02561 | 0.05342 | 0.01553 | 0.00786 | 0.00751 | 0.00834 | 0.01971 |
| neural (final) | 0.02578 | 0.05291 | 0.01636 | 0.00903 | 0.00894 | 0.01022 | 0.02054 |
| neural final / neural at init | 1.005 | 0.940 | 0.941 | 0.885 | 0.886 | 0.934 | 0.932 |
| neural final / geometry only | 1.006 | 0.939 | 0.940 | 0.885 | 0.886 | 0.934 | 0.932 |
| neural final / variance guide | 1.006 | 0.990 | 1.054 | 1.149 | 1.191 | 1.225 | 1.103 |

### Notes

Raw console log: perf_tests/neural_runs/run005.log. A/B of the recurrent latent against run 001 (same data perf_tests/full, targets v2, curriculum, 400 iterations). Without the recurrent chain only the target frame is evaluated per sample.


---

## Run 006: SH feature set v2 STATELESS (no recurrent latent), otherwise as run 003

Logged 2026-10-23 03:58.

Command:

```
python -P src/rendering/rtgi/rt_lab/neural/train/train.py --data "perf_tests/full3/cam*_s94" --holdout cam4 --features v2 --recurrent 0 --iterations 400 --eval-every 100 --profile 1 --cache-frames gpu --out src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run006.bin --log-json perf_tests/neural_runs/run006.json
```

### Configuration

- features `v2` (input 64, 3742 parameters, weights feature mask 64), recurrent False, bptt 4
- rotation tied first layer: False
- iterations 400, batch 8, crop 192, lr 0.001 (cosine to 5 %), w3 gain 0.02, seed 0, loss power 2, normalize levels True
- spatial filter in the loss: 4 a-trous passes (geometry x learned affinity, target samples 1000000); curriculum: first 60 % of the iterations with 2 passes
- frames cached on the GPU: gpu
- training views: cam0_s94, cam1_s94, cam2_s94, cam3_s94
- validation views: cam4_s94@55,92, cam4_s94@287,92 (holdout `cam4`)
- weights written to `src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run006.bin`

### Time and memory

- total wall time 241 s (4.0 min), of which evaluation 4 s
- first iteration 0.7 s (Slang kernel compile / cache load, once per module change)
- steady state 0.593 s per iteration over 399 iterations (= 4.0 min for the iterations)
- peak GPU memory allocated by torch 5.82 GB, skipped non finite steps 0
- profiled split of a training step: crop 0.9 s (0 %), chain 30.0 s (13 %), filter 59.8 s (25 %), backward 146.2 s (62 %)

### Validation (perceptual log error, MSE vs the ground truth; lower is better)

**at init** (untrained network, starts as the geometry only filter)

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02413 | 0.01287 | 0.01111 | 0.00998 | 0.00991 | 0.01076 | 0.01313 |

**after 100 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02419 | 0.01249 | 0.01022 | 0.00912 | 0.00896 | 0.00968 | 0.01244 |

**after 200 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02415 | 0.01281 | 0.01072 | 0.00937 | 0.00922 | 0.00997 | 0.01271 |

**after 300 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02426 | 0.01266 | 0.00923 | 0.00741 | 0.00704 | 0.00783 | 0.01140 |

**after 400 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02424 | 0.01256 | 0.00915 | 0.00739 | 0.00706 | 0.00785 | 0.01137 |

### Summary

| | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide (lab custom guide, 2 sigma) | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural (final) | 0.02424 | 0.01256 | 0.00915 | 0.00739 | 0.00706 | 0.00785 | 0.01137 |
| neural final / neural at init | 1.004 | 0.976 | 0.823 | 0.740 | 0.712 | 0.730 | 0.831 |
| neural final / geometry only | 1.004 | 0.976 | 0.823 | 0.740 | 0.712 | 0.729 | 0.831 |
| neural final / variance guide | 1.005 | 1.015 | 0.985 | 1.000 | 1.000 | 0.968 | 0.995 |

### Notes

Raw console log: perf_tests/neural_runs/run006.log. A/B of the recurrent latent against run 003 (same data perf_tests/full3).


---

## Run 004: SH feature set v2 + variance guide style weighting, STATELESS (latent dims 0..3 means, 4..7 log sigmas; mean / AND / trimmed z^2 gates, compact support, gaussian)

Logged 2026-10-23 04:11.

Command:

```
python -P src/rendering/rtgi/rt_lab/neural/train/train.py --data "perf_tests/full3/cam*_s94" --holdout cam4 --features v2 --weighting v2 --recurrent 0 --iterations 400 --eval-every 100 --profile 1 --cache-frames gpu --out src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run004_v2_w2.bin --log-json perf_tests/neural_runs/run004.json
```

### Configuration

- features `v2` (input 64, 3742 parameters, weights feature mask 192), recurrent False, bptt 4
- rotation tied first layer: False
- iterations 400, batch 8, crop 192, lr 0.001 (cosine to 5 %), w3 gain 0.02, seed 0, loss power 2, normalize levels True
- spatial filter in the loss: 4 a-trous passes (geometry x learned affinity, target samples 1000000); curriculum: first 60 % of the iterations with 2 passes
- frames cached on the GPU: gpu
- training views: cam0_s94, cam1_s94, cam2_s94, cam3_s94
- validation views: cam4_s94@55,92, cam4_s94@287,92 (holdout `cam4`)
- weights written to `src/rendering/rtgi/rt_lab/neural/weights/rt_lab_neural_run004_v2_w2.bin`

### Time and memory

- total wall time 725 s (12.1 min), of which evaluation 4 s
- first iteration 484.3 s (Slang kernel compile / cache load, once per module change)
- steady state 0.594 s per iteration over 399 iterations (= 4.0 min for the iterations)
- peak GPU memory allocated by torch 5.82 GB, skipped non finite steps 0
- profiled split of a training step: crop 0.9 s (0 %), chain 29.9 s (4 %), filter 60.1 s (8 %), backward 630.3 s (87 %)

### Validation (perceptual log error, MSE vs the ground truth; lower is better)

**at init** (untrained network, starts as the geometry only filter)

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |

**after 100 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02414 | 0.01246 | 0.01030 | 0.00915 | 0.00893 | 0.00959 | 0.01243 |

**after 200 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02414 | 0.01259 | 0.01036 | 0.00892 | 0.00865 | 0.00929 | 0.01233 |

**after 300 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02422 | 0.01343 | 0.00912 | 0.00734 | 0.00692 | 0.00766 | 0.01145 |

**after 400 iterations**

| method | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural | 0.02420 | 0.01324 | 0.00902 | 0.00734 | 0.00697 | 0.00769 | 0.01141 |

### Summary

| | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| temporal only | 0.45836 | 0.08516 | 0.03429 | 0.01453 | 0.00821 | 0.00756 | 0.10135 |
| geometry only | 0.02413 | 0.01287 | 0.01112 | 0.00998 | 0.00992 | 0.01077 | 0.01313 |
| variance guide (lab custom guide, 2 sigma) | 0.02412 | 0.01237 | 0.00929 | 0.00739 | 0.00706 | 0.00811 | 0.01139 |
| neural (final) | 0.02420 | 0.01324 | 0.00902 | 0.00734 | 0.00697 | 0.00769 | 0.01141 |
| neural final / neural at init | 1.003 | 1.029 | 0.811 | 0.735 | 0.703 | 0.715 | 0.833 |
| neural final / geometry only | 1.003 | 1.029 | 0.811 | 0.735 | 0.703 | 0.715 | 0.833 |
| neural final / variance guide | 1.003 | 1.070 | 0.971 | 0.993 | 0.987 | 0.948 | 0.996 |

### Notes

Raw console log: perf_tests/neural_runs/run004.log. A/B against run 006 (same data, stateless, same trainer): the only change is the weighting functions (--weighting v2). The functions the network switched off in runs 001 / 003 / 006 (exponential, cauchy, student-t) are replaced by variance guide style tests on the latent vector (z^2 = d^2 / (sigma_c^2 + sigma_t^2 + floor), sigmas predicted per pixel in latent dims 4..7): soft steps on the mean, on the worst dimension (AND) and on the trimmed mean, compact support on mean and worst dimension, and the gaussian likelihood. The first start of this run was recurrent and was cancelled during the kernel compile (the recurrent latent does not matter, see runs 005 / 006). Trainer only, the engine does not run it yet.

