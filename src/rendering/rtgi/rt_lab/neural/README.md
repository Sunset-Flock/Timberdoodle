# RT Lab neural affinity

A small recurrent network predicts a latent feature vector `f` (8 values) per lab pixel. The lab's a-trous passes weight every
tap with a learned function of `|f_center - f_tap|^2` on top of the geometric tests (plane distance, normal); none of the lab's guides
(variance, ray length, SH) enters the neural filter. The filter
stays a normalized convex combination of the temporal output: the network only decides WHICH taps are averaged.

Everything neural is in this folder. Outside it: `rt_lab.cpp` (~60 lines in `tasks_rt_lab_main`, `rt_lab_frame_update`),
`rt_lab_state.hpp` (`RtLabState::neural`), `renderer.cpp` (pipelines), `ui.cpp` (settings block), `application_test_plan.cpp`
(`"neural"` key), `scene_renderer_context.hpp` (timer `NEURAL_LATENT`), `CMakeLists.txt`, and the daxa patch below.

## One shared source of truth

`rt_lab_neural_core.slang` has the feature builder, the MLP (32-32-32-8, tanh latent), the affinity and the whole a-trous pass
(ring budget, count propagation). It has no engine includes. The engine (`rt_lab_neural.hlsl`) includes it and supplies
textures through small interfaces. The trainer (`train/rt_lab_neural_train.slang`) supplies tensors and differentiates the
SAME code with Slang autodiff through SlangPy. Change the core and both sides change.

Data flow per frame: temporal output -> **latent pass** (features + MLP + reprojected previous latent -> packed f16 latent,
persistent double buffer) -> **neural a-trous passes** (replace `entry_rt_lab_atrous`; same inputs / outputs / history
feedback). Only the Custom preset uses it, the progressive reference never does.

## Workflow

```
# 1. capture (Debug exe, repo root; ~10 min for 16 sequences)
python -P src/rendering/rtgi/rt_lab/neural/train/make_dump_plan.py --out perf_tests/neural > plan.json
build/cl-x86_64-windows-msvc/bin/Debug/TimberDoodle.exe --test-plan plan.json
# 2. pair custom + reference dumps into sequences
python -P src/rendering/rtgi/rt_lab/neural/train/convert_dumps.py perf_tests/neural
# 3. train (several at once: sweep.py), weights land in neural/weights/
python -P src/rendering/rtgi/rt_lab/neural/train/train.py --data "perf_tests/neural/*.pt" --iterations 1500
# 4. A/B in the engine (neural off / on, reference error + instability + timers)
python -P src/rendering/rtgi/rt_lab/neural/train/make_ab_plan.py --weights <weights.bin> > ab.json
```

In the app: RT Lab window -> Settings -> Neural Affinity.

## Traps (all hit while building this)

- **Slang autodiff + local arrays:** loops that write local arrays MUST be `[ForceUnroll]` in `[Differentiable]` code,
  otherwise the backward pass is silently wrong (first MLP layer off by 40 %, last layer exact). Verified against torch to 1e-6.
- **SlangPy index order:** index vectors are innermost first. A `[H, W, C]` torch tensor is indexed `uint3(c, x, y)`, grid
  pixels are `(x, y)`, `Tensor.shape` is in storage order. `DiffTensor` args need a gradient buffer on the Vulkan backend
  (constants are plain `Tensor`). `out` parameters are not returned: return the value.
- **SlangPy startup:** the first process compiles the unrolled autodiff kernels (~4 minutes). `make_device()` enables the
  module + shader caches (`train/.cache`, untracked): later processes start in seconds. A Slang change recompiles.
- **`python -P`:** run the scripts with `-P` (no cwd on `sys.path`; a stray `copy.py` in `%TEMP%` shadowed the stdlib when
  Python ran from there). The scripts add their own folder to `sys.path`.
- **SlangPy device:** CUDA backend needs the CUDA toolkit's NVRTC (not installed); the trainer uses Vulkan + CUDA interop with
  `set_allow_torch_fallback(True)` (the native `slangpy-torch` bridge needs a build).
- **Tensor cores:** daxa needs `VK_NV_cooperative_vector` (patched in `deps/daxa`, implicit feature `COOPERATIVE_VECTOR`).
  Slang 2025.11 aborts ("Unhandled global inst in spirv-emit") when a cooperative vector reads from daxa's bindless
  `__DynamicResource` buffer table: `rt_lab_neural.hlsl` declares a typed `RWByteAddressBuffer` alias of the same binding.
  Weights are f16 row major (all offsets are multiples of 64 bytes). The CoopVec pipeline is only registered when the
  device has the feature. `coopVecMatMulAdd` does not take pointers in this Slang version.
- **Auto exposure** ramps from 0 over hundreds of frames: the capture plan settles 400 frames first, the trainer uses the
  per frame exposure, the engine clamps it to >= 1e-3.
- The dump needs `frames > first + dump frames + 8` (flush happens a few frames after the last block).
