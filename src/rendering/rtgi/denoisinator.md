# Denoisinator

The realtime ray traced diffuse indirect GI tracer + denoiser powering Timberdoodle's RTGI.

---

## Why does this exist?

I built the Denoisinator because I was disappointed with basically every other realtime GI denoiser, for a few concrete reasons.

**Fixed rays-per-pixel is inflexible.** Almost every common denoiser shoots a constant `X` rays per pixel. But the amount of information a pixel needs is *not* constant. A pixel that has been on screen and converging for 60 frames needs almost nothing — one ray, sometimes not even that. A freshly *disoccluded* pixel has zero history and needs a whole burst of rays *right now* to look acceptable. Spending the same budget on both is wasteful where it's converged and starved where it matters. The Denoisinator spends its ray budget where it is actually needed.

**The performance is almost always terrible, because everything is built for full resolution.** This is the big one. Gamers do not care about mathematically perfect, ground-truth indirect lighting. They care about the game looking good while running fast. If you trace and denoise at **half resolution** (and lower for the slowly-varying spatial signal that indirect diffuse *is*), you get ~90% of the visual benefit of RT GI at a *fraction* of the cost. Good-performance RT for games is absolutely possible — I want that. I dislike that every game's "RT option" is an ultra-expensive toggle that tanks your framerate. **It does not have to be this way.**

**A note on ReSTIR:** it's very cool tech and I respect it a lot — but it makes disocclusions *worse*, not better. Reservoir resampling leans on temporal/spatial history that a disoccluded pixel simply doesn't have, so exactly the case I care most about is the case it handles worst. Not for me.

**Purpose-built beats general-purpose.** The Denoisinator is not a general reflection/GI/AO denoiser. It is built *specifically* for **diffuse indirect GI**, and that focus unlocks optimizations a general denoiser can't take:

- It **blurs consistently across frames** (ignoring the optional disocclusion-blur ramp). Diffuse indirect is smooth and slowly varying, so a stable, unchanging filter footprint avoids the shimmering you get when a denoiser keeps changing its blur width frame to frame.
- It **does not use variance guiding.** Variance-driven filtering is far too unstable for indirect GI — the signal is too noisy for the variance estimate itself to be trustworthy, so it just chases noise. Instead it guides on *geometric (log) radiance means* and *ray-length/occlusion*, which are stable.

---

## Design philosophy: fast spatial → temporal → cleanup spatial

Denoisers usually pick one of two orderings:

1. **spatial → temporal** — fast reaction, cheap stochastic spatial filters, but shimmers in motion and bands on disocclusion.
2. **temporal → spatial** — resolves to ground truth but has poor reaction time and needs an expensive fully-smooth post filter.

The Denoisinator does **both**: a cheap stochastic **pre-blur** *before* temporal, then a small **post-blur** *after*.

- The pre-blur result is what gets temporally integrated, so the pre-blur is allowed to be a sloppy, sample-efficient stochastic blur — the temporal pass stabilizes it.
- The temporal pass integrates *already spatially-filtered* values, so its fast history is reliable and its reaction time is good.
- Because the image is already pre-integrated, the **post-blur can be tiny and smooth** — just enough to clean up residual shimmer, without the huge expensive filter a pure temporal→spatial design needs.
- The post-blur can also alter the result temporarily (e.g. widen on disocclusion) *without polluting the temporal history*.
- It integrates naturally with the upscale at the end, which is really just another spatial blur.

---

## The pipeline (walking the default settings)

Everything runs at **half resolution** and is upscaled at the very end. Default ray budget is `1.0` rays/pixel with a `0.5` guaranteed minimum, repacked ray dispatch and ray redistribution **on**.

### 1. Temporal Reproject
Runs *first*, before tracing. For each pixel it finds where its history lives in the previous frame (bilinear footprint + custom weights), computes the reprojected sample count, and — crucially — the **ray demand**: how far below the target history each pixel is. A parallax-stretch penalty catches grazing surfaces that get smeared when the camera moves. This pass writes only addressing + weights + counts, so the trace can read them to decide how many rays to shoot.

### 2. Distribute Rays
The flexible ray budget lives here. A global budget is split across 8×8 tiles:
- A **rotating checkerboard floor** (`min_ray_budget = 0.5`) guarantees half the geometry pixels always get their base ray — no pixel drops below half coverage over two frames, even under heavy drain.
- The **discretionary pool** on top is handed out proportional to the *squared* history deficit, so the freshest disocclusions win the scarce rays instead of everyone getting an even, useless sprinkle.
- Distribution is done cooperatively by one wave using prefix sums, and rays are written into a flat ray list traced via an indirect dispatch sized to the actual ray count.

This is the whole point: **converged pixels cost ~1 ray, disoccluded regions get a burst, and the total stays within budget.**

### 3. Trace
Ray-traced (or optional compute/inline) diffuse rays from the ray list. Each ray stores its radiance, its **raw hit distance**, and the geometric mean of the pixel's rays in log space. Ray directions are seeded with frame index *and history length* so a pixel doesn't re-shoot identical directions across frames.

### 4. Blend Rays
Averages a pixel's traced rays into a single directional-SH diffuse + CoCg chroma value, converts each ray's hit distance into a bounded `[0,1]` **ray shortness**, and averages that too. Writes the per-pixel **ray count** (the single source of truth used everywhere downstream).

### 5. Pre-filter (firefly clamp + guides)
The energy-preserving firefly stage:
- **Firefly clamp** against a geometric-mean ceiling built from the *surrounding* neighborhood only (a pixel never contributes to the ceiling used to clamp itself). The ceiling tightens by `1/sqrt(rays)` — well-sampled pixels need less headroom.
- **Center blur** recovers the energy the clamp removed by averaging a quad *before* clamping — but only enough to keep the pre-clamp averaged ray count ~4 everywhere, so disocclusions (many rays/pixel) don't get over-averaged and read too bright.
- Produces the **ambient occlusion guide** (from ray shortness — short/contact rays mean less blur), **footprint quality**, and the geometric radiance mean used for guiding.
- Uses an **exposure-aware perceptual radiance floor** (Weber-Fechner) when building log-radiance means so tiny inputs don't explode the geometric mean below anything a viewer could perceive.

### 6. Pre-blur (default: 2 iterations, 10 samples, base width 64)
A cheap, sample-efficient **stochastic Poisson-disc blur** with per-frame rotation. This is the "fast spatial" pass — it's allowed to be noisy because temporal cleans it up. Its radius is scaled by the AO guide (occluded/contact areas blur less) with a floor, and it guides on the geometric radiance mean.

### 7. Temporal Accumulate (history 64 frames, fast history 4 frames)
Reprojects color, AO guide, geometric-mean and fast-history statistics, then blends the new pre-blurred sample in. The blend is **batch-weighted by `sqrt(rays_shot)`** rather than the raw ray count: the rays within a frame are correlated (and already pre-blurred), so their *effective* independent count is closer to `sqrt(k)` — this stops disocclusion bursts from hijacking history and keeps convergence smooth. A short **fast history** enables an aggressive temporal firefly clamp, and the parallax penalty drops smeared grazing history. **No variance guiding** — the AO guide and geo-radiance mean are the stable signals used instead.

### 8. Post-blur (default: bilateral, width 10, stride 2)
The small "cleanup spatial" pass — vertical + horizontal separable bilateral (groupshared 16×4 tiles, with a swizzled vertical pass for cache coherence). It blurs **consistently every frame** (the disocclusion-blur ramp is the only exception), guided by the AO guide radius scaling (with its own floor) and an optional geometric-radiance weight guide. The center taps are always trusted to hide the quad-prefilter's 2×2 boundary artifacts.

### 9. Upscale + SH resolve
Upscales the denoised half-res diffuse to full resolution and resolves the directional SH against the full-res surface. Since indirect diffuse is smooth, this final spatial step is also effectively free denoising.

---

## The short version

Trace *adaptively* (cheap where converged, bursty on disocclusion), denoise at *half res* with a *fast stochastic spatial → temporal → tiny cleanup spatial* pipeline, guide on *stable* signals (occlusion + geometric radiance, never variance), preserve firefly energy instead of just clamping it, and keep the filter *consistent* frame to frame. The result is RT diffuse GI that actually fits a game's frame budget — because it doesn't have to be ultra-expensive to look good.
