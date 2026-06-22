# RTGI Denoiser

## Philosophy and Goals

This denoiser is built around a specific set of priorities that differ from what most GI denoiser write-ups focus on.

**Performance above all.** The entire pipeline runs at half resolution. A single integrated upscale pass at the end brings the result to full resolution. On an RTX 4080 the full denoiser — trace included — runs in under 400 microseconds. The goal is to demonstrate that RTGI does not have to be a premium-only feature reserved for cinematic workloads. It can run in fast-paced games on mainstream hardware.

**Built for real games, not tech demos.** Fast camera movement, disocclusions every frame, dynamic scenes, first-person shooters — these are first-class concerns, not edge cases. The denoiser does not assume a slow-moving camera or a mostly-static world.

**First frame after disocclusion must be acceptable.** Temporal accumulation is the backbone of almost all real-time denoisers, but it inherently fails on newly revealed geometry. This denoiser puts heavy emphasis on spatial filtering so that the result on frame zero of a disocclusion is already plausible, not a noisy mess waiting for history to build up.

**Accuracy is a goal, not a constraint.** The denoiser aims to be physically plausible and energy-conserving where possible, but it is fully willing to use dirty tricks when they produce better perceptual results. Heuristics, biased estimators, and empirical tuning are all on the table. A beautiful image that is slightly wrong beats a correct image that looks bad.

**Specialized for indirect diffuse only.** This denoiser does not handle specular, shadows, or any other signal — only cosine-weighted hemisphere indirect diffuse radiance. That specialization is a deliberate source of power. Every design decision, every tuning choice, every dirty trick can be optimized for the specific properties of indirect diffuse: it is a low-frequency signal, it does not contain hard specular edges or contact shadows, and small amounts of spatial blur are almost never perceptible. A general-purpose denoiser cannot exploit any of that. This one can, and does.

**Document the underdiscussed stuff.** Most GI denoiser write-ups cover temporal accumulation and bilateral filtering. This document also covers the smaller, less-published techniques — firefly energy redistribution, surface detail metrics, fast history for reactive blending, stochastic pre-blur for temporal stability, and others — that collectively account for a large share of the quality.

---

Real-Time Global Illumination in Timberdoodle uses a single ray per pixel at half resolution. The denoiser carries almost the full burden of producing stable, sharp, and energy-conserving indirect lighting from this extremely sparse signal.

## Ray Tracing

One cosine-weighted hemisphere ray is cast per half-resolution pixel per frame. On a miss the renderer falls back to probe-based GI radiance from the nearest cascade, which bounds ray length and gives the denoiser a useful short-ray signal for filter guidance.

Radiance is stored in a compact YCoCg + spherical-harmonic-Y representation: one coefficient of the second-order SH basis encodes directional luminance, and separate chroma channels carry color. This lets the temporal and spatial filters operate on a single luminance scalar for most decisions, keeping the heavy lifting cheap.

---

## Pre-Filtering

The pre-filter runs on a small groupshared tile and performs most of the heavy analytical work before any blurring or temporal integration happens.

### Firefly Suppression

Most production denoisers — NRD, SVGF, and others — apply firefly suppression as a variance clamp inside or just after the temporal pass. This is well-understood and works for moderate noise, but it has a stability problem: a bright outlier that enters the temporal accumulator in one frame takes several frames to bleed back out, causing a visible brightness pulse. For a denoiser targeting fast-paced games this is unacceptable. This denoiser instead runs a spatial firefly filter **before** temporal accumulation, every frame, so the energy budget entering the accumulator is always consistent and the image never pulses.

Variance clamping on a 5×5 window also simply does not suppress heavy outliers aggressively enough — the variance estimate is too noisy at this sample count and strong fireflies survive. Instead, this denoiser uses a technique more common in audio processing: clamping under the **geometric mean** of the filter window. Computed in log space, the geometric mean is extraordinarily resistant to outliers — a single pixel ten times brighter than its neighbors barely shifts it, because in log space that is just an additive offset. A multiplicative ceiling factor is applied on top of the geometric mean (values of 8–32 work well in practice), allowing genuine lighting variation through while reliably killing isolated spikes.

The geometric mean is conservative enough that a lone legitimately-bright pixel — one that traced a ray toward a small emitter — would be almost entirely killed. To combat this, the raw signal is first blurred with a minimal **5-tap star kernel** (center plus four cardinal neighbors, equal weight) before the clamp is applied. This spreads the energy of a bright pixel across five pixels, effectively letting five times more energy survive the same ceiling. The star-blur pixels are deliberately excluded from the geometric mean calculation so that the ceiling estimate stays statistically independent of the signal being clamped.

```
# # # # #       # = geometric mean ring  (firefly ceiling estimate)
# # * # #       * = star blur arm        (energy distribution)
# * @ * #       @ = center pixel
# # * # #
# # # # #
```

The star blur introduces an unconditional minimum blur that no downstream pass can remove — a dirty trick by conventional denoiser standards. For indirect diffuse, a low-frequency signal where sub-pixel sharpness is never meaningful, this trade-off is entirely acceptable. Specializing for one signal type makes compromises like this available.

The clamp is still conservative overall. To recover the remaining suppressed energy, the pre-filter stores the per-pixel clamping factor — the ratio of raw to clamped value — in a texture. A later pass reads this and redistributes the deferred energy back into the image. Energy is not lost, only deferred. More on this in the pre-blur section.

### Surface Detail Metric

To decide how aggressively to blur a pixel, the denoiser needs to know how much albedo detail the underlying surface carries. On a brick wall or a mesh grille, the albedo pattern hides noise and we can use a tighter blur to preserve more lighting detail. On a plain white wall, any denoising imperfection is immediately visible, so we must blur wide and soft.

A 3×3 window of neighbors is examined. Each neighbor's depth is used to reconstruct its view-space position, which is then tested against the plane defined by the center pixel's depth and face normal. Neighbors that lie close enough to that plane are considered to be on the same surface; those that don't are on a different geometric surface and are excluded. A small tolerance is allowed to handle gently curved surfaces without over-rejecting valid neighbors.

For the valid neighbors, albedo values are loaded. The **coefficient of variation** — the ratio of standard deviation to mean — is computed over those albedo values. The coefficient of variation is a dimensionless measure of relative spread: a value near zero means all neighbors have nearly identical albedo (uniform surface), a high value means there is significant albedo variation across the window (textured or patterned surface). Unlike raw variance, it is scale-invariant, so a white brick wall and a grey brick wall with the same pattern produce the same metric value. The coefficient of variation is also commonly used in variable rate shading to detect perceptually complex regions and decide where to reduce shading rate — the same property that makes it useful there makes it a natural fit as a surface detail metric here.

This coefficient of variation is used directly as the surface detail metric. On a brick wall we can use a smaller blur radius and show more lighting detail — the albedo pattern hides residual noise. On a plain white wall we must blur softly and wide — with no albedo detail to distract the eye, any imperfection in the indirect lighting is immediately visible. So high coefficient of variation allows a tighter kernel; low coefficient of variation demands a wider one.

### Ray Length Guide and Contact Hardening

By default the denoiser uses a wide blur. The surface detail metric tightens it where surface complexity allows. But surface complexity alone is not the only signal available — geometric proximity is another. Areas where GI rays hit nearby geometry are in contact zones: darker, geometrically busy, and naturally noise-hiding. We can exploit this to tighten the blur further.

Ray length is blurred across the same 3×3 window and plane-test validity mask already computed for the surface detail metric, reusing the work for free. The resulting blurred ray length serves as a coarse AO and geometric tightness estimate — short average ray lengths indicate the pixel is close to occluding geometry. To make the metric more sensitive to close hits, the blurred ray length is squared twice (raised to the fourth power), aggressively pushing small values toward zero while leaving distant hits largely unaffected. The result drives the blur radius smaller in contact and corner regions, producing a cheap contact-hardening effect.

This must still be capped by the surface detail level. A smooth white surface close to a wall has short rays but no albedo complexity — noise is just as visible there and the blur must stay wide. The surface detail metric provides this cap, preventing ray length from over-tightening on geometrically close but visually flat surfaces.

### Footprint Quality

The fraction of inner neighbors that pass a plane-distance test measures how geometrically coherent the pixel's local neighbourhood is. A low value means the pixel sits in a complex region — a corner, a thin edge — where stochastic spatial samples likely come from unrelated surfaces. This shrinks the downstream denoising kernel to keep mixing local.

### Filter Guide

The three signals above — footprint quality, ray length guide, and surface detail metric — are combined into a single scalar that controls denoising kernel width throughout the rest of the pipeline. It is also temporally accumulated so that post-blur has a stable, flicker-free version of this guide across frames.

---

## Pre-Temporal Stochastic Blur

Before temporal integration, a stochastic disc blur recovers brightness lost to the firefly clamp and dramatically improves the stability of the temporal pass input.

Eight random disc samples are drawn per pixel using a center-focused concentric distribution. The disc radius is driven by the filter guide squared — squaring gives a sharper radius reduction in geometrically complex regions. Each sample is weighted by plane distance and normal similarity, and critically, by the firefly suppression ratio of the *sampled* pixel. Bright pixels that were clamped donate their full pre-clamp energy outward, recovering suppressed energy across the neighborhood rather than losing it.

This pass is the primary mechanism for converting the raw 1-SPP signal into something the temporal accumulator can track. Without it, the accumulator constantly chases per-frame noise and either over-ghosts or stays too reactive.

---

## Temporal Accumulation

The temporal pass reprojects the previous frame's half-resolution accumulation and blends the new pre-blurred signal against it.

### Reprojection

World position is reconstructed from depth and reprojected into the previous frame's clip space. Four bilinear history neighbors are gathered and each is tested against the current surface using plane distance and normal similarity. The resulting validity weights determine whether reprojection succeeded — below a threshold, the pixel is treated as disoccluded and history is discarded.

Sample count accumulation uses a soft-normalization scheme: a small fractional power of the total validity weight is used to partially normalize the count. Full normalization causes streaking on thin geometry; no normalization causes perpetually low counts on partial disocclusions. The compromise significantly improves temporal stability and any remaining streaking is largely hidden by post-blur.

### Fast History

A short exponential moving average of luminance mean and relative variance runs alongside the main accumulation, inspired by the "Stochastic all the things" technique. It converges in roughly four frames, giving a reactive picture of recent signal behavior.

- **Mean divergence** between the fast and the slow history reduces confidence in the slow history, increasing blend weight and making the accumulator reactive to genuine lighting changes.
- **Fast relative variance** increases confidence when the signal is noisy, which keeps the accumulator stable and prevents chasing sensor noise.

### Temporal Firefly Filter

Once the fast history has had a few frames to warm up, a secondary firefly suppression clamps the incoming frame's luminance against a ceiling derived from the fast mean and fast standard deviation. This catches bright outliers that slip through the spatial firefly filter, particularly around disocclusions and animated lights.

### Slow Statistics

A second pair of statistics — mean and relative variance — accumulates at the same blend rate as the color itself, giving a stable longer-term picture of the signal's noise level. The variance updates slightly faster than the color to remain reactive to changing conditions. These statistics are written to a persistent texture and consumed by post-blur.

---

## Post Blur

A two-pass separable bilateral filter applied after temporal accumulation provides the primary spatial denoising of the accumulated signal.

### Kernel Width

Kernel width is driven jointly by the accumulated filter guide and temporal stability. Young pixels with little history receive a wide kernel regardless of the guide — spatial coverage compensates for low temporal confidence. Mature pixels follow the filter guide directly: tight on textured surfaces and complex geometry, wide on flat open surfaces.

### Sample Weighting

Each tap is weighted by:
- **Plane-distance** — a hard binary test; samples on a different surface plane are fully rejected.
- **Normal similarity** — penalises mismatched surface orientation.
- **Gaussian envelope** — standard spatial falloff scaled to the current kernel width.
- **Sample count** — upweights pixels with more temporal history to hide disocclusion flicker.

### Luminance One-Sided Clamp

Samples brighter than the center pixel beyond a tolerance derived from the slow temporal variance receive a reduced weight. The tolerance widens when variance is high (noisy signal, more permissive) and narrows when variance is low (stable signal, tighter). Samples darker than the center are never penalised — valid shadow samples on the same surface pass freely. This prevents bright background luminance from leaking into darker foreground regions while leaving shadow contact detail intact.

---

## Upscale and SH Resolve

The final pass runs at full resolution and jointly upscales the half-resolution result and resolves the directional SH representation.

### Upscale

A tent filter in half-resolution space approximates a Gaussian-5 with only three taps. Weights are modulated by the sub-pixel position of each full-resolution sample within its half-resolution texel, giving sub-pixel-accurate reconstruction. A plane-distance test and normal similarity penalty prevent half-resolution samples from bleeding across depth or normal discontinuities. A geometric-distance-only fallback accumulator handles thin geometry where the primary accumulator finds no valid neighbors.

### SH Resolve

At resolve time the full-resolution detail normal — higher frequency than the half-resolution face normal used during tracing — is dotted against the SH coefficient to recover directionally weighted irradiance. This lets the indirect lighting respond to fine normal-map detail despite being reconstructed at half resolution.
