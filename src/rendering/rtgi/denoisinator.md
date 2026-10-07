# Denoisinator

The realtime ray traced indirect GI tracer + denoiser powering Timberdoodle's RTGI. It carries two signals through one pipeline: **diffuse indirect** (its original reason to exist) and **specular indirect** (reflections), both traced, filtered and accumulated at half resolution under one shared ray budget.

---

## Why not use existing denoisers such as NRD?

I built the Denoisinator because I was disappointed with most other realtime GI denoisers, for a few concrete reasons:

**They all assume to get one X rays per pixel** Almost every common denoiser shoots a constant `X` rays per pixel. Most raytracing systems also always shoot rays evenly distributed across the screen. 

But this is highly inefficient. T amount of new rays a pixel needs is *not* constant. A pixel that has been on screen and converging for 60 frames needs almost nothing — maybe a ray every few frames just for temporal reaction. A freshly *disoccluded* pixel has zero history and needs a burst of rays *immediately* to look acceptable. 

Off the shelve denoisers like NRD, even DLSS-RR can not properly handle differing ray amounts over the screen, they are simply not build for it. This denoiser heavily relies on dynamic ray allocation to reduce cost and increase quality.

**They are almost always quite costly, because everything is built for full resolution.** This is the big one. Gamers do not care about mathematically perfect, ground-truth indirect lighting. They care about the game looking good while running fast. If you trace and denoise at **half resolution**, you get ~90% of the visual benefit of RT GI at a *fraction* of the cost. Well performing RT for games is absolutely possible — I want that. I dislike that every game's "RT option" is an ultra-expensive toggle that tanks your framerate. **It does not have to be this way.**

**A note on ReSTIR:** it's very cool tech and I respect it a lot — but it makes disocclusions *worse*, not better. Reservoir resampling leans on temporal/spatial history that a disoccluded pixel simply doesn't have, so exactly the case I care most about is the case it handles worst. The Denoisinator does borrow one idea from that family — a sparse, *spatially* resampled pioneer guide (see Pioneer guiding) — but never depends on per-pixel temporal reservoirs.

**Purpose-built beats general-purpose.** The Denoisinator is built around **indirect GI**, and that focus unlocks optimizations a general denoiser can't take. Specular rides along through every pass.

- It **blurs consistently across frames** (ignoring the optional disocclusion-blur ramp). Diffuse indirect is smooth and slowly varying, so a stable, unchanging filter footprint avoids the shimmering you get when a denoiser keeps changing its blur width frame to frame.
- It **does not use variance-guided blurring.** Variance-driven filtering is far too unstable for indirect GI — the signal is too noisy for the variance estimate itself to be trustworthy, so it just chases noise. The blurs guide on *geometry*, *perceptual (log) radiance means* and *ray length / occlusion*, which are stable. (Variance is only used inside the temporal fast history, for anti-lag and temporal firefly clamping — never to size a blur.)

---

## Denoiser structure

Each frame runs nine stages, all at half resolution until the final upscale:

1. **Pioneer rays** — a sparse trace of uniform-hemisphere rays (one per 4×4 cell), spatially resampled into one bright hit per pioneer cell. Every main ray fetches one of those hits from a randomly rotated neighborhood and bends toward it.
2. **Reprojection** — finds each pixel's history in the previous frame and applies the parallax penalty.
3. **Ray allocation** — decides how many rays each pixel requests per signal (history deficit × diffuse/specular brightness share × material factors, capped per pixel), then splits the hard ray budget across the screen: a guaranteed base minimum, extras by request, and leftover raising the base. The result is written as a flat ray list.
4. **Ray tracing** — traces the ray list (guided diffuse rays, GGX specular rays), then blends each pixel's rays into compact guides (log radiance mean, ray shortness, hit distance).
5. **Prefilter** — firefly clamp against a neighborhood ceiling, energy bookkeeping, AO and radiance guides, and filling ray-less pixels from same-surface neighbors.
6. **Preblur** — a pre-temporal stabilizer: ReSTIR-like stochastic spatial reuse over rotating Poisson taps, which lowers frame-to-frame variance before accumulation and puts the clamped firefly energy back. Its output is what gets accumulated.
7. **Temporal** — accumulates the pre-blurred signal into history. Diffuse uses a fast history for anti-lag; specular uses a ReBLUR-style surface + virtual motion reprojection.
8. **Post blur** — a small, frame-consistent bilateral cleanup that never feeds back into the history.
9. **Upscale** — brings both signals to full resolution, resolves the diffuse SH against the full-res normal, and composites diffuse and specular.

---

## Core building blocks

These ideas show up in nearly every pass.

**Two signals.**
- *Diffuse* is stored as a directional **SH-Y + CoCg** pair: `float4 sh_y` (direction × luma in `.xyz`, luma in `.w`) plus `float2 cocg` chroma. The direction lets the final upscale resolve the irradiance against the full-res normal.
- *Specular* is an **RGBA16F**: rgb radiance, `.a` = hit distance (clamped to 1000), which drives the reflection blur radius and the virtual-motion reprojection.
- All radiance is stored pre-scaled by `RTGI_RADIANCE_SCALE = 1e4` to keep half floats away from denormals; the upscale divides it back out.

**Perceptual (log) space.** Eyes perceive brightness logarithmically (Weber–Fechner), so filter decisions — firefly ceilings, radiance guides, the diffuse/specular comparison — are made on `log(max(v, floor))`. Averaging in log space is a **geometric mean**, which is robust against the very outliers the filters are fighting. The floor is exposure-aware (`inv_exposure × RTGI_RADIANCE_SCALE × 1e-3`): anything darker is indistinguishable from black at the current exposure, so it must not drag the log mean towards −∞.

**Half resolution.** Everything from reproject to post-blur runs at half resolution. `gen_gbuffer` picks one *representative* full-res sub-pixel per 2×2 quad and writes half-res **linear view depth**, face normal, packed detail normal + roughness (one `R32_UINT`, persistent so the previous frame is available) and albedo + metalness. All half-res position reconstruction goes through the RTGI depth helpers.

**History counts are counted in rays, not frames.** A pixel that shoots 4 rays in a frame adds 4 samples of history. Up to `max_temporal_samples` (64) the count grows linearly; above it, the history decays by one window per frame, so more rays per frame genuinely mean less noise rather than just a shorter window.

---

## The pipeline (walking the default settings)

Order in `tasks_rtgi_main`: **pioneer guide → reproject → distribute → trace → blend → pre-filter → pre-blur → accumulate → post-blur → upscale.**

Default budget: **0.6 rays per half-res pixel** (diffuse + specular combined), `min_ray_budget = 0.125`, `max_rays_per_pixel = 12`, repacked ray dispatch and ray redistribution **on**, specular **on**.

At **full resolution** that's **0.15 rays per pixel**, about one ray per 6–7 screen pixels, shared between diffuse *and* reflections. The pioneer trace adds 1/64 ray per full-res pixel (one per 4×4 half-res cell), for roughly **0.17 rays per pixel in total**. At 1440p that's about 550k main rays plus 58k pioneers per frame.

### 0. Pioneer guiding (pioneer trace → horizontal resample → vertical resample → per-ray fetch in the trace)
A cheap, sparse exploration pass that tells the real rays where the light is.

- **Pioneer trace.** One pioneer ray per 4×4 half-res cell (`RTGI_GUIDE_PIONEER_GRID_DIV = 4`), with the cell position rotating every frame. Pioneers shoot **uniform-hemisphere** directions, *not* cosine: the main rays are already cosine distributed and cover the pole well, so the pioneers' job is to *complement* them. The light the main rays miss is grazing light — its contribution is `L·cos`, but cosine sampling only hits it with `cos` density, so it is rare and bright, i.e. noise. Uniform pioneers find it; cosine pioneers would only re-find what the main rays already see.
- **Resample (RIS).** Each pioneer stores a (hit position, brightness) pair — never a direction. A separable horizontal + vertical reservoir pick over a 64-pixel window (stride 2, position-gated with a 32-pixel distance threshold) chooses one bright hit per **pioneer cell** (`pioneer_guide_hit_y`). Splitting H then V gives exactly the same odds as one flat 2D pick, at `2N` instead of `N²` taps.
- **Per-ray fetch = reconnection.** There is no per-pixel resolve pass. Each main ray picks one cell's hit itself (`rtgi_fetch_ray_guide`, details below) and only then turns it into a direction: `normalize(hit − ray origin)`, computed for that exact ray. One pioneer sample can therefore serve every nearby pixel correctly, not just the one it was cast from.
- **The guide must be fuzzy.** Guiding has to be spread out spatially across pixels. If a compact group of nearby pixels all bends toward the same direction, the guide's error is shared by the whole group: when the pick changes from frame to frame, the whole group brightens or darkens together. That shows up as very visible **bubbling at the resolution of the pioneer grid**, blobs the size of a few pioneer cells that no blur can remove, because the noise is correlated across exactly the area a blur would average over. Making the guide fuzzy, so neighboring pixels follow different picks, turns that blob-shaped error into fine per-pixel noise that the pre-blur and temporal passes average away. That is why every stage is deliberately wide in screen space:
  - the pioneer resample gathers candidates over a **64-pixel window** (with a 32-pixel distance gate), so each cell's pick comes from a large, varied pool rather than its immediate surroundings;
  - the per-ray fetch reads from a **randomly rotated disc of about 9×9 cells** (±16 half-res pixels) around each pixel, so neighboring pixels and the rays within one pixel land on different cells.
- **Use.** Main diffuse rays are **bent** toward the guide with a disc warp (Schlick-style pull, `guide_concentration = 0.92`) and re-weighted by `pdf_cosine / pdf_guide`, so the estimate stays unbiased. Rough specular rays mix the guide lobe with GGX VNDF sampling via one-sample MIS (`specular_guide_mix = 0.5`).

**Why redirect the rays at all.** Guiding is not only about catching grazing lights: it gives a much clearer image overall. Every bent ray is spent on a direction where the pioneers actually found light, instead of being spread evenly over a hemisphere that is mostly dim. The same ray budget then lands where it reduces the most visible noise. That matters a lot at our rates (well under one ray per screen pixel), where a wasted ray is a large fraction of a pixel's information.

**Why not reuse the pioneer hits directly, like ReSTIR.** ReSTIR-style methods reuse the *paths* they already traced: a pixel picks a good neighbor's sample and uses that sample's radiance as its own estimate. That only works with enough candidates per pixel. We have one pioneer per 4×4 half-res cell, about 1/64 of a ray per screen pixel. Reusing those few hits directly would give every pixel in a region the same handful of samples, which looks blotchy and isn't anywhere near a clear image.

So the pioneers don't *provide* the radiance; they only provide a **direction**. Each pixel's own main rays are bent toward that direction and traced fresh from that pixel, with the pdf weight keeping the estimate unbiased. This way:
- every pixel still gets its own independent samples, so coverage and image clarity come from the full main ray budget, not from the sparse pioneers;
- the pioneers only decide *where* those samples go, and a stale or wrong guide costs variance, never correctness.

**Every ray picks its own guide — why this makes guiding work with dynamic ray allocation.**

The trap: a guide is shared information. If every ray of a pixel, or every pixel of a neighborhood, reads the *same* guide in one frame, all those rays bend toward the same direction. They are no longer independent samples. A pixel that receives a burst of 12 rays then integrates one frame's guide 12 times, and a whole region of fresh pixels integrates the same handful of guides at once. A lucky or unlucky pick gets baked into many samples simultaneously, which shows up as splotches and stripes exactly where the allocator concentrated rays. A per-pixel resolved guide (an earlier design) had this problem: neighbouring pixels picked from nearly the same candidates and ended up with nearly identical guides.

The fix is to make the guide a **per-ray random choice from a wide neighborhood** (`rtgi_fetch_ray_guide`):

1. **Many candidates.** The vertical resample leaves one RIS-picked bright hit per pioneer cell. A ray chooses among the cells within `RTGI_GUIDE_STOCHASTIC_FETCH_CELL_RADIUS` = 4 cells around its pixel (about ±16 half-res pixels), roughly 9×9 cells, each holding an independently resampled pick.
2. **Different taps per ray.** A pixel walks one Poisson disc sequence (`g_Poisson16`, the same pattern the pre-blur uses) across its rays: ray `i`, attempt `a` uses tap `(i·3 + a) mod 16`. A burst of rays therefore reads well-spread, *distinct* cells and bends toward *different* guides.
3. **Different kernels per pixel.** The disc is rotated by a random per-pixel angle and centered on the pixel's exact position inside its cell. Without that, all pixels of a 4×4 block would walk the same taps, read the same cells and the guide field would be spatially uniform again. With it, neighbouring pixels land on different cells, so the guides are decorrelated spatially as well.
4. **Exact direction.** The chosen hit is reconnected to the ray's own position, so borrowing a hit from 16 pixels away does not tilt the direction. A cell is rejected if its reference pixel is sky, fails the resample's distance gate, or the reconnected direction points below the surface; after 3 failed taps the ray samples unguided.

The result is that the rays of a super-sampled pixel behave like independent samples again: each one follows a different guide, and neighbouring pixels follow different guides too. Their errors average out within the burst and within the pre-blur kernel instead of adding up. That is what lets guiding and dynamic ray allocation coexist: the allocator can hand a disocclusion a burst of rays in one frame without the burst inheriting a single frame's guide, and without needing to fall back to unguided rays for everything but the first one. (An earlier attempt guided only the first ray per pixel; it avoided the splotches but threw away the guide for most of a burst and caused a slight glow-up on disocclusion.)

The wide radius is the important knob: widening it from 1 to 4 cells, together with the per-pixel rotation, improved temporal stability a lot. The per-pixel ray cap (`max_rays_per_pixel`) still helps on top by spreading bursts over more frames, i.e. over more independent sets of pioneer picks.

Correctness doesn't depend on any of this: whatever guide a ray uses, its pdf weight is computed for that guide, so the estimate stays unbiased. The fetch only decides how correlated the samples are.

### 1. Temporal reproject — where each pixel's history lives
Runs *before* tracing, so the trace already knows each pixel's history.

- **Addressing.** Finds where the pixel's history lives in the previous frame: a bilinear footprint with custom geometric + normal weights, addressed from the half-res pixel **center** (an off-center address makes a static scene read its own history a fraction of a texel off every frame, which compounds into visible drift). Writes the history corner + weights for the accumulate pass and the packed history count.
- **Carrying the history count: only occluders count as disocclusion.** Each pixel's history count is reprojected from the 2×2 footprint in the previous frame. A footprint that is only *partly* valid has two very different causes, and telling them apart is what keeps thin geometry stable in motion:
  - *Edge overhang.* The pixel sits on a surface whose silhouette the footprint hangs over: a leaf, a twig, any edge in slow motion. The taps that fail the surface test hit what was **behind** that surface, or sky. The point itself was visible last frame, and the valid taps describe it completely, so its history is genuine.
  - *Real disocclusion.* The pixel was hidden last frame. The failed taps hit the **occluder**, something closer to the previous camera than the expected point. The valid taps are only neighbors of the revealed area, so inheriting their count would leave streaks.

  So the count is the fully normalized mean of the valid taps' counts, reduced only by the bilinear coverage of **occluder** taps: failed taps lying more than two pixel widths in front of the expected point. The penalty is `(1 − occluder coverage)^0.34`, with a small floor of `max(valid tap count, 4)`. Failed taps behind the point, or sky, cost nothing (`RTGI_REPROJECT_OCCLUDER_AWARE_COUNT`).

  Why this matters: the earlier version (and the usual approach in denoisers such as NRD's footprint quality) penalized *every* partial footprint by its total valid weight, and did so again every frame. At half resolution almost every pixel on thin geometry is an edge pixel, so foliage and slow-moving silhouettes decayed to about 4–5 samples and never converged in motion. Worse, the bilinear average then blurred those low edge counts into the interior of leaves. With the front/behind distinction, which is the classic TAA/TSR disocclusion test applied to the count, edges keep their full history and real reveals still restart.

  The same rule applies to the specular history: its surface-motion footprint quality (NRD's `sqrt` of the valid footprint weight) is replaced by `sqrt(1 − occluder coverage)`. Specular decides its history in the accumulate pass, so reproject hands the occluder coverage over in a small R8 image. The lobe test and the virtual-motion history are unchanged.

  Known weak spot: a revealed wall pixel right next to the boundary may have only one small-weight occluder tap, so the penalty barely applies and the pixel keeps the stable wall's high count. The planned refinement is to treat the pixel as disoccluded whenever the tap it actually reprojects into (the one with the largest bilinear weight) is an occluder.
- **Parallax penalty.** A surface seen nearly edge-on that becomes much more face-on as the camera moves gets its thin strip of history stretched across many new pixels. The penalty measures that stretch from the change in foreshortening between the two camera positions (deadzone 2×, then a ramp). When active, it first caps the history at the ray demand target and *then* scales it down, so a stretched pixel always requests fresh rays.
- **Specular history count** is gathered along the surface motion with its own parallax penalty.
- **Ray request.** Because reproject already has the history counts and the material, it also computes each pixel's ray request (see Distribute rays) and sums it per tile, so the distribute pass only has to read it.

### 2. Distribute rays — what each pixel gets
The flexible budget lives here. The budget is **hard**: the frame never shoots more than `budget × half-res pixels` rays.

#### What each pixel requests
```
request_s = 1 base ray + round( amplitude · exp(−4 · history_s / T_s) · share_s · material_s )
```
for each signal `s` (diffuse, specular), with the total capped at `max_rays_per_pixel`.

- **Curve.** Exponential in the history relative to the **fast convergence target** `T` (32 samples for both signals). It is zero once the history reaches `T`: the request is *what this pixel still needs to look converged*, not what it needs to reach full history.
- **Amplitude.** Derived from the per-pixel cap: `(max_rays_per_pixel − base) / 2` with specular on. A fully fresh pixel therefore asks for exactly the cap (12 rays), and the curve keeps its shape instead of being flattened by the cap.
- **Why cap at 12.** All rays a pixel shoots in one frame share that frame's noise pattern and that frame's pioneer guide. Thirty rays in one frame are too temporally similar — they form stripes and bake one guide in hard, which reads as splotches. The cap forces a fresh pixel to build its history over more frames, each with a different guide, so the guide's errors average out. It also stops a disocclusion from grabbing a huge burst and draining the budget from the rest of the screen.
- **Diffuse / specular share.** Each signal's *visible* radiance is estimated as last frame's history luma (nearest reprojected texel) × the material's reflectance for that signal (diffuse: `albedo · (1 − metalness)`; specular: environment BRDF with `F0 = lerp(0.04, albedo, metalness)`). They are compared in stops in perceptual space and turned into `share_spec = 1 / (1 + 2^(−2 · stops))`; the factors are `2 · share` and always sum to 2, so the share only ever **moves** rays between the signals, never adds any. On a total disocclusion no history is read at all (the nearest texel would belong to the occluder) and the share falls back to the material reflectances alone.
- **Material factors** (no history needed, so they also apply on disocclusions):
  - *Diffuse vs metalness:* leaving diffuse out would change the pixel by `E = log2((D + S) / S)` stops. While `E` is above a just-noticeable difference of 0.1 stops the diffuse matters; the factor is `saturate(E / 0.1)`. Since `E ≈ −log2(metalness)`, this is 1 up to metalness ~0.93 and 0 for pure metal, almost independent of albedo.
  - *Specular vs roughness:* `1 − smoothstep(0.8, 1.0, roughness)`. Near roughness 1 the lobe is as wide as the cosine lobe; extra specular rays buy almost nothing visible.
  - Both only scale the *extra* rays; base rays stay.

#### How the budget is handed out
1. **Base minimum first.** Every geometry pixel is guaranteed `min_ray_budget` (1/8) base coverage per signal, chosen by a frame-rotated 8×8 Bayer dither, so base coverage is spatially even at any rate.
2. **Extras get the rest.** The remaining budget is spread at a uniform fraction of every tile's extra demand; inside a 8×8 tile, a cumulative scan hands the extras out in proportion to each pixel's extra request (linear priority), capped at that request.
3. **Leftover raises the base.** Whatever the extras don't use raises base coverage above the minimum, up to full coverage — so a static, converged view still spends its budget, just on base rays.
4. **Split per pixel.** A pixel's granted extras are split between diffuse and specular in proportion to their extra requests. Ray-list entries per pixel are `[diffuse…][specular…]`.

If the budget covers every request in full, every pixel simply gets what it asked for. The tile budgeting runs cooperatively in one wave with prefix sums; a small per-tile dither slack is reserved and the ray list itself is clamped to the budget, so rounding can never overshoot it.

**What this buys in practice.** Converged pixels need almost nothing, so the budget they leave is enough for the allocator to give disoccluded pixels a real burst. In practice, every pixel on screen sits at a history of roughly **6–12 rays at all times, even in the frame right after it was disoccluded**. There is no visible "fresh pixel" phase where a newly revealed area shows up with a single noisy sample and has to converge over many frames. A fixed-rate tracer at the same average budget (0.6 rays per half-res pixel) would leave a disoccluded pixel at less than one sample.

Effectively, disocclusions resolve as fast as with **more than 6 rays per pixel**, at a fraction of the cost: the frame as a whole pays for 0.6 rays per half-res pixel (0.15 per full-res pixel), and that budget is shared between diffuse *and* specular. That's roughly a 10× budget saving in the case that matters most.

### 3. Trace
The repacked path traces the flat ray list with an indirect dispatch sized to the real ray count. (A classic one-thread-per-pixel path remains as an option.)

- **Diffuse rays:** cosine-hemisphere directions, each ray bent toward its own fetched pioneer guide (see Pioneer guiding and below). Seeds fold in the frame index *and* the history length so a pixel doesn't redraw near-identical directions on consecutive frames, and each ray advances its own RNG slot rather than reseeding.
- **Specular rays:** GGX VNDF directions on their own RNG stream, each mixing its own fetched guide lobe in for rough surfaces. Hits are shaded with the full `shade_material` (but with low-quality material evaluation: face normal and a fixed coarse texture mip).
- Each ray stores its radiance and hit distance.

#### How diffuse rays are bent toward the guide (`rtgi_guided_sampling.hlsl`)
Plain cosine sampling uses **Malley's method**: pick a uniform point on the unit disc in the surface's tangent plane and lift it onto the hemisphere, `(x, y, √(1 − x² − y²))`. The bend keeps that lift and only changes *where on the disc* the point lands.

1. **Project the guide.** The guide direction (the reconnected pioneer hit) is projected onto the same disc, giving a point `pd`.
2. **Uniform disc point, seen from `pd`.** A uniform disc point `p0` is drawn as usual, then expressed in polar coordinates around `pd`: an angle `φ` and a normalized distance `u = (|p0 − pd| / t_max(φ))²`, where `t_max(φ)` is the distance from `pd` to the disc rim in that direction. For any `pd`, `u` is exactly uniform in `[0, 1)` and independent of `φ`.
3. **Pull.** Only `u` is warped, with a Schlick-style rational pull:
   ```
   u' = u / (1 + κ · (1 − u))
   ```
   This squeezes points toward `pd` (`u' ≤ u`) without `pow`, `log` or `exp`. The angle is kept, and the pull is measured against the rim distance, so the bent point never leaves the disc and the lifted ray never leaves the hemisphere. A lobe built directly on the sphere around the guide could point below the horizon when the guide is oblique; this one can't.
4. **Lift** the bent point exactly like Malley's method.

**Strength.** `κ` comes from `guide_concentration` through the von Mises–Fisher approximation `κ = R(3 − R²) / (1 − R²)`. The default `R = 0.92` gives `κ ≈ 12.9`; `R = 0` gives `κ = 0`, which is exactly plain cosine sampling. The mapping is very nonlinear, so any per-pixel strength scales `κ` linearly rather than the concentration.

**Why it's unbiased.** Bending changes the sampling density, so the cosine term no longer cancels for free. The pdf of the pull is
```
pdf_pull(u') = (1 + κ) / (1 + κ · u')²
```
and because only the radial coordinate is warped (the angular part and the disc → hemisphere Jacobian are untouched), the direction pdf is
```
pdf_guide(ω) = pdf_pull(u') · cos θ / π
```
where `θ` is the angle between the ray and the surface normal.
Every guided ray's radiance is multiplied by `pdf_cosine / pdf_guide`, the ratio of the density the estimator expects (cosine) to the density the ray was actually drawn from. The cosine terms cancel, so the weight is simply
```
weight = 1 / pdf_pull(u') = (1 + κ · u')² / (1 + κ)
```
Rays near the guide (small `u'`) are drawn more often and weighted down; rays far from it are drawn less often and weighted up. In expectation the result is identical to plain cosine sampling, just with much less variance when the guide points at the light. A wrong guide costs variance, never correctness. At `κ = 0` the weight is exactly 1.

### 4. Blend rays
Turns each pixel's rays into compact guides: the **geometric mean** of the diffuse rays in log rgb plus the mean **ray shortness** (`[0,1]`: 1 = hit right next to the surface, 0 at `max_visibility_pixel_range`), and the specular log radiance + hit distance. The per-pixel ray counts written by distribute are the single source of truth downstream.

### 5. Pre-filter — firefly clamp, guides and filling the gaps
The big pass. It preloads world position and normal for the tile into groupshared, builds a geometry-aware 2×2-blurred log radiance grid, then filters each pixel against a 5×5 neighborhood of quads.

- **Firefly clamp.** Each ray is clamped to a ceiling = neighborhood log-rgb mean + a perceptual tolerance (4.0 for diffuse, 3.0 for specular), hue-preserving (all channels scaled by one ratio). The ceiling is built from the *surrounding* quads only — a pixel never contributes to the ceiling that clamps it, or a firefly would pass itself. Where the neighborhood has few rays, the ceiling reference is tightened. Each signal has its own per-cell validity, so a quad without specular rays never feeds a placeholder into the specular ceiling (and vice versa).
- **Energy bookkeeping.** The clamp writes `firefly_factor = energy_before / energy_after` per pixel for each signal, used by the pre-blur to put the energy back.
- **Blend.** The clamped rays are averaged straight from the ray list into the diffuse SH-Y + CoCg and the specular radiance — this is the sole source of the filtered signals.
- **Guides.** Writes the **AO guide** (from ray shortness: contact/occluded areas blur less) and the perceptual radiance used by the blurs.
- **Imposter fill.** At sub-1 ray rates many pixels shot no ray. A ray-less pixel that sits on the same surface as its quad **copies** the value of the first same-surface quad-mate that did shoot (via quad ops after a thread→pixel reswizzle that makes the hardware quad a real screen 2×2). It must be a *copy*, not a blend: the goal is only that the pre-blur never samples an empty pixel, without inflating the neighborhood mean as the ray rate drops.
- **Quad-pair fill.** If a whole quad has no value for a signal, it copies from the horizontally neighboring quad (via wave lane exchange) when that quad is on the same surface. Together these let each signal run below one ray per quad.

### 6. Pre-blur (default: 1 iteration, 8 samples, base width 20)
The cheap "fast spatial" pass: an adaptive, stochastic **Poisson-disc** blur, rotated every frame and every iteration (golden angle), so the temporal pass averages it over many orientations.

**A pre-temporal stabilizer.** The pre-blur's main job isn't to produce a clean image, it's to make what the temporal pass receives *stable*. At sub-1 ray rates a single pixel's value swings wildly from frame to frame: sometimes it shot a ray, sometimes it only carries a copied neighbor, and one lucky ray toward a bright light dominates it completely. Feeding that straight into the history makes the temporal pass chase noise.

The pre-blur works like **ReSTIR-style spatial reuse**:
- Every pixel borrows the samples of a handful of same-surface neighbors, picked stochastically (rotated Poisson taps).
- Each borrowed sample is weighted by how much it should count: geometry, normal, how many rays it carries, and how much firefly energy it stands for.
- Different neighbors get picked every frame, so over a few frames each pixel effectively sees a much larger, well-mixed neighborhood than the 8 taps of any single frame.

The temporal pass then integrates many small, already-averaged estimates instead of a few large, noisy ones, which gives much lower frame-to-frame variance at the same ray count. Like ReSTIR's spatial reuse, it lifts the effective sample count per pixel without tracing more rays. Unlike ReSTIR, it doesn't rely on per-pixel temporal reservoirs, so a disoccluded pixel benefits fully from frame one.

**Retaining firefly energy.** The pre-filter clamps bright outliers to keep them from popping, but the energy they carried is real light. The pre-blur is where it comes back (see firefly energy compensation below): a clamped firefly gets a larger weight in its neighbors' averages, so its energy is spread over the kernel instead of being thrown away. Combined with the per-frame tap rotation, a rare bright path shows up as a soft, stable contribution across many pixels and frames, not a dot that flashes in one pixel.

- **Tap weight** = geometry × normal × ray count × perceptual-difference guide (diffuse only).
- **Radius** is scaled by the AO guide (contact areas blur less) with a floor.
- **Specular** has no kernel of its own: it walks the *same* taps and keeps only those inside its own radius, which comes from the reflection lobe and hit distance (`hit · lobe_tan / pixel width`, × `specular_pre_blur_scale`), with a 1-pixel soft edge. A sharp reflection ends up using only the center and innermost taps. Specular taps add detail-normal and roughness weights.
- **Firefly energy compensation (iteration 0 only).** The firefly factor is multiplied into the blur **weight** (center and taps), not into the value. Multiplying the value would just undo the clamp. Multiplying the weight lets a clamped firefly *dominate* the mean of every pixel its kernel touches, spreading its energy into a wide, ceiling-bounded blob — like a photon-map density estimate. The output stays a convex mix of clamped values, so no pixel ever exceeds the ceiling, and the energy of a rare bright event is spread over many pixels instead of popping in one. Applying it again in a later iteration would compound the boost, hence iteration 0 only.
- Optionally the pre-blur can run *inline inside the temporal accumulate* (`pre_blur_in_accumulate`), skipping its own pass and intermediate image.

### 7. Temporal accumulate (diffuse history 64 samples, fast history 4 frames)

**Diffuse.**
- Gathers color, AO guide, perceptual mean and fast-history statistics through the reproject corner + weights, then blends the pre-blurred sample in with `blend = rays_this_frame / (1 + history)`.
- A short **fast history** (brightness mean + relative variance, 4 frames) clamps temporal fireflies (3.5 standard deviations) and provides **anti-lag**: where the slow history's brightness diverges from the fast mean, the slow history is trusted less; plain noise (high relative variance) earns some of that trust back.
- The parallax penalty also cuts the fast history.

**Specular — a port of NVIDIA ReBLUR's specular temporal accumulation**, adapted to the half-res layout:
- **Two candidate histories every frame:** *surface motion* (the diffuse reprojection) and *virtual motion* — a reflection moves like a point *behind* the mirror, at the hit distance along the dominant reflection direction. Each gets its own frame count × footprint quality, capped by a confidence (virtual: parallax and normal agreement inside the lobe; surface: view-angle change versus the lobe's half-angle, relaxed for very rough surfaces). A selector blends them, favoring the virtual history where it is at least as long; a static camera uses the surface history.
- **Lobe overlap test.** Gloss and normal changes are judged by the normalized overlap of the two reflection lobes, each modeled as a von Mises–Fisher lobe around the detail normal (`κ = 2/α²`). Roughness is floored for the comparison so near-mirrors aren't rejected by tiny normal noise. All rejections ramp in only once the camera has moved more than about half a pixel, so slow motion keeps its history.
- **Disagreement cut.** When both reprojections are valid but fetch very different reflections (more than a small deadzone in stops), the carried history is shortened.
- **Footprint quality** shrinks the history from the *geometric* weights only — putting the lobe test in there too compounded every moving frame and collapsed the history in slow motion.
- **Catmull-Rom** history fetch keeps reflections sharp, and a **specular fast history** (mean, relative variance, 4 frames) mirrors the diffuse one: temporal firefly clamp + anti-lag.
- Specular history is capped at `specular_max_temporal_frames` (32).

### 8. Post-blur (default: separable bilateral, max width 16, stride 2)
The small "cleanup spatial" pass — horizontal + vertical bilateral (groupshared variant by default; an à-trous mode exists as an alternative). It blurs **consistently every frame** (the optional disocclusion-blur ramp is the only exception), guided by the AO guide radius scaling (floor 0.2) and a perceptual radiance guide. The innermost taps are always sampled at stride 1 to hide the pre-filter's 2×2 quad structure. Specular reuses the same passes with its own lobe-and-hit-distance width (× `specular_post_blur_scale`), also scaled by the AO guide.

### 9. Upscale + resolve
Upscales both signals to full resolution with geometry-aware tent weights over the nearest half-res texels.

- **Diffuse:** resolves the directional SH against the full-res surface normal, so normal-map detail comes back even though the GI was filtered at half res.
- **Specular:** weights its taps additionally by the full-res detail normal and roughness versus each half-res texel's reflection lobe. Where almost nothing matches, it blends in a lenient fallback of the same taps instead of leaving a hole.
- **Composite:** `diffuse_resolved · albedo · (1 − metalness)` + `specular_resolved · environment BRDF`.

---

## Observability

- **Statistics** (Render System Statistics → Ray Traced Global Illumination):
  - **Rays this frame:** requested vs shot per signal, plus the hard budget.
  - **History length distribution:** relative to the fast convergence target, i.e. how much of the screen is still asking for extra rays.
- **Debug views** (RTGI dropdown):
  - Ray allocation: per-pixel rays shot, per-tile rays, diffuse/specular share, material ray factors.
  - Pioneer guide: confidence and direction.
  - Pre-filter guides: AO, hit distance, perceptual mean.
  - Temporal: history lengths, reactivity, specular virtual-motion amount.
  - Lighting composition: direct / indirect / all, for both diffuse and specular.

---

## The short version

Trace *adaptively* — a bounded burst on disocclusion, almost nothing where converged, and diffuse or specular depending on which one the pixel actually shows. Guide the rays with a sparse, uniform pioneer trace that finds the light cosine sampling misses. Denoise at *half res* with a *fast stochastic spatial → temporal → tiny cleanup spatial* pipeline. Guide on *stable* signals (geometry, occlusion, geometric radiance means — never variance-driven blurs). Clamp fireflies, but *spread* their energy instead of throwing it away. Keep the filter *consistent* frame to frame. The result is RT GI — diffuse and reflections — that actually fits a game's frame budget, because it doesn't have to be ultra-expensive to look good.
