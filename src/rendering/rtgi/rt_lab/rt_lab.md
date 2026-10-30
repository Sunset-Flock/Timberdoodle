# RT Lab Findings

## Dear Claude, when you read this

- This file collects the findings of the RT Lab experiments (diffuse RTGI denoising, `src/rendering/rtgi/rt_lab/`).
- The user tells you over time what to add. Write it down when asked, in the user's sense, without inventing findings.
- Keep it tight and concise, but do not drop details that matter: numbers, conditions, what was measured how, caveats.
- Structure: fundamentals first, then one section per topic. Each finding states the observation, why it happens and
  what follows from it. Mark things that are being tried but not yet evaluated as such.

## Fundamentals

- **Denoising = raising the effective sample (ray) count per pixel.** A massive spatial filter collects hundreds of
  samples per pixel, temporal accumulation multiplies that by 16x to 64x, so a pixel ends up with thousands of samples:
  a stable, noise free image. Every denoising technique is judged by how many samples it gathers per pixel (and at what
  bias). Measured live by the `RT_LAB_SPATIAL_SAMPLE_COUNT` debug view: sum of weight / center weight x temporal count
  over all spatial taps, not normalized.
- **Smooth geometry (flat wall) and detailed geometry (foliage) must be treated differently.** The eye notices error
  less as geometric complexity rises. Massive grain of salt: detailed areas are usually MORE starved for samples, they
  disocclude more (fewer temporal samples) and the spatial filter fails more there (fewer spatial samples), by more
  than the perceptual effect compensates. Takeaway: detailed areas still NEED SIMILAR SAMPLE COUNTS as smooth ones, but
  the way they get them may be more biased (e.g. laxer spatial filter, stronger firefly clamp).
- **The input signal must always be variance reduced (pre filtering).** The raw rays always have edge cases, e.g. one
  bright ray every 500 frames somewhere in 25x25 pixels. Physically it should light up the scene, but resolving that
  properly in real time is hopeless: the denoiser would need far more samples than it ever gets, and temporal reaction
  becomes impossible because catching these rays takes many frames. So real time denoising falls back to massaging and
  pre filtering the ray radiance before the denoiser sees it. A pre filter lowers the variance of the input so the
  denoiser converges to a coherent, stable image quicker: fewer disocclusion artifacts, more temporal stability, easier
  (faster) temporal reaction. The measure of a good pre filter: how much it reduces variance (at what bias). The firefly
  filter every denoiser has is exactly such a pre filter: a firefly filter is really only a variance reducing pre filter.

## Tracing rays

- **Ray origins need the RTGI specular offset, else rays leak.** The lab rays started with the RTGI diffuse offset,
  which only pulls the origin back along the view ray by one pixel width. That lifts the origin off the surface by only
  `pixel_width * NoV`, almost nothing at grazing views, so near-tangent rays start on or below the surface and leak
  through it (worse at half res: the pixel position comes from one representative sub-pixel depth at the pixel center,
  off the true surface). Fix (`rt_lab_ray_launch`, rt_lab_shared.hlsl, used by the trace and the temporal gradient
  records): the RTGI specular origin (`rtgi_specular_ray_desc`), view ray pullback AND a lift along the face normal by
  one pixel width. Not yet evaluated visually.

## Pre filtering

### Tricks

- **Firefly energy recovery: put the clipped energy into the filter WEIGHTS, not the values.** A ray pre filter (firefly
  clamp, `ray_filter_*`) makes the input stable but loses the energy it clips. Trick (RTGI pre-blur iteration 0 trick,
  moved post temporal): the gather writes the raw (unclamped) mean luminance next to the filtered color, the temporal
  pass accumulates it with the color's blend, factor = accumulated raw / filtered luminance (>= 1, = how much energy the
  clamp ate here). The FIRST fine and the first coarse spatial pass multiply each tap's color weight by its factor;
  sample counts are not scaled.
  - Why weights: scaling the value would just undo the clamp. A bigger weight lets the clipped tap DOMINATE the
    weighted mean of every pixel whose kernel touches it, so its energy spreads over the footprint as a wide blob
    bounded by the clamped values (output stays a convex mix of clamped taps, never above the ceiling, no popping dot).
  - Only the first pass: after one pass the energy has already moved, applying it again compounds it.
  - Recovers energy only where the factor varies over the footprint (isolated clipped events); where every pixel's
    history clipped about equally the factor is uniform and cancels out of `sum(f x) / sum(f)`. Measured 2026-10 with
    tight clamps (perceptual tolerance 2, 3 bistro views): no energy back for that reason.
  - Setting `firefly_weight_enabled` (UI Trace > Ray Filter, "Firefly Energy Weighting"), default OFF.
    Acts only when the ray filter runs (needs temporal + a spatial filter).

## Temporal accumulation

- **Responsiveness vs stability is a fine line.** Anti-lag (temporal gradient cuts / caps / rescale, coarse anti-lag)
  that reacts too eagerly makes even SMALL lighting changes cut history across nearly the whole screen, which shows up
  as a short burst of jitter (a few frames) on most pixels. Tuning only for temporal lag pushes exactly there: the lag
  metric rewards fast reaction, and the instability metric (EMA std over a window, judged after the move) barely
  sees short bursts during a change. Judge anti-lag work on lag AND on frame-to-frame jitter during the change.

### Tricks

- **Reproject the sample count occluder aware soft normalized.** The count is reprojected with the history's bilinear
  footprint, taps masked by the disocclusion test. Partially valid footprints (edges, thin geometry, partial
  disocclusions) break the usual ways:
  - Normalized over valid taps: revealed pixels inherit the full count, stick to their first history frames, streak.
  - Not normalized: every partial footprint loses count every frame; thin moving geometry never converges and the low
    counts bleed inward.
  - Soft normalized `sum(w * count) / W^0.66` (old RTGI, ReLAX like): better, but still penalizes EVERY partial
    footprint (`W^0.34` per frame), so foliage and slow moving edges decay to a handful of samples.
  - Occluder aware soft normalized (tido RTGI, RT Lab default): separate the two causes of a failed tap. Behind the
    point / sky = the footprint overhangs an edge, the point was visible: no penalty. More than 2 pixel widths in front
    of it (seen from the previous camera) = it was the occluder, real disocclusion: penalize. Count = normalized valid
    mean x `(1 - occluder coverage)^0.34`, floor `max(min(max valid count, 4), ...)`. Edges converge in motion, real
    disocclusions still restart. Weak spot: occluders with small bilinear weight barely penalize.

## Spatial filtering

- **Edge stopping starves pixels of samples.** The spatial filter must not blur over geometry or normal discontinuities
  (plane distance + normal weights), so taps across edges are rejected. Pixels near edges, on thin geometry and on
  small surfaces then gather only a few hundred effective samples instead of many thousands, and stay noisy while the
  rest of the image is clean (visible as low count bands along every edge in `RT_LAB_SPATIAL_SAMPLE_COUNT`).
- **Effective sample count as guide.** Where the fine filter's effective sample count (temporal x fine spatial) is
  low, switch to a coarse spatial filter. Fixes exactly the starved pixels: fine geometry (edge stopping rejects most
  taps) and disocclusions (no temporal history). The switch is driven by a statistically sound metric: the effective
  sample count directly sets the noise level (variance ~ 1/N), unlike guides such as luminance difference or AO.
  - Two spatial filters on the temporal output: fine (detailed edge stopping) and coarse (wide a-trous, sloppy edge
    stopping). Collected separately, only lerped at the very end (resolve): `lerp(fine, coarse, t)`, so the coarse
    result never contaminates the fine collection nor the temporal history (pre-spatial).
  - The coarse filter gathers the diffuse as RTGI SH-Y + CoCg (accumulated in the temporal pass next to the color) and
    the resolve evaluates it for the receiver's normal: with its loose normal test it gathers from surfaces facing
    elsewhere, the SH keeps that light directional instead of smearing it as plain color.
  - Coarse edge stopping: no plane test. Smooth distance test: world distance relative to what a camera facing surface
    at that screen offset would give, `exp(-(max(d_rel - 1, 0) / tol)^2)`. Smooth normal test `((1 + n.n') / 2)^p`,
    0 at 180 degrees.
  - Blend = top up: just enough coarse to bring the pixel's effective sample count up to a target. Fine / coarse
    treated as estimates with variance 1/N, so the blend has `N_eff = 1 / ((1-t)^2 / N_fine + t^2 / N_coarse)`; t is the
    smallest t with `N_eff = target` (unreachable: the minimum variance mix `N_coarse / (N_fine + N_coarse)`). A linear
    lerp of the counts fails: the a-trous count compounds (~7x per pass), N_coarse ~1M gives t ~0. With the variance
    form a huge N_coarse just means `t ~ 1 - sqrt(N_fine / target)`. Ignores that fine and coarse share samples.
  - REMOVED 2026-10 (cleanup): the coarse filter, its settings `coarse_*` and debug views are gone from the code.
- **Being tried: variance guide (fine filter).** The temporal pass estimates the per sample rgb variance from the 5x5
  raw samples on the same surface and accumulates it like the color. Center and tap are treated as estimates of their
  means, `N(mean, variance / n)` with n = the effective sample count, and the tap is weighted by how likely both share a
  mean: `exp(-z^2 / (2 k^2))`, `z^2` = mean over rgb of `(mean_c - mean_t)^2 / (var_c / n_c + var_t / n_t)`. Few samples
  = loose test, many samples = tight test. Rejected taps also do not count into the effective sample count.

### Tricks

- **Gradient aware variance guide: compare against a color plane, not a constant.** The plain guide tests "same
  mean". Smooth lighting ramps (falloffs, bounce regions) violate that, and with high effective counts the test turns
  near binary (the gaussian weight goes from fully in to gone between ~3 and 6 sigma, the noisy z flips taps across it),
  so the filter breaks smooth regions into blotches with hard edges. Instead fit a local rgb gradient per screen axis
  and compare every tap with `center mean + gradient x offset`, adding the gradient's variance (a plane distance test in
  color space). Observed (user, visual): reduces the splotchy hard edges quite a lot. Bistro street static 128 frames
  a-trous 5, 1 run: instability p50 0.53 -> 0.48 %, p90 and split error unchanged. Setup that matters:
  - Fit the gradient ONCE, in the first fine pass (5x5 weighted least squares, geometric weights only,
    `var(g) = 2 var_c sum w^2 o^2 / (sum w o^2)^2`). That first pass filters WITHOUT a gradient (constant model).
  - Do not refit in later passes. Each later pass uses the incoming gradient as is (pass 2 the raw fit) and passes it on
    filtered with its own final color weights, variance guide included (variance with the squared weights): taps the
    guide rejects, e.g. across an edge, do not feed into the gradient, so it stays sharp at edges. A refit per pass
    measures the slope over the pass's growing step on increasingly blurred color and swings between passes.
  - Shrink the gradient per channel by its significance, `g x max(0, 1 - var(g) / g^2)`: noise and the fake slope a
    step edge produces fall back to the constant model.
  - Setting `variance_guide_gradient`, debug view `RT_LAB_COLOR_GRADIENT` (relative slope after the shrinkage). The
    averaging itself is not gradient corrected (a lopsided kernel still shifts the mean; not tried yet).
