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

## Temporal accumulation

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
  - Settings `coarse_*` (Spatial > Coarse Fallback), debug view `RT_LAB_COARSE_BLEND`.
- **Being tried: variance guide (fine filter).** The temporal pass estimates the per sample rgb variance from the 5x5
  raw samples on the same surface and accumulates it like the color. Center and tap are treated as estimates of their
  means, `N(mean, variance / n)` with n = the effective sample count, and the tap is weighted by how likely both share a
  mean: `exp(-z^2 / (2 k^2))`, `z^2` = mean over rgb of `(mean_c - mean_t)^2 / (var_c / n_c + var_t / n_t)`. Few samples
  = loose test, many samples = tight test. Rejected taps also do not count into the effective sample count.
