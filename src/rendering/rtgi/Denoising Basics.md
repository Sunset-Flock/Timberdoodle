# Denoising at Lightspeed — Bottom Up

## Motivation

Real-time global illumination is one of the most expensive things a renderer can do. The naive approach — many rays per pixel, many bounces, high resolution — produces beautiful results at film speeds. Getting it into a game means accepting a far more brutal budget: one ray per pixel, a handful of microseconds for the entire denoising pipeline.

Most published denoisers are general-purpose. They handle specular, shadows, AO, and diffuse GI with the same set of passes, tuned to be inoffensive across all of them. That generality has a cost. A general denoiser cannot assume the signal is low-frequency, cannot exploit the fact that indirect diffuse has no hard shadow edges, cannot use wide spatial blurs that would smear specular highlights, and cannot commit to dirty tricks that only work for one signal type. It has to stay conservative.

This denoiser is not general-purpose. It handles exactly one thing: cosine-weighted hemisphere indirect diffuse radiance. That specialization is where the performance and quality both come from.

Indirect diffuse is not a low-frequency signal but it is generally acceptable to only reconstruct the low-frequency part of it — unlike specular, small amounts of spatial blur are nearly invisible in indirect diffuse. The eye integrates indirect lighting over large surface regions and is insensitive to soft gradients in it. This means we can blur wide, blur hard, and blur often, in ways a general denoiser simply cannot.

The denoiser is also built for games, not to render an ideal demo scene. Fast camera movement, disocclusions every frame, dynamic lights, first-person gameplay — these are the primary test cases, not edge cases. Every design decision is evaluated against the worst-case game scenario, not the best-case benchmark screenshot.

On an RTX 4080 the full pipeline — trace included — runs in under 400 microseconds. The goal is to show that ray-traced global illumination does not have to be a premium feature. It can run in fast-paced games on mainstream hardware, and it can look good doing it.

This document builds that denoiser from the ground up, one problem at a time. The pipeline it arrives at:

1. **Trace** — one ray per pixel
2. **Temporal accumulation** — reproject and blend history over many frames
3. **Firefly filter** — suppress isolated bright spikes that resist temporal averaging
4. **Post filter** — bilateral spatial filter to clean remaining noise and disocclusion artifacts

[Denoising Tricks](Denoising%20Tricks.md) extends each of these passes with further improvements.

> DEV NOTE: pos view for screenshot series 0 bistro: -26.356014 6.4785852 11.6196 0.86602473 -0.32465482 -0.38026375
> DEV NOTE: pos view for screenshot series 5 bistro: -11.937591 7.509383 3.3489404 0.99683386 0.076705284 0.020941874

## Understanding the Input Signal

> **Scope note:** This document covers the denoising pipeline only. The ray tracing itself — BVH traversal, ray generation, hit shading, emissive evaluation — is not described here.

Without indirect lighting the scene is dark and flat — only surfaces in direct line-of-sight to a light source are lit:

![direct lighting only, no indirect](showcase_images/bistro%20series%200%2011.png)

Adding indirect diffuse lighting achieves something like this — traced rays, denoised:

![denoised RTGI — goal](showcase_images/bistro%20series%200%2010.png)

In practice we can only afford a tiny number of rays per frame — one or fewer per pixel. Blending that raw single-sample indirect contribution on top of the direct result looks like this:

![direct + raw traced indirect, one ray per pixel](showcase_images/bistro%20series%200%200.png)

The indirect contribution is present, but the image is overwhelmed by noise. Turning that into something close to the goal image above is what the denoiser does.

Each ray carries **pure incident radiance** — the light arriving at the surface from the sampled direction, with no albedo applied. Albedo is multiplied in at the very end, after denoising. Working on albedo-free radiance means the denoiser never smears texture or color detail across surfaces — blurring `light * albedo` would soften sharp albedo transitions, while blurring just `light` and multiplying albedo afterward preserves all that detail for free.

Rays are shot with a cosine-weighted importance sampling distribution over the hemisphere. This matches the cosine term in the rendering equation, so every sample carries equal expected contribution and no explicit cosine weight is needed when accumulating.

```hlsl
// Build an orthonormal basis (tangent, bitangent) around the surface normal
float3 up        = abs(normal.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
float3 tangent   = normalize(cross(up, normal));
float3 bitangent = cross(normal, tangent);

// Project a uniform disc sample up to the hemisphere — gives cosine distribution
float2 disc      = rand_concentric_disc(rand2());
float3 local_dir = float3(disc.x, disc.y, sqrt(max(0.0f, 1.0f - dot(disc, disc))));

// Transform from local to world space
float3 ray_dir   = normalize(local_dir.x * tangent + local_dir.y * bitangent + local_dir.z * normal);
```

The projected disc construction (`z = sqrt(1 - x² - y²)`) naturally produces a cosine-weighted distribution: samples cluster near the normal (where the cosine term is large) and thin out toward the horizon.

With only one ray per pixel, the chance of any given ray hitting a bright light source is low — most rays return near-zero radiance, while the occasional lucky hit comes back orders of magnitude brighter. The result is high-contrast salt-and-pepper noise dominated by isolated bright spikes.

Offline renderers solve this by averaging thousands of rays per pixel. In real-time we do the same thing temporally: accumulate ray results over many frames instead of many rays within one frame.

A per-pixel sample count image tracks how many frames have contributed to each pixel. Each frame, the new ray result is blended into the existing accumulated value using a weight that shrinks as the count grows:

```hlsl
accumulated = lerp(history, new_sample, 1.0f / (1.0f + sample_count));
sample_count = min(max_history_frames, sample_count + 1);
```

At count 0 the blend weight is 1 — the new sample is taken directly with no history. At count 31 it is 1/32, so 31 previous frames dominate. The sample count resets to zero on disocclusion so newly revealed surfaces converge from scratch.

![screenshot of raw image](showcase_images/bistro%20series%200%201.png)

With 255 accumulated frames the image looks far cleaner — but only when the camera is still.

## Temporal Reprojection

Accumulating on fixed screen pixels works while the camera does not move. The moment it does, old history data gets mapped onto whatever new geometry now occupies those pixels. Here the camera has moved to look down and to the right; without geometric reprojection the denoiser has no idea the view has changed, and the accumulated wall lighting is ghosted onto the floor:

![camera moved — accumulated lighting splatted onto floor](showcase_images/bistro%20series%200%2012.png)

The fix is reprojection: instead of always reading history from the same screen-space pixel, figure out where the current pixel was in the previous frame and read from there.

Every pixel has a depth value. Unproject it through the inverse view-projection matrix to recover world-space position, then project that position through the previous frame's view-projection matrix to get a UV into the previous frame's accumulation buffer.

This works cleanly for indirect diffuse because diffuse is **view-independent**: the amount of indirect light arriving at a surface point does not change when the camera moves. History accumulated at that world-space point last frame is just as valid this frame. Specular is the opposite — the reflection changes entirely with view angle — which is why temporal accumulation is far more powerful for diffuse than for specular.

When the reprojected position fails its geometric validity tests — none of the four neighboring history texels pass the surface and normal checks — the pixel is treated as a fresh disocclusion and its history is reset to zero. The pixel then converges from scratch over the following frames.

**Previous frame UV**

```hlsl
// Unproject current pixel to world space
const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
const float4 world_pos_h = mul(camera.inv_view_proj, float4(ndc, 1.0f));
const float3 world_pos   = world_pos_h.xyz / world_pos_h.w;

// Project into previous frame's clip space
const float4 ndc_prev_h  = mul(prev_camera.view_proj, float4(world_pos, 1.0f));
const float3 ndc_prev    = ndc_prev_h.xyz / ndc_prev_h.w;
const float2 uv_prev     = ndc_prev.xy * 0.5f + 0.5f;
```

For a static scene with a moving camera this precisely tracks where a surface point appeared one frame ago. Dynamic objects would also need motion vectors, but that is outside this denoiser's current scope.

**Manual bilinear — fetching all four neighbors**

The reprojected UV typically lands between four history texels. A hardware bilinear sampler would blend them automatically, but that gives no control over which neighbors are valid. Instead `get_bilinear_filter` decomposes the UV into a texel origin and fractional weights, and all four texels are fetched at once with `Gather` for individual testing.

```hlsl
struct Bilinear { float2 origin; float2 weights; };

Bilinear get_bilinear_filter(float2 uv, float2 texSize)
{
    Bilinear ret;
    ret.origin  = floor(uv * texSize - 0.5f);  // top-left texel of the 2x2 quad
    ret.weights = frac (uv * texSize - 0.5f);  // fractional offset within the quad
    return ret;
}

const Bilinear bilinear = get_bilinear_filter(saturate(uv_prev), half_res_size);
const float2 gather_uv  = (float2(bilinear.origin) + 1.0f) * inv_half_res_size;

// Fetch all four neighbors in one instruction per channel; .wzxy = top-left, top-right, bottom-left, bottom-right
const float4 depth4  = depth_history .GatherRed(linear_clamp_s, gather_uv).wzxy;
const uint4  normal4 = normal_history.GatherRed(linear_clamp_s, gather_uv).wzxy;
const float4 count4  = count_history .GatherRed(linear_clamp_s, gather_uv).wzxy;
```

**Geometric and normal validity weights**

Each of the four gathered texels is tested against the current pixel's surface before its value is used. `surface_weight` runs three binary tests — plane distance bidirectionally (neighbor projected onto center plane, and center projected onto neighbor plane) to catch glancing surfaces, plus a loose raw distance cap for parallel surfaces that somehow satisfy both plane tests. Any single failure returns 0.

```hlsl
func surface_weight(float2 inv_render_target_size, float near_plane, float depth,
    float3 vs_position, float3 vs_normal,
    float3 other_vs_position, float3 other_vs_normal,
    float threshold_scale = 2.0f) -> float
{
    const float pixel_size = get_pixel_width_ws(inv_render_target_size, near_plane, depth);
    const float plane_distanceA = abs(dot(other_vs_position - vs_position, vs_normal));
    const float plane_distanceB = abs(dot(vs_position - other_vs_position, other_vs_normal));
    const float dist            = abs(distance(vs_position, other_vs_position));
    const float plane_threshold = pixel_size * threshold_scale;
    const float dist_threshold  = pixel_size * 32.0f * threshold_scale;
    return step(dist, dist_threshold) *
           step(plane_distanceA, plane_threshold) *
           step(plane_distanceB, plane_threshold);
}
```

The threshold scales with `get_pixel_width_ws` — the world-space size of one pixel at the current depth — so the cutoff automatically tightens for close geometry and loosens for distant geometry.

A normal similarity weight is multiplied in on top to catch two surfaces meeting at a shallow angle, which the plane test alone would pass.

```hlsl
func normal_similarity_weight(float3 normal, float3 other_normal) -> float
{
    const float validity       = (max(0.1f, dot(normal, other_normal) + 0.85f)) * (1.0f / 1.85f);
    const float tight_validity = validity * validity;
    return tight_validity;
}
```

The `+0.85` shift means even opposing normals get a small non-zero weight rather than being hard-rejected. Squaring sharpens the response — normals that agree closely get near-full weight, those that diverge even moderately are suppressed.

These two functions are reused as-is in the post filter. The validity tests are then applied to the gathered texels:

```hlsl
float4 surface_weights = {
    surface_weight(inv_size, near, depth, world_pos, pixel_normal, texel_ws[0], prev_normal[0]),
    surface_weight(inv_size, near, depth, world_pos, pixel_normal, texel_ws[1], prev_normal[1]),
    surface_weight(inv_size, near, depth, world_pos, pixel_normal, texel_ws[2], prev_normal[2]),
    surface_weight(inv_size, near, depth, world_pos, pixel_normal, texel_ws[3], prev_normal[3]),
};
float4 sample_weights = get_bilinear_custom_weights(bilinear, surface_weights * normal_similarity);
```

`get_bilinear_custom_weights` multiplies the standard bilinear coefficients by the per-tap validity, zeroing out rejected taps while keeping the correct spatial weighting for surviving ones. `apply_bilinear_custom_weights` then sums and renormalizes, redistributing rejected taps' contribution to their neighbors rather than darkening the result.

```hlsl
float4 get_bilinear_custom_weights(Bilinear f, float4 customWeights)
{
    float4 weights;
    weights.x = (1.0f - f.weights.x) * (1.0f - f.weights.y);  // top-left
    weights.y =         f.weights.x  * (1.0f - f.weights.y);  // top-right
    weights.z = (1.0f - f.weights.x) *         f.weights.y;   // bottom-left
    weights.w =         f.weights.x  *         f.weights.y;   // bottom-right
    return weights * customWeights;
}

float4 apply_bilinear_custom_weights(float4 s00, float4 s10, float4 s01, float4 s11, float4 w, bool normalize = true)
{
    float4 wsum = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    return wsum * (normalize ? rcp(dot(w, 1.0f)) : 1.0f);
}
```

**Sample count and color blend**

A *soft-normalized* blend accumulates the reprojected sample count to avoid two failure modes: full normalization gives thin geometry (which rarely gathers all four valid taps) an artificially high count on the first valid frame, locking it in immediately; no normalization leaves thin geometry permanently low-count. The soft version partially compensates.

```hlsl
float reprojected_count = apply_bilinear_custom_weights_soft_normalize(
    count4.x, count4.y, count4.z, count4.w, sample_weights
);

// On disocclusion reset to zero; otherwise increment up to the configured maximum
float accumulated_count = disocclusion ? 0.0f : min(history_frames, reprojected_count + 1.0f);

// Blend weight derived from count: aggressive at low count, conservative when stable
float blend = 1.0f / (1.0f + accumulated_count);

// On disocclusion there is no valid history — use the new frame directly
float4 accumulated_diffuse = disocclusion
    ? new_diffuse
    : lerp(reprojected_diffuse, new_diffuse, blend);
```

At `accumulated_count = 1` the blend is `0.5`. At count 31 it is `1/32 ≈ 0.03`, letting history dominate. This is the foundation the rest of the denoiser builds on: a buffer that grows stable every frame a surface stays visible, and resets gracefully when it is newly revealed. The EMA property means no explicit history buffer is needed — the accumulated value already encodes the weighted average of all past frames implicitly.

Here the camera is moving laterally to the right. The heatmap in the top left shows the per-pixel accumulated sample count — red is 255 frames, dark blue is 0. Most of the scene is saturated red: reprojection is tracking surfaces correctly and history carries over from frame to frame. The lamp pole in the center creates a disocclusion; the geometry it was occluding is newly visible, so those pixels reset to zero and must re-accumulate from scratch. That band shows up as dark blue in the heatmap and as raw noise in the lighting — the same one-sample-per-pixel variance seen before any accumulation:

![reprojection working; lamp pole disocclusion with noisy re-accumulation](showcase_images/bistro%20series%205%200.png)

With reprojection in place, history correctly follows geometry as the camera moves. The remaining problems are independent of camera motion.

## Firefly Filter

Looking back at the image accumulated over 255 still frames — and especially at the disocclusion zones where pixels have no history at all — isolated bright spikes are still visible that temporal accumulation cannot suppress. These are fireflies: ray hits on extremely bright surfaces so rare that even across hundreds of frames they remain incoherent, appearing only once every few hundred samples. Freshly disoccluded pixels are the worst case: every frame is a one-sample draw with full variance.

A temporal firefly clamp — clamping the incoming sample against a neighborhood mean before blending — is a common response. The problem is reactivity. On the first frame after a disocclusion the clamp has no valid history to reference, so it either lets spikes through or suppresses too aggressively. Worse, as history builds the clamp behavior shifts: the image arrives darker immediately after disocclusion and gradually brightens toward the correct value as the clamp relaxes. This slow brightness drift during the ramp-up period is visible in fast-paced gameplay where disocclusions are constant.

For best reactivity this denoiser uses only a spatial firefly filter — always-on, zero history required, identical behavior on the first frame and the ten-thousandth.

**Filter size.** The kernel must stay small for performance. A 5×5 neighborhood is the practical ceiling.

**Why not variance clamping?** The standard approach computes luma mean and standard deviation in the neighborhood and clamps to `mean + k * stddev`. When variance is already high — which it is everywhere in raw one-sample indirect — the standard deviation is large, and the clamp is loose enough to pass the very outliers it should be catching.

**Geometric mean clamp.** The geometric mean is far more outlier-resistant than the arithmetic mean. A single very bright value barely shifts the geometric mean because it contributes only additively in log space. This makes it a stable reference: compute the geometric mean of luma in the 5×5 neighborhood and clamp the center pixel to `ceiling_factor * geometric_mean`.

`ceiling_factor` controls the tradeoff between energy preservation and spike rejection. At 8 the filter is very conservative — aggressive suppression, slightly darker result. At 128 it is loose — more light energy preserved but more spikes let through. The 8–32 range works well for most scenes; push toward 64–128 when bright emissives need to contribute more freely.

The geometric mean is computed in log space: accumulate `log(sample)` across the neighborhood, average, then exponentiate back. A small constant multiplier before the log and dividing after keeps values numerically stable in fp16.

```hlsl
static const float PERCEPTUAL_SPACE_MULTIPLIER = 1e1f;

func linear_to_perceptual(float v) -> float {
    return log(max(v, 1e-8f) * PERCEPTUAL_SPACE_MULTIPLIER);
}
func perceptual_to_linear(float v) -> float {
    return exp(v) / PERCEPTUAL_SPACE_MULTIPLIER;
}

// accumulate in log space across the 5x5 neighborhood
y_mean_geometric_acc += linear_to_perceptual(sample_y);

// convert back: exp(mean(log(x))) = geometric mean
const float y_mean_geometric = perceptual_to_linear(y_mean_geometric_acc * rcp(valid_samples));
const float clamp_factor = min(1.0f, (y_mean_geometric * ceiling_factor) / pixel_y);
filtered_diffuse *= clamp_factor;
```

The result on the static scene — 255 accumulated frames, firefly filter applied:

![firefly filtered — 255 accumulated frames](showcase_images/bistro%20series%200%202.png)

Fireflies are gone. Remaining noise is broad, low-frequency, and spatially coherent — the kind a spatial blur can handle. The image is also noticeably darker overall — the aggressive clamp removes a lot of energy along with the spikes. This is an acceptable trade for now, and there are tricks to recover most of that lost energy at nearly no extra cost; [Denoising Tricks](Denoising%20Tricks.md) covers this.

The benefit is even more apparent on moving camera scenes. Without the filter, the rare but extreme brightness spikes in freshly disoccluded pixels drag up the exponential moving average and hold the accumulated value far from the correct answer for many frames — convergence in the disocclusion trail takes much longer than it should. With the filter those spikes are removed before they ever enter history. The disocclusion zone behind the lamp pole converges noticeably faster:

![moving camera with firefly filter — disocclusion trail converges quickly](showcase_images/bistro%20series%205%201.png)

## Post Filter

Temporal accumulation integrates samples over time — but we can also integrate spatially. Neighboring pixels observe the same scene from nearly the same angle, so their indirect radiance is likely similar. Sharing samples across neighbors effectively multiplies the sample count per pixel for free.

The simplest version is a plain Gaussian blur applied after temporal accumulation:

![post-temporal Gaussian blur](showcase_images/bistro%20series%200%203.png)

Finally noise-free — but the blur smears across geometry edges and erases surface detail. A wall bleeds light onto the floor next to it; a pillar leaks light around its silhouette. We need the blur to stop at geometry boundaries.

**Geometric and normal edge stops.** Each neighbor is tested with `surface_weight` and `normal_similarity_weight` — the same two functions introduced in the Temporal Reprojection section. Any neighbor that fails the plane distance or normal similarity check is excluded from the blur.

![post filter with geometric and normal edge stops](showcase_images/bistro%20series%200%204.png)

Clean and edge-preserving.

**Separable approximation.** A 2D bilateral blur with geometric edge-stopping terms is not mathematically separable — the plane distance test depends on both axes simultaneously. However, when the filter radius stays around 16 pixels or below, splitting the blur into a horizontal pass followed by a vertical pass produces results that are visually indistinguishable from the full 2D kernel. The geometric error from the approximation is small relative to the noise being removed. The performance advantage is significant: two linear passes over N pixels instead of one quadratic pass over N² pixels. Compared to alternatives like à-trous (which spreads taps across multiple dilated passes), the separable Gaussian is faster and simpler to implement.

The same moving camera scene from the temporal reprojection section, now with the spatial filter enabled. The disoccluded areas behind the lamp pole are immediately acceptable — some low-frequency noise remains where history is still young, but there are no more raw single-sample spikes and the lighting reads correctly at a glance:

![moving camera with full post filter — disocclusion areas immediately acceptable](showcase_images/bistro%20series%205%202.png)

## Where To Go From Here

What has been built here is one of the simplest viable denoisers for indirect diffuse. It works — but it leaves a lot on the table. With the right additions the same pipeline can produce noticeably less noise, less temporal boiling, faster reactivity to lighting changes, sharper results near geometry edges, and significantly more recovered light energy, all at the same or better temporal stability.

[Denoising Tricks](Denoising%20Tricks.md) covers exactly that: a set of targeted improvements to each pass that make the denoiser self-regulating, scene-independent, and robust in the conditions a real game actually throws at it.
