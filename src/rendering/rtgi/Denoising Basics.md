# Denoising at Lightspeed — Bottom Up

## Motivation

Real-time global illumination is one of the most expensive things a renderer can do. The naive approach — many rays per pixel, many bounces, high resolution — produces beautiful results at film speeds. Getting it into a game means accepting a far more brutal budget: one ray per pixel, half resolution, a handful of microseconds for the entire denoising pipeline.

Most published denoisers are general-purpose. They handle specular, shadows, AO, and diffuse GI with the same set of passes, tuned to be inoffensive across all of them. That generality has a cost. A general denoiser cannot assume the signal is low-frequency, cannot exploit the fact that indirect diffuse has no hard shadow edges, cannot use wide spatial blurs that would smear specular highlights, and cannot commit to dirty tricks that only work for one signal type. It has to stay conservative.

This denoiser is not general-purpose. It handles exactly one thing: cosine-weighted hemisphere indirect diffuse radiance. That specialization is where the performance and quality both come from.

Indirect diffuse is not a low-frequency signal but it is generally acceptable to only reconstruct the low-frequency part of it — unlike specular, small amounts of spatial blur are nearly invisible in indirect diffuse. The eye integrates indirect lighting over large surface regions and is insensitive to soft gradients in it. This means we can blur wide, blur hard, and blur often, in ways a general denoiser simply cannot.

The denoiser is also built for games, not to render an ideal demo scene. Fast camera movement, disocclusions every frame, dynamic lights, first-person gameplay — these are the primary test cases, not edge cases. Every design decision is evaluated against the worst-case game scenario, not the best-case benchmark screenshot.

On an RTX 4080 the full pipeline — trace included — runs in under 400 microseconds at half resolution. The goal is to show that ray-traced global illumination does not have to be a premium feature. It can run in fast-paced games on mainstream hardware, and it can look good doing it.

This document builds that denoiser from the ground up, one problem at a time. The pipeline it arrives at:

1. **Trace** — one ray per pixel at half resolution
2. **Firefly filter** — suppress isolated bright spikes before they enter history
3. **Temporal accumulation** — reproject and blend history over many frames
4. **Post-blur** — bilateral spatial filter to share samples across neighbors

[Denoising Tricks](Denoising%20Tricks.md) extends each of these passes with further improvements.

> DEV NOTE: pos view for screenshot series 0 bistro: -26.356014 6.4785852 11.6196 0.86602473 -0.32465482 -0.38026375

## Understanding the Input Signal

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

![screenshot of raw image](showcase_images/bistro%20series%200%200.png)

The raw image is extremely noisy. With only one ray per pixel, the chance of any given ray hitting a bright light source is low — most rays return near-zero radiance, while the occasional lucky hit comes back orders of magnitude brighter. The result is high-contrast salt-and-pepper noise dominated by isolated bright spikes called fireflies.

Offline renderers solve this by averaging thousands of rays per pixel. In real-time we do the same thing temporally: accumulate ray results over many frames instead of many rays within one frame.

A per-pixel sample count image tracks how many frames have contributed to each pixel. Each frame, the new ray result is blended into the existing accumulated value using a weight that shrinks as the count grows:

```hlsl
accumulated = lerp(history, new_sample, 1.0f / (1.0f + sample_count));
sample_count = min(max_history_frames, sample_count + 1);
```

At count 0 the blend weight is 1 — the new sample is taken directly with no history. At count 31 it is 1/32, so 31 previous frames dominate. The sample count resets to zero on disocclusion so newly revealed surfaces converge from scratch.

![screenshot of raw image](showcase_images/bistro%20series%200%201.png)

With 255 accumulated frames the image looks far cleaner — but a handful of pixels still stand out dramatically. These are fireflies: ray hits so bright and so rare that even across hundreds of frames they remain temporally incoherent, appearing only one frame in several hundred. They create a permanently unstable, flickering image in their vicinity.

A temporal firefly filter could suppress them, but temporal filters need several frames to build confidence after a disocclusion — when the camera moves and a surface that was hidden behind other geometry becomes newly visible, it has no accumulated history yet. During that ramp-up window a temporal filter either passes the firefly through (noise spikes) or suppresses too aggressively (dark banding). Neither is acceptable in a fast-paced game. So instead this denoiser relies on a conservative spatial firefly filter that is always-on and requires zero history.

**Guided filtering — geometric mean clamp.** Standard firefly filters compute the luma mean and variance in a neighborhood and clamp within some multiple of the standard deviation. That still passes large outliers when variance is high. A geometric mean is far more outlier-resistant. The filter computes the geometric mean in a 5×5 neighborhood around each pixel and clamps brightness to `ceiling_factor * geometric_mean`. A good baseline ceiling factor is in the 8–32 range — 8 is aggressive and dark, 32 lets more energy through but also more fireflies. This is very aggressive — it deletes a lot of firefly energy and makes the image darker — but the denoiser compensates for that energy loss in a later pass, so the aggressiveness is the right call.

The geometric mean is computed in log space: accumulate `log(sample)` across the neighborhood, average, then exponentiate back. Multiplying by a small constant before the log and dividing after keeps values numerically well-behaved in fp16.

```hlsl
static const float PERCEPTUAL_SPACE_MULTIPLIER = 1e1f;

func linear_to_perceptual(float v) -> float {
    return log(max(v, 1e-8f) * PERCEPTUAL_SPACE_MULTIPLIER);
}
func perceptual_to_linear(float v) -> float {
    return exp(v) / PERCEPTUAL_SPACE_MULTIPLIER;
}

// accumulate in log space across the neighborhood
y_mean_geometric_acc += linear_to_perceptual(sample_y);

// convert back: exp(mean(log(x))) = geometric mean
const float y_mean_geometric = perceptual_to_linear(y_mean_geometric_acc * rcp(valid_samples));
const float clamp_factor = min(1.0f, (y_mean_geometric * ceiling_factor) / pixel_y);
filtered_diffuse *= clamp_factor;
```

![screenshot of raw image](showcase_images/bistro%20series%200%202.png)

Fireflies are gone. Remaining noise is now broad, low-frequency, and spatially coherent — the kind of noise a spatial blur can handle.

**Spatial filtering — post-blur.** Instead of accumulating only temporally, we share ray samples across neighboring pixels in a bilateral blur — the post-blur. A bilateral blur is like a Gaussian blur but with an additional edge-stopping term: samples that lie on a different surface are rejected, so the blur never crosses geometry boundaries. With a radius of even 8 pixels, the effective sample count per pixel jumps dramatically.

![screenshot of raw image](showcase_images/bistro%20series%200%203.png)

The image is noise-free, but the lighting bleeds over geometry edges. We need edge-stopping.

**Geometric edge stop.** For each sample neighbor, reconstruct its world-space position from depth, project it onto the virtual plane defined by the center pixel's position and normal, and measure the perpendicular distance. If that distance exceeds a threshold (proportional to pixel footprint size), the neighbor is rejected. This is a simple and effective way to test whether two pixels lie on the same surface.

`surface_weight` runs three binary tests — any one failing returns 0. Plane distance is tested bidirectionally (A: neighbor projected onto center plane, B: center projected onto neighbor plane) to catch glancing surfaces where a one-sided test would pass. A loose raw distance cap handles parallel surfaces that somehow satisfy both plane tests.

```hlsl
func surface_weight(float2 inv_render_target_size, float near_plane, float depth,
    float3 vs_position, float3 vs_normal,
    float3 other_vs_position, float3 other_vs_normal,
    float threshold_scale = 2.0f) -> float
{
    const float pixel_size = ws_pixel_size(inv_render_target_size, near_plane, depth);
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

The threshold scales with `ws_pixel_size` — the world-space size of one pixel at the current depth — so the cutoff automatically tightens for close geometry and loosens for distant geometry.

**Normal edge stop.** The plane test alone will pass two surfaces meeting at a shallow angle. Adding a normal similarity weight catches those: `dot(n_center, n_sample)` raised to a power to sharpen the cutoff. Together the two tests cleanly stop the blur at geometry boundaries.

```hlsl
func normal_similarity_weight(float3 normal, float3 other_normal) -> float
{
    const float validity       = (max(0.1f, dot(normal, other_normal) + 0.85f)) * (1.0f / 1.85f);
    const float tight_validity = validity * validity;
    return tight_validity;
}
```

The `+0.85` shift moves the zero crossing so that even opposing normals get a small non-zero weight rather than being hard-rejected. Squaring sharpens the response — normals that agree closely get near-full weight, those that diverge even moderately are suppressed.

![screenshot of raw image](showcase_images/bistro%20series%200%204.png)

Now the image is clean and edge-preserving. The final problem is temporal smearing. Accumulating on fixed screen pixels means that when the camera moves, old history data — accumulated for geometry now off-screen — gets applied to whatever new geometry occupies those pixels. The result is visible ghosting.

![screenshot of raw image](showcase_images/bistro%20series%200%205.png)

## Temporal Reprojection

The fix for temporal smearing is reprojection: instead of always reading history from the same screen-space pixel, figure out where the current pixel was in the previous frame and read from there.

Every pixel has a depth value. Unproject it through the inverse view-projection matrix to recover world-space position, then project that position through the previous frame's view-projection matrix to get a UV into the previous frame's accumulation buffer. The reprojected UV generally lands between four history texels, each of which must be tested for geometric validity before being blended.

This works cleanly for indirect diffuse because diffuse is **view-independent**: the amount of indirect light arriving at a surface point does not change when the camera moves. History accumulated at that world-space point last frame is just as valid this frame. Specular is the opposite — the reflection changes entirely with view angle — which is why temporal accumulation is far more powerful for diffuse than for specular.

When the reprojected position fails its geometric validity tests — none of the four neighboring history texels pass the surface and normal checks — the pixel is treated as a fresh disocclusion and its history is reset to zero. The pixel then converges from scratch over the following frames.

A per-pixel sample count tracks how long history has been accumulating. The blend weight is `1 / (1 + count)` — an exponential moving average (EMA) that starts aggressive (50% new at count 1) and grows increasingly conservative as history builds up (3% new at count 31). Young pixels converge fast; stable pixels change slowly.

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

Each of the four gathered texels is tested with `surface_weight` (introduced above) after unprojecting its depth to world space via the previous camera. A soft normal similarity weight is multiplied in on top.

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

## Where To Go From Here

What has been built here is one of the simplest viable denoisers for indirect diffuse. It works — but it leaves a lot on the table. With the right additions the same pipeline can produce noticeably less noise, less temporal boiling, faster reactivity to lighting changes, sharper results near geometry edges, and significantly more recovered light energy, all at the same or better temporal stability.

[Denoising Tricks](Denoising%20Tricks.md) covers exactly that: a set of targeted improvements to each pass that make the denoiser self-regulating, scene-independent, and robust in the conditions a real game actually throws at it.
