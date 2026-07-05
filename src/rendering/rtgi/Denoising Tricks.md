# Denoising Tricks

## Motivation

[Denoising Basics](Denoising%20Basics.md) builds a working denoiser from first principles: geometric edge stops, temporal reprojection, a sample count driven EMA blend. It works. But working is not the same as robust.

The basic denoiser exposes a set of parameters that interact in ways that make them hard to set globally:

- **Blur radius** — the right radius depends heavily on scene content. A radius that cleans up an open outdoor scene will smear a corridor full of fine geometry. Getting it wrong means either residual noise or blurry mush, with no single value that works everywhere.
- **Max sample count** — too low and the image stays noisy even when the camera is still; too high and the denoiser becomes sluggish to react to lighting changes or motion, producing ghosting that takes seconds to resolve.
- **Firefly clamp** — the geometric mean clamp is deliberately aggressive. In well-lit areas this is fine, but in very dark regions it can over-suppress valid bright hits, making the image unnecessarily dark or causing energy loss that reads as incorrect shadowing.
- **Temporal instability** — reprojection failures, partial disocclusions, and thin geometry all produce pixels with very low sample counts that never stabilize, causing persistent shimmer even when nothing is moving.
- **Temporal slowness** — a high max sample count makes stable regions look clean but means the denoiser reacts slowly to sudden illumination changes. The same blend weight that gives stability also gives lag.

Each of these is a real problem in practice. This document goes through each pass in the pipeline and describes the tricks used to address them — adaptive filter guiding, fast history for reactive blending, soft-normalized sample counts, energy compensation for the firefly clamp, and more. None of them are fundamental changes to the denoiser structure; they are targeted improvements that make each stage more self-regulating and scene-independent.

> DEV NOTE: pos view for screenshot series 0 bistro: -13.808407 9.0285425 5.2979064 0.9969183 0.062728465 0.04710621

## Half Resolution and Upscaling

The single most impactful performance decision in the pipeline is running everything at half resolution in each axis — one quarter of the full pixel count. Tracing, firefly filtering, pre-blur, temporal accumulation, and post-blur all operate on half-res buffers. Without this, the 400µs total budget on an RTX 4080 would not be achievable. Every other trick in this document operates on top of this foundation.

**Downscaling depth, normals, and albedo**

Depth, face normals, and albedo are downscaled from full resolution inside the same GBuffer generation pass, using groupshared memory to avoid a separate pass. Each 2×2 block of full-res pixels writes into shared memory; once the block is complete, the pixel at position `(0,0)` within the block selects a single representative sample. Rather than a plain closest-depth pick, the selection finds the depth closest to the RMS average of the four — this biases toward surfaces that are representative of the block rather than always picking foreground geometry, which would cause thin objects to consistently dominate.

```hlsl
// RMS depth: L2 power mean of the four depths
const float4 scaled_depths = square(depths);
const float avg_scaled_depth = sqrt(dot(scaled_depths, 1.0f) * 0.25f);

// Pick the depth closest to the RMS average
const float4 scaled_depth_differences = abs(depths - avg_scaled_depth);
int best_depth_index = /* index of minimum */;

// Normal and albedo follow the selected depth — they stay consistent with it
float closest_depth = depths[best_depth_index];
uint closest_face_normal = normals[best_depth_index];
float4 closest_albedo = albedos[best_depth_index];
```

Tying normal and albedo to the selected depth index keeps the three G-buffer values geometrically consistent with each other, which matters for the surface_weight tests run throughout the denoiser.

**Upscaling**

The upscaler runs at full resolution and reads from the half-res denoised buffers. It preloads a `(group_size/2 + 2)²` tile of half-res sh_y, CoCg, face normals, and view-space positions into groupshared memory so the 3×3 neighborhood lookup for each full-res pixel costs no global memory bandwidth.

The filter is a 3×3 tent (derived from collapsing a 5-tap Gaussian): the kernel weights are not symmetric but subpixel-aware. Each full-res pixel knows which quadrant of its parent half-res texel it sits in via `full_res_pixel_index & 1`. If the pixel is in the left half, the kernel is weighted more heavily toward the left half-res neighbor; if it is in the right half, the weights flip. This means neighboring full-res pixels that fall in the same half-res texel but different quadrants use slightly different kernels, letting the tent filter reconstruct sub-texel variation rather than all four full-res pixels inside a half-res texel getting identical results.

```hlsl
// Determine which quadrant of the half-res texel this full-res pixel is in
const uint2 rtgi_subpixel_index = (full_res_pixel_index & 0x1);

// Tent weights derived from a 5-tap Gaussian, collapsed to 3 taps by folding the outer pair
static const float3 TENT_WEIGHTS_LEFT_3 = { 5.0/16.0, 10.0/16.0, 4.0/16.0 };

// Flip the weight order based on subpixel position
const float3 tent_weights_x = rtgi_subpixel_index.x == 0 ? TENT_WEIGHTS_LEFT_3 : TENT_WEIGHTS_LEFT_3.zyx;
const float3 tent_weights_y = rtgi_subpixel_index.y == 0 ? TENT_WEIGHTS_LEFT_3 : TENT_WEIGHTS_LEFT_3.zyx;
```

Each of the 9 half-res samples in the 3×3 tent receives a geometry weight — the planar surface distance between the half-res sample's view-space position and the full-res pixel's position, normalized to pixel footprint size, with a 3-pixel threshold. Normal weight is a 4th-power dot product (`square(square(dot(...)))`), sharpening the cutoff on diverging surfaces. When the geometry tests fail for too many taps the filter falls back to a soft distance-weighted blend, preventing black holes at geometry boundaries where no half-res neighbor is on the same surface.

**SH resolve at full resolution**

The SH resolve — evaluating the L1 SH probe against the detail normal — happens at the very end of the upscaler, after the weighted tent filter has accumulated the blended sh_y and CoCg across the neighborhood. Critically, it uses the full-resolution detail normal (from the normal map), not the half-res face normal used during tracing and denoising.

This is what makes the half-resolution rendering practical from a quality standpoint. Any spatially blurred or upscaled flat radiance value will look soft and washed out on surfaces with strong normal maps — the indirect lighting will not respond to the bumps at all. By carrying directional information through the entire pipeline as an SH probe and only resolving it at full resolution against the high-detail normal, the final indirect lighting remains crisp and responsive to normal map detail regardless of how much spatial blurring happened upstream. The shading variation from the normal map is essentially free — it costs nothing in the denoiser, adds no noise, and produces results visually close to tracing at full resolution.

## Firefly Filter

> DEV NOTE: pos-view bistro series 1: -6.044629 -15.205926 4.6312356 0.80867165 0.56309754 0.17020945

Raising the geometric mean ceiling lets in more energy but also more fireflies — going from 24 to 108 makes the image brighter but noticeably noisier and less stable. Simply cranking it is not a viable solution.

Blurring the image before clamping also helps, but creates a mismatch: the geometric mean is now computed from blurred values, which are quieter than the raw signal, so the ceiling is miscalibrated and stability suffers.

**The star blur trick** keeps the two separate: compute the geometric mean from the raw signal, but clamp a blurred version of the center pixel. The mean stays accurate and well-calibrated; the blurred input lets more energy through. No extra prepass needed — the existing 5×5 window is split: the inner 5 taps form a geometry-weighted + that produces the blurred input, the outer 20 compute the geometric mean. The star taps must be excluded from the mean — a firefly pixel in the mean would pull the ceiling up with it, defeating the clamp.

```hlsl
// Star shape: center + 4 cardinal neighbors
func is_part_of_center_blur(int2 index) -> bool {
    return (abs(index.x) + abs(index.y) <= 1);
}

// outer 20 taps → geometric mean on the raw signal
// inner 5 star taps → blurred input to clamp against it
if (!is_part_of_center_blur(int2(x, y)))
    y_mean_geometric_acc += linear_to_perceptual(sample_y);
else
    star_blurred_diffuse_acc += geometric_weight * sample_diffuse;

filtered_diffuse *= min(1.0f, (y_mean_geometric * ceiling_factor) / filtered_diffuse.w);
```

No screenshots here — the difference between a higher ceiling factor and the star blur is nearly invisible in a still frame. The real tell is temporal: with a raised ceiling the image boils noticeably as fireflies flicker in and out across frames, while the star blur keeps the same energy level without that instability.

Denoiser before Star Blur (geometric mean cap factor 32):

![screenshot](showcase_images/bistro%20series%201%200.png)

Denoiser after Star Blur (geometric mean cap factor 32):

![screenshot](showcase_images/bistro%20series%201%201.png)

## Spatial Filter Width Guiding

A spatial filter needs to be wide. The human eye is very good at detecting noise on large flat surfaces — an untextured wall or a floor lit by a single indirect bounce will show residual shimmer even at blur radii that look perfectly clean on detailed geometry. To keep those areas noise-free the filter radius needs to be on the order of 32–128 pixels in the worst case.

But a uniform wide radius looks bad everywhere else. Detailed geometry, contact shadows, and areas where indirect light has sharp transitions all suffer from over-blurring if the same radius is applied. A single global radius is always a compromise that is too narrow in open areas and too wide near detail.

The solution is to guide the filter width per pixel based on what the local signal actually needs. Instead of a fixed radius, compute a per-pixel scale factor between 0 and 1 that the filter radius is multiplied by. Open, flat areas get a factor near 1 and use the full wide radius. Areas where a tighter filter is both permissible and desirable get a lower factor and shrink accordingly.

**Geometric Proximity guide (AO-inspired contact hardening).** One highly effective metric comes from the ray lengths recorded during tracing. Short ray lengths mean the traced rays hit nearby geometry — the pixel is in a contact region, under an overhang, or in a tight corner. Long ray lengths mean the rays escaped into open space. This is directly analogous to ambient occlusion: short rays indicate occluded, contact-heavy regions; long rays indicate open exposure.

In the same inner 3×3 loop used for firefly statistics, ray lengths are averaged in a way that heavily promotes short values — the shortness (`1 - normalized_length`) is raised to a high power before accumulating, then the inverse power is taken on readout. This means even a single very short ray in the neighborhood pulls the guide value down significantly. The result is a per-pixel guide that smoothly scales the filter radius down near geometry and lets it expand freely in open areas, producing sharper contact regions and cleaner open surfaces simultaneously — contact hardening essentially for free as a side effect of the filter width guide.

The filter guide must also be temporally accumulated. A raw per-frame guide is driven by a single noisy ray per pixel and would cause the spatial filter width to wobble from frame to frame, which is visible as shimmering blur radius changes. Accumulating the guide through the same temporal pass smooths it out and makes the filter width stable across frames.

**Code**

Ray length is normalized against a maximum visibility distance — the world-space size of a pixel times the maximum expected filter radius in pixels. The normalized shortness (`1 - normalized_length`, clamped to 0–1) is then raised to a high power (triple-squared = x^8) before accumulating. This extreme compression heavily promotes near-zero values so that even one short ray in the neighborhood dominates the average.

```hlsl
// Maximum ray length that the filter could ever reach — anything beyond is "open space"
const float max_visibility_raylen = pixel_width_ws * max_filter_radius_pixels;

// Per tap in the inner loop: accumulate shortness with L8 power to promote short rays
const float shortness = 1.0f - min(1.0f, sample_ray_length * rcp(max_visibility_raylen));
ray_shortness_acc += square(square(square(shortness))) * geometric_weight;  // shortness^8
```

After the loop the inverse power is applied to recover the mean shortness, then the guide value is lerped from 1 (full width, open area) down toward a minimum of 0.33. Going below 0.33 causes too much noise on untextured surfaces — there is simply not enough signal in a single ray to support a very tight filter on a flat area where noise is maximally visible.

```hlsl
// Inverse of x^8 accumulation: take triple sqrt to get the power-mean
const float ray_shortness_mean = sqrt(sqrt(sqrt(ray_shortness_acc * rcp(valid_footprint_samples))));

// Guide scales from 1 (open) down to 0.33 minimum — tighter causes noise on untextured surfaces
const float raylength_filter_guide = lerp(1.0f, 0.33f, ray_shortness_mean);
```

The guide is then written out and temporally accumulated the same way as the color values.

```hlsl
// Accumulate albedo luminance stats in the inner 3x3 loop
const float neighbor_lum = dot(neighbor_albedo, float3(0.2126f, 0.7152f, 0.1722f));
pp_alb_lum_acc    += neighbor_lum;
pp_alb_lum_sq_acc += neighbor_lum * neighbor_lum;

// After the loop: coefficient of variation = std_dev / mean
const float lum_mean     = pp_alb_lum_acc    * rcp(sample_count);
const float lum_sq_mean  = pp_alb_lum_sq_acc * rcp(sample_count);
const float lum_variance = max(0.0f, lum_sq_mean - lum_mean * lum_mean);
const float albedo_cv    = saturate(sqrt(lum_variance) / max(lum_mean, 0.05f));

// sqrt-compress, then invert: high detail → low guide (tighter filter)
const float surface_detail_guide = 1.0f - sqrt(albedo_cv);
```

The two guides are multiplied together into a single filter guide value, so either one can independently tighten the filter radius — a contact region on a flat wall gets both, an open textured surface gets only the surface guide.

```hlsl
const float filter_guide = raylength_filter_guide * surface_detail_guide;
```

Here an example without geometric proximity filter guiding:

![screenshot](showcase_images/luminara%20series%200%200.png)

Here the same scene with geometric proximity guiding:

![screenshot](showcase_images/luminara%20series%200%201.png)

The Filter guide displayed as a heatmap (Red -> max filter radius, Blue -> min filter radius) 

![screenshot](showcase_images/luminara%20series%200%202.png)

**Surface detail guide**

The ray length guide handles open vs. contact regions well, but it cannot tell the difference between a flat untextured wall and a heavily textured one at the same depth. Both get the same ray lengths, but the textured surface can tolerate a much tighter filter without looking blurry — the texture itself masks residual noise. The flat wall cannot.

A second guide addresses this using albedo as a proxy for surface detail. In the same inner 3×3 loop, the luminance of each neighbor's albedo is accumulated along with its square. After the loop, these give a mean and variance from which the coefficient of variation is computed: `std_dev / mean`. High CoV means the albedo is changing quickly — the surface has detail that will mask blur. Low CoV means the surface is flat and uniform, so the filter needs to stay wide.

On its own, the surface detail guide produces a similarly strong improvement to image quality as the geometric proximity guide — tightening the filter on textured geometry and keeping it wide on flat areas independently does a lot. But the two guides are most powerful in combination: geometric proximity narrows the filter in contact regions, and surface detail unlocks how far that narrowing can go. Together they cover cases that neither handles well alone.

The CoV is compressed with a square root before being inverted into a guide. The sqrt re-scales the metric so that moderate texture detail counts meaningfully, while extreme high-contrast edges (which would otherwise dominate the CoV) are not weighted disproportionately — without it, a single sharp albedo transition could collapse the filter to minimum width over a large area.

The surface detail guide also feeds back into the ray length guide. The ray length guide alone is capped at a minimum of 0.33, but on textured surfaces noise is masked by the texture and a much tighter filter is both safe and desirable. The surface detail guide unlocks this: on high-detail surfaces it drives the ray length guide minimum all the way down to 0, allowing full contact hardening. On low-detail surfaces the minimum stays near 0.33 and the filter stays wide enough to keep things clean.

This only needs to activate for the most extreme texture variation — squaring or cubing the surface detail guide concentrates the effect at the high-detail end, leaving moderately textured surfaces unaffected.

```hlsl
// Square to push the effect toward only the most detailed surfaces
const float smooth_scale = surface_detail_guide * surface_detail_guide;
// min_guide_value: 0.33 for flat surfaces, approaches 0 for highly detailed ones
const float min_guide_value = lerp(0.33f, 0.0f, smooth_scale);
const float raylength_filter_guide = lerp(1.0f, min_guide_value, ray_shortness_mean);
```

```hlsl
// Accumulate albedo luminance and its square in the inner 3x3 loop
const float neighbor_lum = dot(neighbor_albedo, float3(0.2126f, 0.7152f, 0.1722f));
pp_alb_lum_acc    += neighbor_lum;
pp_alb_lum_sq_acc += neighbor_lum * neighbor_lum;

// After the loop: CoV = std_dev / mean  (clamped mean avoids div-by-zero on black surfaces)
const float lum_mean    = pp_alb_lum_acc    * rcp(sample_count);
const float lum_sq_mean = pp_alb_lum_sq_acc * rcp(sample_count);
const float std_dev     = sqrt(max(0.0f, lum_sq_mean - lum_mean * lum_mean));
const float albedo_cv   = saturate(std_dev / max(lum_mean, 0.05f));

// sqrt compresses the CoV for robustness, then invert: high detail → low guide
const float surface_detail_guide = 1.0f - sqrt(albedo_cv);
```

Here the scene with only geometric proximity guiding:

![screenshot](showcase_images/luminara%20series%200%201.png)

Here the scene with geometric proximity and surface detail guiding:

![screenshot](showcase_images/luminara%20series%200%205.png)

Here the the filger guide with combined geometric proximity and surface detail guiding:

![screenshot](showcase_images/luminara%20series%200%206.png)

> DEV NOTE: pos view for screenshot series 0 luminara: 144.73042 101.39061 10.705887 0.9114913 0.3362906 -0.23683819

**Variance guiding**

Unlike the geometric proximity and surface detail guides — which control the filter radius — variance guiding does not change the width of the post-blur at all. It changes the per-sample weight inside the filter based on how much the sample's luminance deviates from the center pixel.

Variance guiding is explicitly not the primary spatial guide. The reason is a hard constraint: it requires temporally accumulated, already-converged variance. That variance is computed from the slow history and takes many frames to stabilize — on the first frame after a disocclusion, or any time history is young, the variance estimate is unreliable noise. Using it aggressively in those conditions would cause the filter to flicker visibly as the variance itself thrashes from frame to frame. It simply cannot do the heavy lifting that the geometric proximity and surface detail guides handle from frame zero.

Instead it kicks in after a few frames as a final sharpener, once the slow variance has had time to settle. Its role at that point is narrow but useful: the post-blur accumulates a slow temporal variance of radiance over the full history, and the guiding is one-sided — samples brighter than the center pixel are penalized relative to the slow standard deviation, while samples darker than the center pass through freely. This asymmetry is intentional — a dark sample blending into a bright region is a valid shadow, and suppressing it would brighten shadow edges. A bright sample bleeding into a dark region is far more likely to be residual noise or a leaking bright background.

What variance guiding does well at that stage, and cheaply, is act as a final fine-tune sharpener on dark areas where the post-blur would otherwise over-blend: deep shadow contact regions, distant shadowed geometry, corners where the filter tends to pull in bright surrounding samples. In those areas the one-sided clamp quietly prevents the brightest neighborhood samples from contributing, keeping the shadows crisp without affecting stable well-lit regions at all. The effect is visible in the sponza comparisons below — distant shadow boundaries that smear together without it stay distinctly separated, and contact shadow detail in the foreground is tighter.

```hlsl
// In the temporal pass: accumulate slow relative variance (same fp16 trick as fast history —
// store (radiance - mean)²/mean² rather than radiance² to avoid overflow at high radiance values)
const float slow_new_relative_variance = min(
    square((new_radiance - reprojected_slow_mean) / max(reprojected_slow_mean, 1e-6f)), 4.0f);
accumulated_slow_relative_variance = lerp(
    reprojected_slow_relative_variance, slow_new_relative_variance, slow_variance_blend);
```

```hlsl
// In the post-blur: one-sided luminance weight per tap
const float slow_relative_std_dev = sqrt(pixel_statistics.w);  // .w = slow_rel_var

// Ramp variance influence in over frames 4–12; below 4 the variance estimate is too
// young to trust and the guide contributes nothing.
const float variance_guide_ramp = saturate((accumulated_sample_count - 4.0f) / 8.0f);

// Fold the ramp into the tolerance: at ramp=0 max_allowed_y == pixel_y and every
// sample passes at weight 1 (guide inactive). At ramp=1 the full std_dev tolerance applies.
const float max_allowed_y = pixel_y * (1.0f + 4.0f * slow_relative_std_dev * variance_guide_ramp);
const float one_sided_luminance_weight = min(1.0f, max_allowed_y / (sample_sh_y.w + 1e-4f));
```

Samples at or below the tolerance pass at weight 1; samples brighter than the ceiling are suppressed with a 1/x falloff — cheap (one divide, one min) and never hard-rejects a sample. The ramp ensures the guide has zero influence for the first four frames while the slow variance is still settling, then linearly reaches full strength by frame 12.

Without variance guiding (shadows in distance blend together, contact details soft):

![screenshot](showcase_images/sponza%20series%200%200.png)

With variance guiding (distant shadow edges sharper, contact details preserved):

![screenshot](showcase_images/sponza%20series%200%201.png)

## Pre-Blur

The pre-blur is named for where it sits in the pipeline: it runs before temporal accumulation, while the post-blur you already know from the basics runs after. This ordering matters — the pre-blur shapes the signal that goes into the temporal accumulator, while the post-blur refines what comes out of it.

Temporal accumulation has a fundamental weakness on disocclusion: newly revealed pixels have zero history and are shown raw. After firefly filtering, the remaining noise is not high-frequency salt-and-pepper — it is broad, low-frequency blobs that spread across large areas. This is exactly the kind of noise the post-blur spatial filter cannot handle.

The post-blur uses a separable horizontal/vertical split to keep costs low. At small radii this approximation is accurate enough and the artifacts are invisible. But killing low-frequency disocclusion noise requires a very wide radius, and at wide radii the axis-split separable filter produces cross-shaped smearing artifacts that are far more objectionable than the noise they are replacing. You cannot simply widen the post-blur to deal with disocclusions — the filter breaks before the noise is gone.

The pre-blur runs before temporal accumulation and uses a different filter that does not have this limitation. It blurs the firefly-filtered signal with a wide radius, absorbing the low-frequency noise before it ever enters the temporal accumulator. Disoccluded pixels arrive already smoothed and temporal accumulation only needs to handle the residual, which it converges on in far fewer frames. The overall temporal stability also improves — a smoother incoming signal means less variance for the accumulator to fight in every frame, not just on disocclusions.

The pre-blur filter is stochastic and anisotropic. Samples are drawn from a disc around each pixel, but the disc is stretched and oriented along the surface normal projected into screen space — a surface seen at a shallow angle gets a kernel that follows its orientation rather than cutting across it. With only 8–32 samples the kernel is far too sparse to avoid aliasing on its own, but since it runs before temporal accumulation those sparse samples are amortized over many frames.

The image below visualizes the filter kernel for a single center pixel on the side of a shop viewed at an angle — the white dots show the sample taps. The oval shape is the anisotropy in action: the kernel stretches along the surface rather than sampling uniformly in screen space. The tap count here is 255 for visualization purposes only; in practice 8–32 taps are sufficient and the temporal accumulator fills in the rest over subsequent frames.

![pre-blur kernel visualized — anisotropic oval on shop wall viewed at angle](showcase_images/bistro%20series%205%203.png)

The filter guide is squared before being applied to the radius (`square(filter_guide)`). The base radius is set for the absolute worst case — a completely flat, untextured surface, where low-frequency noise is maximally visible and there is nothing to mask it. That worst-case radius is massive, 100–200 pixels. With a linear scale, even moderately textured surfaces would still receive a very large radius, losing detail unnecessarily. Squaring the guide makes it shrink much faster away from 1: a guide value of 0.5 gives only 0.25 of the full radius rather than half. This means even a small amount of surface detail drives the radius down dramatically, preserving detail that would otherwise be lost to the enormous worst-case blur. The clean flat face of a shop wall gets the full radius; anything with texture collapses to a tight kernel almost immediately.

```hlsl
// Filter guide squared for aggressive width adaptation
const float pixel_filter_guide = square(filter_guide);
float blur_radius = max(2.0f, pre_blur_base_width * pixel_filter_guide);

// Sample count also scales with guide: fewer samples where radius is small
uint samples = lerp(8u, 16u, pixel_filter_guide);
```

**Uniform disc sampling**

Samples are drawn from a uniform disc using the concentric mapping. A small twist: the random radius `r` is drawn linearly rather than with the standard `sqrt` remapping, biasing samples toward the center. This gives more weight to close neighbors which tend to be more geometrically similar, reducing the number of rejected taps at geometry edges.

```hlsl
float2 rand_concentric_sample_disc_center_focus()
{
    float r     = rand();                   // linear r — biases toward center
    float theta = rand() * 2.0f * PI;
    return float2(cos(theta), sin(theta)) * r;
}
```

**Anisotropic kernel**

The disc is not applied uniformly in screen space. The surface normal is projected to view space and its x and y components are used to compute a per-axis screen-space gradient — how many pixels the surface moves per unit of the disc offset in each axis. Multiplying the disc sample by this gradient stretches and rotates the kernel to follow the surface orientation. A shallow-angle surface gets a very elongated kernel that runs along the surface rather than across it. A small view-direction bias is applied to the normal first to prevent the kernel from collapsing to zero width on surfaces that face exactly sideways.

```hlsl
// Bias normal slightly toward camera to prevent kernel collapsing at grazing angles
const float3 biased_vs_normal = lerp(vs_normal, float3(0, 0, 1), 0.01f);

// Screen-space gradient: how much the surface moves in screen space per world-space unit
const float2 ss_gradient = float2(
    sin(acos(biased_vs_normal.x)),
    sin(acos(biased_vs_normal.y))
) * inv_render_target_size;

// Apply disc sample through gradient to get anisotropic screen-space offset
const float2 disc_noise  = rand_concentric_sample_disc_center_focus();
const float2 sample_2d   = disc_noise * blur_radius;
const float3 sample_ndc  = ndc + float3(ss_gradient * sample_2d, 0.0f);
```

Without pre-blur (camera moving left, pillar reveals disoccluded shop — low-frequency noise bubbles in the newly revealed area):

![screenshot](showcase_images/bistro%20series%202%200.png)

With pre-blur (same motion — disocclusion resolves quite clearly within the first few frames):

![screenshot](showcase_images/bistro%20series%202%201.png)

The pre-blur result is noticeably brighter than the basic denoised image. This is not a mistake — it is energy compensation, explained next.

**Energy recovery**

The pre-blur also solves the energy loss from the firefly clamp. Consider what the signal looks like at each stage. After firefly clamping the raw signal is still extremely sparse and noisy — the bright spikes are suppressed but most pixels carry near-zero energy:

![screenshot](showcase_images/bistro%20series%201%203.png)

After star blurring on top of the firefly clamp, the energy is lower but the signal is more manageable to filter:

![screenshot](showcase_images/bistro%20series%201%204.png)

The final denoised image without energy compensation looks good in terms of noise, but it is too dark — the aggressive firefly clamp deleted a lot of real energy:

![screenshot](showcase_images/bistro%20series%201%205.png)

The pre-blur recovers this energy by using the firefly suppression ratio as a per-sample weight. When the firefly clamp reduced a pixel to one third of its value, that pixel's suppression ratio is 3. During the pre-blur, samples contribute to the blur weighted by both their geometric validity and this suppression ratio — so heavily clamped pixels spread three times as much weight to their neighbors as unclamped ones. The energy that was removed from the bright pixel gets redistributed into the surrounding area rather than disappearing.

There is a principled reason this works. The firefly clamp is a biased estimator: for a pixel with true radiance `v` and clamping factor `k`, the clamped output is `v * k`, which systematically underestimates `v` whenever `k < 1`. When the pre-blur computes a weighted average of clamped values, the numerator accumulates `sum(base_weight_i * v_true_i * k_i)` — biased low, because the brightest pixels (with the smallest `k`) contribute the least. Weighting each sample by `1/k_i` instead changes the numerator to `sum(base_weight_i * (1/k_i) * v_clamped_i) = sum(base_weight_i * v_true_i)` — the `k_i` cancels and the numerator becomes the unbiased weighted sum of true radiance values. The denominator inflates slightly (by `sum(base_weight_i / k_i)` rather than `sum(base_weight_i)`), but fireflies are rare so most pixels have `k = 1`, the inflation is small, and the numerator correction dominates. This is essentially inverse-probability weighting: when samples are downweighted by a known factor, multiplying by the reciprocal recovers an unbiased estimate of the original signal.

This also explains why the approach is fundamentally more stable than simply raising the firefly clamp ceiling. A higher ceiling lets more energy through unconditionally — any pixel that happens to be bright gets to contribute more, whether or not the spatial filter has touched it. The pre-blur energy recovery only releases a firefly's suppressed energy when the stochastic filter actually samples that pixel. A firefly that is never hit by any pre-blur sample stays dark and suppressed; its energy is not recovered and it does not appear in the output. This gating is critical: when a firefly pixel IS sampled by the pre-blur, it is also blended with its neighbors and spatially flattened in the process. The energy is recovered precisely when and because the firefly has been stabilized. A firefly that was never hit remains exactly as suppressed as it was after the clamp — no boiling, no leakage.

Naturally, when there are few fireflies and little clamping is happening, the suppression ratios are near 1 and the weighting has almost no effect — the filter behaves as a plain low-frequency smoother. The energy recovery only activates where it is actually needed.

Denoised, firefly filter only — no star blur, no pre-blur:

![screenshot](showcase_images/bistro%20series%201%200.png)

Denoised, firefly filter + star blur, no pre-blur:

![screenshot](showcase_images/bistro%20series%201%201.png)

Denoised, firefly filter + star blur + pre-blur with energy recovery:

![screenshot](showcase_images/bistro%20series%201%202.png)

```hlsl
// Firefly suppression ratio used as weight multiplier during pre-blur accumulation.
// Pixels clamped to 1/N of their value get N× more influence in the blur,
// redistributing their lost energy into the surrounding area.
const float firefly_power = firefly_factor_image[sample_index] * RTGI_MAX_FIREFLY_FACTOR;
const float weight = geometric_weight * normal_weight * firefly_power;
```

## Fast Temporal History Clamping

A fast temporal history is only viable because of all the spatial blurring before the temporal pass. A 4-frame EMA on the raw signal after just the firefly clamp would track noise and spikes rather than scene brightness:

![screenshot](showcase_images/bistro%20series%201%203.png)

The pre-blurred signal entering the temporal pass is stable enough to actually build a meaningful short-window history from:

![screenshot](showcase_images/bistro%20series%201%205.png)

With pre-blurred input, 4 frames is enough to accumulate a reliable radiance mean and variance. With the raw signal the fast history would have to be so conservative it would barely react to anything.

A fast history EMA of radiance mean and relative variance is accumulated in parallel with the main color history, capped at ~4 frames. These two values then modulate the slow history's blend weight in opposite directions:

**Mean divergence → blend toward new frames.** The fast mean tracks the current scene brightness. If the slow history mean and the fast mean diverge significantly, it means the slow history is holding onto outdated brightness — lighting has changed and the slow history hasn't caught up. The larger the ratio between them, the more confidence is reduced, pushing the blend toward the incoming frame so the slow history converges faster.

**High variance → blend toward history.** High relative variance means the fast mean itself is wobbly and unreliable — the signal is noisy or the scene is in flux and the mean changes can't be fully trusted. In this case variance scaling boosts confidence, pulling the blend back toward history to avoid overreacting to noisy mean fluctuations. Low variance means the fast mean is trustworthy and stable, so the mean divergence signal can be acted on more directly.

```hlsl
const float FAST_HISTORY_FRAMES = 4.0f;
const float fast_blend = 1.0f / (1.0f + min(accumulated_sample_count, FAST_HISTORY_FRAMES));

// Accumulate fast mean as EMA
accumulated_fast_mean = lerp(reprojected_fast_mean, new_radiance, fast_blend);

// Relative variance: (radiance - mean)² / mean² — stored relative to avoid fp16 overflow.
// Raw squared radiance would explode for bright values; dividing by mean keeps it dimensionless
// and always within a safe fp16 range regardless of scene brightness.
const float new_rel_variance = min(square((new_radiance - reprojected_fast_mean) / max(reprojected_fast_mean, 1e-6f)), 4.0f);
accumulated_fast_variance = lerp(reprojected_fast_variance, new_rel_variance, fast_blend);

// Mean scaling: ratio of fast to slow mean — large divergence reduces confidence (blend toward new)
const float mean_ratio       = max(slow_mean, fast_mean) / (min(slow_mean, fast_mean) + 1e-8f);
const float mean_scaling     = square(1.0f / max(1.0f, mean_ratio));

// Variance scaling: high variance boosts confidence (blend toward history — mean changes less trustworthy)
const float variance_scaling = square(1.0f + sqrt(accumulated_fast_variance) * 2.0f);

// Cap at 2x sample count; both scalings applied multiplicatively inside the cap
history_confidence = min(accumulated_sample_count * 2.0f,
                         history_confidence * variance_scaling * mean_scaling);
float blend = 1.0f / (1.0f + history_confidence);
```

Without fast history reactivity scaling, areas near a fast-moving bright light source adapt too slowly. As the emissive ball moves, the surrounding pixels hold onto the bright lighting from when the ball was closer to their position — their slow history was built up while the ball was nearby and now cannot let go fast enough. Meanwhile the fresh pixels in the trail of the ball have short history and are darker because they never accumulated the ball being that far out yet. The result is a visible bright ghost trail left behind the ball:

![screenshot](showcase_images/bistro%20series%203%200.png)

With the mean divergence scaling active, pixels whose slow history has fallen behind the fast mean are detected and their blend is pushed toward the incoming frame. The surroundings update fast enough that no trail is left. The debug overlay shows the reactivity scaling heatmap — areas near the ball light up as high-reactivity, confirming the scaling is activating exactly where needed:

![screenshot](showcase_images/bistro%20series%203%201.png)

**Temporal firefly filter**

The fast history mean also enables a lightweight secondary firefly clamp inside the temporal pass. Before blending the new frame into the slow history, the incoming radiance is tested against the fast mean: if it exceeds `fast_mean * (1 + 0.5 * fast_std_dev)`, the entire new sample is scaled down to match. Unlike the spatial firefly filter, this only activates once the fast history has accumulated enough frames to be reliable.

```hlsl
// Only runs once fast history is built up (> FAST_HISTORY_FRAMES accumulated)
if (accumulated_sample_count > FAST_HISTORY_FRAMES && temporal_firefly_filter_enabled)
{
    const float brightness_ratio = reprojected_fast_temporal_mean
        * (1.0f + sqrt(reprojected_fast_temporal_variance) * 0.5f)
        / new_diffuse.w;
    const float clamp_factor = min(1.0f, brightness_ratio);
    new_diffuse  *= clamp_factor;
    new_diffuse2 *= clamp_factor;
}
```

This is not strictly necessary — the spatial filter catches most fireflies before they reach the temporal pass — but it noticeably reduces boiling in scenes with a lot of high-frequency emissive detail where a few bright samples survive the spatial clamp.

## Spherical Harmonics Denoising

The spatial and temporal blur passes smear lighting values across a surface. For a pixel using a flat face normal this is fine, but surfaces with high-frequency normal maps lose the directional variation in their lighting — the denoised value represents the average incoming light over a large neighborhood, and evaluating that flat average against a bumpy normal map gives flat-looking indirect lighting.

The fix is to not store a flat radiance value at all. Instead, project each ray's result onto a per-pixel spherical harmonic probe. The SH probe stores not just how much light arrived, but from which directions it came. After denoising, the probe is evaluated against the high-detail normal, recovering the directional shading that blurring would otherwise erase.

The SH values go through the denoiser exactly like color values — they are just floats and can be lerped with the same blend weights. The only things that change are what gets written at the start of the pipeline and what gets read at the end.

**YCoCg color space**

Before projecting onto SH, radiance values are converted from RGB into YCoCg: a radiance channel Y and two chroma difference channels Co and Cg.

```hlsl
float3 linear_to_y_co_cg(float3 color)
{
    float y  = dot(color, float3( 0.25,  0.5,  0.25));
    float Co = dot(color, float3( 0.5,   0.0, -0.5 ));
    float Cg = dot(color, float3(-0.25,  0.5, -0.25));
    return float3(y, Co, Cg);
}

float3 y_co_cg_to_linear(float3 color)
{
    float t = color.x - color.z;
    float3 r;
    r.y = color.x + color.z;
    r.x = t + color.y;
    r.z = t - color.y;
    return max(r, 0.0);
}
```

Y is by far the perceptually dominant channel — the eye is far more sensitive to radiance variation than to chroma variation, and all the high-frequency directional lighting detail lives in Y. Co and Cg tend to vary slowly and smoothly across surfaces; they also stay numerically smaller than Y for typical scene colors, which makes them more fp16-friendly. This is why all the interesting per-pixel work — the SH projection, the firefly clamp, the relative variance accumulation — operates on Y, while Co and Cg are carried as flat values without any directional encoding. Storing flat chroma instead of SH chroma saves significant bandwidth: the color pipeline carries `float4 sh_y + float2 co_cg` per pixel rather than `float4 sh_y + float4 sh_co + float4 sh_cg`.

**Projecting a ray result onto SH**

Only radiance is projected onto SH; the chroma channels are stored flat because they carry far less perceptual high-frequency detail.

```hlsl
// Converts a linear RGB radiance sample and its incoming direction to YCoCg-space L1 SH
void radiance_to_y_co_cg_sh(float3 radiance, float3 direction, out float4 sh_y, out float2 co_cg)
{
    float3 y_co_cg = linear_to_y_co_cg(radiance);
    co_cg = y_co_cg.gb;                  // chroma stored flat
    // L1 SH projection: sh.xyz = direction * Y (directional coefficients), sh.w = Y (DC term)
    sh_y = float4(direction * y_co_cg.x, y_co_cg.x);
}

// In the tracer, accumulate over all samples per pixel:
float4 sh_y_acc  = float4(0, 0, 0, 0);
float2 co_cg_acc = float2(0, 0);
for (uint i = 0; i < samples; ++i)
{
    // ... trace ray, get payload.color and sample_dir ...
    float4 sh_y_new; float2 co_cg_new;
    radiance_to_y_co_cg_sh(payload.color, sample_dir, sh_y_new, co_cg_new);
    sh_y_acc  += sh_y_new  * rcp(samples);
    co_cg_acc += co_cg_new * rcp(samples);
}
// Write sh_y_acc (float4) and co_cg_acc (float2) to separate textures
```

**Resolving SH against the high-detail normal**

After the denoising pipeline, the accumulated SH probe is evaluated against the pixel's full-detail normal (from the normal map, not the face normal used during tracing). The dot of the normal with the L1 directional coefficients recovers the directional radiance, and the chroma is rescaled relative to the DC term to stay color-accurate.

```hlsl
float3 sh_resolve_diffuse(float4 sh_y, float2 co_cg, float3 normal)
{
    // Evaluate L1 SH against the normal: dot(normal, sh_y.xyz) gives directional radiance
    // sh_y.w is the DC term (omnidirectional average)
    float y = max(dot(normal, sh_y.xyz) + 0.5f * sh_y.w, sh_y.w * 0.1f);
    // Rescale chroma relative to the DC term to keep color correct after directional modulation
    co_cg *= (y + 1e-6f) / (sh_y.w + 1e-6f);
    return y_co_cg_to_linear(float3(y, co_cg));
}
```

The `0.5 * sh_y.w` offset accounts for the cosine-weighted hemisphere: a perfectly diffuse surface receives half its energy from the hemisphere centered on the normal. The floor at `sh_y.w * 0.1` prevents the result from going negative when the normal points away from the dominant light direction.

Before SH Denoising:

![screenshot](showcase_images/bistro%20series%204%200.png)

After SH Denoising:

![screenshot](showcase_images/bistro%20series%204%201.png)

> DEV NOTE: pos view for screenshot series 2 bistro: 0.4744511 20.577366 1.1684266 0.8606896 -0.50898343 0.012216632

## The Full Pipeline

After adding all the tricks, the complete pipeline looks like this:

1. **Downsample** depth, normals, and albedo from full to half resolution, using RMS depth selection so no single surface type always dominates.

2. **Trace** one ray per half-res pixel, sampling a cosine-weighted hemisphere. Store pure incident radiance — no albedo multiplication. Project into YCoCg L1 SH.

3. **Firefly pre-filter** — star blur: blend the center pixel with a tight 5-tap cross. Then compute the geometric mean of an outer ring of 20 samples and clamp the blended center down to a ceiling multiple of that mean. Inverse-probability weighting rescales the surviving energy so suppressed fireflies contribute back when actually hit by the pre-blur.

4. **Pre-blur** — stochastic anisotropic disc filter, 8–16 samples, running on the firefly-filtered signal. Radius is driven by `square(filter_guide)` — worst-case on a flat surface the radius is very large, but even small surface detail collapses it fast. The kernel is tilted along the screen-space surface gradient to avoid bleeding across depth edges. This stabilises disocclusions within a few frames and removes low-frequency noise that a separable post-blur cannot safely remove after accumulation.

5. **Temporal accumulation** — reproject the previous frame using the inverse-then-forward view-projection chain. Bilinear validity weights and geometric+normal tests gate which history is usable. EMA blend at `1/(1+count)` accumulates SH and chroma separately. Disocclusion resets count to zero. Fast EMA radiance statistics (mean + relative variance) drive confidence: high divergence reduces blend toward history; high variance increases it. A temporal firefly filter additionally clamps incoming radiance against the fast mean before blending into the slow history, suppressing boiling on high-frequency emissive detail.

6. **Post-blur** — separable horizontal then vertical bilateral filter. Gaussian kernel width is modulated by the temporally accumulated filter guide (ray shortness × surface detail CoV), so smooth surfaces with distant hits get wide blurs and detailed or near-geometry surfaces stay sharp. Variance guiding adds a one-sided per-sample luminance weight, penalising samples brighter than the center pixel scaled by local relative variance — a cheap final sharpener on dark areas.

7. **Upscale** — full-resolution pass reads a groupshared tile of half-res buffers. Subpixel-aware 3×3 tent filter weights shift based on which quadrant of the parent half-res texel the full-res pixel falls in. Geometry and normal weights gate which half-res neighbours contribute; fallback to distance-weighted blend when too few pass. SH is resolved here against the full-resolution detail normal, recovering normal-map sharpness for free regardless of how much spatial blurring happened upstream.

8. **Albedo multiply** — the denoised radiance is multiplied by full-resolution albedo as a final step, preserving texture and colour detail that would otherwise be blurred with the lighting.