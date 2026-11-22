#pragma once

#include "rtgi_post_blur.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiAtrousPostBlurPush rtgi_atrous_post_blur_push;

// DESIGN NOTE (idea, not implemented): run the WIDEST a-trous passes BEFORE the temporal accumulate, in place of the adaptive pre-blur.
//
// How it came up:
//   - The pre-blur (rtgi_pre_blur.hlsl) is already something like a wide a-trous: a big spatial low pass on the raw, noisy ray result, before temporal.
//   - Its weak spot: it picks 8 random Poisson taps per pixel. At a wide radius every pixel gets a different sparse tap set, so it is noisy and it cannot
//     resolve wide areas consistently. This shows on disocclusions, where the history is empty and the pre-blur is all there is.
//   - An a-trous pass is deterministic: the same footprint everywhere, no per pixel random tap set. The wide taps do not have that problem, so a fully
//     disoccluded area comes out smooth immediately.
//
// Benefits:
//   - Disocclusions are smooth at once (see above).
//   - Variance guiding (future): for the WIDE passes a SPATIAL variance is enough, each pixel pools many neighbors, so no temporal variance is needed.
//     The per pixel ray count is known, the variance of the mean is sigma^2 / n. Only the finer a-trous passes and the upscale pass (after the temporal
//     accumulate) use the sharp TEMPORAL variance.
//   - The small tap jitter of the wide passes (rtgi_settings.post_blur_atrous_jitter) jiggles the result a little. Before the temporal accumulate that is
//     averaged out by the temporal accumulation: the jiggle is a free spatial dither of the wide kernels, the same trick the pre-blur uses with its
//     per frame disc rotation.
//
// Risks / things to port when this is done:
//   - A blur before the temporal accumulate is baked into the history: the history converges to the BLURRED signal, later passes cannot bring the detail
//     back (a blur after the temporal only touches the output). The pre-blur has this already. REPLACE the pre-blur, do not add on top of it.
//     Idea: gate the wide passes by the reprojected history count (wide on disocclusions, skipped where the history is converged): fixes the baked-in blur and saves time.
//   - Raw 1 ray input: no luminance guide (unreliable), geometry / normal / depth / AO guide + ray count tap weights only.
//   - Port from the pre-blur: firefly energy compensation (weight on iteration 0), AO radius guide, ray count tap weights.
//   - The temporal accumulate assumes a 1 sample input. After a wide pass the input variance is much lower, it may blend too conservatively (lag):
//     it needs the effective sample count of the input.
//   - Specular stays on the pre-blur path (a wide spatial blur on glossy history is wrong).
//   - The FINE a-trous passes stay AFTER the temporal accumulate on purpose: a spatial pass after the temporal hides disocclusion streaking
//     (slight brightness / tint differences between the fresh and the reprojected pixels). Only the widest passes move before it.
//   - TRIED and REMOVED: a setting ("Atrous Passes Before Temporal") that ran the widest N passes before the temporal accumulate. Result: the pre temporal result was not
//     smooth and disocclusions showed strong splodges (also with the guides off, with the pre blur off). Why, as far as understood:
//       * the post temporal wide passes weight taps by the sample count, so they FILL a disoccluded strip from the converged neighbors (inpainting from smooth history);
//         before the temporal there is no converged data, only raw noise.
//       * a wide low pass of raw white noise makes LOW FREQUENCY noise (blobs the size of the kernel): the narrow passes after it cannot remove it, the temporal only slowly.
//       * the post passes are recomputed every frame (a bad pixel costs one frame), the pre passes are accumulated into the history.
//       * the low weight fallback of the pass (lerp back to the raw pixel when the tap weight sum is below 2) assumes count weights >= 1: with the counts of fresh pixels
//         (0) at least half of the raw pixel survived.
//     Idea instead: initialize disoccluded pixels from the converged neighbors' history (like the hash grid revive).
//   - Plan (original): setting for the number of pre-temporal a-trous passes + pre-blur switchable off (A/B), reuse entry_atrous with the same head, keep step 4 and
//     finer after the temporal and the merged finest step in the upscale pass.

// Specular post blur state of one pixel. The specular channel rides along the diffuse tap loops: it reuses
// their geometry weight and adds detail-normal + roughness similarity, with its own (lobe footprint) width.
struct RtgiSpecularBlur
{
    float  roughness;
    float3 normal;
    float  width;   // in taps along the blur axis; < 1 == center only
    float4 acc;
    float  weight;
};

// ao_guide: the pixel's (temporal) AO guide. The shared post blur AO radius scaling + floor shrink the specular width
// exactly like the diffuse one (General setting, both signals).
func rtgi_specular_blur_begin(RtgiSettings settings, float4 center_spec, float roughness, float3 normal, float pixel_width_ws, float ao_guide) -> RtgiSpecularBlur
{
    RtgiSpecularBlur b;
    b.roughness = roughness;
    b.normal = normal;
    const float ao_radius_scale = settings.post_blur_ao_guiding ? lerp(settings.post_blur_ao_guide_floor, 1.0f, ao_guide) : 1.0f;
    b.width = settings.specular_enabled != 0
        ? rtgi_specular_blur_radius_px(roughness, center_spec.a, pixel_width_ws, (float)settings.post_blur_max_width) * settings.specular_post_blur_scale * ao_radius_scale
        : 0.0f;
    b.acc = float4(0, 0, 0, 0);
    b.weight = 0.0f;
    return b;
}

func rtgi_specular_blur_tap(inout RtgiSpecularBlur b, float tap_distance, float geometric_weight, float3 sample_normal, float sample_roughness, float4 sample_spec)
{
    if (tap_distance > max(b.width, 0.0f)) { return; }
    const float gauss = tap_distance == 0.0f ? 1.0f : calc_gaussian_weight(tap_distance / max(b.width, 1.0f));
    const float w = geometric_weight * gauss *
        rtgi_specular_normal_weight(b.normal, sample_normal, b.roughness) *
        rtgi_specular_roughness_weight(b.roughness, sample_roughness);
    b.acc += sample_spec * w;
    b.weight += w;
}

func rtgi_specular_blur_end(RtgiSpecularBlur b, float4 center_spec) -> float4
{
    return b.weight > 1e-6f ? b.acc / b.weight : center_spec;
}



// groupshared float gs_depth_preload[PRELAOD_WIDTH][RTGI_POST_BLUR_X];
// groupshared float4 gs_diffuse_preload[PRELAOD_WIDTH][RTGI_POST_BLUR_X];
// groupshared float2 gs_diffuse2_preload[PRELAOD_WIDTH][RTGI_POST_BLUR_X];

// Sparse far taps (see rtgi.inl): the 4 taps are weighted by an equal share of the 8 ring taps' total weight (0.75).
static const float2 RTGI_ATROUS_MSAA4_PATTERN[4] = { float2(-0.125f, -0.375f), float2(0.375f, -0.125f), float2(-0.375f, 0.125f), float2(0.125f, 0.375f) };

// Taps whose weight is below this before the color / guide fetches are skipped without loading them.
#define RTGI_ATROUS_MIN_TAP_WEIGHT 1e-4f
// The callback dispatches with a square block size (dispatch_image_relative uses one block size for x and y), so the group must be square.
[shader("compute")]
[numthreads(RTGI_POST_BLUR_X, RTGI_POST_BLUR_X, 1)]
func entry_atrous_post_blur(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_atrous_post_blur_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_size = rcp(half_res_size);
    const uint2 pixel_index = dtid;

    // Jitter amplitude in texels, 0 when jitter is off: the jitter is multiplied by it, the setting is read once.
    const float jitter_amplitude = rtgi_settings.post_blur_atrous_jitter != 0 ? RTGI_ATROUS_TAP_JITTER_AMPLITUDE * float(push.step_size) : 0.0f;
    const bool sparse_taps_enabled = rtgi_settings.post_blur_atrous_sparse_taps != 0;

    if (any(dtid.xy >= push.size))
        return;

    const PixelData pixel = calc_pixel_data(dtid, inv_half_res_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
    const float pixel_width_ws = rtgi_half_res_pixel_width_ws(inv_half_res_size, camera.near_plane, pixel.depth_vs);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);

    if (pixel.depth_vs == 0.0f)
        return;

    const float pixel_samplecnt = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[pixel_index]);

    const float temporal_stability_scale = rtgi_settings.post_blur_disocclusion_blur_enabled ? 1.0f - square(saturate(pixel_samplecnt * rcp(16.0f))) : 0.0f;
    const float ao_guide    = push.attach.ao_guide_image.get()[dtid];

    // Ambient occlusion guide is calibrated for the pre blur that has a radius of 64, we take the lowest 25% of the guide as that rouughly matches our range here of 8-16 pixels.
    const float ao_guide_radius_scale = min(0.25f, rtgi_settings.post_blur_ao_guiding ? lerp(rtgi_settings.post_blur_ao_guide_floor, 1.0f, ao_guide) : 1.0f) * 4.0f;
    // Temporal stability pushes toward full blur when sample count is low (disocclusion)
    const float effective_ao_guide_radius_scale = lerp(ao_guide_radius_scale, 1.0f, temporal_stability_scale);

    const float pixel_perceptual_radiance_atrous = push.attach.temporal_perceptual_radiance.get()[pixel_index];

    const float sample_count_ramp = rtgi_settings.max_temporal_samples * 0.1f;
    const float guide_ramp = saturate((pixel_samplecnt - sample_count_ramp) / max(sample_count_ramp, 1.0f));

    float weight_accum = 0.0f;
    float4 diffuse_accum = float4(0, 0, 0, 0);
    float2 diffuse2_accum = float2(0, 0);

    // Per pixel, per pass, per frame seed for the tap jitter.
    rand_seed(dtid.x * 9629u + dtid.y * 10069u + uint(push.step_size) * 1743216u + (rtgi_settings.animate_noise ? uint(push.attach.globals.trunk_flt_frame_index) * 257u : 0u));

    // Specular joins an a-trous level only while the step still fits inside its lobe footprint.
    const float4 pixel_spec = push.attach.rtgi_specular_before.get()[pixel_index];
    RtgiSpecularBlur spec_blur = rtgi_specular_blur_begin(rtgi_settings, pixel_spec,
        unpack_normal_roughness_roughness(push.attach.specular_normal_roughness.get()[pixel_index]),
        unpack_normal_roughness_normal(push.attach.specular_normal_roughness.get()[pixel_index]),
        pixel_width_ws, ao_guide);

    const bool sparse_taps = sparse_taps_enabled && push.step_size >= RTGI_ATROUS_SPARSE_TAPS_MIN_STEP;
    const int tap_count = sparse_taps ? 5 : 9;

    for (int tap = 0; tap < tap_count; ++tap)
    {
        {
            // Tap 0 of the sparse pattern is the center, the dense pattern is the 3x3 grid (tap 4 is the center).
            int2 tap_offset;
            float tap_gauss;
            bool is_center;
            if (sparse_taps)
            {
                is_center = tap == 0;
                tap_offset = is_center ? int2(0, 0) : int2(round(RTGI_ATROUS_MSAA4_PATTERN[max(tap - 1, 0)] * (RTGI_ATROUS_SPARSE_PATTERN_SCALE * float(push.step_size))));
                tap_gauss = is_center ? 0.25f : 0.1875f;
            }
            else
            {
                const int dx = tap % 3 - 1;
                const int dy = tap / 3 - 1;
                is_center = dx == 0 && dy == 0;
                tap_offset = int2(dx, dy) * push.step_size;
                // 3x3 separable Gaussian: [0.25, 0.5, 0.25] per axis.
                tap_gauss = (dx == 0 ? 0.5f : 0.25f) * (dy == 0 ? 0.5f : 0.25f);
            }

            // Tap jitter of up to +-0.25 * step_size texels, rounded to whole texels:
            // step 1 and 2 round to 0 (no effect), from step 4 (third pass) on taps move by up to 1 texel and so on.
            // Center tap stays put.
            int2 tap_jitter = int2(0, 0);
            if (!is_center)
            {
                tap_jitter = int2(round((float2(rand(), rand()) * 2.0f - 1.0f) * jitter_amplitude));
            }
            const int2 raw_index = int2(pixel_index) + tap_offset + tap_jitter;
            // Out-of-bounds taps contribute nothing.
            const bool out_of_bounds = any(raw_index < int2(0, 0)) || any(raw_index >= int2(push.size));
            if (out_of_bounds)
            {
                continue;
            }
            const uint2 sample_index = uint2(raw_index);

            const PixelData sample = calc_pixel_data(sample_index, inv_half_res_size, camera, push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get());
            if (sample.depth_vs == 0.0f)
            {
                continue;
            }

            const float geometric_weight = calc_similar_surface_weight(pixel_width_ws_rcp, pixel.position_ws, pixel.normal_ws, sample.position_ws, sample.normal_ws);
            if (geometric_weight < RTGI_ATROUS_MIN_TAP_WEIGHT)
            {
                continue;
            }
            const float normal_weight = calc_similar_normal_weight(pixel.normal_ws, sample.normal_ws);

            const float spec_tap_distance = is_center ? 0.0f : (sparse_taps ? length(float2(tap_offset)) : float(push.step_size));
            if (spec_tap_distance <= spec_blur.width)
            {
                rtgi_specular_blur_tap(spec_blur, spec_tap_distance, geometric_weight,
                    unpack_normal_roughness_normal(push.attach.specular_normal_roughness.get()[sample_index]),
                    unpack_normal_roughness_roughness(push.attach.specular_normal_roughness.get()[sample_index]),
                    push.attach.rtgi_specular_before.get()[sample_index]);
            }

            // Geometric guide scales down non-center tap contributions.
            const float geometric_factor = is_center ? 1.0f : effective_ao_guide_radius_scale;
            const float gauss_weight = tap_gauss * geometric_factor;

            // The remaining factors (count weight >= 1, guide weight <= 1) cannot lift a tiny weight, skip the color fetches.
            if (geometric_weight * normal_weight * gauss_weight < RTGI_ATROUS_MIN_TAP_WEIGHT)
            {
                continue;
            }

            const float4 sample_sh_y = push.attach.rtgi_diffuse_before.get()[sample_index];
            const float2 sample_cocg = push.attach.rtgi_diffuse2_before.get()[sample_index].rg;
            const float sample_samplecnt = rtgi_unpack_normal_count(push.attach.rtgi_sample_count.get()[sample_index]);
            const float sample_count_weight = sample_samplecnt + 1.0f;

            float extra_weight = 1.0f;
            if (!is_center)
            {
                float raw_weight = 1.0f;
                if (rtgi_settings.post_blur_perceptual_difference_guiding)
                {
                    const float sample_perceptual_radiance = push.attach.temporal_perceptual_radiance.get()[sample_index];
                    raw_weight = calc_perceptual_difference_weight(pixel_perceptual_radiance_atrous, sample_perceptual_radiance, rtgi_settings.post_blur_perceptual_radiance_guide_tolerance);
                }
                extra_weight = lerp(1.0f, raw_weight, guide_ramp);
            }

            const float weight = geometric_weight * normal_weight * gauss_weight * sample_count_weight * extra_weight;

            weight_accum += weight;
            diffuse_accum += weight * sample_sh_y;
            diffuse2_accum += weight * sample_cocg;
        }
    }

    const float low_weight_fallback_blend = max(0.0f, 1.0f - (weight_accum / (8.0f / 4.0f)));
    const float4 blurry_sh_y = diffuse_accum * rcp(weight_accum + 0.0001f);
    const float2 blurry_cocg = diffuse2_accum * rcp(weight_accum + 0.0001f);
    const float4 pixel_sh_y = push.attach.rtgi_diffuse_before.get()[pixel_index];
    const float2 pixel_cocg = push.attach.rtgi_diffuse2_before.get()[pixel_index].rg;

    push.attach.rtgi_diffuse_blurred.get()[pixel_index] = lerp(blurry_sh_y, pixel_sh_y, low_weight_fallback_blend);
    push.attach.rtgi_diffuse2_blurred.get()[pixel_index] = lerp(blurry_cocg, pixel_cocg, low_weight_fallback_blend);
    push.attach.rtgi_specular_blurred.get()[pixel_index] = rtgi_specular_blur_end(spec_blur, pixel_spec);
}
