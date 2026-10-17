#pragma once

#include "rt_lab_resolve.inl"

#include "rt_lab_shared.hlsl"
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtLabResolvePush rt_lab_resolve_push;

func rt_lab_resolve_luminance(float3 v) -> float
{
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

// Live post-spatial split history error views: the split difference went through the same blur + upsample weights as
// the diffuse, so |d| is the error estimate of THIS output.
// Effective sample count view: heatmap of count / sample_count_debug_max.
func rt_lab_resolve_sample_count_debug(uint2 dtid, float count, bool sky)
{
    let push = rt_lab_resolve_push;
    if (push.attach.globals.settings.debug_draw_mode != DEBUG_DRAW_MODE_RT_LAB_SPATIAL_SAMPLE_COUNT)
    {
        return;
    }
    const float maximum = max(push.attach.globals.rt_lab_settings.sample_count_debug_max, 1e-3f);
    const float3 color = sky ? float3(0.0f, 0.0f, 0.0f) : Heatmap(saturate(count / maximum));
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
}

// Coarse fallback's own effective sample count (its a-trous chain), same scale as the spatial sample count view.
func rt_lab_resolve_coarse_count_debug(uint2 dtid, float count, bool sky)
{
    let push = rt_lab_resolve_push;
    if (push.attach.globals.settings.debug_draw_mode != DEBUG_DRAW_MODE_RT_LAB_COARSE_SAMPLE_COUNT)
    {
        return;
    }
    const float maximum = max(push.attach.globals.rt_lab_settings.sample_count_debug_max, 1e-3f);
    const float3 color = sky ? float3(0.0f, 0.0f, 0.0f) : (push.coarse_enabled == 0u ? float3(1.0f, 0.0f, 1.0f) * 0.25f : Heatmap(saturate(count / maximum)));
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
}

// Coarse fallback blend view: heatmap of t (0 = fine, 1 = coarse).
func rt_lab_resolve_coarse_debug(uint2 dtid, float blend, bool sky)
{
    let push = rt_lab_resolve_push;
    if (push.attach.globals.settings.debug_draw_mode != DEBUG_DRAW_MODE_RT_LAB_COARSE_BLEND)
    {
        return;
    }
    const float3 color = sky ? float3(0.0f, 0.0f, 0.0f) : (push.coarse_enabled == 0u ? float3(1.0f, 0.0f, 1.0f) * 0.25f : Heatmap(blend));
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
}

func rt_lab_resolve_split_debug(uint2 dtid, float3 resolved_radiance, float2 split_difference, bool sky)
{
    let push = rt_lab_resolve_push;
    const int debug_mode = push.attach.globals.settings.debug_draw_mode;
    if (debug_mode != DEBUG_DRAW_MODE_RT_LAB_SPLIT_ERROR && debug_mode != DEBUG_DRAW_MODE_RT_LAB_SPLIT_RELATIVE_ERROR)
    {
        return;
    }
    float3 color = float3(0.0f, 0.0f, 0.0f);
    if (sky)
    {
        color = float3(0.0f, 0.0f, 0.0f);
    }
    else if (push.split_enabled == 0u)
    {
        color = float3(1.0f, 0.0f, 1.0f) * 0.25f; // split history off
    }
    else if (split_difference.y < 0.999f)
    {
        color = float3(1.0f, 0.0f, 1.0f); // part of the footprint has a half without samples: error unknown
    }
    else
    {
        const float exposure_scale = PI * push.attach.globals.exposure;
        const float error = abs(split_difference.x) / RTGI_RADIANCE_SCALE * exposure_scale;
        if (debug_mode == DEBUG_DRAW_MODE_RT_LAB_SPLIT_ERROR)
        {
            color = Heatmap(error * push.attach.globals.settings.debug_visualization_scale);
        }
        else
        {
            // |A - B| / min(A, B). Post-spatial only d = (A - B) / 2 is carried; the output is ~(A + B) / 2, so
            // A, B ~ output +- |d|.
            const float luminance = rt_lab_resolve_luminance(resolved_radiance * exposure_scale);
            color = rt_lab_relative_difference_heatmap(rt_lab_relative_difference(luminance + error, luminance - error));
        }
    }
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend));
}

// Resolve of one full res pixel (no writes): the diffuse the opaque shading reads (mean incoming radiance,
// RTGI_RADIANCE_SCALE divided out), the split difference, the effective sample counts and the coarse blend.
// Full res lab: a copy. Half res lab: the RTGI upscale weights (3x3 wide gaussian footprint x plane x normal, see below).
// The split difference (split history on) takes the same path with the same weights (still RTGI_RADIANCE_SCALE scaled).
#define RT_LAB_UPSCALE_SIGMA 1.0f           // = RTGI_UPSCALE_WIDE_FOOTPRINT_SIGMA
#define RT_LAB_UPSCALE_PLANE_TOLERANCE 3.0f // half res pixel widths, = rtgi_upscale.hlsl

struct RtLabResolved
{
    float3 radiance;
    float2 split_difference;
    float count;         // effective sample count of the output
    float blend;         // coarse fallback blend
    float coarse_count;  // coarse chain's own count (coarse_ran)
    bool coarse_ran;
    bool sky;
};

func rt_lab_resolve_pixel(uint2 dtid) -> RtLabResolved
{
    let push = rt_lab_resolve_push;
    RtLabResolved r = {};
    r.radiance = float3(0.0f, 0.0f, 0.0f);
    r.split_difference = float2(0.0f, 0.0f);
    r.count = 0.0f;
    r.blend = 0.0f;
    r.coarse_count = 0.0f;
    r.coarse_ran = false;
    r.sky = false;
    const CameraInfo camera = push.attach.globals.view_camera;
    const bool split = push.split_enabled != 0u;

    const float ndc_depth = push.attach.view_cam_depth.get()[dtid];
    if (ndc_depth == 0.0f)
    {
        r.sky = true;
        return r;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const bool coarse = push.coarse_enabled != 0u;
    const bool clamp_count = push.clamp_input_count != 0u;
    const float count_clamp = push.attach.globals.rt_lab_settings.sample_count_input_clamp;

    uint2 lab_size;
    push.attach.lab_diffuse.get().GetDimensions(lab_size.x, lab_size.y);
    if (all(lab_size == push.size))
    {
        const float4 lab_value = push.attach.lab_diffuse.get()[dtid];
        const float fine_count = rt_lab_spatial_read_count(lab_value.a, clamp_count, count_clamp);
        float3 value = lab_value.rgb;
        r.count = fine_count;
        r.split_difference = split ? push.attach.lab_split_difference.get()[dtid] : float2(0.0f, 0.0f);
        if (coarse)
        {
            const float4 coarse_cocg_count = push.attach.lab_coarse_cocg_count.get()[dtid];
            r.coarse_count = coarse_cocg_count.z * RT_LAB_SPATIAL_COUNT_SCALE;
            r.coarse_ran = true;
            const float3 lab_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
            const float3 coarse_value = rt_lab_sh_resolve(push.attach.lab_coarse_sh_y.get()[dtid], coarse_cocg_count.xy, lab_normal);
            r.blend = rt_lab_coarse_top_up(fine_count, r.coarse_count, settings.coarse_target_samples, settings.coarse_max_blend, r.count);
            value = lerp(value, coarse_value, r.blend);
            if (split)
            {
                r.split_difference = lerp(r.split_difference, push.attach.lab_coarse_split_difference.get()[dtid], r.blend);
            }
        }
        r.radiance = value / RTGI_RADIANCE_SCALE;
        return r;
    }

    // Half res lab: the RTGI upscale weights (rtgi_upscale.hlsl, wide footprint): 3x3 half res taps around the pixel,
    // per axis gaussian (sigma RT_LAB_UPSCALE_SIGMA half res px) of the distance sub-pixel center -> tap center,
    // x binary plane test (RT_LAB_UPSCALE_PLANE_TOLERANCE half res pixel widths) x face normal dot^4. No matching tap:
    // RTGI fallback weights (gaussian x 1 / (1 + dist^4) x (0.1 + dot)).
    const float2 size = float2(push.size);
    const float depth = camera.near_plane / ndc_depth;
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(size)), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.view_cam_face_normals.get()[dtid]);
    // = rtgi_upscale.hlsl: calc_pixel_width_ws(inv half res size, near, ndc depth) = 2 x linear depth / half res height.
    const float rcp_half_pixel_width_ws = rcp(calc_pixel_width_ws(rcp(float2(lab_size)), camera.near_plane, ndc_depth));
    const float2 inv_lab_size = rcp(float2(lab_size));

    // Distance (half res px) from a left / top sub-pixel center to the taps at offset -1, 0, +1; mirrored for right / bottom.
    const float3 tap_distances_left = float3(0.75f, 0.25f, 1.25f);
    const float3 gauss_left_unnormalized = exp(-(tap_distances_left * tap_distances_left) / (2.0f * RT_LAB_UPSCALE_SIGMA * RT_LAB_UPSCALE_SIGMA));
    const float3 gauss_left = gauss_left_unnormalized / dot(gauss_left_unnormalized, 1.0f);
    const float3 gauss_x = (dtid.x & 1u) == 0u ? gauss_left : gauss_left.zyx;
    const float3 gauss_y = (dtid.y & 1u) == 0u ? gauss_left : gauss_left.zyx;

    float3 acc = float3(0.0f, 0.0f, 0.0f);
    float2 acc_split = float2(0.0f, 0.0f);
    float acc_count = 0.0f;
    float3 acc_coarse = float3(0.0f, 0.0f, 0.0f);
    float2 acc_coarse_split = float2(0.0f, 0.0f);
    float acc_coarse_count = 0.0f;
    float acc_weight = 0.0f;
    float3 fallback_acc = float3(0.0f, 0.0f, 0.0f);
    float2 fallback_acc_split = float2(0.0f, 0.0f);
    float fallback_acc_count = 0.0f;
    float3 fallback_acc_coarse = float3(0.0f, 0.0f, 0.0f);
    float2 fallback_acc_coarse_split = float2(0.0f, 0.0f);
    float fallback_acc_coarse_count = 0.0f;
    float fallback_acc_weight = 0.0f;
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            const int2 tap = clamp(int2(dtid / 2u) + int2(x - 1, y - 1), int2(0, 0), int2(lab_size) - 1);
            const float tap_depth = push.attach.lab_depth.get()[tap];
            if (tap_depth == 0.0f)
            {
                continue;
            }
            const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[tap]);
            const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_lab_size), tap_depth);
            const float3 offset = tap_position - position;
            const float gauss_weight = gauss_x[x] * gauss_y[y];
            const float plane_distance = abs(dot(offset, normal)) * rcp_half_pixel_width_ws;
            const float geometry_weight = plane_distance < RT_LAB_UPSCALE_PLANE_TOLERANCE ? 1.0f : 0.0f;
            const float normal_dot = max(0.0f, dot(tap_normal, normal));
            const float normal_weight = (normal_dot * normal_dot) * (normal_dot * normal_dot);
            const float weight = gauss_weight * geometry_weight * normal_weight;
            const float distance_squared = dot(offset, offset);
            const float fallback_weight = gauss_weight * rcp(1.0f + distance_squared * distance_squared) * (0.1f + normal_dot);

            const float4 lab_value = push.attach.lab_diffuse.get()[tap];
            const float count = rt_lab_spatial_read_count(lab_value.a, clamp_count, count_clamp);
            acc += weight * lab_value.rgb;
            acc_count += weight * count;
            fallback_acc += fallback_weight * lab_value.rgb;
            fallback_acc_count += fallback_weight * count;
            if (split)
            {
                const float2 split_value = push.attach.lab_split_difference.get()[tap];
                acc_split += weight * split_value;
                fallback_acc_split += fallback_weight * split_value;
            }
            if (coarse)
            {
                // Each tap's SH resolved for this full res pixel's normal, then interpolated.
                const float4 coarse_cocg_count = push.attach.lab_coarse_cocg_count.get()[tap];
                const float3 coarse_value = rt_lab_sh_resolve(push.attach.lab_coarse_sh_y.get()[tap], coarse_cocg_count.xy, normal);
                const float coarse_count = coarse_cocg_count.z * RT_LAB_SPATIAL_COUNT_SCALE;
                acc_coarse += weight * coarse_value;
                acc_coarse_count += weight * coarse_count;
                fallback_acc_coarse += fallback_weight * coarse_value;
                fallback_acc_coarse_count += fallback_weight * coarse_count;
                if (split)
                {
                    const float2 coarse_split = push.attach.lab_coarse_split_difference.get()[tap];
                    acc_coarse_split += weight * coarse_split;
                    fallback_acc_coarse_split += fallback_weight * coarse_split;
                }
            }
            acc_weight += weight;
            fallback_acc_weight += fallback_weight;
        }
    }
    // Only fall back when NO tap matched (thin geometry that does not exist at half res).
    if (!(acc_weight > 1e-5f))
    {
        acc = fallback_acc;
        acc_split = fallback_acc_split;
        acc_count = fallback_acc_count;
        acc_coarse = fallback_acc_coarse;
        acc_coarse_split = fallback_acc_coarse_split;
        acc_coarse_count = fallback_acc_coarse_count;
        acc_weight = fallback_acc_weight;
    }
    if (!(acc_weight > 0.0f))
    {
        return r; // all taps sky
    }
    // Normalized weights: upsampling interpolates the counts, no new samples.
    const float rcp_weight = rcp(acc_weight);
    float3 result = acc * rcp_weight;
    const float fine_count = acc_count * rcp_weight;
    r.count = fine_count;
    if (split)
    {
        r.split_difference = acc_split * rcp_weight;
    }
    if (coarse)
    {
        r.coarse_count = acc_coarse_count * rcp_weight;
        r.coarse_ran = true;
        r.blend = rt_lab_coarse_top_up(fine_count, r.coarse_count, settings.coarse_target_samples, settings.coarse_max_blend, r.count);
        result = lerp(result, acc_coarse * rcp_weight, r.blend);
        if (split)
        {
            r.split_difference = lerp(r.split_difference, acc_coarse_split * rcp_weight, r.blend);
        }
    }
    r.radiance = result / RTGI_RADIANCE_SCALE;
    return r;
}

// == Post stabilization (push.stabilize, RtLabState::post_stabilization_*) ===========================================
// TAA style temporal step on the resolved FULL RES output, the last thing the resolve does: every group resolves its
// 8x8 pixels + a halo of `radius` pixels into groupshared, then each pixel clamps its reprojected stabilization history
// into the (2 radius + 1)^2 neighborhood box mean +- gamma x sigma (YCoCg, TAA variance clipping; skipped with
// stabilize_disable_clamp) and blends by alpha.
// Short variance history (RT_LAB_STABILIZE_VARIANCE_*): per pixel exponential mean of the NEIGHBORHOOD variance over at
// most RT_LAB_STABILIZE_VARIANCE_MAX_FRAMES frames, reprojected with the same weights as the history. With
// RT_LAB_STABILIZE_VARIANCE_CLAMP the box half width becomes gamma x sqrt(max(current var, history var)): a frame whose
// neighborhood spreads less than recently gets the recent (more lenient) box instead of a hard clamp. Never wider than
// the neighborhoods actually were. (Not the temporal variance of the input: that IS the flicker, a box covering it
// never clamps.)
#define RT_LAB_STABILIZE_TILE (RT_LAB_RESOLVE_X + 2 * RT_LAB_STABILIZE_MAX_RADIUS)
#define RT_LAB_STABILIZE_PLANE_TOLERANCE 2.0f // history reprojection: plane distance in pixel widths at which a tap's weight reaches 0
groupshared float4 gs_stabilize_tile[RT_LAB_STABILIZE_TILE * RT_LAB_STABILIZE_TILE]; // .rgb radiance, .a 1 = surface, 0 = sky / outside

func rt_lab_rgb_to_ycocg(float3 c) -> float3
{
    return float3(
        0.25f * c.r + 0.5f * c.g + 0.25f * c.b,
        0.5f * c.r - 0.5f * c.b,
        -0.25f * c.r + 0.5f * c.g - 0.25f * c.b);
}

func rt_lab_ycocg_to_rgb(float3 c) -> float3
{
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Short history of the NEIGHBORHOOD (clamp box) variance: per channel YCoCg variance + frames.
struct RtLabStabilizeVariance
{
    float3 variance; // YCoCg
    float frames;
};

func rt_lab_stabilize_variance_pack(RtLabStabilizeVariance v) -> uint4
{
    const float3 std_dev = sqrt(max(v.variance, 0.0f)); // std dev, not variance: f16 range of radiance, not of its square
    return uint4(
        f32tof16(std_dev.x) | (f32tof16(std_dev.y) << 16u),
        f32tof16(std_dev.z),
        0u,
        asuint(v.frames));
}

func rt_lab_stabilize_variance_unpack(uint4 packed) -> RtLabStabilizeVariance
{
    RtLabStabilizeVariance v;
    const float3 std_dev = float3(f16tof32(packed.x), f16tof32(packed.x >> 16u), f16tof32(packed.y));
    v.variance = std_dev * std_dev;
    v.frames = asfloat(packed.w);
    return v;
}

// Variance estimator radius of this dispatch (push value 0 = 1, bounded).
func rt_lab_stabilize_radius() -> uint
{
    return clamp(rt_lab_resolve_push.stabilize_radius, 1u, uint(RT_LAB_STABILIZE_MAX_RADIUS));
}

// The stabilized radiance of pixel dtid (current = its resolved radiance, the neighborhood from the tile).
func rt_lab_stabilize(uint2 dtid, uint2 gtid, float3 current) -> float3
{
    let push = rt_lab_resolve_push;
    const CameraInfo camera = push.attach.globals.view_camera;
    CameraInfo* prev_camera = &push.attach.globals.view_camera_prev_frame;
    const float ndc_depth = push.attach.view_cam_depth.get()[dtid];
    const float linear_depth = camera.near_plane / ndc_depth;
    const float2 size = float2(push.size);
    const float2 inv_size = rcp(size);
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, inv_size), linear_depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.view_cam_face_normals.get()[dtid]);
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, size.y, linear_depth));

    // (2 radius + 1)^2 neighborhood statistics (YCoCg), sky / outside taps skipped.
    const int radius = int(rt_lab_stabilize_radius());
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    float3 sum_squared = float3(0.0f, 0.0f, 0.0f);
    float count = 0.0f;
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            const int2 cell = int2(gtid) + RT_LAB_STABILIZE_MAX_RADIUS + int2(x, y);
            const float4 tap = gs_stabilize_tile[cell.y * RT_LAB_STABILIZE_TILE + cell.x];
            if (tap.a > 0.0f)
            {
                const float3 value = rt_lab_rgb_to_ycocg(tap.rgb);
                sum += value;
                sum_squared += value * value;
                count += 1.0f;
            }
        }
    }
    const float3 spatial_mean = sum / max(count, 1.0f);
    const float3 spatial_sigma = sqrt(max(sum_squared / max(count, 1.0f) - spatial_mean * spatial_mean, 0.0f));

    // History: bilinear at the previous camera's projection, plane test against each tap's stored linear depth (.a).
    const bool variance_bound = (push.stabilize & RT_LAB_STABILIZE_VARIANCE_BOUND) != 0u;
    float3 history = float3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    // Short neighborhood variance history, same taps + weights.
    float3 variance_history = float3(0.0f, 0.0f, 0.0f);
    float variance_frames = 0.0f;
    if (push.attach.globals.rt_lab_settings.reset_history == 0)
    {
        const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
        if (prev_clip.w > 0.0f)
        {
            const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
            const Bilinear bilinear = get_bilinear_filter(prev_uv, size);
            const float4 bilinear_weights = get_bilinear_custom_weights(bilinear, float4(1.0f, 1.0f, 1.0f, 1.0f));
            [unroll] for (uint i = 0u; i < 4u; ++i)
            {
                const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
                if (any(tap < 0) || any(tap >= int2(push.size)))
                {
                    continue;
                }
                const float4 tap_history = push.attach.stabilize_history.get()[tap];
                if (!(tap_history.a > 0.0f) || !all(isfinite(tap_history)))
                {
                    continue;
                }
                const float3 tap_position = linear_depth_to_world_space(*prev_camera, rt_lab_ndc(uint2(tap), inv_size), tap_history.a);
                const float weight = bilinear_weights[i] *
                    rt_lab_plane_weight(position, normal, tap_position, normal, rcp_pixel_width_ws, RT_LAB_STABILIZE_PLANE_TOLERANCE);
                history += tap_history.rgb * weight;
                weight_sum += weight;
                // weight > 0: the history is not initialized on the first frame, 0 x inf garbage = NaN (whole frame black via exposure).
                if (variance_bound && weight > 0.0f)
                {
                    const RtLabStabilizeVariance tap_variance = rt_lab_stabilize_variance_unpack(push.attach.stabilize_variance.get()[tap]);
                    if (tap_variance.frames > 0.0f && tap_variance.frames <= RT_LAB_STABILIZE_VARIANCE_MAX_FRAMES && all(isfinite(tap_variance.variance)))
                    {
                        variance_history += tap_variance.variance * weight;
                        variance_frames += tap_variance.frames * weight;
                    }
                }
            }
        }
    }
    const float3 current_ycocg = rt_lab_rgb_to_ycocg(current);
    float3 box_min = spatial_mean - push.stabilize_gamma * spatial_sigma;
    float3 box_max = spatial_mean + push.stabilize_gamma * spatial_sigma;
    if (variance_bound)
    {
        // Exponential mean of the neighborhood variance, window capped at RT_LAB_STABILIZE_VARIANCE_MAX_FRAMES.
        const float3 spatial_variance = spatial_sigma * spatial_sigma;
        RtLabStabilizeVariance v;
        v.variance = spatial_variance;
        v.frames = 1.0f;
        if (weight_sum > 1e-3f && variance_frames > 0.0f)
        {
            const float rcp_weight_sum = rcp(weight_sum);
            v.frames = min(variance_frames * rcp_weight_sum + 1.0f, RT_LAB_STABILIZE_VARIANCE_MAX_FRAMES);
            v.variance = lerp(variance_history * rcp_weight_sum, spatial_variance, rcp(v.frames));
        }
        if (!all(isfinite(v.variance)))
        {
            v.variance = spatial_variance;
            v.frames = 1.0f;
        }
        push.attach.stabilize_variance_out.get()[dtid] = rt_lab_stabilize_variance_pack(v);
        if ((push.stabilize & RT_LAB_STABILIZE_VARIANCE_CLAMP) != 0u)
        {
            // Loosen only: a frame whose neighborhood happens to spread less than recently does not clamp harder.
            const float3 sigma = sqrt(max(spatial_variance, v.variance));
            box_min = spatial_mean - push.stabilize_gamma * sigma;
            box_max = spatial_mean + push.stabilize_gamma * sigma;
        }
    }
    float3 result = current;
    if (weight_sum > 1e-3f)
    {
        const float3 history_ycocg = rt_lab_rgb_to_ycocg(history / weight_sum);
        const float3 clamped = push.stabilize_disable_clamp != 0u ? history_ycocg : clamp(history_ycocg, box_min, box_max);
        result = max(rt_lab_ycocg_to_rgb(lerp(clamped, current_ycocg, saturate(push.stabilize_alpha))), 0.0f);
    }
    push.attach.stabilize_history_out.get()[dtid] = float4(result, linear_depth);
    return result;
}

[shader("compute")]
[numthreads(RT_LAB_RESOLVE_X, RT_LAB_RESOLVE_Y, 1)]
func entry_rt_lab_resolve(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint group_index : SV_GroupIndex)
{
    let push = rt_lab_resolve_push;
    const bool stabilize = push.stabilize != 0u;
    // No early outs before the barrier: out of bounds threads still help with the halo.
    const bool in_bounds = all(dtid < push.size);
    RtLabResolved r = {};
    r.radiance = float3(0.0f, 0.0f, 0.0f);
    r.split_difference = float2(0.0f, 0.0f);
    r.count = 0.0f;
    r.blend = 0.0f;
    r.coarse_count = 0.0f;
    r.coarse_ran = false;
    r.sky = true;
    if (in_bounds)
    {
        r = rt_lab_resolve_pixel(dtid);
    }
    float3 output = r.radiance;
    if (stabilize)
    {
        const uint max_radius = uint(RT_LAB_STABILIZE_MAX_RADIUS);
        gs_stabilize_tile[(gtid.y + max_radius) * RT_LAB_STABILIZE_TILE + gtid.x + max_radius] = float4(r.radiance, (in_bounds && !r.sky) ? 1.0f : 0.0f);
        // Halo: the ring of `radius` cells around the 8x8 block, spread over the group's threads.
        const uint radius = rt_lab_stabilize_radius();
        const uint region = RT_LAB_RESOLVE_X + 2u * radius;
        for (uint index = group_index; index < region * region; index += RT_LAB_RESOLVE_X * RT_LAB_RESOLVE_Y)
        {
            const uint2 cell = uint2(max_radius - radius) + uint2(index % region, index / region);
            if (all(cell >= max_radius) && all(cell < max_radius + RT_LAB_RESOLVE_X))
            {
                continue; // inside the block: written by its own thread
            }
            const int2 pixel = int2(dtid - gtid) - int(max_radius) + int2(cell);
            float4 value = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (all(pixel >= 0) && all(pixel < int2(push.size)))
            {
                const RtLabResolved halo = rt_lab_resolve_pixel(uint2(pixel));
                value = float4(halo.radiance, halo.sky ? 0.0f : 1.0f);
            }
            gs_stabilize_tile[cell.y * RT_LAB_STABILIZE_TILE + cell.x] = value;
        }
        GroupMemoryBarrierWithGroupSync();
        if (in_bounds && !r.sky)
        {
            output = rt_lab_stabilize(dtid, gtid, r.radiance);
        }
        else if (in_bounds)
        {
            push.attach.stabilize_history_out.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            if ((push.stabilize & RT_LAB_STABILIZE_VARIANCE_BOUND) != 0u)
            {
                push.attach.stabilize_variance_out.get()[dtid] = uint4(0u, 0u, 0u, 0u);
            }
        }
    }
    if (!in_bounds)
    {
        return;
    }

    push.attach.resolved.get()[dtid] = float4(output, 1.0f);
    if (push.split_enabled != 0u)
    {
        push.attach.split_difference_resolved.get()[dtid] = r.split_difference;
    }
    rt_lab_resolve_split_debug(dtid, r.radiance, r.split_difference, r.sky);
    rt_lab_resolve_sample_count_debug(dtid, r.count, r.sky);
    rt_lab_resolve_coarse_debug(dtid, r.blend, r.sky);
    if (r.sky || r.coarse_ran)
    {
        rt_lab_resolve_coarse_count_debug(dtid, r.coarse_count, r.sky);
    }
}
