#pragma once

#include "rt_lab_blur.inl"

#include "rt_lab_shared.hlsl"

[[vk::push_constant]] RtLabBlurPush rt_lab_blur_push;

// 5x5 B3 spline kernel weights by |offset| (1/16, 1/4, 3/8, 1/4, 1/16 per axis).
static const float RT_LAB_ATROUS_KERNEL[3] = {3.0f / 8.0f, 1.0f / 4.0f, 1.0f / 16.0f};

// Weighted sums of a group of taps: color / split by the filter weight (normalized at the end), count by the weight
// relative to the center weight (the effective sample count, not normalized). Gradient carry (push.gradient_mode
// RT_LAB_GRADIENT_CARRY): the taps' gradients with the same weights, their variances with the squared weights.
struct RtLabSpatialSum
{
    float3 color;
    float2 split;
    float weight;
    float count;
    float3 gradient_x;
    float3 gradient_y;
    float2 gradient_variance; // sum w^2 var (x, y)
};

func rt_lab_spatial_sum_zero() -> RtLabSpatialSum
{
    const float3 zero = float3(0.0f, 0.0f, 0.0f);
    return RtLabSpatialSum(zero, float2(0.0f, 0.0f), 0.0f, 0.0f, zero, zero, float2(0.0f, 0.0f));
}

// Sample count budget (fine filter, RtLabSettings::fine_target_samples > 0): taps are accepted ring by ring from the
// center outwards (closest first). A ring that fits is accepted fully; the ring that would cross the target is scaled
// so the pass reaches the target exactly; the rest is dropped. Returns true when the budget is used up.
func rt_lab_accept_ring(inout RtLabSpatialSum total, RtLabSpatialSum ring, float target) -> bool
{
    float accept = 1.0f;
    if (target > 0.0f && ring.count > 0.0f)
    {
        accept = saturate((target - total.count) / ring.count);
    }
    total.color += ring.color * accept;
    total.split += ring.split * accept;
    total.weight += ring.weight * accept;
    total.count += ring.count * accept;
    total.gradient_x += ring.gradient_x * accept;
    total.gradient_y += ring.gradient_y * accept;
    total.gradient_variance += ring.gradient_variance * (accept * accept);
    return target > 0.0f && total.count >= target * 0.9999f;
}

// Variance guide of the fine filter (RtLabSettings::variance_guide_*): the center's mean and the uncertainty of it.
struct RtLabVarianceGuide
{
    bool enabled;
    float3 mean;
    float3 mean_variance; // per sample variance / effective sample count
    float rcp_two_k_sq;   // 1 / (2 k^2)
    float3 variance_floor;
    float tail_nu;        // Student-t nu, 0 = gaussian
    // Gradient aware guide (RtLabSettings::variance_guide_gradient): the center's local rgb gradient per screen axis
    // (per pixel of offset) after the significance shrinkage, and its remaining variance. The tap is compared with the
    // center's mean PLUS the gradient times its offset (a plane in color space instead of a constant).
    float3 gradient_x;
    float3 gradient_y;
    float3 gradient_variance_x;
    float3 gradient_variance_y;
};

func rt_lab_variance_guide_none() -> RtLabVarianceGuide
{
    const float3 zero = float3(0.0f, 0.0f, 0.0f);
    return RtLabVarianceGuide(false, zero, zero, 0.0f, zero, 0.0f, zero, zero, zero, zero);
}

// The center's per sample variance / effective sample count (variance guide + gradient fit).
func rt_lab_center_mean_variance(uint2 dtid, float count) -> float3
{
    let push = rt_lab_blur_push;
    return max(push.attach.lab_variance.get()[dtid].rgb, 0.0f) / rt_lab_variance_guide_count(&push.attach.globals.rt_lab_settings, count);
}

func rt_lab_variance_guide_begin(uint2 dtid, float3 mean, float count) -> RtLabVarianceGuide
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (push.variance_bound == 0u || settings.variance_guide_enabled == 0 || count <= 0.0f)
    {
        return rt_lab_variance_guide_none();
    }
    RtLabVarianceGuide guide = rt_lab_variance_guide_none();
    guide.enabled = true;
    guide.mean = mean;
    guide.mean_variance = rt_lab_center_mean_variance(dtid, count);
    const float k = max(settings.variance_guide_sigmas, 1e-3f);
    guide.rcp_two_k_sq = rcp(2.0f * k * k);
    guide.variance_floor = rt_lab_variance_guide_floor(push.attach.globals.exposure);
    guide.tail_nu = rt_lab_variance_guide_tail_nu(&push.attach.globals.rt_lab_settings);
    return guide;
}

// == Gradient (push.gradient_mode) ===================================================================================
// FIT (first fine pass): weighted least squares over the 5x5 step 1 neighborhood (geometric weights only), per screen
// axis: g = sum w delta o / sum w o^2, var(g) = 2 var_c sum w^2 o^2 / (sum w o^2)^2 (stored as a scalar, mean over rgb).
// CARRY (later passes): no refit; the gradient image is filtered with the color's own final weights (variance guide
// included: rejected taps, e.g. across an edge, do not feed in), its variance with the squared weights.
// The FIT pass's own guide uses no gradient (constant model); the CARRY passes use the incoming gradient after a per
// channel significance shrinkage g x max(0, 1 - var / g^2).

struct RtLabGradientSum
{
    float3 delta_offset_x;
    float3 delta_offset_y;
    float2 offset_sq;    // sum w o^2 (x, y)
    float2 offset_sq_w2; // sum w^2 o^2 (x, y)
};

// The gradient as stored: .rgb slope per pixel, .a its variance (scalar).
struct RtLabGradient
{
    float4 x;
    float4 y;
};

func rt_lab_gradient_fit(uint2 dtid, float4 center, float center_count, float3 position, float3 normal, float rcp_pixel_width_ws,
    float2 inv_size, bool clamp_count) -> RtLabGradient
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    RtLabGradientSum sum = RtLabGradientSum(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), float2(0.0f, 0.0f), float2(0.0f, 0.0f));
    [unroll] for (int y = -2; y <= 2; ++y)
    {
        [unroll] for (int x = -2; x <= 2; ++x)
        {
            if (x == 0 && y == 0)
            {
                continue;
            }
            const int2 tap = int2(dtid) + int2(x, y);
            if (any(tap < 0) || any(tap >= int2(push.size)))
            {
                continue;
            }
            const float tap_depth = push.attach.lab_depth.get()[tap];
            if (tap_depth == 0.0f)
            {
                continue;
            }
            const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[tap]);
            const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
            const float weight = RT_LAB_ATROUS_KERNEL[abs(x)] * RT_LAB_ATROUS_KERNEL[abs(y)] *
                                 rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, settings.blur_plane_tolerance) *
                                 pow(saturate(dot(normal, tap_normal)), settings.blur_normal_power);
            if (weight <= 0.0f)
            {
                continue;
            }
            const float4 value = push.attach.src.get()[tap];
            if (rt_lab_spatial_read_count(value.a, clamp_count, settings.sample_count_input_clamp) <= 0.0f)
            {
                continue;
            }
            const float2 offset = float2(x, y);
            const float3 delta = value.rgb - center.rgb;
            sum.delta_offset_x += delta * (weight * offset.x);
            sum.delta_offset_y += delta * (weight * offset.y);
            sum.offset_sq += weight * offset * offset;
            sum.offset_sq_w2 += weight * weight * offset * offset;
        }
    }
    const float3 variance_rgb = 2.0f * (rt_lab_center_mean_variance(dtid, center_count) + rt_lab_variance_guide_floor(push.attach.globals.exposure));
    const float variance = dot(variance_rgb, 1.0f / 3.0f);
    RtLabGradient gradient;
    gradient.x = sum.offset_sq.x > 1e-6f
        ? float4(sum.delta_offset_x / sum.offset_sq.x, variance * sum.offset_sq_w2.x / (sum.offset_sq.x * sum.offset_sq.x))
        : float4(0.0f, 0.0f, 0.0f, 0.0f);
    gradient.y = sum.offset_sq.y > 1e-6f
        ? float4(sum.delta_offset_y / sum.offset_sq.y, variance * sum.offset_sq_w2.y / (sum.offset_sq.y * sum.offset_sq.y))
        : float4(0.0f, 0.0f, 0.0f, 0.0f);
    return gradient;
}

// The center's gradient for this pass: fitted (FIT) or read (CARRY). Zero when off.
func rt_lab_gradient_center(uint2 dtid, float4 center, float center_count, float3 position, float3 normal, float rcp_pixel_width_ws,
    float2 inv_size, bool clamp_count) -> RtLabGradient
{
    let push = rt_lab_blur_push;
    if (push.gradient_mode == RT_LAB_GRADIENT_FIT)
    {
        return rt_lab_gradient_fit(dtid, center, center_count, position, normal, rcp_pixel_width_ws, inv_size, clamp_count);
    }
    if (push.gradient_mode == RT_LAB_GRADIENT_CARRY)
    {
        return RtLabGradient(push.attach.gradient_x_src.get()[dtid], push.attach.gradient_y_src.get()[dtid]);
    }
    return RtLabGradient(float4(0.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 0.0f));
}

// Significance shrinkage per channel, then into the guide.
func rt_lab_variance_guide_set_gradient(inout RtLabVarianceGuide guide, RtLabGradient gradient)
{
    const float3 keep_x = saturate(1.0f - gradient.x.a / max(gradient.x.rgb * gradient.x.rgb, 1e-20f));
    const float3 keep_y = saturate(1.0f - gradient.y.a / max(gradient.y.rgb * gradient.y.rgb, 1e-20f));
    guide.gradient_x = gradient.x.rgb * keep_x;
    guide.gradient_y = gradient.y.rgb * keep_y;
    guide.gradient_variance_x = gradient.x.a * keep_x * keep_x;
    guide.gradient_variance_y = gradient.y.a * keep_y * keep_y;
}

// Likelihood that center and tap estimate the same mean (rt_lab_variance_guide_likelihood). Gradient aware: the center
// side is the mean predicted at the tap's offset (pixels) from the local gradient, plus the gradient's variance.
func rt_lab_variance_guide_weight(RtLabVarianceGuide guide, int2 tap, float3 tap_mean, float tap_count, float2 offset) -> float
{
    let push = rt_lab_blur_push;
    const float3 tap_mean_variance = max(push.attach.lab_variance.get()[tap].rgb, 0.0f) / rt_lab_variance_guide_count(&push.attach.globals.rt_lab_settings, tap_count);
    const float3 predicted = guide.mean + guide.gradient_x * offset.x + guide.gradient_y * offset.y;
    const float3 predicted_variance = guide.mean_variance + guide.gradient_variance_x * (offset.x * offset.x) + guide.gradient_variance_y * (offset.y * offset.y);
    return rt_lab_variance_guide_likelihood(predicted, predicted_variance, tap_mean, tap_mean_variance, guide.variance_floor, guide.rcp_two_k_sq, guide.tail_nu);
}

// Ray length guide radius of a pixel (world units): accumulated AO guide (RTGI style, 1 - sqrt(mean shortness)) x the
// pixel's max distance R, floored at a tiny fraction of R (log ratios stay finite). 0 = guide off / no estimate.
func rt_lab_spatial_ray_length_radius(int2 pixel) -> float
{
    let push = rt_lab_blur_push;
    if (push.ray_length_bound == 0u || push.attach.globals.rt_lab_settings.ray_length_guide_mode == 0)
    {
        return 0.0f;
    }
    const float ao_guide = push.attach.lab_ray_length.get()[pixel].x;
    const float depth = push.attach.lab_depth.get()[pixel];
    if (!(ao_guide >= 0.0f) || depth == 0.0f)
    {
        return 0.0f;
    }
    return max(ao_guide, 1e-3f) * rt_lab_ray_length_max_distance(push.attach.globals, depth);
}

// Ray length guide weight of a tap (RtLabSettings::ray_length_guide_mode): DISTANCE = donors beyond the center's radius
// (floored at one pixel width) fade, SIMILARITY = donors with a different radius (log2 ratio) fade. 1 without estimates.
func rt_lab_ray_length_guide_weight(float center_radius, int2 tap, float tap_distance, float rcp_pixel_width_ws) -> float
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (!(center_radius > 0.0f))
    {
        return 1.0f;
    }
    float weight = 1.0f;
    if ((settings.ray_length_guide_mode & RT_LAB_RAY_LENGTH_GUIDE_DISTANCE) != 0)
    {
        const float radius = max(center_radius, rcp(rcp_pixel_width_ws));
        weight *= exp(-square(max(tap_distance / radius - 1.0f, 0.0f) / max(settings.ray_length_guide_distance_falloff, 1e-3f)));
    }
    if ((settings.ray_length_guide_mode & RT_LAB_RAY_LENGTH_GUIDE_SIMILARITY) != 0)
    {
        const float tap_radius = rt_lab_spatial_ray_length_radius(tap);
        if (tap_radius > 0.0f)
        {
            weight *= exp(-square(log2(center_radius / tap_radius) / max(settings.ray_length_guide_similarity_tolerance, 1e-3f)));
        }
    }
    return weight;
}

// Firefly energy weighting (push.firefly_weight, first fine pass only: after one pass the energy has moved): the pixel's
// accumulated energy_pre / energy_post (lab_variance .a = accumulated raw luminance, color = the filtered history).
func rt_lab_spatial_firefly_factor(int2 pixel, float3 color) -> float
{
    let push = rt_lab_blur_push;
    if (push.firefly_weight == 0u)
    {
        return 1.0f;
    }
    return rt_lab_firefly_factor(push.attach.lab_variance.get()[pixel].a, rt_lab_luminance(color));
}

// One spatial tap: filter weight (kernel x edge stopping (plane distance + normal power), 0 = rejected) and its color /
// count / split (and carried gradient) added to ring.
func rt_lab_spatial_tap(inout RtLabSpatialSum ring, int2 tap, float2 offset, float kernel, float count_scale,
    float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size, bool clamp_count, RtLabVarianceGuide guide, float center_ray_length_radius)
{
    let push = rt_lab_blur_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (any(tap < 0) || any(tap >= int2(push.size)))
    {
        return;
    }
    const float tap_depth = push.attach.lab_depth.get()[tap];
    if (tap_depth == 0.0f)
    {
        return;
    }
    const CameraInfo camera = push.attach.globals.view_camera;
    const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[tap]);
    const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
    float weight = kernel *
                   rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, settings.blur_plane_tolerance) *
                   pow(saturate(dot(normal, tap_normal)), settings.blur_normal_power) *
                   rt_lab_ray_length_guide_weight(center_ray_length_radius, tap, distance(tap_position, position), rcp_pixel_width_ws);
    if (weight <= 1e-6f)
    {
        return;
    }
    const float4 value = push.attach.src.get()[tap];
    const float tap_count = rt_lab_spatial_read_count(value.a, clamp_count, settings.sample_count_input_clamp);
    if (tap_count <= 0.0f)
    {
        return; // no samples (ray redistribution left the pixel without a ray)
    }
    // A rejected tap is not gathered: the variance guide weight also scales the effective sample count.
    if (guide.enabled)
    {
        weight *= rt_lab_variance_guide_weight(guide, tap, value.rgb, tap_count, offset);
        if (weight <= 1e-6f)
        {
            return;
        }
    }
    // Firefly energy weighting: the color (and split / gradient) weight grows by the tap's clipped energy factor, the
    // sample count keeps the plain weight.
    const float color_weight = weight * rt_lab_spatial_firefly_factor(tap, value.rgb);
    ring.color += value.rgb * color_weight;
    ring.weight += color_weight;
    ring.count += tap_count * weight * count_scale;
    if (push.split_enabled != 0u)
    {
        ring.split += push.attach.split_src.get()[tap] * color_weight;
    }
    if (push.gradient_mode == RT_LAB_GRADIENT_CARRY)
    {
        const float4 tap_gradient_x = push.attach.gradient_x_src.get()[tap];
        const float4 tap_gradient_y = push.attach.gradient_y_src.get()[tap];
        ring.gradient_x += tap_gradient_x.rgb * color_weight;
        ring.gradient_y += tap_gradient_y.rgb * color_weight;
        ring.gradient_variance += float2(tap_gradient_x.a, tap_gradient_y.a) * (color_weight * color_weight);
    }
}

// Relative color gradient view (DEBUG_DRAW_MODE_RT_LAB_COLOR_GRADIENT, drawn by the LAST fine pass): |grad luminance| /
// luminance per pixel of offset, the gradient the next consumer sees (after the significance shrinkage), log heatmap
// 0.1 % .. 100 % per pixel. Dim magenta: gradient aware guide off.
func rt_lab_spatial_gradient_debug(uint2 dtid, float3 color, RtLabGradient gradient, bool sky)
{
    let push = rt_lab_blur_push;
    if (push.last_pass == 0u || push.attach.globals.settings.debug_draw_mode != DEBUG_DRAW_MODE_RT_LAB_COLOR_GRADIENT)
    {
        return;
    }
    float3 debug_color = float3(0.0f, 0.0f, 0.0f);
    if (push.gradient_mode == RT_LAB_GRADIENT_OFF)
    {
        debug_color = float3(1.0f, 0.0f, 1.0f) * 0.25f;
    }
    else if (!sky)
    {
        const float3 keep_x = saturate(1.0f - gradient.x.a / max(gradient.x.rgb * gradient.x.rgb, 1e-20f));
        const float3 keep_y = saturate(1.0f - gradient.y.a / max(gradient.y.rgb * gradient.y.rgb, 1e-20f));
        const float2 luminance_gradient = float2(rt_lab_luminance(gradient.x.rgb * keep_x), rt_lab_luminance(gradient.y.rgb * keep_y));
        const float luminance_floor = RT_LAB_RELATIVE_DIFFERENCE_FLOOR * RTGI_RADIANCE_SCALE / (PI * push.attach.globals.exposure);
        debug_color = rt_lab_relative_difference_heatmap(length(luminance_gradient) / max(rt_lab_luminance(color), luminance_floor));
    }
    const uint resolution_scale = max(push.attach.globals.settings.render_target_size.x / push.size.x, 1u);
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(debug_color, 1.0f + push.attach.globals.settings.debug_visualization_blend), resolution_scale);
}

// Shared prologue of both filters: false = sky (output written).
func rt_lab_spatial_begin(uint2 dtid, out float4 center, out float3 position, out float3 normal, out float rcp_pixel_width_ws, out float2 inv_size) -> bool
{
    let push = rt_lab_blur_push;
    center = push.attach.src.get()[dtid];
    position = float3(0.0f, 0.0f, 0.0f);
    normal = float3(0.0f, 0.0f, 1.0f);
    rcp_pixel_width_ws = 0.0f;
    inv_size = rcp(float2(push.size));
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        push.attach.dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (push.split_enabled != 0u)
        {
            push.attach.split_dst.get()[dtid] = float2(0.0f, 0.0f);
        }
        if (push.gradient_mode != RT_LAB_GRADIENT_OFF)
        {
            push.attach.gradient_x_dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            push.attach.gradient_y_dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        rt_lab_spatial_gradient_debug(dtid, float3(0.0f, 0.0f, 0.0f), RtLabGradient(float4(0.0f, 0.0f, 0.0f, 0.0f), float4(0.0f, 0.0f, 0.0f, 0.0f)), true);
        return false;
    }
    const CameraInfo camera = push.attach.globals.view_camera;
    position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, inv_size), depth);
    normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.size.y), depth));
    return true;
}

// Writes the pass output. FIT: the fitted center gradient is the output gradient; CARRY: the weighted mean of the taps'.
func rt_lab_spatial_end(uint2 dtid, RtLabSpatialSum total, RtLabGradient center_gradient)
{
    let push = rt_lab_blur_push;
    // No samples anywhere in the footprint: count 0, the next pass / the resolve skip it.
    const float rcp_weight = total.weight > 0.0f ? rcp(total.weight) : 0.0f;
    const float3 color = total.color * rcp_weight;
    push.attach.dst.get()[dtid] = float4(color, total.count / RT_LAB_SPATIAL_COUNT_SCALE);
    if (push.split_enabled != 0u)
    {
        push.attach.split_dst.get()[dtid] = total.split * rcp_weight;
    }
    RtLabGradient gradient = center_gradient;
    if (push.gradient_mode == RT_LAB_GRADIENT_CARRY && total.weight > 0.0f)
    {
        gradient.x = float4(total.gradient_x * rcp_weight, total.gradient_variance.x * rcp_weight * rcp_weight);
        gradient.y = float4(total.gradient_y * rcp_weight, total.gradient_variance.y * rcp_weight * rcp_weight);
    }
    if (push.gradient_mode != RT_LAB_GRADIENT_OFF)
    {
        push.attach.gradient_x_dst.get()[dtid] = gradient.x;
        push.attach.gradient_y_dst.get()[dtid] = gradient.y;
    }
    rt_lab_spatial_gradient_debug(dtid, color, gradient, false);
}

// The center tap: always fully accepted (relative count weight 1), unless it holds no samples (weight 0).
func rt_lab_spatial_center(uint2 dtid, float4 center, float weight_in, bool clamp_count, RtLabGradient center_gradient) -> RtLabSpatialSum
{
    let push = rt_lab_blur_push;
    RtLabSpatialSum total = rt_lab_spatial_sum_zero();
    total.count = rt_lab_spatial_read_count(center.a, clamp_count, push.attach.globals.rt_lab_settings.sample_count_input_clamp);
    const float weight = total.count > 0.0f ? weight_in * rt_lab_spatial_firefly_factor(int2(dtid), center.rgb) : 0.0f;
    total.color = center.rgb * weight;
    total.weight = weight;
    if (push.split_enabled != 0u)
    {
        total.split = push.attach.split_src.get()[dtid] * weight;
    }
    total.gradient_x = center_gradient.x.rgb * weight;
    total.gradient_y = center_gradient.y.rgb * weight;
    total.gradient_variance = float2(center_gradient.x.a, center_gradient.y.a) * (weight * weight);
    return total;
}

// Center gradient + guide setup shared by both filters. The FIT pass (first fine pass) only estimates the gradient for
// the next pass, its own guide stays the constant model; every CARRY pass uses the previous pass's gradient as is and
// passes it on filtered with its own weights.
func rt_lab_spatial_guide(uint2 dtid, float4 center, float3 position, float3 normal, float rcp_pixel_width_ws, float2 inv_size,
    bool clamp_count, out RtLabGradient center_gradient) -> RtLabVarianceGuide
{
    let push = rt_lab_blur_push;
    const float center_count = rt_lab_spatial_read_count(center.a, clamp_count, push.attach.globals.rt_lab_settings.sample_count_input_clamp);
    center_gradient = rt_lab_gradient_center(dtid, center, center_count, position, normal, rcp_pixel_width_ws, inv_size, clamp_count);
    RtLabVarianceGuide guide = rt_lab_variance_guide_begin(dtid, center.rgb, center_count);
    if (guide.enabled && push.gradient_mode == RT_LAB_GRADIENT_CARRY)
    {
        rt_lab_variance_guide_set_gradient(guide, center_gradient);
    }
    return guide;
}

// One a-trous pass: 5x5 B3 spline taps spaced push.step pixels apart, rings (max(|x|, |y|) = 1, then 2) accepted from
// the center outwards under the fine filter's sample count budget. The split difference takes the same weights.
// Alpha: effective sample count.
[shader("compute")]
[numthreads(RT_LAB_BLUR_X, RT_LAB_BLUR_Y, 1)]
func entry_rt_lab_atrous(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_blur_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    float4 center;
    float3 position;
    float3 normal;
    float rcp_pixel_width_ws;
    float2 inv_size;
    if (!rt_lab_spatial_begin(dtid, center, position, normal, rcp_pixel_width_ws, inv_size))
    {
        return;
    }
    const bool clamp_count = push.clamp_input_count != 0u;
    const float target = push.attach.globals.rt_lab_settings.fine_target_samples;
    const float center_kernel = RT_LAB_ATROUS_KERNEL[0] * RT_LAB_ATROUS_KERNEL[0];
    RtLabGradient center_gradient;
    const RtLabVarianceGuide guide = rt_lab_spatial_guide(dtid, center, position, normal, rcp_pixel_width_ws, inv_size, clamp_count, center_gradient);
    RtLabSpatialSum total = rt_lab_spatial_center(dtid, center, center_kernel, clamp_count, center_gradient);
    const float center_ray_length_radius = rt_lab_spatial_ray_length_radius(int2(dtid));
    if (!(target > 0.0f && total.count >= target))
    {
        [unroll] for (int r = 1; r <= 2; ++r)
        {
            RtLabSpatialSum ring = rt_lab_spatial_sum_zero();
            [unroll] for (int y = -2; y <= 2; ++y)
            {
                [unroll] for (int x = -2; x <= 2; ++x)
                {
                    if (max(abs(x), abs(y)) != r)
                    {
                        continue;
                    }
                    const float kernel = RT_LAB_ATROUS_KERNEL[abs(x)] * RT_LAB_ATROUS_KERNEL[abs(y)];
                    rt_lab_spatial_tap(ring, int2(dtid) + int2(x, y) * push.step, float2(x, y) * float(push.step), kernel, 1.0f / center_kernel,
                        position, normal, rcp_pixel_width_ws, inv_size, clamp_count, guide, center_ray_length_radius);
                }
            }
            if (rt_lab_accept_ring(total, ring, target))
            {
                break;
            }
        }
    }
    rt_lab_spatial_end(dtid, total, center_gradient);
}

// One pass of the separable bilateral gaussian. Edge stopping is purely geometric: plane distance (in pixel widths)
// and face normal similarity, plus the variance guide. Taps +-d are accepted from d = 1 outwards under the fine
// filter's sample count budget. Alpha: effective sample count (rt_lab_blur.inl).
[shader("compute")]
[numthreads(RT_LAB_BLUR_X, RT_LAB_BLUR_Y, 1)]
func entry_rt_lab_blur(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_blur_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    float4 center;
    float3 position;
    float3 normal;
    float rcp_pixel_width_ws;
    float2 inv_size;
    if (!rt_lab_spatial_begin(dtid, center, position, normal, rcp_pixel_width_ws, inv_size))
    {
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    const int radius = clamp(settings.blur_radius, 0, 32);
    const float sigma = max(float(radius) * 0.5f, 0.5f);
    const float rcp_two_sigma_sq = rcp(2.0f * sigma * sigma);
    const bool clamp_count = push.clamp_input_count != 0u;
    const float target = settings.fine_target_samples;

    // The center weight is 1, so the relative count weight is the weight itself.
    RtLabGradient center_gradient;
    const RtLabVarianceGuide guide = rt_lab_spatial_guide(dtid, center, position, normal, rcp_pixel_width_ws, inv_size, clamp_count, center_gradient);
    RtLabSpatialSum total = rt_lab_spatial_center(dtid, center, 1.0f, clamp_count, center_gradient);
    const float center_ray_length_radius = rt_lab_spatial_ray_length_radius(int2(dtid));
    if (!(target > 0.0f && total.count >= target))
    {
        [loop] for (int d = 1; d <= radius; ++d)
        {
            const float kernel = exp(-float(d * d) * rcp_two_sigma_sq);
            RtLabSpatialSum ring = rt_lab_spatial_sum_zero();
            rt_lab_spatial_tap(ring, int2(dtid) - push.direction * d, -float2(push.direction * d), kernel, 1.0f, position, normal, rcp_pixel_width_ws, inv_size, clamp_count, guide, center_ray_length_radius);
            rt_lab_spatial_tap(ring, int2(dtid) + push.direction * d, float2(push.direction * d), kernel, 1.0f, position, normal, rcp_pixel_width_ws, inv_size, clamp_count, guide, center_ray_length_radius);
            if (rt_lab_accept_ring(total, ring, target))
            {
                break;
            }
        }
    }
    rt_lab_spatial_end(dtid, total, center_gradient);
}
