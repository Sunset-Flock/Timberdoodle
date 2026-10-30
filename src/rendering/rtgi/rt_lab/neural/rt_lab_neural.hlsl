#pragma once

#include "rt_lab_neural.inl"

#include "../rt_lab_shared.hlsl"
#include "../rt_lab_reproject.hlsl"

// The network, the features and the a-trous pass are the shared core: the trainer (neural/train) differentiates this exact
// code. Here it only gets engine backends (textures, weights buffer).
#include "rt_lab_neural_core.slang"

[[vk::push_constant]] RtLabNeuralLatentPush rt_lab_neural_latent_push;
[[vk::push_constant]] RtLabNeuralAtrousPush rt_lab_neural_atrous_push;
[[vk::push_constant]] RtLabNeuralDumpPush rt_lab_neural_dump_push;
[[vk::push_constant]] RtLabNeuralRayStatsPush rt_lab_neural_ray_stats_push;

// ---------------------------------------------------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------------------------------------------------

float rt_lab_neural_load_weight(float* weights, int index)
{
    return weights[index];
}

struct RtLabNeuralEngineParams : IRtLabNeuralParams
{
    float* weights;

    [Differentiable]
    float get(int index)
    {
        return no_diff rt_lab_neural_load_weight(weights, index);
    }
}

func rt_lab_neural_pack_latent(rt_real latent[RT_LAB_NEURAL_LATENT]) -> uint4
{
    return uint4(
        f32tof16(float(latent[0])) | (f32tof16(float(latent[1])) << 16u),
        f32tof16(float(latent[2])) | (f32tof16(float(latent[3])) << 16u),
        f32tof16(float(latent[4])) | (f32tof16(float(latent[5])) << 16u),
        f32tof16(float(latent[6])) | (f32tof16(float(latent[7])) << 16u));
}

func rt_lab_neural_unpack_latent(uint4 packed) -> rt_real[RT_LAB_NEURAL_LATENT]
{
    rt_real r[RT_LAB_NEURAL_LATENT];
    r[0] = RT_H(f16tof32(packed.x & 0xFFFFu));
    r[1] = RT_H(f16tof32(packed.x >> 16u));
    r[2] = RT_H(f16tof32(packed.y & 0xFFFFu));
    r[3] = RT_H(f16tof32(packed.y >> 16u));
    r[4] = RT_H(f16tof32(packed.z & 0xFFFFu));
    r[5] = RT_H(f16tof32(packed.z >> 16u));
    r[6] = RT_H(f16tof32(packed.w & 0xFFFFu));
    r[7] = RT_H(f16tof32(packed.w >> 16u));
    return r;
}

func rt_lab_neural_load_latent(Texture2D<uint4> latents, int2 pixel) -> rt_real[RT_LAB_NEURAL_LATENT]
{
    return rt_lab_neural_unpack_latent(latents[pixel]);
}

// Network outputs [8, 16) packed as 8 f16 (the same packing as the latent).
func rt_lab_neural_pack_affinity(rt_real result[RT_LAB_NEURAL_OUT]) -> uint4
{
    rt_real raw[RT_LAB_NEURAL_LATENT];
    [unroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        raw[k] = RT_H(clamp(float(result[RT_LAB_NEURAL_LATENT + k]), -60000.0f, 60000.0f));
    }
    return rt_lab_neural_pack_latent(raw);
}

func rt_lab_neural_load_affinity_raw(Texture2D<uint4> affinity, int2 pixel) -> rt_real[RT_LAB_NEURAL_AFFINITY_VALUES]
{
    return rt_lab_neural_unpack_latent(affinity[pixel]);
}

struct RtLabNeuralEngineLatents : IRtLabNeuralLatents
{
    Texture2D<uint4> latents;
    Texture2D<uint4> affinity;

    [Differentiable]
    rt_real[RT_LAB_NEURAL_LATENT] load(int2 pixel)
    {
        return no_diff rt_lab_neural_load_latent(latents, pixel);
    }

    [Differentiable]
    rt_real[RT_LAB_NEURAL_AFFINITY_VALUES] load_affinity_raw(int2 pixel)
    {
        return no_diff rt_lab_neural_load_affinity_raw(affinity, pixel);
    }
}

struct RtLabNeuralEnginePixels : IRtLabNeuralPixels
{
    Texture2D<float> depth;
    Texture2D<uint> normal;
    Texture2D<float4> accumulated;
    Texture2D<float4> raw;
    Texture2D<float4> variance;
    Texture2D<float2> ray_length;
    bool ray_length_bound;
    int2 size;
    Texture2D<uint> detail_normal;
    bool detail_normal_bound;
    Texture2D<float> ray_capped;
    bool ray_capped_bound;

    RtLabNeuralPixel load(int2 pixel)
    {
        RtLabNeuralPixel r = {};
        const float4 a = accumulated[pixel];
        r.acc = a.rgb;
        r.acc_length = a.a;
        const float4 w = raw[pixel];
        r.raw = w.rgb;
        r.raw_count = w.a;
        r.variance = variance[pixel].rgb;
        r.ray_ao = -1.0f;
        r.ray_shortness = 0.0f;
        if (ray_length_bound)
        {
            const float2 g = ray_length[pixel];
            r.ray_ao = g.x;
            r.ray_shortness = g.y;
        }
        r.depth = depth[pixel];
        r.normal = uncompress_normal_octahedral_32(normal[pixel]);
        // Lite feature set inputs (zero / the face normal when unavailable, like the training data of lab runs without them).
        r.detail_normal = detail_normal_bound ? uncompress_normal_octahedral_32(detail_normal[pixel]) : float3(0.0f, 0.0f, 0.0f);
        r.ray_capped = ray_capped_bound ? max(ray_capped[pixel], 0.0f) : 0.0f;
        return r;
    }

    int2 extent()
    {
        return size;
    }

    // The SH feature set (v2) is not run by the engine yet (the weights loader refuses its weights): no footprint weights.
    float footprint_geometry_weight(int2 center, int2 tap)
    {
        return 1.0f;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Latent pass
// ---------------------------------------------------------------------------------------------------------------------

// Previous frame's latent at this pixel: bilinear reprojection with the temporal pass's surface test, normalized over the
// valid taps (zeros when nothing matches or after a reset).
func rt_lab_neural_previous_latent(uint2 dtid, float3 position, float3 normal, float rcp_pixel_width_ws) -> rt_real[RT_LAB_NEURAL_LATENT]
{
    let push = rt_lab_neural_latent_push;
    float accumulated[RT_LAB_NEURAL_LATENT];
    rt_real result[RT_LAB_NEURAL_LATENT];
    [unroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        accumulated[k] = 0.0f;
        result[k] = RT_H(0.0f);
    }
    if (push.reset != 0u)
    {
        return result;
    }
    CameraInfo* prev_camera = &push.attach.globals.view_camera_prev_frame;
    const float2 size_f = float2(push.size);
    const float2 inv_size = rcp(size_f);
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    const Bilinear bilinear = get_bilinear_filter(prev_uv, size_f);
    float4 tap_weights = float4(0.0f, 0.0f, 0.0f, 0.0f);
    [unroll]
    for (uint i = 0u; i < 4u; ++i)
    {
        const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
        bool occluder;
        tap_weights[i] = rt_lab_reproject_tap(push.attach.lab_depth_history.get(), push.attach.lab_face_normal_history.get(), prev_camera,
            &push.attach.globals.rt_lab_settings, int2(push.size), tap, position, normal, rcp_pixel_width_ws, inv_size, occluder);
    }
    const float4 weights = get_bilinear_custom_weights(bilinear, tap_weights);
    const float weight_sum = dot(weights, 1.0f);
    if (weight_sum <= 1e-3f)
    {
        return result;
    }
    [unroll]
    for (uint i = 0u; i < 4u; ++i)
    {
        if (weights[i] <= 0.0f)
        {
            continue;
        }
        const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
        const float tap_weight = weights[i] / weight_sum;
        const rt_real tap_latent[RT_LAB_NEURAL_LATENT] = rt_lab_neural_unpack_latent(push.attach.latent_history.get()[tap]);
        [unroll]
        for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
        {
            accumulated[k] += float(tap_latent[k]) * tap_weight;
        }
    }
    [unroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        result[k] = RT_H(accumulated[k]);
    }
    return result;
}

// MLP backend of the latent pass: the plain loops of the shared core, or cooperative vectors (tensor cores).
interface IRtLabNeuralMlp
{
    void evaluate(rt_real x[RT_LAB_NEURAL_IN], out rt_real result[RT_LAB_NEURAL_OUT]);
}

struct RtLabNeuralPlainMlp : IRtLabNeuralMlp
{
    RtLabNeuralEngineParams params;

    void evaluate(rt_real x[RT_LAB_NEURAL_IN], out rt_real result[RT_LAB_NEURAL_OUT])
    {
        rt_lab_neural_mlp<RtLabNeuralEngineParams>(params, x, result);
    }
}

// Typed alias of daxa's bindless storage buffer table (same set / binding): daxa's own table is a __DynamicResource array and
// Slang's cooperative vector emitter aborts on it ("Unhandled global inst in spirv-emit", Slang 2025.11).
[[vk::binding(DAXA_STORAGE_BUFFER_BINDING, 0)]] RWByteAddressBuffer rt_lab_neural_buffers[];

// VK_NV_cooperative_vector: every layer is one per thread matrix x vector product on the tensor cores. Weights f16, row
// major, at the RT_LAB_NEURAL_* offsets (x 2 bytes: all matrix / bias offsets are multiples of 64 bytes).
struct RtLabNeuralCoopVecMlp : IRtLabNeuralMlp
{
    daxa_BufferId weights_id;

    void evaluate(rt_real x[RT_LAB_NEURAL_IN], out rt_real result[RT_LAB_NEURAL_OUT])
    {
        RWByteAddressBuffer weights = rt_lab_neural_buffers[weights_id.index()];
        CoopVec<half, RT_LAB_NEURAL_IN> input;
        [unroll]
        for (int i = 0; i < RT_LAB_NEURAL_IN; ++i)
        {
            input[i] = half(x[i]);
        }
        CoopVec<half, RT_LAB_NEURAL_HID> h1 = coopVecMatMulAdd<half, RT_LAB_NEURAL_HID>(input, CoopVecComponentType::Float16, weights, RT_LAB_NEURAL_W1 * 2,
            CoopVecComponentType::Float16, weights, RT_LAB_NEURAL_B1 * 2, CoopVecComponentType::Float16, CoopVecMatrixLayout::RowMajor, false, RT_LAB_NEURAL_IN * 2);
        [unroll]
        for (int i = 0; i < RT_LAB_NEURAL_HID; ++i)
        {
            h1[i] = max(h1[i], h1[i] * half(RT_LAB_NEURAL_LEAK)); // leaky relu
        }
        CoopVec<half, RT_LAB_NEURAL_HID> h2 = coopVecMatMulAdd<half, RT_LAB_NEURAL_HID>(h1, CoopVecComponentType::Float16, weights, RT_LAB_NEURAL_W2 * 2,
            CoopVecComponentType::Float16, weights, RT_LAB_NEURAL_B2 * 2, CoopVecComponentType::Float16, CoopVecMatrixLayout::RowMajor, false, RT_LAB_NEURAL_HID * 2);
        [unroll]
        for (int i = 0; i < RT_LAB_NEURAL_HID; ++i)
        {
            h2[i] = max(h2[i], h2[i] * half(RT_LAB_NEURAL_LEAK));
        }
        CoopVec<half, RT_LAB_NEURAL_OUT> out_vector = coopVecMatMulAdd<half, RT_LAB_NEURAL_OUT>(h2, CoopVecComponentType::Float16, weights,
            RT_LAB_NEURAL_W3 * 2, CoopVecComponentType::Float16, weights, RT_LAB_NEURAL_B3 * 2, CoopVecComponentType::Float16, CoopVecMatrixLayout::RowMajor, false,
            RT_LAB_NEURAL_HID * 2);
        [unroll]
        for (int i = 0; i < RT_LAB_NEURAL_OUT; ++i)
        {
            result[i] = RT_H(i < RT_LAB_NEURAL_LATENT ? tanh(float(out_vector[i])) : float(out_vector[i]));
        }
    }
}

func rt_lab_neural_latent_run<M : IRtLabNeuralMlp>(uint2 dtid, M mlp)
{
    let push = rt_lab_neural_latent_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        push.attach.latent.get()[dtid] = uint4(0u, 0u, 0u, 0u);
        push.attach.affinity.get()[dtid] = uint4(0u, 0u, 0u, 0u);
        return;
    }
    const CameraInfo camera = push.attach.globals.view_camera;
    const float2 inv_size = rcp(float2(push.size));
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, inv_size), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.size.y), depth));
    const rt_real previous[RT_LAB_NEURAL_LATENT] = rt_lab_neural_previous_latent(dtid, position, normal, rcp_pixel_width_ws);

    RtLabNeuralEnginePixels pixels = {
        push.attach.lab_depth.get(), push.attach.lab_face_normal.get(), push.attach.accumulated.get(), push.attach.lab_raw_diffuse.get(),
        push.attach.lab_variance.get(), push.attach.lab_ray_length.get(), push.ray_length_bound != 0u, int2(push.size), push.attach.detail_normal.get(),
        push.detail_normal_bound != 0u, push.attach.ray_capped.get(), push.ray_capped_bound != 0u};
    rt_real x[RT_LAB_NEURAL_IN];
    RtLabNeuralEngineParams feature_params = {push.attach.weights};
    rt_lab_neural_features<RtLabNeuralEnginePixels, RtLabNeuralEngineParams>(pixels, feature_params, int2(dtid), max(push.attach.globals.exposure, 1e-3f), previous, x);
    rt_real result[RT_LAB_NEURAL_OUT];
    mlp.evaluate(x, result);
    rt_real latent[RT_LAB_NEURAL_LATENT];
    [unroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        latent[k] = result[k];
    }
    push.attach.latent.get()[dtid] = rt_lab_neural_pack_latent(latent);
    push.attach.affinity.get()[dtid] = rt_lab_neural_pack_affinity(result);
}

[shader("compute")]
[numthreads(RT_LAB_NEURAL_X, RT_LAB_NEURAL_Y, 1)]
func entry_rt_lab_neural_latent(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_neural_latent_push;
    RtLabNeuralPlainMlp mlp = {{push.attach.weights}};
    rt_lab_neural_latent_run<RtLabNeuralPlainMlp>(dtid, mlp);
}

// Same pass with the tensor core MLP. Only registered when the device has VK_NV_cooperative_vector.
[shader("compute")]
[numthreads(RT_LAB_NEURAL_X, RT_LAB_NEURAL_Y, 1)]
func entry_rt_lab_neural_latent_coopvec(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_neural_latent_push;
    RtLabNeuralCoopVecMlp mlp = {push.attach.weights_f16};
    rt_lab_neural_latent_run<RtLabNeuralCoopVecMlp>(dtid, mlp);
}

// ---------------------------------------------------------------------------------------------------------------------
// Ray statistics (lite feature set): accumulated capped normalized ray length
// ---------------------------------------------------------------------------------------------------------------------

// The history value at this pixel: bilinear reprojection with the temporal pass's surface test, normalized over the taps that hold a value
// (-1 = never hit by a ray). -1 when nothing matches or after a reset.
func rt_lab_neural_previous_ray_capped(uint2 dtid, float3 position, float3 normal, float rcp_pixel_width_ws) -> float
{
    let push = rt_lab_neural_ray_stats_push;
    if (push.reset != 0u)
    {
        return -1.0f;
    }
    CameraInfo* prev_camera = &push.attach.globals.view_camera_prev_frame;
    const float2 size_f = float2(push.size);
    const float2 inv_size = rcp(size_f);
    const float4 prev_clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    const float2 prev_uv = (prev_clip.xy / prev_clip.w) * 0.5f + 0.5f;
    const Bilinear bilinear = get_bilinear_filter(prev_uv, size_f);
    float4 tap_weights = float4(0.0f, 0.0f, 0.0f, 0.0f);
    [unroll]
    for (uint i = 0u; i < 4u; ++i)
    {
        const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
        bool occluder;
        tap_weights[i] = rt_lab_reproject_tap(push.attach.lab_depth_history.get(), push.attach.lab_face_normal_history.get(), prev_camera,
            &push.attach.globals.rt_lab_settings, int2(push.size), tap, position, normal, rcp_pixel_width_ws, inv_size, occluder);
    }
    const float4 weights = get_bilinear_custom_weights(bilinear, tap_weights);
    float sum = 0.0f;
    float valid_weight = 0.0f;
    [unroll]
    for (uint i = 0u; i < 4u; ++i)
    {
        if (weights[i] <= 0.0f)
        {
            continue;
        }
        const int2 tap = int2(bilinear.origin) + int2(i & 1u, i >> 1u);
        const float value = push.attach.ray_capped_history.get()[tap];
        if (value < 0.0f)
        {
            continue;
        }
        sum += value * weights[i];
        valid_weight += weights[i];
    }
    return valid_weight > 1e-3f ? sum / valid_weight : -1.0f;
}

[shader("compute")]
[numthreads(RT_LAB_NEURAL_X, RT_LAB_NEURAL_Y, 1)]
func entry_rt_lab_neural_ray_stats(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_neural_ray_stats_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        push.attach.ray_capped.get()[dtid] = -1.0f;
        return;
    }
    const CameraInfo camera = push.attach.globals.view_camera;
    const float2 inv_size = rcp(float2(push.size));
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, inv_size), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(push.size.y), depth));
    float state = rt_lab_neural_previous_ray_capped(dtid, position, normal, rcp_pixel_width_ws);
    const uint2 alloc = push.attach.ray_alloc.get()[dtid];
    if (alloc.y > 0u)
    {
        const float cap = 32.0f * rt_lab_pixel_width_ws(camera, float(push.size.y), depth);
        float sum_capped = 0.0f;
        for (uint k = 0u; k < alloc.y; ++k)
        {
            sum_capped += min(push.attach.ray_lengths[alloc.x + k], cap) / cap;
        }
        const float frame_mean = sum_capped / float(alloc.y);
        if (state < 0.0f)
        {
            state = frame_mean;
        }
        else
        {
            const float rays = push.attach.lab_raw_diffuse.get()[dtid].a;
            const float length = max(push.attach.accumulated.get()[dtid].a, 1.0f);
            state += (frame_mean - state) * saturate(rays / length);
        }
    }
    push.attach.ray_capped.get()[dtid] = state;
}

// ---------------------------------------------------------------------------------------------------------------------
// Neural a-trous
// ---------------------------------------------------------------------------------------------------------------------

struct RtLabNeuralEngineAtrousSource : IRtLabNeuralAtrousSource
{
    Texture2D<float> depth;
    Texture2D<uint> normal;
    Texture2D<float4> src;
    Texture2D<float2> split_src;
    Texture2D<float4> variance;
    int2 size;
    bool first_pass;
    bool split_enabled;

    // Pass input convention of rt_lab_blur.hlsl: first pass = temporal history (plain length, sample_count_input_clamp),
    // later passes = alpha x RT_LAB_SPATIAL_COUNT_SCALE.
    float4 load_color_count(int2 pixel)
    {
        let push = rt_lab_neural_atrous_push;
        const float4 value = src[pixel];
        return float4(value.rgb, rt_lab_spatial_read_count(value.a, first_pass, push.attach.globals.rt_lab_settings.sample_count_input_clamp));
    }

    [Differentiable]
    float4 color_count(int2 pixel)
    {
        const float4 value = no_diff load_color_count(pixel);
        return float4(value.rgb, value.w);
    }

    float2 split(int2 pixel)
    {
        return split_enabled ? split_src[pixel] : float2(0.0f, 0.0f);
    }

    bool geometry_valid(int2 pixel)
    {
        return depth[pixel] != 0.0f;
    }

    float geometry_weight(int2 center, int2 tap, int tap_index)
    {
        let push = rt_lab_neural_atrous_push;
        let settings = push.attach.globals.rt_lab_settings;
        const CameraInfo camera = push.attach.globals.view_camera;
        const float2 inv_size = rcp(float2(size));
        const float center_depth = depth[center];
        const float tap_depth = depth[tap];
        const float3 center_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(center), inv_size), center_depth);
        const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), inv_size), tap_depth);
        const float3 center_normal = uncompress_normal_octahedral_32(normal[center]);
        const float3 tap_normal = uncompress_normal_octahedral_32(normal[tap]);
        const float rcp_pixel_width_ws = rcp(rt_lab_pixel_width_ws(camera, float(size.y), center_depth));
        return rt_lab_plane_weight(center_position, center_normal, tap_position, tap_normal, rcp_pixel_width_ws, settings.blur_plane_tolerance) *
               pow(saturate(dot(center_normal, tap_normal)), settings.blur_normal_power);
    }

    int2 extent()
    {
        return size;
    }
}

[shader("compute")]
[numthreads(RT_LAB_NEURAL_X, RT_LAB_NEURAL_Y, 1)]
func entry_rt_lab_neural_atrous(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_neural_atrous_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    if (push.attach.lab_depth.get()[dtid] == 0.0f)
    {
        push.attach.dst.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (push.split_enabled != 0u)
        {
            push.attach.split_dst.get()[dtid] = float2(0.0f, 0.0f);
        }
        if (push.history_feedback != 0u)
        {
            push.attach.history_feedback.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        return;
    }
    let settings = push.attach.globals.rt_lab_settings;
    RtLabNeuralEngineAtrousSource source = {
        push.attach.lab_depth.get(), push.attach.lab_face_normal.get(), push.attach.src.get(), push.attach.split_src.get(), push.attach.lab_variance.get(),
        int2(push.size), push.clamp_input_count != 0u, push.split_enabled != 0u};
    RtLabNeuralEngineLatents latents = {push.attach.latent.get(), push.attach.affinity.get()};
    const RtLabNeuralAtrousResult r = rt_lab_neural_atrous<RtLabNeuralEngineAtrousSource, RtLabNeuralEngineLatents>(
        source, latents, int2(dtid), push.step, push.target_samples, true);
    const float rcp_weight = r.weight > 0.0f ? rcp(r.weight) : 0.0f;
    push.attach.dst.get()[dtid] = float4(r.color * rcp_weight, r.count / RT_LAB_SPATIAL_COUNT_SCALE);
    if (push.split_enabled != 0u)
    {
        push.attach.split_dst.get()[dtid] = r.split * rcp_weight;
    }
    if (push.history_feedback != 0u)
    {
        // The filtered color with the center's own temporal history length; no samples in the footprint: the center stays.
        const float4 center = push.attach.src.get()[dtid];
        push.attach.history_feedback.get()[dtid] = float4(r.weight > 0.0f ? r.color * rcp_weight : center.rgb, center.a);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Dump
// ---------------------------------------------------------------------------------------------------------------------

func rt_lab_neural_pack_f16(float a, float b) -> uint
{
    return f32tof16(clamp(a, -65000.0f, 65000.0f)) | (f32tof16(clamp(b, -65000.0f, 65000.0f)) << 16u);
}

[shader("compute")]
[numthreads(RT_LAB_NEURAL_X, RT_LAB_NEURAL_Y, 1)]
func entry_rt_lab_neural_dump(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_neural_dump_push;
    if (any(dtid >= push.crop_size))
    {
        return;
    }
    const uint2 pixel = min(push.crop_origin + dtid, push.size - 1u);
    const uint crop_pixels = push.crop_size * push.crop_size;
    const uint index = dtid.y * push.crop_size + dtid.x;
    const float depth = push.attach.lab_depth.get()[pixel];
    if (push.block == 0u)
    {
        // Geometry block: [0] position.xyz, depth   [1] normal.xyz, rcp pixel width.
        const CameraInfo camera = push.attach.globals.view_camera;
        const float2 inv_size = rcp(float2(push.size));
        const float3 position = depth > 0.0f ? linear_depth_to_world_space(camera, rt_lab_ndc(pixel, inv_size), depth) : float3(0.0f, 0.0f, 0.0f);
        const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[pixel]);
        const float rcp_width = depth > 0.0f ? rcp(rt_lab_pixel_width_ws(camera, float(push.size.y), depth)) : 0.0f;
        float3 detail_normal = float3(0.0f, 0.0f, 0.0f);
        if (push.detail_normal_bound != 0u)
        {
            detail_normal = uncompress_normal_octahedral_32(push.attach.detail_normal.get()[pixel]);
        }
        float3 albedo = float3(0.0f, 0.0f, 0.0f);
        if (push.albedo_bound != 0u)
        {
            albedo = push.attach.half_res_albedo.get()[pixel >> push.albedo_shift].rgb;
        }
        push.attach.dump[index * 3u + 0u] = uint4(asuint(position.x), asuint(position.y), asuint(position.z), asuint(depth));
        push.attach.dump[index * 3u + 1u] = uint4(asuint(normal.x), asuint(normal.y), asuint(normal.z), asuint(rcp_width));
        push.attach.dump[index * 3u + 2u] = uint4(rt_lab_neural_pack_f16(detail_normal.x, detail_normal.y), rt_lab_neural_pack_f16(detail_normal.z, albedo.x),
            rt_lab_neural_pack_f16(albedo.y, albedo.z), 0u);
        return;
    }
    const uint block_pixels = 1u + crop_pixels * 4u;
    const uint base = crop_pixels * 3u + (push.block - 1u) * block_pixels;
    if (all(dtid == uint2(0u, 0u)))
    {
        push.attach.dump[base] = uint4(asuint(push.attach.globals.exposure), push.frame_index, 0u, 0u);
    }
    const float4 acc = push.attach.accumulated.get()[pixel];
    const float4 raw = push.attach.lab_raw_diffuse.get()[pixel];
    const float4 variance = push.attach.lab_variance.get()[pixel];
    float2 ray = float2(-1.0f, 0.0f);
    if (push.ray_length_bound != 0u)
    {
        ray = push.attach.lab_ray_length.get()[pixel];
    }
    push.attach.dump[base + 1u + index * 4u + 0u] = uint4(rt_lab_neural_pack_f16(acc.x, acc.y), rt_lab_neural_pack_f16(acc.z, acc.w),
        rt_lab_neural_pack_f16(raw.x, raw.y), rt_lab_neural_pack_f16(raw.z, raw.w));
    // This frame's rays of the pixel: mean log2(1 + hit distance) and mean capped normalized hit distance (cap = 32 pixel widths).
    float2 ray_stats = float2(-1.0f, -1.0f);
    float ray_count = 0.0f;
    if (push.ray_stats_bound != 0u && depth > 0.0f)
    {
        const uint2 alloc = push.attach.ray_alloc.get()[pixel];
        const float cap = 32.0f * rt_lab_pixel_width_ws(push.attach.globals.view_camera, float(push.size.y), depth);
        float sum_log = 0.0f;
        float sum_capped = 0.0f;
        for (uint k = 0u; k < alloc.y; ++k)
        {
            const float t = push.attach.ray_lengths[alloc.x + k];
            sum_log += log2(1.0f + min(t, 1000.0f));
            sum_capped += min(t, cap) / cap;
        }
        if (alloc.y > 0u)
        {
            ray_stats = float2(sum_log, sum_capped) / float(alloc.y);
        }
        ray_count = float(alloc.y);
    }
    push.attach.dump[base + 1u + index * 4u + 1u] = uint4(rt_lab_neural_pack_f16(variance.x, variance.y), rt_lab_neural_pack_f16(variance.z, ray.x),
        rt_lab_neural_pack_f16(ray.y, ray_stats.x), rt_lab_neural_pack_f16(ray_stats.y, ray_count));
    // v3: the temporal SH-Y + CoCg (f16: the history images are f16 too) and the f32 squared moments (Y^2 overflows f16).
    float4 sh_y = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 cocg = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sq = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (push.sh_bound != 0u)
    {
        sh_y = push.attach.lab_sh_y.get()[pixel];
        cocg = push.attach.lab_cocg.get()[pixel];
        sq = push.attach.lab_sq.get()[pixel];
    }
    push.attach.dump[base + 1u + index * 4u + 2u] = uint4(rt_lab_neural_pack_f16(sh_y.x, sh_y.y), rt_lab_neural_pack_f16(sh_y.z, sh_y.w),
        rt_lab_neural_pack_f16(cocg.x, cocg.y), 0u);
    push.attach.dump[base + 1u + index * 4u + 3u] = uint4(asuint(sq.x), asuint(sq.y), asuint(sq.z), 0u);
}
