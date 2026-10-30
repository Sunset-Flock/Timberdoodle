import slangpy;

// Trainer side of the RT Lab neural affinity (train.py). Everything that matters (features, MLP, affinity, the a-trous
// pass) is rt_lab_neural_core.slang, the same file the engine includes. This file only maps SlangPy tensors onto the core's
// interfaces, so training differentiates exactly the code that ships.
//
// SlangPy index vectors are innermost first: a [H, W, C] tensor is indexed uint3(c, x, y), grid pixels are (x, y), and
// Tensor.shape is in storage order (shape[0] = H).

#include "../rt_lab_neural_core.slang"

// Plane tolerance (pixel widths) and normal power of the lab's fine filter (rt_lab_settings.inl blur_plane_tolerance / blur_normal_power).
static const float TRAIN_PLANE_TOLERANCE = 1.5f;
static const float TRAIN_NORMAL_POWER = 8.0f;
// Variance guide floor: 0.1 % of display white per channel (the sigma is a runtime argument: 2 = the lab's custom preset, 0.5 = the
// strict guide of the reference filter).
static const float TRAIN_RELATIVE_FLOOR = 0.01f;

struct TrainParams : IRtLabNeuralParams
{
    DiffTensor<float, 1> t;

    [Differentiable]
    float get(int index)
    {
        return t[uint(index)];
    }
}

// Constant per sequence geometry: position [H, W, 3], normal [H, W, 3], rcp pixel width [H, W, 1] (0 depth = sky via depth).
struct TrainGeometry
{
    Tensor<float, 3> position;
    Tensor<float, 3> normal;
    Tensor<float, 3> rcp_width;
    Tensor<float, 3> depth;
    Tensor<float, 3> detail_normal; // [H, W, 3], raw 3x3 feature set
    Tensor<float, 3> albedo;        // [H, W, 3], raw 3x3 feature set

    int2 extent()
    {
        return int2(depth.shape[1], depth.shape[0]);
    }

    bool valid(int2 p)
    {
        return depth[uint3(0, p.x, p.y)] > 0.0f;
    }

    float3 position_at(int2 p)
    {
        return float3(position[uint3(0, p.x, p.y)], position[uint3(1, p.x, p.y)], position[uint3(2, p.x, p.y)]);
    }

    float3 normal_at(int2 p)
    {
        return float3(normal[uint3(0, p.x, p.y)], normal[uint3(1, p.x, p.y)], normal[uint3(2, p.x, p.y)]);
    }

    float weight(int2 center, int2 tap)
    {
        const float3 pa = position_at(center);
        const float3 pb = position_at(tap);
        const float3 na = normal_at(center);
        const float3 nb = normal_at(tap);
        const float3 d = pb - pa;
        const float dist = max(abs(dot(d, na)), abs(dot(d, nb))) * rcp_width[uint3(0, center.x, center.y)];
        return saturate(1.0f - dist / TRAIN_PLANE_TOLERANCE) * pow(saturate(dot(na, nb)), TRAIN_NORMAL_POWER);
    }
}

// Constant per frame lab data.
struct TrainFrame
{
    Tensor<float, 3> acc;      // [H, W, 4] rgb, history length
    Tensor<float, 3> raw;      // [H, W, 4] rgb, rays
    Tensor<float, 3> variance; // [H, W, 4] rgb per sample variance, unused
    Tensor<float, 3> ray;      // [H, W, 2] ao guide (< 0 none), shortness
    Tensor<float, 3> ray_stats; // [H, W, 2] accumulated per pixel mean log2(1 + hit distance), mean capped normalized hit distance
    Tensor<float, 3> sh_y;      // [H, W, 4] SH feature set (v2): temporal SH-Y, xyz = mean of direction x Y, w = mean Y
    Tensor<float, 3> cocg;      // [H, W, 2] mean Co, Cg
    Tensor<float, 3> sq;        // [H, W, 3] mean of Y^2, Co^2, Cg^2
    float exposure;
}

struct TrainPixels : IRtLabNeuralPixels
{
    TrainGeometry geometry;
    TrainFrame frame;

    RtLabNeuralPixel load(int2 p)
    {
        RtLabNeuralPixel r = {};
        r.acc = float3(frame.acc[uint3(0, p.x, p.y)], frame.acc[uint3(1, p.x, p.y)], frame.acc[uint3(2, p.x, p.y)]);
        r.acc_length = frame.acc[uint3(3, p.x, p.y)];
        r.raw = float3(frame.raw[uint3(0, p.x, p.y)], frame.raw[uint3(1, p.x, p.y)], frame.raw[uint3(2, p.x, p.y)]);
        r.raw_count = frame.raw[uint3(3, p.x, p.y)];
        r.variance = float3(frame.variance[uint3(0, p.x, p.y)], frame.variance[uint3(1, p.x, p.y)], frame.variance[uint3(2, p.x, p.y)]);
        r.ray_ao = frame.ray[uint3(0, p.x, p.y)];
        r.ray_shortness = frame.ray[uint3(1, p.x, p.y)];
        r.depth = geometry.depth[uint3(0, p.x, p.y)];
        r.normal = geometry.normal_at(p);
        r.detail_normal = float3(geometry.detail_normal[uint3(0, p.x, p.y)], geometry.detail_normal[uint3(1, p.x, p.y)], geometry.detail_normal[uint3(2, p.x, p.y)]);
        r.albedo = float3(geometry.albedo[uint3(0, p.x, p.y)], geometry.albedo[uint3(1, p.x, p.y)], geometry.albedo[uint3(2, p.x, p.y)]);
        r.ray_log_distance = frame.ray_stats[uint3(0, p.x, p.y)];
        r.ray_capped = frame.ray_stats[uint3(1, p.x, p.y)];
        r.sh_y = float4(frame.sh_y[uint3(0, p.x, p.y)], frame.sh_y[uint3(1, p.x, p.y)], frame.sh_y[uint3(2, p.x, p.y)], frame.sh_y[uint3(3, p.x, p.y)]);
        r.cocg = float2(frame.cocg[uint3(0, p.x, p.y)], frame.cocg[uint3(1, p.x, p.y)]);
        r.sq = float3(frame.sq[uint3(0, p.x, p.y)], frame.sq[uint3(1, p.x, p.y)], frame.sq[uint3(2, p.x, p.y)]);
        return r;
    }

    float footprint_geometry_weight(int2 center, int2 tap)
    {
        if (!geometry.valid(center) || !geometry.valid(tap))
        {
            return 0.0f;
        }
        return geometry.weight(center, tap);
    }

    int2 extent()
    {
        return geometry.extent();
    }
}

struct TrainLatents : IRtLabNeuralLatents
{
    DiffTensor<float, 3> latent;   // [H, W, 8]
    DiffTensor<float, 3> affinity; // [H, W, 8] raw network outputs [8, 16)

    [Differentiable]
    float[RT_LAB_NEURAL_AFFINITY_VALUES] load_affinity_raw(int2 p)
    {
        float r[RT_LAB_NEURAL_AFFINITY_VALUES];
        [ForceUnroll]
        for (int k = 0; k < RT_LAB_NEURAL_AFFINITY_VALUES; ++k)
        {
            r[k] = affinity[uint3(uint(k), uint(p.x), uint(p.y))];
        }
        return r;
    }

    [Differentiable]
    float[RT_LAB_NEURAL_LATENT] load(int2 p)
    {
        float r[RT_LAB_NEURAL_LATENT];
        [ForceUnroll]
        for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
        {
            r[k] = latent[uint3(uint(k), uint(p.x), uint(p.y))];
        }
        return r;
    }
}

// The neural a-trous pass input: color + effective count [H, W, 4] (rgb gradients flow through the pass chain) and the PRECOMPUTED
// geometric edge stopping of the pass: geo_table [H, W, 25] holds plane distance x normal for the 25 taps of the 5x5 window at the
// pass step (index (dy + 2) * 5 + (dx + 2), 0 for invalid taps). It does not depend on the parameters, so it is computed once per crop
// (geometry_table) instead of in the forward and again in the backward of every pass.
struct TrainAtrousSource : IRtLabNeuralAtrousSource
{
    Tensor<float, 3> depth;
    Tensor<float, 3> geo_table;
    DiffTensor<float, 3> color_count_t;

    [Differentiable]
    float4 color_count(int2 p)
    {
        return float4(color_count_t[uint3(0, p.x, p.y)], color_count_t[uint3(1, p.x, p.y)], color_count_t[uint3(2, p.x, p.y)], color_count_t[uint3(3, p.x, p.y)]);
    }

    float2 split(int2 p)
    {
        return float2(0.0f);
    }

    bool geometry_valid(int2 p)
    {
        return depth[uint3(0, p.x, p.y)] > 0.0f;
    }

    float geometry_weight(int2 center, int2 tap, int tap_index)
    {
        return geo_table[uint3(uint(tap_index), uint(center.x), uint(center.y))];
    }

    int2 extent()
    {
        return int2(depth.shape[1], depth.shape[0]);
    }
}

// Baseline / ground truth cleanup source (NOT part of the neural filter): geometry x the lab's variance guide
// (rt_lab_variance_guide_likelihood), folded into the edge stopping weight.
struct TrainGuideSource : IRtLabNeuralAtrousSource
{
    TrainGeometry geometry;
    DiffTensor<float, 3> color_count_t;
    Tensor<float, 3> variance; // [H, W, 4] per sample variance, rgb
    float exposure;
    float guide_sigma;

    [Differentiable]
    float4 color_count(int2 p)
    {
        return float4(color_count_t[uint3(0, p.x, p.y)], color_count_t[uint3(1, p.x, p.y)], color_count_t[uint3(2, p.x, p.y)], color_count_t[uint3(3, p.x, p.y)]);
    }

    float2 split(int2 p)
    {
        return float2(0.0f);
    }

    bool geometry_valid(int2 p)
    {
        return geometry.valid(p);
    }

    float geometry_weight(int2 center, int2 tap, int tap_index)
    {
        const float4 c = color_count(center);
        const float4 t = color_count(tap);
        const float luma_floor = 0.1f * TRAIN_RELATIVE_FLOOR * RT_LAB_NEURAL_RADIANCE_SCALE / (RT_LAB_NEURAL_PI * exposure);
        const float3 floor_v = float3(luma_floor * luma_floor);
        const float3 vc = max(float3(variance[uint3(0, center.x, center.y)], variance[uint3(1, center.x, center.y)], variance[uint3(2, center.x, center.y)]), 0.0f) / max(c.w, 1e-3f);
        const float3 vt = max(float3(variance[uint3(0, tap.x, tap.y)], variance[uint3(1, tap.x, tap.y)], variance[uint3(2, tap.x, tap.y)]), 0.0f) / max(t.w, 1e-3f);
        const float3 diff = c.xyz - t.xyz;
        const float3 z_sq = diff * diff / (vc + vt + floor_v);
        return geometry.weight(center, tap) * exp(-dot(z_sq, float3(1.0f / 3.0f)) / (2.0f * guide_sigma * guide_sigma));
    }

    int2 extent()
    {
        return geometry.extent();
    }
}

// Latents of the filters that do not use any (use_latent = false never loads them).
struct TrainNoLatents : IRtLabNeuralLatents
{
    [Differentiable]
    float[RT_LAB_NEURAL_AFFINITY_VALUES] load_affinity_raw(int2 p)
    {
        float r[RT_LAB_NEURAL_AFFINITY_VALUES];
        [ForceUnroll]
        for (int k = 0; k < RT_LAB_NEURAL_AFFINITY_VALUES; ++k)
        {
            r[k] = 0.0f;
        }
        return r;
    }

    [Differentiable]
    float[RT_LAB_NEURAL_LATENT] load(int2 p)
    {
        float r[RT_LAB_NEURAL_LATENT];
        [ForceUnroll]
        for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
        {
            r[k] = 0.0f;
        }
        return r;
    }
}

// Latent of this frame from the frame's lab data and the previous latent (backprop through time through latent_prev).
// Flat tensor arguments (SlangPy maps each torch tensor to one parameter).
[Differentiable]
float[RT_LAB_NEURAL_OUT] latent_step(uint2 pix, DiffTensor<float, 1> params, Tensor<float, 3> g_position, Tensor<float, 3> g_normal, Tensor<float, 3> g_rcp_width,
    Tensor<float, 3> g_depth, Tensor<float, 3> g_detail_normal, Tensor<float, 3> g_albedo, Tensor<float, 3> f_acc, Tensor<float, 3> f_raw,
    Tensor<float, 3> f_variance, Tensor<float, 3> f_ray, Tensor<float, 3> f_ray_stats, Tensor<float, 3> f_sh_y, Tensor<float, 3> f_cocg, Tensor<float, 3> f_sq, float exposure,
    DiffTensor<float, 3> latent_prev,
    Tensor<float, 1> input_mask)
{
    TrainGeometry geometry = { g_position, g_normal, g_rcp_width, g_depth, g_detail_normal, g_albedo };
    TrainFrame frame = { f_acc, f_raw, f_variance, f_ray, f_ray_stats, f_sh_y, f_cocg, f_sq, exposure };
    TrainParams p = { params };
    TrainPixels pixels = { geometry, frame };
    TrainLatents previous = { latent_prev, latent_prev }; // only load() is used
    float x[RT_LAB_NEURAL_IN];
    float prev[RT_LAB_NEURAL_LATENT] = previous.load(int2(pix));
    rt_lab_neural_features<TrainPixels, TrainParams>(pixels, p, int2(pix), exposure, prev, x);
    // Input groups can be zeroed (ablation of a trained network): the mask is all ones in training.
    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_IN; ++k)
    {
        x[k] = x[k] * RT_H(no_diff train_input_mask(input_mask, k));
    }
    float result[RT_LAB_NEURAL_OUT];
    rt_lab_neural_mlp<TrainParams>(p, x, result);
    if (!no_diff geometry.valid(int2(pix)))
    {
        [ForceUnroll]
        for (int k = 0; k < RT_LAB_NEURAL_OUT; ++k)
        {
            result[k] = 0.0f;
        }
    }
    return result;
}

// Constant read helper (the mask carries no gradient).
float train_input_mask(Tensor<float, 1> input_mask, int k)
{
    return input_mask[uint(k)];
}

// Geometric edge stopping (plane distance x normal) of the 25 taps of the 5x5 window at `step` for every pixel: [H, W, 25],
// index (dy + 2) * 5 + (dx + 2); 0 for the sky, taps outside the image and sky taps.
float[25] geometry_table(uint2 pix, int step, Tensor<float, 3> g_position, Tensor<float, 3> g_normal, Tensor<float, 3> g_rcp_width, Tensor<float, 3> g_depth)
{
    TrainGeometry geometry = { g_position, g_normal, g_rcp_width, g_depth, g_normal, g_normal }; // detail normal / albedo unused
    float table[25];
    [ForceUnroll]
    for (int i = 0; i < 25; ++i)
    {
        table[i] = 0.0f;
    }
    const int2 center = int2(pix);
    const int2 extent = geometry.extent();
    if (!geometry.valid(center))
    {
        return table;
    }
    [ForceUnroll]
    for (int y = -2; y <= 2; ++y)
    {
        [ForceUnroll]
        for (int x = -2; x <= 2; ++x)
        {
            const int2 tap = center + int2(x, y) * step;
            if (any(tap < 0) || any(tap >= extent) || !geometry.valid(tap))
            {
                continue;
            }
            table[(y + 2) * 5 + (x + 2)] = geometry.weight(center, tap);
        }
    }
    return table;
}

// One neural a-trous pass (rgb + effective count out, [H, W, 4]). use_latent = false: the geometry only baseline.
[Differentiable]
float4 atrous_pass(uint2 pix, int step, float target, bool use_latent, Tensor<float, 3> g_depth, Tensor<float, 3> g_geo_table, DiffTensor<float, 3> color_in,
    DiffTensor<float, 3> latent, DiffTensor<float, 3> affinity)
{
    TrainAtrousSource source = { g_depth, g_geo_table, color_in };
    TrainLatents latents = { latent, affinity };
    const int2 center = int2(pix);
    if (!no_diff source.geometry_valid(center))
    {
        return float4(0.0f);
    }
    RtLabNeuralAtrousResult r = rt_lab_neural_atrous<TrainAtrousSource, TrainLatents>(source, latents, center, step, target, use_latent);
    const float rcp_weight = r.weight > 0.0f ? 1.0f / r.weight : 0.0f;
    return float4(r.color * rcp_weight, r.count);
}

// One pass of the variance guide filter (geometry x the lab's variance guide): the evaluation baseline and the ground truth cleanup.
[Differentiable]
float4 guide_pass(uint2 pix, int step, float target, Tensor<float, 3> g_position, Tensor<float, 3> g_normal, Tensor<float, 3> g_rcp_width,
    Tensor<float, 3> g_depth, DiffTensor<float, 3> color_in, Tensor<float, 3> variance, float exposure, float guide_sigma)
{
    TrainGeometry geometry = { g_position, g_normal, g_rcp_width, g_depth, g_normal, g_normal }; // detail normal / albedo unused
    TrainGuideSource source = { geometry, color_in, variance, exposure, guide_sigma };
    TrainNoLatents latents = {};
    const int2 center = int2(pix);
    if (!no_diff geometry.valid(center))
    {
        return float4(0.0f);
    }
    RtLabNeuralAtrousResult r = rt_lab_neural_atrous<TrainGuideSource, TrainNoLatents>(source, latents, center, step, target, false);
    const float rcp_weight = r.weight > 0.0f ? 1.0f / r.weight : 0.0f;
    return float4(r.color * rcp_weight, r.count);
}

// One pass of the PROGRESSIVE preset's spatial filter (rt_lab_blur.hlsl entry_rt_lab_blur, the separable gaussian): taps d = 1 .. radius on
// both sides along `direction`, weight = exp(-d^2 / (2 sigma^2)), sigma = radius / 2, x plane distance x normal x the variance guide. The
// center always counts (weight 1) and the effective count is the sum of weight x tap count. Used for the ground truth cleanup (convert_full.py).
[Differentiable]
float4 gauss_guide_pass(uint2 pix, int2 direction, int radius, Tensor<float, 3> g_position, Tensor<float, 3> g_normal, Tensor<float, 3> g_rcp_width,
    Tensor<float, 3> g_depth, DiffTensor<float, 3> color_in, Tensor<float, 3> variance, float exposure, float guide_sigma)
{
    TrainGeometry geometry = { g_position, g_normal, g_rcp_width, g_depth, g_normal, g_normal }; // detail normal / albedo unused
    TrainGuideSource source = { geometry, color_in, variance, exposure, guide_sigma };
    const int2 center = int2(pix);
    if (!no_diff geometry.valid(center))
    {
        return float4(0.0f);
    }
    const int2 extent = geometry.extent();
    const float4 center_value = source.color_count(center);
    const float center_weight = no_diff rt_lab_neural_constant(center_value.w) > 0.0f ? 1.0f : 0.0f;
    float3 color = center_value.xyz * center_weight;
    float weight = center_weight;
    float count = no_diff rt_lab_neural_constant(center_value.w);
    const float sigma = max(float(radius) * 0.5f, 0.5f);
    const float rcp_two_sigma_sq = 1.0f / (2.0f * sigma * sigma);
    [MaxIters(32)]
    for (int d = 1; d <= radius; ++d)
    {
        const float kernel = exp(-float(d * d) * rcp_two_sigma_sq);
        [ForceUnroll]
        for (int side = -1; side <= 1; side += 2)
        {
            const int2 tap = center + direction * (d * side);
            if (any(tap < 0) || any(tap >= extent) || !no_diff geometry.valid(tap))
            {
                continue;
            }
            const float4 tap_value = source.color_count(tap);
            const float tap_count = no_diff rt_lab_neural_constant(tap_value.w);
            if (tap_count <= 0.0f)
            {
                continue;
            }
            const float w = kernel * no_diff source.geometry_weight(center, tap, 0); // geometry x variance guide
            if (w <= 1e-6f)
            {
                continue;
            }
            color += tap_value.xyz * w;
            weight += w;
            count += tap_count * no_diff rt_lab_neural_constant(w);
        }
    }
    return float4(weight > 0.0f ? color / weight : float3(0.0f), count);
}
