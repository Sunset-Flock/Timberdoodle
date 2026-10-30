#pragma once

// RT Lab neural affinity: the part of the neural denoiser that is SHARED between the engine and the trainer.
//
// Idea: a small MLP predicts, per pixel, a latent feature vector f (8 values) from the temporally accumulated radiance, the
// frame's new samples, the ray length guide, geometry and (optionally) its own latent of the previous frame. The spatial
// a-trous filter weights every tap with w(|f_center - f_tap|^2) on top of its geometric tests. The network ALSO picks the
// weighting function w: it outputs a gate per candidate function (softmax) and a sharpness, and the pixel being filtered
// blends the candidates with its own gates (gaussian, exponential, cauchy, student-t, compact support, soft step). The
// learned weight replaces the lab's guides (variance, ray length, SH): only the geometric edge stopping stays. The filter stays a normalized convex combination of the accumulated
// data, so the network can only change WHICH taps are averaged, never invent energy.
//
// This file has no engine includes and no daxa types on purpose: the trainer (neural/train, SlangPy) differentiates
// exactly this code with Slang autodiff, the engine (rt_lab_neural.hlsl) includes it and supplies textures through the
// small interfaces below. Compiles with the engine's Slang and with SlangPy's newer one.
//
// AUTODIFF TRAP: loops that write local arrays MUST be [ForceUnroll] inside [Differentiable] code, otherwise the
// backward pass silently returns wrong gradients for everything upstream of the array (measured: first layer of a
// 2 layer MLP off by 40 %, last layer exact).

// Feature set (compile time, selected per pipeline / per trainer run; the weights file must match, see rt_lab_neural.cpp):
//   0: 28 inputs: per pixel data + 3x3 neighborhood statistics (IN 32)
//   1: 5x5 window: the raw log-luminance of the 24 neighbors relative to the center instead of the 3x3 statistics (IN 64)
#ifndef RT_LAB_NEURAL_WINDOW
#define RT_LAB_NEURAL_WINDOW 0
#endif
// 1 = no input derived from the lab's variance / ray length guides (the noise estimate [6] and the center ray length [7, 9] stay 0).
#ifndef RT_LAB_NEURAL_NO_GUIDE_INPUTS
#define RT_LAB_NEURAL_NO_GUIDE_INPUTS 0
#endif
// 1 = additionally the ray length guide (ao) of every pixel of the 3x3 neighborhood as input [32, 41) (IN 48, feature set 0 only).
#ifndef RT_LAB_NEURAL_RAYLENGTH3
#define RT_LAB_NEURAL_RAYLENGTH3 0
#endif
// 1 = raw 3x3 data feature set: for each of the 9 pixels of the 3x3 neighborhood 19 plain values (RT_LAB_NEURAL_RAW3_PER_PIXEL: valid,
// depth, face normal, detail normal, albedo, radiance rgb, perceptual radiance rgb, ray length, capped ray length), no hand made
// statistics; plus the center's history length, absolute depth, the previous latent and a bias (IN 192).
#ifndef RT_LAB_NEURAL_RAW3
#define RT_LAB_NEURAL_RAW3 0
#endif
#define RT_LAB_NEURAL_RAW3_PER_PIXEL 19
// 1 = lite raw data feature set: only the plus shaped neighborhood (center, left, right, up, down), 5 plain values per pixel (relative
// depth with sky encoded, perceptual radiance rgb, capped normalized ray length) + the center's detail normal, history length, absolute
// depth, previous latent and a bias (IN 40). What the filter's own geometric edge stopping already knows (face normal, depth planes)
// and what irradiance does not depend on (albedo) is left out. Needs the same per pixel data as the raw 3x3 set.
#ifndef RT_LAB_NEURAL_LITE
#define RT_LAB_NEURAL_LITE 0
#endif
// 1 = the SH feature set (v2): the 3x3 footprint, every pixel of it described by 12 values (the temporal SH-Y relative to its mean, the log mean Y,
// the CoCg chroma ratios, the second moment ratios of Y / Co / Cg, the capped ray length, the geometric weight against the center pixel and the cosine of
// its light direction with the center's detail normal) and encoded by ONE shared small network (12 -> 6 leaky ReLU, parameters at the END of the parameter
// vector); the MLP sees the 9 encodings in a fixed order (center, left, right, up, down, top left, top right, bottom left, bottom right: 54 inputs), the
// square root of the history length, the previous latent and a constant 1 (IN 64). No depth, no radiance, no normal of the neighbors.
#ifndef RT_LAB_NEURAL_V2
#define RT_LAB_NEURAL_V2 0
#endif
#define RT_LAB_NEURAL_V2_PIXEL_INPUTS 12
#define RT_LAB_NEURAL_V2_ENC 6
#if RT_LAB_NEURAL_V2
#define RT_LAB_NEURAL_IN 64
#elif RT_LAB_NEURAL_LITE
#define RT_LAB_NEURAL_IN 40
#elif RT_LAB_NEURAL_RAW3
#define RT_LAB_NEURAL_IN 192
#elif RT_LAB_NEURAL_WINDOW
#define RT_LAB_NEURAL_IN 64
#elif RT_LAB_NEURAL_RAYLENGTH3
#define RT_LAB_NEURAL_IN 48
#else
#define RT_LAB_NEURAL_IN 32
#endif

// Precision of the network math (MLP, latents, affinity, weighting functions): the engine's fast variant defines
// RT_LAB_NEURAL_HALF (f16), the trainer keeps f32 (autodiff). Radiometric work (radiance, exposure, logs) always stays f32:
// radiance x exposure overflows f16.
#ifdef RT_LAB_NEURAL_HALF
typealias rt_real = half;
#else
typealias rt_real = float;
#endif
#define RT_H(x) rt_real(x)
#define RT_LAB_NEURAL_HID 32
#define RT_LAB_NEURAL_LATENT 8
#define RT_LAB_NEURAL_GATES 6
// Network outputs: [0, 8) latent (tanh), [8, 14) weighting function gate logits, [14] log sharpness, [15] reserved.
#define RT_LAB_NEURAL_OUT 16
// 1 = variance guide style weighting (feature bit 128, "w2"), same output layout. The latent vector carries the noise estimate itself: dimensions 0..3 are mean
// features mu_k, dimensions 4..7 log standard deviations s_k (sigma_k = exp(RT_LAB_NEURAL_SIGMA_BIAS + RT_LAB_NEURAL_SIGMA_GAIN s_k)). A tap is tested like the lab's
// variance guide tests two estimates of a mean: z_k^2 = (mu_c,k - mu_t,k)^2 / (sigma_c,k^2 + sigma_t,k^2 + floor), both pixels contribute their own predicted noise.
// The weighting functions act on S = mean z_k^2, P = max z_k^2 (the worst dimension) and T = trimmed mean (all but the worst dimension), in units of kappa^2
// (the network's tolerance in sigmas, the old sharpness output): multi dimensional rejection that the latent vector controls.
#ifndef RT_LAB_NEURAL_WEIGHTING_V2
#define RT_LAB_NEURAL_WEIGHTING_V2 0
#endif
#define RT_LAB_NEURAL_SIGMA_BIAS (-1.6094379f) // ln 0.2: sigma 0.2 at s = 0
#define RT_LAB_NEURAL_SIGMA_GAIN 1.6094379f    // sigma 0.04 .. 1.0 for s = -1 .. 1
#define RT_LAB_NEURAL_SIGMA_FLOOR 1e-4f        // added to the variance sum of the test
#define RT_LAB_NEURAL_AFFINITY_VALUES (RT_LAB_NEURAL_OUT - RT_LAB_NEURAL_LATENT) // raw outputs the filter needs from the center pixel

// Parameter layout (floats, row major [out][in]): W1, b1, W2, b2, W3, b3.
#define RT_LAB_NEURAL_W1 0
#define RT_LAB_NEURAL_B1 (RT_LAB_NEURAL_W1 + RT_LAB_NEURAL_HID * RT_LAB_NEURAL_IN)
#define RT_LAB_NEURAL_W2 (RT_LAB_NEURAL_B1 + RT_LAB_NEURAL_HID)
#define RT_LAB_NEURAL_B2 (RT_LAB_NEURAL_W2 + RT_LAB_NEURAL_HID * RT_LAB_NEURAL_HID)
#define RT_LAB_NEURAL_W3 (RT_LAB_NEURAL_B2 + RT_LAB_NEURAL_HID)
#define RT_LAB_NEURAL_B3 (RT_LAB_NEURAL_W3 + RT_LAB_NEURAL_OUT * RT_LAB_NEURAL_HID)
#if RT_LAB_NEURAL_V2
// The shared pixel encoder follows the MLP (the MLP offsets keep their alignment for the cooperative vector path): [6][12] weights, [6] biases.
#define RT_LAB_NEURAL_ENC_W (RT_LAB_NEURAL_B3 + RT_LAB_NEURAL_OUT)
#define RT_LAB_NEURAL_ENC_B (RT_LAB_NEURAL_ENC_W + RT_LAB_NEURAL_V2_ENC * RT_LAB_NEURAL_V2_PIXEL_INPUTS)
#define RT_LAB_NEURAL_PARAM_COUNT (RT_LAB_NEURAL_ENC_B + RT_LAB_NEURAL_V2_ENC)
#else
#define RT_LAB_NEURAL_PARAM_COUNT (RT_LAB_NEURAL_B3 + RT_LAB_NEURAL_OUT)
#endif

// Sharpness s = RT_LAB_NEURAL_AFFINITY_SCALE x exp(clamp(raw, +-3)); latents are tanh bounded to [-1, 1].
#define RT_LAB_NEURAL_AFFINITY_SCALE 4.0f
#define RT_LAB_NEURAL_LEAK 0.05f

// Must equal RTGI_RADIANCE_SCALE (rtgi_shared.hlsl): the lab stores radiance x this. The engine side static asserts it.
#define RT_LAB_NEURAL_RADIANCE_SCALE 10000.0f
#define RT_LAB_NEURAL_PI 3.14159265358979f

// ---------------------------------------------------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------------------------------------------------

interface IRtLabNeuralParams
{
    [Differentiable]
    float get(int index);
}

// One lab pixel as the lab holds it after the temporal pass (all radiance in RTGI_RADIANCE_SCALE units).
struct RtLabNeuralPixel
{
    float3 acc;       // temporally accumulated radiance
    float acc_length; // its history length (samples)
    float3 raw;       // this frame's mean sample
    float raw_count;  // rays of this frame
    float3 variance;  // accumulated per sample rgb variance
    float ray_ao;     // ray length guide (1 - sqrt(mean shortness)), < 0 = none
    float ray_shortness;
    float depth;      // linear view depth, 0 = sky
    float3 normal;    // face normal
    // Raw 3x3 feature set only (0 where the source has no data):
    float3 detail_normal;    // normal map applied shading normal
    float3 albedo;           // linear albedo (half res g-buffer sampled at the pixel)
    float ray_log_distance;  // mean over the pixel's rays (accumulated over frames) of log2(1 + min(hit distance, 1000 m))
    float ray_capped;        // mean of min(hit distance, 32 pixel widths) / (32 pixel widths), pixel width = world size of the pixel at its depth
    // SH feature set (v2) only (0 where the source has no data):
    float4 sh_y;             // temporal SH-Y: xyz = mean of direction x Y, w = mean Y (RTGI_RADIANCE_SCALE units)
    float2 cocg;             // temporal mean Co, Cg
    float3 sq;               // mean of Y^2, Co^2, Cg^2 (the second moments; the first moments are sh_y.w and cocg)
};

interface IRtLabNeuralPixels
{
    RtLabNeuralPixel load(int2 pixel);
    int2 extent();
    // Geometric edge stopping (plane distance x normal) between the center pixel and a pixel of its 3x3 footprint, 0 = a different surface or sky (v2).
    float footprint_geometry_weight(int2 center, int2 tap);
}

// Weighting function selection of a pixel (decoded from the network outputs [8, 16)).
struct RtLabNeuralAffinity : IDifferentiable
{
    vector<rt_real, 4> gates_a; // softmax gates of weighting functions 0..3
    vector<rt_real, 2> gates_b; // ... 4..5
    rt_real sharpness;
}

// A latent image (previous frame reprojected for the feature builder, current frame for the filter).
interface IRtLabNeuralLatents
{
    [Differentiable]
    rt_real[RT_LAB_NEURAL_LATENT] load(int2 pixel);
    // The raw affinity outputs of a pixel (network outputs [8, 16)): decoded with rt_lab_neural_decode_affinity.
    [Differentiable]
    rt_real[RT_LAB_NEURAL_AFFINITY_VALUES] load_affinity_raw(int2 pixel);
}

// Spatial filter side of the a-trous pass.
interface IRtLabNeuralAtrousSource
{
    // xyz = color, w = effective sample count (0 = the pixel holds no samples). Differentiable: in training the colors are
    // the previous pass's output, gradients flow back through the whole pass chain. The count channel is used as a constant.
    [Differentiable]
    float4 color_count(int2 pixel);
    float2 split(int2 pixel);
    bool geometry_valid(int2 pixel); // false = sky
    // Edge stopping between two pixels (plane distance x normal), 0 = reject. Includes nothing else: no variance / ray length /
    // SH guide of the lab enters the neural filter. tap_index = (dy + 2) * 5 + (dx + 2) in units of the pass step, for sources that
    // precompute the weights of a window (the trainer; the engine ignores it).
    float geometry_weight(int2 center, int2 tap, int tap_index);
    int2 extent();
}

// ---------------------------------------------------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------------------------------------------------

// Identity that blocks the gradient: no_diff only decorates calls, so a constant copy of a value goes through this.
float rt_lab_neural_constant(float v)
{
    return v;
}

[Differentiable]
rt_real rt_lab_neural_leaky(rt_real v)
{
    return v > RT_H(0.0f) ? v : v * RT_H(RT_LAB_NEURAL_LEAK);
}

#ifndef RT_LAB_NEURAL_EXTERNAL_MLP
// Reference MLP: 32 -> 32 -> 32 -> 16 (first 8 tanh = latent, the rest raw affinity outputs). The engine has a cooperative vector version.
[Differentiable]
void rt_lab_neural_mlp<P : IRtLabNeuralParams>(P params, rt_real x[RT_LAB_NEURAL_IN], out rt_real result[RT_LAB_NEURAL_OUT])
{
    rt_real h1[RT_LAB_NEURAL_HID];
    [ForceUnroll]
    for (int o = 0; o < RT_LAB_NEURAL_HID; ++o)
    {
        rt_real a = RT_H(params.get(RT_LAB_NEURAL_B1 + o));
        [ForceUnroll]
        for (int i = 0; i < RT_LAB_NEURAL_IN; ++i)
        {
            a += RT_H(params.get(RT_LAB_NEURAL_W1 + o * RT_LAB_NEURAL_IN + i)) * x[i];
        }
        h1[o] = rt_lab_neural_leaky(a);
    }
    rt_real h2[RT_LAB_NEURAL_HID];
    [ForceUnroll]
    for (int o = 0; o < RT_LAB_NEURAL_HID; ++o)
    {
        rt_real a = RT_H(params.get(RT_LAB_NEURAL_B2 + o));
        [ForceUnroll]
        for (int i = 0; i < RT_LAB_NEURAL_HID; ++i)
        {
            a += RT_H(params.get(RT_LAB_NEURAL_W2 + o * RT_LAB_NEURAL_HID + i)) * h1[i];
        }
        h2[o] = rt_lab_neural_leaky(a);
    }
    [ForceUnroll]
    for (int o = 0; o < RT_LAB_NEURAL_OUT; ++o)
    {
        rt_real a = RT_H(params.get(RT_LAB_NEURAL_B3 + o));
        [ForceUnroll]
        for (int i = 0; i < RT_LAB_NEURAL_HID; ++i)
        {
            a += RT_H(params.get(RT_LAB_NEURAL_W3 + o * RT_LAB_NEURAL_HID + i)) * h2[i];
        }
        result[o] = o < RT_LAB_NEURAL_LATENT ? tanh(a) : a;
    }
}
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Features
// ---------------------------------------------------------------------------------------------------------------------

[Differentiable]
float rt_lab_neural_luma(float3 v)
{
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

// Log of the exposed luminance (display white ~ 1), the perceptual axis everything is compared on, scaled to ~[-1, 1].
func rt_lab_neural_log_luma(float3 scaled_radiance, float exposure) -> float
{
    const float exposed = rt_lab_neural_luma(scaled_radiance) * RT_LAB_NEURAL_PI * exposure / RT_LAB_NEURAL_RADIANCE_SCALE;
    return (log(max(exposed, 1e-4f)) + 4.0f) * 0.25f;
}

func rt_lab_neural_chroma(float3 scaled_radiance) -> float2
{
    const float l = max(rt_lab_neural_luma(scaled_radiance), 1e-6f);
    return clamp(log2(max(float2(scaled_radiance.r, scaled_radiance.b), 1e-6f) / l), -3.0f, 3.0f) * (1.0f / 3.0f);
}

// Network input of one pixel. Feature set 0: the pixel's data, 3x3 neighborhood statistics and the previous latent. Feature set 1
// (RT_LAB_NEURAL_WINDOW): the same pixel data, but the raw 5x5 window of log luminances relative to the center instead of the
// statistics. Constant data (pixels) carries no gradient, only the previous latent does (backprop through time).
#if RT_LAB_NEURAL_V2
// The 12 inputs of the shared pixel encoder for one pixel of the 3x3 footprint (constant data, no gradient).
struct RtLabNeuralV2PixelInputs
{
    float v[RT_LAB_NEURAL_V2_PIXEL_INPUTS];
};

func rt_lab_neural_v2_pixel_inputs(RtLabNeuralPixel t, float geometry, float3 center_detail_normal, float exposure) -> RtLabNeuralV2PixelInputs
{
    RtLabNeuralV2PixelInputs r;
    [ForceUnroll]
    for (int i = 0; i < RT_LAB_NEURAL_V2_PIXEL_INPUTS; ++i)
    {
        r.v[i] = 0.0f;
    }
    if (!(t.depth > 0.0f))
    {
        return r; // sky: all zeros (geometry weight 0)
    }
    const float radiance_to_exposed = RT_LAB_NEURAL_PI * exposure / RT_LAB_NEURAL_RADIANCE_SCALE;
    const float y_floor = 1e-3f / radiance_to_exposed; // 0.1 % of display white in Y units
    const float y = max(t.sh_y.w, 0.0f);
    const float3 direction = y > y_floor ? clamp(t.sh_y.xyz / y, float3(-1.0f), float3(1.0f)) : float3(0.0f); // first ring relative to the mean
    r.v[0] = direction.x;
    r.v[1] = direction.y;
    r.v[2] = direction.z;
    r.v[3] = (log(max(y * radiance_to_exposed, 1e-4f)) + 4.0f) * 0.25f;
    r.v[4] = clamp(t.cocg.x / max(y, y_floor), -2.0f, 2.0f) * 0.5f;
    r.v[5] = clamp(t.cocg.y / max(y, y_floor), -2.0f, 2.0f) * 0.5f;
    // Second moments as the ratio E[x^2] / E[x]^2 (>= 1, floored): the network gets the mean and the ratio, so it forms the variance (ratio - 1) x mean^2 itself.
    const float y_floor_sq = y_floor * y_floor;
    r.v[6] = clamp(log2(max((t.sq.x + y_floor_sq) / (y * y + y_floor_sq), 1.0f)) * 0.25f, 0.0f, 2.0f);
    r.v[7] = clamp(log2(max((t.sq.y + y_floor_sq) / (t.cocg.x * t.cocg.x + y_floor_sq), 1.0f)) * 0.25f, 0.0f, 2.0f);
    r.v[8] = clamp(log2(max((t.sq.z + y_floor_sq) / (t.cocg.y * t.cocg.y + y_floor_sq), 1.0f)) * 0.25f, 0.0f, 2.0f);
    r.v[9] = t.ray_capped;
    r.v[10] = geometry;
    r.v[11] = clamp(dot(direction, center_detail_normal), -1.0f, 1.0f);
    return r;
}

// The shared pixel encoder: 12 -> 6, leaky ReLU. Parameters at RT_LAB_NEURAL_ENC_W / _B.
[Differentiable]
void rt_lab_neural_v2_encode<P : IRtLabNeuralParams>(P params, RtLabNeuralV2PixelInputs inputs, out rt_real e[RT_LAB_NEURAL_V2_ENC])
{
    [ForceUnroll]
    for (int o = 0; o < RT_LAB_NEURAL_V2_ENC; ++o)
    {
        rt_real a = RT_H(params.get(RT_LAB_NEURAL_ENC_B + o));
        [ForceUnroll]
        for (int i = 0; i < RT_LAB_NEURAL_V2_PIXEL_INPUTS; ++i)
        {
            a += RT_H(params.get(RT_LAB_NEURAL_ENC_W + o * RT_LAB_NEURAL_V2_PIXEL_INPUTS + i)) * RT_H(inputs.v[i]);
        }
        e[o] = rt_lab_neural_leaky(a);
    }
}
#endif

[Differentiable]
void rt_lab_neural_features<S : IRtLabNeuralPixels, P : IRtLabNeuralParams>(S pixels, P params, int2 pixel, float exposure, rt_real previous_latent[RT_LAB_NEURAL_LATENT],
    out rt_real x[RT_LAB_NEURAL_IN])
{
    const RtLabNeuralPixel center = no_diff pixels.load(pixel);
    const int2 extent = no_diff pixels.extent();
    const float la = no_diff rt_lab_neural_log_luma(center.acc, exposure);
    const float2 chroma = no_diff rt_lab_neural_chroma(center.acc);
    const float luma = rt_lab_neural_luma(center.acc);
    const float rel_std = sqrt(max(rt_lab_neural_luma(center.variance), 0.0f) / max(center.acc_length, 1.0f)) / max(luma, 1e-3f * RT_LAB_NEURAL_RADIANCE_SCALE / (RT_LAB_NEURAL_PI * exposure));
    const bool has_ao = center.ray_ao >= 0.0f;

    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_IN; ++k)
    {
        x[k] = RT_H(0.0f);
    }
#if RT_LAB_NEURAL_V2
    // [0, 54): the 9 pixels of the 3x3 footprint, each through the shared encoder (6 values): center, left, right, up, down, top left, top right,
    // bottom left, bottom right. [54] sqrt(history length) / 8, [55, 63) previous latent, [63] constant 1.
    [ForceUnroll]
    for (int p = 0; p < 9; ++p)
    {
        const int2 offset = int2(p == 1 || p == 5 || p == 7 ? -1 : (p == 2 || p == 6 || p == 8 ? 1 : 0), p == 3 || p == 5 || p == 6 ? -1 : (p == 4 || p == 7 || p == 8 ? 1 : 0));
        const int2 tap = pixel + offset;
        RtLabNeuralV2PixelInputs inputs;
        [ForceUnroll]
        for (int i = 0; i < RT_LAB_NEURAL_V2_PIXEL_INPUTS; ++i)
        {
            inputs.v[i] = 0.0f;
        }
        if (!(any(tap < 0) || any(tap >= extent)))
        {
            RtLabNeuralPixel t = center;
            float geometry = 1.0f;
            if (p != 0)
            {
                t = no_diff pixels.load(tap);
                geometry = no_diff pixels.footprint_geometry_weight(pixel, tap);
            }
            inputs = no_diff rt_lab_neural_v2_pixel_inputs(t, geometry, center.detail_normal, exposure);
        }
        rt_real e[RT_LAB_NEURAL_V2_ENC];
        rt_lab_neural_v2_encode<P>(params, inputs, e);
        [ForceUnroll]
        for (int c = 0; c < RT_LAB_NEURAL_V2_ENC; ++c)
        {
            x[p * RT_LAB_NEURAL_V2_ENC + c] = e[c];
        }
    }
    x[54] = RT_H(sqrt(max(center.acc_length, 0.0f)) * 0.125f);
    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        x[55 + k] = previous_latent[k];
    }
    x[63] = RT_H(1.0f);
#elif RT_LAB_NEURAL_LITE
    // [0, 25): the 5 pixels of the plus shaped neighborhood (0 center, 1 left, 2 right, 3 up, 4 down), 5 values each:
    //   +0 (log2(depth) - log2(center depth)) x 8 clamped to +-4, 6 = sky / outside the image,
    //   +1..+3 perceptual radiance (ln(max(e, 1e-4)) + 4) / 4 per channel, e = exposed radiance, +4 capped normalized ray length.
    // [25, 28) detail normal of the center, [28] history length, [29] absolute log depth, [30, 38) previous latent, [38] constant 1.
    const float log_depth_center = log2(max(center.depth, 1e-2f));
    const float radiance_to_exposed = RT_LAB_NEURAL_PI * exposure / RT_LAB_NEURAL_RADIANCE_SCALE;
    [ForceUnroll]
    for (int p = 0; p < 5; ++p)
    {
        const int dx = p == 1 ? -1 : (p == 2 ? 1 : 0);
        const int dy = p == 3 ? -1 : (p == 4 ? 1 : 0);
        const int base = p * 5;
        x[base] = RT_H(6.0f);
        const int2 tap = pixel + int2(dx, dy);
        if (any(tap < 0) || any(tap >= extent))
        {
            continue;
        }
        RtLabNeuralPixel t = center;
        if (p != 0)
        {
            t = no_diff pixels.load(tap);
        }
        if (!(t.depth > 0.0f))
        {
            continue;
        }
        const float3 e = max(t.acc, 0.0f) * radiance_to_exposed;
        x[base + 0] = RT_H(clamp((log2(max(t.depth, 1e-2f)) - log_depth_center) * 8.0f, -4.0f, 4.0f));
        x[base + 1] = RT_H((log(max(e.x, 1e-4f)) + 4.0f) * 0.25f);
        x[base + 2] = RT_H((log(max(e.y, 1e-4f)) + 4.0f) * 0.25f);
        x[base + 3] = RT_H((log(max(e.z, 1e-4f)) + 4.0f) * 0.25f);
        x[base + 4] = RT_H(t.ray_capped);
    }
    x[25] = RT_H(center.detail_normal.x);
    x[26] = RT_H(center.detail_normal.y);
    x[27] = RT_H(center.detail_normal.z);
    x[28] = RT_H(log2(1.0f + center.acc_length) * (1.0f / 9.0f));
    x[29] = RT_H(log_depth_center * (1.0f / 8.0f));
    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        x[30 + k] = previous_latent[k];
    }
    x[38] = RT_H(1.0f);
#elif RT_LAB_NEURAL_RAW3
    // [0, 171): the 9 pixels row major (dy -1..1, dx -1..1, center = pixel 4), 19 values each, sky / outside the image = all 0 (valid 0):
    //   +0 valid, +1 log2(depth) - log2(center depth) (x 8), +2..+4 face normal, +5..+7 detail normal, +8..+10 albedo,
    //   +11..+13 exposed radiance e / (1 + e), +14..+16 perceptual radiance (ln(max(e, 1e-4)) + 4) / 4, +17 ray length log2(1 + t) / 10,
    //   +18 capped normalized ray length.
    // [171] history length, [172] absolute log depth, [173, 181) previous latent, [181] constant 1.
    const float log_depth_center = log2(max(center.depth, 1e-2f));
    const float radiance_to_exposed = RT_LAB_NEURAL_PI * exposure / RT_LAB_NEURAL_RADIANCE_SCALE;
    [ForceUnroll]
    for (int dy = -1; dy <= 1; ++dy)
    {
        [ForceUnroll]
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int base = ((dy + 1) * 3 + (dx + 1)) * RT_LAB_NEURAL_RAW3_PER_PIXEL;
            const int2 tap = pixel + int2(dx, dy);
            if (any(tap < 0) || any(tap >= extent))
            {
                continue;
            }
            RtLabNeuralPixel t = center;
            if (dx != 0 || dy != 0)
            {
                t = no_diff pixels.load(tap);
            }
            if (!(t.depth > 0.0f))
            {
                continue;
            }
            const float3 e = max(t.acc, 0.0f) * radiance_to_exposed;
            x[base + 0] = RT_H(1.0f);
            x[base + 1] = RT_H(clamp((log2(max(t.depth, 1e-2f)) - log_depth_center) * 8.0f, -4.0f, 4.0f));
            x[base + 2] = RT_H(t.normal.x);
            x[base + 3] = RT_H(t.normal.y);
            x[base + 4] = RT_H(t.normal.z);
            x[base + 5] = RT_H(t.detail_normal.x);
            x[base + 6] = RT_H(t.detail_normal.y);
            x[base + 7] = RT_H(t.detail_normal.z);
            x[base + 8] = RT_H(t.albedo.x);
            x[base + 9] = RT_H(t.albedo.y);
            x[base + 10] = RT_H(t.albedo.z);
            x[base + 11] = RT_H(e.x / (1.0f + e.x));
            x[base + 12] = RT_H(e.y / (1.0f + e.y));
            x[base + 13] = RT_H(e.z / (1.0f + e.z));
            x[base + 14] = RT_H((log(max(e.x, 1e-4f)) + 4.0f) * 0.25f);
            x[base + 15] = RT_H((log(max(e.y, 1e-4f)) + 4.0f) * 0.25f);
            x[base + 16] = RT_H((log(max(e.z, 1e-4f)) + 4.0f) * 0.25f);
            x[base + 17] = RT_H(t.ray_log_distance * 0.1f);
            x[base + 18] = RT_H(t.ray_capped);
        }
    }
    x[171] = RT_H(log2(1.0f + center.acc_length) * (1.0f / 9.0f));
    x[172] = RT_H(log_depth_center * (1.0f / 8.0f));
    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        x[173 + k] = previous_latent[k];
    }
    x[181] = RT_H(1.0f);
#elif RT_LAB_NEURAL_WINDOW
    // [0] center log luminance, [1, 25) the 24 neighbors of the 5x5 window minus the center (0 for sky / outside), [25, 39) the
    // pixel data, [39, 47) the previous latent, [47] constant 1.
    x[0] = RT_H(la);
    [ForceUnroll]
    for (int dy = -2; dy <= 2; ++dy)
    {
        [ForceUnroll]
        for (int dx = -2; dx <= 2; ++dx)
        {
            if (dx == 0 && dy == 0)
            {
                continue;
            }
            const int2 tap = pixel + int2(dx, dy);
            float relative = 0.0f;
            if (!(any(tap < 0) || any(tap >= extent)))
            {
                const RtLabNeuralPixel t = no_diff pixels.load(tap);
                if (t.depth > 0.0f)
                {
                    relative = clamp((no_diff rt_lab_neural_log_luma(t.acc, exposure) - la) * 2.0f, -2.0f, 2.0f);
                }
            }
            const int window_index = (dy + 2) * 5 + (dx + 2);
            x[window_index < 12 ? window_index + 1 : window_index] = RT_H(relative); // the center (index 12) is skipped
        }
    }
    x[25] = RT_H(chroma.x);
    x[26] = RT_H(chroma.y);
    x[27] = RT_H(log2(1.0f + center.acc_length) * (1.0f / 9.0f));
    x[28] = RT_H(no_diff rt_lab_neural_log_luma(center.raw, exposure));
    x[29] = RT_H(min(center.raw_count, 4.0f) * 0.25f);
#if !RT_LAB_NEURAL_NO_GUIDE_INPUTS
    x[30] = RT_H(clamp(log2(max(rel_std, 1e-3f)) * (1.0f / 8.0f) + 0.5f, -1.0f, 1.0f));
    x[31] = RT_H(has_ao ? center.ray_ao : 0.0f);
    x[32] = RT_H(has_ao ? center.ray_shortness : 0.0f);
    x[33] = RT_H(has_ao ? 1.0f : 0.0f);
#endif
    x[34] = RT_H(log2(max(center.depth, 1e-2f)) * (1.0f / 8.0f));
    x[35] = RT_H(center.normal.x);
    x[36] = RT_H(center.normal.y);
    x[37] = RT_H(center.normal.z);
    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        x[39 + k] = previous_latent[k];
    }
    x[47] = RT_H(1.0f);
#else
    float mean = 0.0f;
    float mean_sq = 0.0f;
    float taps = 0.0f;
    float low = 1e30f;
    float high = -1e30f;
    float la_x[2] = { la, la };
    float la_y[2] = { la, la };
    [ForceUnroll]
    for (int dy = -1; dy <= 1; ++dy)
    {
        [ForceUnroll]
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int2 tap = pixel + int2(dx, dy);
            if (dx == 0 && dy == 0)
            {
                continue;
            }
            if (any(tap < 0) || any(tap >= extent))
            {
                continue;
            }
            const RtLabNeuralPixel t = no_diff pixels.load(tap);
            if (!(t.depth > 0.0f))
            {
                continue;
            }
            const float lt = no_diff rt_lab_neural_log_luma(t.acc, exposure);
            mean += lt;
            mean_sq += lt * lt;
            taps += 1.0f;
            low = min(low, lt);
            high = max(high, lt);
            if (dy == 0)
            {
                la_x[dx < 0 ? 0 : 1] = lt;
            }
            if (dx == 0)
            {
                la_y[dy < 0 ? 0 : 1] = lt;
            }
        }
    }
    const float rcp_taps = taps > 0.0f ? 1.0f / taps : 0.0f;
    mean *= rcp_taps;
    const float variance = max(mean_sq * rcp_taps - mean * mean, 0.0f);
    if (taps == 0.0f)
    {
        low = la;
        high = la;
        mean = la;
    }
    x[0] = RT_H(la);
    x[1] = RT_H(chroma.x);
    x[2] = RT_H(chroma.y);
    x[3] = RT_H(log2(1.0f + center.acc_length) * (1.0f / 9.0f));
    x[4] = RT_H(no_diff rt_lab_neural_log_luma(center.raw, exposure));
    x[5] = RT_H(min(center.raw_count, 4.0f) * 0.25f);
#if !RT_LAB_NEURAL_NO_GUIDE_INPUTS
    x[6] = RT_H(clamp(log2(max(rel_std, 1e-3f)) * (1.0f / 8.0f) + 0.5f, -1.0f, 1.0f));
    x[7] = RT_H(has_ao ? center.ray_ao : 0.0f);
    x[8] = RT_H(has_ao ? center.ray_shortness : 0.0f);
    x[9] = RT_H(has_ao ? 1.0f : 0.0f);
#endif
    x[10] = RT_H(log2(max(center.depth, 1e-2f)) * (1.0f / 8.0f));
    x[11] = RT_H(center.normal.x);
    x[12] = RT_H(center.normal.y);
    x[13] = RT_H(center.normal.z);
    x[14] = RT_H(clamp((mean - la) * 2.0f, -2.0f, 2.0f));
    x[15] = RT_H(clamp(sqrt(variance) * 4.0f, 0.0f, 2.0f));
    x[16] = RT_H(clamp((high - la) * 2.0f, -2.0f, 2.0f));
    x[17] = RT_H(clamp((low - la) * 2.0f, -2.0f, 2.0f));
    x[18] = RT_H(clamp((la_x[1] - la_x[0]) * 2.0f, -2.0f, 2.0f));
    x[19] = RT_H(clamp((la_y[1] - la_y[0]) * 2.0f, -2.0f, 2.0f));
    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        x[20 + k] = previous_latent[k];
    }
    x[28] = RT_H(1.0f); // constant, lets layer 1 learn a bias per input group
#if RT_LAB_NEURAL_RAYLENGTH3
    // [32, 41): the ray length guide (ao, accumulated over frames) of the 3x3 neighborhood incl. the center, row major, 0 for sky /
    // no estimate / outside the image.
    [ForceUnroll]
    for (int dy = -1; dy <= 1; ++dy)
    {
        [ForceUnroll]
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int2 tap = pixel + int2(dx, dy);
            float ao = 0.0f;
            if (!(any(tap < 0) || any(tap >= extent)))
            {
                const RtLabNeuralPixel t = no_diff pixels.load(tap);
                if (t.depth > 0.0f && t.ray_ao >= 0.0f)
                {
                    ao = t.ray_ao;
                }
            }
            x[32 + (dy + 1) * 3 + (dx + 1)] = RT_H(ao);
        }
    }
#endif
#endif
}

// ---------------------------------------------------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------------------------------------------------

[Differentiable]
rt_real rt_lab_neural_distance_sq(rt_real a[RT_LAB_NEURAL_LATENT], rt_real b[RT_LAB_NEURAL_LATENT])
{
    rt_real d2 = RT_H(0.0f);
    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
    {
        const rt_real d = a[k] - b[k];
        d2 += d * d;
    }
    return d2;
}

// Network outputs [8, 16) -> gates (softmax of the 6 logits) + sharpness. The softmax runs in f32 (logits can be large).
[Differentiable]
RtLabNeuralAffinity rt_lab_neural_decode_affinity(rt_real raw[RT_LAB_NEURAL_AFFINITY_VALUES])
{
    float peak = float(raw[0]);
    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_GATES; ++k)
    {
        peak = max(peak, float(raw[k]));
    }
    float sum = 0.0f;
    float e[RT_LAB_NEURAL_GATES];
    [ForceUnroll]
    for (int k = 0; k < RT_LAB_NEURAL_GATES; ++k)
    {
        e[k] = exp(float(raw[k]) - no_diff rt_lab_neural_constant(peak));
        sum += e[k];
    }
    RtLabNeuralAffinity r;
    const float rcp_sum = 1.0f / sum;
    r.gates_a = vector<rt_real, 4>(RT_H(e[0] * rcp_sum), RT_H(e[1] * rcp_sum), RT_H(e[2] * rcp_sum), RT_H(e[3] * rcp_sum));
    r.gates_b = vector<rt_real, 2>(RT_H(e[4] * rcp_sum), RT_H(e[5] * rcp_sum));
#if RT_LAB_NEURAL_WEIGHTING_V2
    // sharpness = 1 / kappa^2, kappa = 2 exp(-raw) sigmas (raw 0 = the lab's default guide tolerance of 2 sigmas).
    r.sharpness = RT_H(0.25f * exp(2.0f * clamp(float(raw[RT_LAB_NEURAL_GATES]), -3.0f, 3.0f)));
#else
    r.sharpness = RT_H(RT_LAB_NEURAL_AFFINITY_SCALE * exp(clamp(float(raw[RT_LAB_NEURAL_GATES]), -3.0f, 3.0f)));
#endif
    return r;
}

// The candidate weighting functions of u = sharpness x |f_c - f_t|^2 (all 1 at u = 0, falling with u), blended by the gates.
// 0 gaussian, 1 exponential (sharp at 0, long tail), 2 cauchy, 3 student-t (heavier tail), 4 compact support, 5 soft step at u = 1.
[Differentiable]
rt_real rt_lab_neural_weighting(RtLabNeuralAffinity a, rt_real d2)
{
    const rt_real u = min(a.sharpness * d2, RT_H(60.0f));
    const rt_real f0 = exp(-u);
    const rt_real eps = RT_H(1e-3f);
    const rt_real f1 = exp(-(sqrt(u + eps) - sqrt(eps)));
    const rt_real f2 = RT_H(1.0f) / (RT_H(1.0f) + u);
    const rt_real t = RT_H(1.0f) + u * RT_H(1.0f / 3.0f);
    const rt_real f3 = RT_H(1.0f) / (t * t);
    const rt_real c = max(RT_H(1.0f) - u, RT_H(0.0f));
    const rt_real f4 = c * c;
    // u is clamped for this branch: exp(8 (u - 1)) overflows for large u and its gradient is inf / inf = NaN (hit in training).
    const rt_real us = min(u, RT_H(3.0f));
    const rt_real f5 = RT_H(1.0f + exp(-8.0f)) / (RT_H(1.0f) + exp(RT_H(8.0f) * (us - RT_H(1.0f))));
    return a.gates_a.x * f0 + a.gates_a.y * f1 + a.gates_a.z * f2 + a.gates_a.w * f3 + a.gates_b.x * f4 + a.gates_b.y * f5;
}

#if RT_LAB_NEURAL_WEIGHTING_V2
// 1 at u = 0 falling steeply to ~0 around u = 1 (a smooth gate; u is clamped: exp(8 (u - 1)) overflows and its gradient is inf / inf = NaN).
[Differentiable]
rt_real rt_lab_neural_soft_step(rt_real u)
{
    const rt_real us = min(u, RT_H(3.0f));
    return RT_H(1.0f + exp(-8.0f)) / (RT_H(1.0f) + exp(RT_H(8.0f) * (us - RT_H(1.0f))));
}

// Variance guide style test of a tap against the center on the latent vectors (see RT_LAB_NEURAL_WEIGHTING_V2). Functions (u in units of kappa^2):
//   0 soft step on S/kappa^2 (the dimensions together)      1 soft step on P/kappa^2 (AND: ONE deviating dimension rejects)
//   2 soft step on T/kappa^2 (one outlier dimension is tolerated)   3 compact support on S, (1 - u)^2   4 compact support on P   5 gaussian on S, exp(-u): the lab's
// variance guide likelihood (the network kept the gaussian and compact support alive and switched off the heavy tails in run 003)
[Differentiable]
rt_real rt_lab_neural_weighting_v2(RtLabNeuralAffinity a, rt_real latent_c[RT_LAB_NEURAL_LATENT], rt_real latent_t[RT_LAB_NEURAL_LATENT])
{
    rt_real sum = RT_H(0.0f);
    rt_real peak = RT_H(0.0f);
    [ForceUnroll]
    for (int k = 0; k < 4; ++k)
    {
        const rt_real sigma_c = exp(RT_H(RT_LAB_NEURAL_SIGMA_BIAS) + RT_H(RT_LAB_NEURAL_SIGMA_GAIN) * latent_c[4 + k]);
        const rt_real sigma_t = exp(RT_H(RT_LAB_NEURAL_SIGMA_BIAS) + RT_H(RT_LAB_NEURAL_SIGMA_GAIN) * latent_t[4 + k]);
        const rt_real d = latent_c[k] - latent_t[k];
        const rt_real z_sq = d * d / (sigma_c * sigma_c + sigma_t * sigma_t + RT_H(RT_LAB_NEURAL_SIGMA_FLOOR));
        sum += z_sq;
        peak = max(peak, z_sq);
    }
    const rt_real s = sum * RT_H(0.25f) * a.sharpness;
    const rt_real p = peak * a.sharpness;
    const rt_real t = max(sum - peak, RT_H(0.0f)) * RT_H(1.0f / 3.0f) * a.sharpness;
    const rt_real f0 = rt_lab_neural_soft_step(s);
    const rt_real f1 = rt_lab_neural_soft_step(p);
    const rt_real f2 = rt_lab_neural_soft_step(t);
    const rt_real cs = max(RT_H(1.0f) - s, RT_H(0.0f));
    const rt_real f3 = cs * cs;
    const rt_real cp = max(RT_H(1.0f) - p, RT_H(0.0f));
    const rt_real f4 = cp * cp;
    const rt_real f5 = exp(-min(s, RT_H(30.0f)));
    return a.gates_a.x * f0 + a.gates_a.y * f1 + a.gates_a.z * f2 + a.gates_a.w * f3 + a.gates_b.x * f4 + a.gates_b.y * f5;
}
#endif

// B3 spline kernel weights by |offset| (1/16, 1/4, 3/8, 1/4, 1/16 per axis), as in rt_lab_blur.hlsl.
[Differentiable]
float rt_lab_neural_kernel(int offset)
{
    return offset == 0 ? 3.0f / 8.0f : (offset == 1 ? 1.0f / 4.0f : 1.0f / 16.0f);
}

struct RtLabNeuralAtrousResult : IDifferentiable
{
    float3 color;
    float2 split;
    float count;  // effective sample count (not normalized, see rt_lab.md SPATIAL SAMPLE COUNT)
    float weight; // sum of the tap weights (0 = nothing gathered)
}

// One a-trous pass (5x5 B3 spline, taps `step` pixels apart) at one pixel. Mirrors entry_rt_lab_atrous: the center is
// always accepted, rings (max(|x|, |y|) = 1, then 2) are accepted from the center outwards under the sample count
// budget `target` (0 = off). use_latent = false gives the geometry only filter, the trainer's baseline.
// The counts are treated as constants (no gradient through the budget decision).
[Differentiable]
RtLabNeuralAtrousResult rt_lab_neural_atrous<S : IRtLabNeuralAtrousSource, L : IRtLabNeuralLatents>(S source, L latents, int2 center, int step,
    float target, bool use_latent)
{
    const float center_kernel = rt_lab_neural_kernel(0) * rt_lab_neural_kernel(0);
    const int2 extent = no_diff source.extent();
    const float4 center_value = source.color_count(center);
    rt_real latent_center[RT_LAB_NEURAL_LATENT];
    RtLabNeuralAffinity affinity;
    affinity.gates_a = vector<rt_real, 4>(RT_H(0.25f), RT_H(0.25f), RT_H(0.25f), RT_H(0.25f));
    affinity.gates_b = vector<rt_real, 2>(RT_H(0.0f), RT_H(0.0f));
    affinity.sharpness = RT_H(RT_LAB_NEURAL_AFFINITY_SCALE);
    if (use_latent)
    {
        latent_center = latents.load(center);
        affinity = rt_lab_neural_decode_affinity(latents.load_affinity_raw(center));
    }
    else
    {
        [ForceUnroll]
        for (int k = 0; k < RT_LAB_NEURAL_LATENT; ++k)
        {
            latent_center[k] = RT_H(0.0f);
        }
    }

    RtLabNeuralAtrousResult total;
    const float center_weight = (no_diff rt_lab_neural_constant(center_value.w)) > 0.0f ? center_kernel : 0.0f;
    total.color = center_value.xyz * center_weight;
    total.split = no_diff source.split(center) * center_weight;
    total.count = no_diff rt_lab_neural_constant(center_value.w);
    total.weight = center_weight;
    if (target > 0.0f && total.count >= target)
    {
        return total;
    }
    [ForceUnroll]
    for (int r = 1; r <= 2; ++r)
    {
        float3 ring_color = float3(0.0f);
        float2 ring_split = float2(0.0f);
        float ring_count = 0.0f;
        float ring_weight = 0.0f;
        [ForceUnroll]
        for (int y = -2; y <= 2; ++y)
        {
            [ForceUnroll]
            for (int x = -2; x <= 2; ++x)
            {
                if (max(abs(x), abs(y)) != r)
                {
                    continue;
                }
                const int2 tap = center + int2(x, y) * step;
                if (any(tap < 0) || any(tap >= extent))
                {
                    continue;
                }
                if (!no_diff source.geometry_valid(tap))
                {
                    continue;
                }
                float weight = rt_lab_neural_kernel(abs(x)) * rt_lab_neural_kernel(abs(y)) *
                               no_diff source.geometry_weight(center, tap, (y + 2) * 5 + (x + 2));
                if (weight <= 1e-6f)
                {
                    continue;
                }
                const float4 tap_value = source.color_count(tap);
                const float tap_count = no_diff rt_lab_neural_constant(tap_value.w);
                if (tap_count <= 0.0f)
                {
                    continue;
                }
                if (use_latent)
                {
                    rt_real latent_tap[RT_LAB_NEURAL_LATENT] = latents.load(tap);
#if RT_LAB_NEURAL_WEIGHTING_V2
                    weight *= float(rt_lab_neural_weighting_v2(affinity, latent_center, latent_tap));
#else
                    weight *= float(rt_lab_neural_weighting(affinity, rt_lab_neural_distance_sq(latent_center, latent_tap)));
#endif
                }
                ring_color += tap_value.xyz * weight;
                ring_weight += weight;
                ring_count += tap_count * (no_diff rt_lab_neural_constant(weight)) / center_kernel;
                ring_split += no_diff source.split(tap) * weight;
            }
        }
        // Budget: a ring that fits is accepted fully, the one crossing the target is scaled to land on it, the rest dropped.
        float raw_accept = 1.0f;
        if (target > 0.0f && ring_count > 0.0f)
        {
            raw_accept = saturate((no_diff rt_lab_neural_constant(target - total.count)) / (no_diff rt_lab_neural_constant(ring_count)));
        }
        const float accept = no_diff rt_lab_neural_constant(raw_accept);
        total.color += ring_color * accept;
        total.split += ring_split * accept;
        total.weight += ring_weight * accept;
        total.count += ring_count * accept;
        if (target > 0.0f && total.count >= target * 0.9999f)
        {
            break;
        }
    }
    return total;
}
