#pragma once

#include "rtgi_pre_filter.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiPreFilterPush rtgi_pre_filter_prepare_push;

// Because of the +1
static const float PERCEPTUAL_SPACE_MULTIPLIER = 1e1f;

__generic<uint N>
func linear_to_perceptual(vector<float, N> v) -> vector<float, N> 
{
    return log(max(v, 1e-8f) * PERCEPTUAL_SPACE_MULTIPLIER);
}

__generic<uint N>
func perceptual_to_linear(vector<float, N> v) -> vector<float, N> 
{
    return (exp(v)) / PERCEPTUAL_SPACE_MULTIPLIER;
}

// The center blur is used to preserve firefly energy by flat filtering all pixels in a star kernel covering 5 pixels.
func is_part_of_center_blur(int2 index) -> bool
{
    return (abs(index.x) + abs(index.y) <= 1);
}

static const int FILTER_STRIDE = 1;
static const int TOTAL_FILTER_REACH = 2;
static const int TOTAL_INNER_FILTER_REACH = 1;                                         // Inner filter calculates geometry aware metrics: footprint valid samples ratio, star blur, ray length average
static const int FILTER_TAPS_TOTAL = square(TOTAL_FILTER_REACH * 2 + 1);
static const int INNER_FILTER_TAPS_TOTAL = square(TOTAL_INNER_FILTER_REACH * 2 + 1);
static const int STAR_BLUR_TAPS = 5;                                                    // 5 generally yields the best cost to use ratio.
static const int GEOMETRIC_MEAN_TAPS_TOTAL = FILTER_TAPS_TOTAL - STAR_BLUR_TAPS;              // 20 geometric mean taps are a good ratio against the 5 center blur taps.

static const int PRELOAD_SIZE_X = RTGI_PRE_BLUR_PREPARE_X + TOTAL_FILTER_REACH * 2;
static const int PRELOAD_SIZE_Y = RTGI_PRE_BLUR_PREPARE_X + TOTAL_FILTER_REACH * 2;
groupshared float4 gs_sample_value_vs[PRELOAD_SIZE_X][PRELOAD_SIZE_Y]; // .xyz = sample vs space pos, .w = depth
// Full PRELOAD_SIZE (20x20) gives a 2-pixel border, supporting stride 1 and stride 2 gradients
// RGB packed as R8G8B8 in the low 24 bits of a uint
groupshared uint gs_albedo_rgb[PRELOAD_SIZE_X][PRELOAD_SIZE_Y];

func src_to_preload_index(int2 src_index, int2 gid) -> int2
{
    const int2 src_group_base_index = int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) - TOTAL_FILTER_REACH;
    return clamp(src_index - src_group_base_index, int2(0,0), int2(PRELOAD_SIZE_X-1, PRELOAD_SIZE_Y-1));
}

    func unpack_rgb(uint p) -> float3 { return float3(p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF) * (1.0f / 255.0f); }

[shader("compute")]
[numthreads(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y,1)]
func entry_prepare(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rtgi_pre_filter_prepare_push;
    RWTexture2D<float4> dbg = push.attach.debug_image.get();

    // Load and precalculate constants
    CameraInfo *camera = &push.attach.globals->view_camera;
    const float2 inv_half_res_render_target_size = rcp(float2(push.size));
    const uint2 half_res_index = dtid.xy;
    const uint2 clamped_index = min( half_res_index, push.size - 1u );      // Can not early out because we perform group shared memory barriers later!

    // Precalculate surrounding view space position data for geometric testing
    {
        [unroll]
        for (uint group_iter_x = 0; group_iter_x < round_up_div(PRELOAD_SIZE_X, RTGI_PRE_BLUR_PREPARE_X); ++group_iter_x)
        {
            [unroll]
            for (uint group_iter_y = 0; group_iter_y < round_up_div(PRELOAD_SIZE_Y, RTGI_PRE_BLUR_PREPARE_Y); ++group_iter_y)
            {
                const int2 in_preload_index = int2(group_iter_x * RTGI_PRE_BLUR_PREPARE_X, group_iter_y * RTGI_PRE_BLUR_PREPARE_Y) + int2(gtid);
                const int2 src_group_base_index = int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) - TOTAL_FILTER_REACH;
                const int2 src_index = src_group_base_index + in_preload_index;
                int2 clamped_src_index = clamp(src_index, int2(0,0), int2(push.size - 1));
                const int2 max_index = push.size - 1;
                clamped_src_index = flip_oob_index(clamped_src_index, max_index);

                if (all(in_preload_index < int2(PRELOAD_SIZE_X, PRELOAD_SIZE_Y)))
                {
                    const float preload_depth = push.attach.view_cam_half_res_depth.get()[clamped_src_index];
                    const float2 uv = (float2(clamped_src_index.xy) + 0.5f) * inv_half_res_render_target_size;
                    const float3 sample_value_ndc = float3(uv * 2.0f - 1.0f, preload_depth);
                    const float4 sample_value_vs_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
                    const float3 sample_value_vs = sample_value_vs_pre_div.xyz * rcp(sample_value_vs_pre_div.w);
                    gs_sample_value_vs[in_preload_index.x][in_preload_index.y] = float4(sample_value_vs, preload_depth);

                    const float3 alb = push.attach.view_cam_half_res_albedo.get()[clamped_src_index].rgb;
                    const uint3 alb_q = (uint3)(saturate(alb) * 255.0f + 0.5f);
                    gs_albedo_rgb[in_preload_index.x][in_preload_index.y] = alb_q.r | (alb_q.g << 8) | (alb_q.b << 16);
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync(); // preload visible

    // gs_albedo_rgb is fully written — center pixel sits at gtid + TOTAL_FILTER_REACH in the 20x20 tile
    const int2 grad_idx = int2(gtid) + TOTAL_FILTER_REACH;

    // Load Pixel Data
    const float depth = push.attach.view_cam_half_res_depth.get()[clamped_index];

    // const float pixel_samplecnt = push.attach.half_res_samplecnt.get()[clamped_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_normals.get()[clamped_index]);

    // Reconstruct pixel positions based on depth
    const float pixel_vs_depth = linearise_depth(depth, camera.near_plane);
    const float2 uv = (float2(clamped_index.xy) + 0.5f) * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 vs_position = mul(camera.view, float4(world_position, 1.0f)).xyz;
    const float3 vs_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;
    const float pixel_ws_size = ws_pixel_size(inv_half_res_render_target_size, camera.near_plane, depth);
    const float pixel_ws_size_rcp = rcp(pixel_ws_size);

    // Geometric mean is used to suppress fireflies
    float y_mean_geometric_acc = 0.0f;
    float valid_geometric_mean_samples = 0.0f;

    // Variance, ray length and valid footprint samples are used for pre blur filter guiding later
    float y_mean_acc = 0.0f;
    float y_variance_acc = 0.0f;
    float y_max = 0.0f;
    float ray_shortness_acc = 0.0f;
    float valid_footprint_samples = 0.0f;

    float3 albedo_mean_acc = float3(0.0f, 0.0f, 0.0f);

    // Per-pixel albedo gradient estimation: accumulate |neighbor_alb - center_alb| from groupshared
    const float3 pp_center_alb = unpack_rgb(gs_albedo_rgb[grad_idx.x][grad_idx.y]);
    float pp_alb_diff_acc = 0.0f;
    float pp_alb_diff_count = 0.0f;
    // Coefficient of variation accumulators (same 3x3 window)
    float pp_alb_lum_acc = 0.0f;
    float pp_alb_lum_sq_acc = 0.0f;

    // The raw signal is pre blurredi in a star shape with 5 taps to conserve energy from the firefly filter.
    float4 star_blurred_diffuse_acc = float4(0,0,0,0);
    float2 star_blurred_diffuse2_acc = float2(0,0);
    float star_blurred_weight_acc = 0.0f;

    const float max_visibility_pixel_range = 32.0f * 1.25f; // Should roughly match 1.5x the max denoiser width (currently always around 32.0f).
    const float max_visibility_raylen = pixel_ws_size * max_visibility_pixel_range;

    [unroll]
    for (int x = -TOTAL_FILTER_REACH; x <= TOTAL_FILTER_REACH; ++x)
    {
        [unroll]
        for (int y = -TOTAL_FILTER_REACH; y <= TOTAL_FILTER_REACH; ++y)
        {
            const int2 max_index = push.size - 1;
            int2 load_idx = int2(x,y) * FILTER_STRIDE + int2(clamped_index);
            load_idx = flip_oob_index(load_idx, max_index);

            const int2 preload_index = src_to_preload_index(load_idx, gid);
            float4 sample_diffuse = push.attach.diffuse_raw.get()[load_idx];                // preloading via shared mem reduces perf, distribute loading onto multiply hw units
            float2 sample_diffuse2 = push.attach.diffuse2_raw.get()[load_idx];              // preloading via shared mem reduces perf, distribute loading onto multiply hw units
            const float sample_ray_length = push.attach.ray_length_image.get()[load_idx];   // preloading via shared mem reduces perf, distribute loading onto multiply hw units
            const float4 preload_v = gs_sample_value_vs[preload_index.x][preload_index.y];
            const float3 sample_value_vs = preload_v.xyz;
            const float sample_depth = preload_v.w;
            float sample_y = sample_diffuse.w;
            sample_y = max(sample_y, 1e-3f);          // Values below this start to break float16 precision, have to clamp radiance up to that value as a minimum for statistic analysis.
            
            const bool is_sky = sample_depth == 0.0f;
            if (is_sky)
            {
                continue;
            }

            if (!is_part_of_center_blur(int2(x,y)))
            {
                float geometric_mean_acc_value = linear_to_perceptual(sample_y);
                y_mean_geometric_acc += geometric_mean_acc_value;
                valid_geometric_mean_samples += 1.0f;
            }

            if (all(abs(int2(x,y)) <= TOTAL_INNER_FILTER_REACH))
            {
                const float GEO_WEIGHT_THRESHOLD = 1.0f * FILTER_STRIDE;
                const float plane_distance = planar_surface_distance(vs_position, vs_normal, sample_value_vs) * pixel_ws_size_rcp;
                const float geometric_weight = abs(plane_distance) < 1.0f ? 1.0f : 0.0f;

                if (is_part_of_center_blur(int2(x,y)))
                {
                    star_blurred_weight_acc += geometric_weight;
                    star_blurred_diffuse_acc += geometric_weight * sample_diffuse;
                    star_blurred_diffuse2_acc += geometric_weight * sample_diffuse2;
                }
                
                y_mean_acc += sample_y * geometric_weight;
                y_variance_acc += square(sample_y) * geometric_weight;
                y_max = max(y_max, sample_y) * geometric_weight;
                // double square the ray shortness (ranges from 0-1) to promote small values in sum. Later we take sqrt(sqrt(acc*weight_sum)) to get a scaled average of ray shortness.
                ray_shortness_acc += square(square(square(1.0f - min(1.0f, sample_ray_length * rcp(max_visibility_raylen))))) * geometric_weight;

                const float3 sample_albedo = push.attach.view_cam_half_res_albedo.get()[load_idx].rgb;
                albedo_mean_acc += sample_albedo * geometric_weight;

                valid_footprint_samples += geometric_weight;

                if (geometric_weight > 0.0f)
                {
                    const float3 pp_neighbor_alb = unpack_rgb(gs_albedo_rgb[preload_index.x][preload_index.y]);
                    pp_alb_diff_acc += dot(abs(pp_neighbor_alb - pp_center_alb), (1.0f / 3.0f).xxx);
                    pp_alb_diff_count += 1.0f;
                    const float pp_neighbor_lum = dot(pp_neighbor_alb, float3(0.2126f, 0.7152f, 0.1722f));
                    pp_alb_lum_acc += pp_neighbor_lum;
                    pp_alb_lum_sq_acc += square(pp_neighbor_lum);
                }
            }
        }
    }

    let rtgi = push.attach.globals->rtgi_settings;

    // --- Footprint Quality ---
    float footprint_quality = 1.0f;
    {
        footprint_quality = valid_footprint_samples * rcp(float(INNER_FILTER_TAPS_TOTAL));
    }

    // --- Firefly Filter ---
    float4 filtered_diffuse;
    float2 filtered_diffuse2;
    float firefly_energy_factor = 1.0f;
    float y_std_dev;
    {
        const float EPSILON = 1e-8f;
        const float4 star_blurred_diffuse = star_blurred_diffuse_acc * rcp(star_blurred_weight_acc + EPSILON);
        const float2 star_blurred_diffuse2 = star_blurred_diffuse2_acc * rcp(star_blurred_weight_acc + EPSILON);

        if (rtgi.firefly_filter_enabled != 0)
        {
            filtered_diffuse = star_blurred_diffuse;
            filtered_diffuse2 = star_blurred_diffuse2;
        }
        else
        {
            filtered_diffuse = push.attach.diffuse_raw.get()[clamped_index];
            filtered_diffuse2 = push.attach.diffuse2_raw.get()[clamped_index];
        }

        const float y_mean = y_mean_acc * rcp(valid_footprint_samples);
        const float y_variance = (y_variance_acc * rcp(valid_footprint_samples)) - square(y_mean);
        y_std_dev = sqrt(y_variance);

        const float y_mean_geometric = perceptual_to_linear(y_mean_geometric_acc * rcp(valid_geometric_mean_samples));
        const float ceiling_factor = max(1.0f,
            rtgi.firefly_filter_ceiling *
            (valid_geometric_mean_samples / float(GEOMETRIC_MEAN_TAPS_TOTAL)) *
            footprint_quality);
        const float geometric_y_clamp_factor = min((y_mean_geometric * ceiling_factor) / (EPSILON + filtered_diffuse.w), 1.0f);

        if (rtgi.firefly_filter_enabled != 0)
        {
            filtered_diffuse  *= geometric_y_clamp_factor;
            filtered_diffuse2 *= geometric_y_clamp_factor;
            firefly_energy_factor = 1.0f / max(1.0f / RTGI_MAX_FIREFLY_FACTOR, geometric_y_clamp_factor);
        }
    }

    // --- Surface Detail ---
    float surface_detail_guide = 1.0f;
    if (rtgi.surface_detail_guiding != 0)
    {
        const float boost = 1.0f;
        const float pp_lum_mean = pp_alb_lum_acc / (pp_alb_diff_count + 1e-8f);
        const float pp_lum_sq_mean = pp_alb_lum_sq_acc / (pp_alb_diff_count + 1e-8f);
        const float pp_lum_variance = max(0.0f, pp_lum_sq_mean - square(pp_lum_mean));
        const float pp_lim_std_dev = sqrt(pp_lum_variance);

        // In practice, the linear covaraince is too localized of a value here. Use sqrt to re-scale the covariance to be less effected by super strong gradients.
        const float surface_albedo_cv = saturate(boost * (pp_lim_std_dev / max(pp_lum_mean, 0.05f)));
        surface_detail_guide = 1.0f - sqrt(surface_albedo_cv);
    }

    // --- Raylength Guide ---
    float raylength_filter_guide = 1.0f;
    if (rtgi.pre_blur_ray_length_guiding != 0)
    {
        const float ray_shortness_mean = sqrt(sqrt(sqrt(ray_shortness_acc * rcp(valid_footprint_samples))));
        // For more detailed surfaces the raylength guide is allowed to scale down further.
        // For very smooth surfaces scaling down too much causes residual noise.
        const float super_smooth_scale = clamp(surface_detail_guide, 0.0f, 0.25f) * 4.0f;
        const float min_guide_value = lerp(0.0f, 0.33f, super_smooth_scale);
        raylength_filter_guide = lerp(1.0f, min_guide_value, ray_shortness_mean);
    }

    // --- Filter Guide ---
    const float filter_guide = raylength_filter_guide * surface_detail_guide * footprint_quality;

    //debug_image_tile_draw(push.attach.debug_image.get(), 0, dtid, float4(TurboColormap(filter_guide), 2.0f), 2);

    push.attach.pre_filtered_diffuse_image.get()[dtid] = filtered_diffuse;
    push.attach.pre_filtered_diffuse2_image.get()[dtid] = filtered_diffuse2;
    push.attach.firefly_factor_image.get()[dtid] = min(1.0f, firefly_energy_factor * (1.0f / RTGI_MAX_FIREFLY_FACTOR));
    push.attach.spatial_std_dev_image.get()[dtid] = y_std_dev;
    push.attach.filter_guide_image.get()[dtid] = filter_guide;
}