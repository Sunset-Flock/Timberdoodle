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

func src_to_preload_index(int2 src_index, int2 gid) -> int2
{
    const int2 src_group_base_index = int2(gid) * int2(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y) - TOTAL_FILTER_REACH;
    return clamp(src_index - src_group_base_index, int2(0,0), int2(PRELOAD_SIZE_X-1, PRELOAD_SIZE_Y-1));
}

[shader("compute")]
[numthreads(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y,1)]
func entry_prepare(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rtgi_pre_filter_prepare_push;

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
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

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
    const float pixel_ws_size_rcp = rcp(ws_pixel_size(inv_half_res_render_target_size, camera.near_plane, depth));

    // Geometric mean is used to suppress fireflies
    float y_mean_geometric_acc = 0.0f;
    float valid_geometric_mean_samples = 0.0f;

    // Variance, ray length and valid footprint samples are used for pre blur filter guiding later
    float y_mean_acc = 0.0f;
    float y_variance_acc = 0.0f;
    float y_max = 0.0f;
    float ray_length_mean_acc = 0.0f;
    float valid_footprint_samples = 0.0f;

    // The raw signal is pre blurredi in a star shape with 5 taps to conserve energy from the firefly filter.
    float4 star_blurred_diffuse_acc = float4(0,0,0,0);
    float2 star_blurred_diffuse2_acc = float2(0,0);
    float star_blurred_weight_acc = 0.0f;

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
            float4 sample_diffuse = push.attach.diffuse_raw.get()[load_idx];                        // preloading via shared mem reduces perf, distribute loading onto multiply hw units
            float2 sample_diffuse2 = push.attach.diffuse2_raw.get()[load_idx];                      // preloading via shared mem reduces perf, distribute loading onto multiply hw units
            const float sample_ray_length = push.attach.ray_length_image.get()[load_idx];           // preloading via shared mem reduces perf, distribute loading onto multiply hw units
            const float4 preload_v = gs_sample_value_vs[preload_index.x][preload_index.y];
            const float3 sample_value_vs = preload_v.xyz;
            const float sample_depth = preload_v.w;
            float sample_y = sample_diffuse.w;
            sample_y = max(sample_y, 1e-4f);          // Values below 1e-4f start to break float16 precision, have to clamp radiance up to that value as a minimum for statistic analysis.
            
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
                ray_length_mean_acc += sample_ray_length * geometric_weight;

                valid_footprint_samples += geometric_weight;
            }
        }
    }

    float y_std_dev = 1.0f;
    float firefly_energy_factor = 1.0f;
    float footprint_quality = 1.0f;
    float4 filtered_diffuse = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float2 filtered_diffuse2 = float2(0.0f, 0.0f);
    {
        const float EPSILON = 0.00000001f;

        const float4 star_blurred_diffuse = star_blurred_diffuse_acc * rcp(star_blurred_weight_acc + EPSILON);
        const float2 star_blurred_diffuse2 = star_blurred_diffuse2_acc * rcp(star_blurred_weight_acc + EPSILON);

        filtered_diffuse = star_blurred_diffuse;
        filtered_diffuse2 = star_blurred_diffuse2;

        const float y_mean = y_mean_acc * rcp(valid_footprint_samples);
        const float y_variance = (y_variance_acc * rcp(valid_footprint_samples)) - square(y_mean);
        const float y_variance_relative = (y_variance) / (y_mean + EPSILON);
        y_std_dev = sqrt(y_variance);
        const float y_std_dev_relative = y_std_dev / y_mean;

        const float y_mean_perceptual = y_mean_geometric_acc * rcp(valid_geometric_mean_samples);
        const float y_mean_geometric = perceptual_to_linear(y_mean_perceptual);

        const float valid_footprint_relative = (valid_footprint_samples * rcp(INNER_FILTER_TAPS_TOTAL));
        footprint_quality = valid_footprint_relative;

        float ray_length_mean = ray_length_mean_acc * rcp(valid_footprint_samples);
        if (push.attach.globals.rtgi_settings.pre_blur_ray_length_guiding != 0)
        {
            footprint_quality *= square(square(ray_length_mean));
        }

        const float valid_geometric_samples_ceiling_factor = (valid_geometric_mean_samples / (GEOMETRIC_MEAN_TAPS_TOTAL));
        const float valid_footprint_samples_ceiling_factor = (valid_footprint_relative);
        const float ceiling_factor = max(1.0f, push.attach.globals.rtgi_settings.firefly_filter_ceiling * valid_geometric_samples_ceiling_factor * valid_footprint_samples_ceiling_factor);
        const float y_center_pixel_perceptual = linear_to_perceptual(filtered_diffuse.w);
        const float y_center_pixel = filtered_diffuse.w;
        const float y_ratio = (y_mean * ceiling_factor) / (EPSILON + y_center_pixel);
        const float y_geometric_ratio = (y_mean_geometric * ceiling_factor) / (EPSILON + y_center_pixel);
        const float y_geometric_perceptual_ratio = (y_mean_perceptual * ceiling_factor) / (EPSILON + y_center_pixel_perceptual);
        const float linear_y_clamp_factor = min(y_ratio, 1.0f);
        const float geometric_y_clamp_factor = min(y_geometric_ratio, 1);

        const float adjustment_factor = geometric_y_clamp_factor;
        if (push.attach.globals.rtgi_settings.firefly_filter_enabled != 0)
        {
            filtered_diffuse *= adjustment_factor;
            filtered_diffuse2 *= adjustment_factor;
            firefly_energy_factor = 1.0f / max(1.0f / RTGI_MAX_FIREFLY_FACTOR, adjustment_factor);
        }

        // push.attach.debug_image.get()[dtid * 2 + uint2(0,0)] = float4(y_mean_geometric, 0, 0, 0);
        // push.attach.debug_image.get()[dtid * 2 + uint2(0,1)] = float4(y_mean_geometric, 0, 0, 0);
        // push.attach.debug_image.get()[dtid * 2 + uint2(1,0)] = float4(y_mean_geometric, 0, 0, 0);
        // push.attach.debug_image.get()[dtid * 2 + uint2(1,1)] = float4(y_mean_geometric, 0, 0, 0);
    } 

    push.attach.pre_filtered_diffuse_image.get()[dtid] = filtered_diffuse;
    push.attach.pre_filtered_diffuse2_image.get()[dtid] = filtered_diffuse2;
    push.attach.firefly_factor_image.get()[dtid] = min(1.0f, firefly_energy_factor * (1.0f / RTGI_MAX_FIREFLY_FACTOR));
    push.attach.spatial_std_dev_image.get()[dtid] = y_std_dev;
    push.attach.footprint_quality_image.get()[dtid] = footprint_quality;
}