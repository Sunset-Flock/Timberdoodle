#pragma once

#include "rtgi_pre_filter.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiPreFilterPreparePush rtgi_pre_filter_prepare_push;

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
    #if RTGI_FIREFLY_ENERGY_HACKS
    return (abs(index.x) + abs(index.y) <= 1);
    #else
    return all(index == 0);
    #endif
}

[shader("compute")]
[numthreads(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y,1)]
func entry_prepare(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID)
{
    let push = rtgi_pre_filter_prepare_push;

    // Load and precalculate constants
    CameraInfo *camera = &push.attach.globals->view_camera;
    const float2 inv_half_res_render_target_size = rcp(float2(push.size));
    const uint2 half_res_index = dtid.xy;
    const uint2 clamped_index = min( half_res_index, push.size - 1u );      // Can not early out because we perform group shared memory barriers later!

    // Load Pixel Data
    const float depth = push.attach.view_cam_half_res_depth.get()[clamped_index];
    float4 diffuse = push.attach.diffuse_raw.get()[clamped_index];
    float2 diffuse2 = push.attach.diffuse2_raw.get()[clamped_index];

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

    // Analyze pixel footprint
    float4 filtered_diffuse = diffuse;
    float2 filtered_diffuse2 = diffuse2;
    float foreground_footprint_quality = 1.0f;

    // Generally, a wider filter is more stable but also kills more light.
    // Bistro interior lit by only emissives for example suffers A LOT when the filter is smaller than 3.
    // Most other locations do not really care for a wide filter, 1 is fine for most situations.
    const int FILTER_STRIDE = 1;
    const int FILTER_WIDTH = 2;
    const int FILTER_TAPS_TOTAL = square(FILTER_WIDTH * 2 + 1) - 1;

    // Sums samples that are either closer or similar to the pixel
    // When low, indicates that the pixel is on a hard to de-noise thin geometry.
    float foreground_sample_weight = 0.0f;  

    // Sums samples that are either further or similar to the pixel
    // When low, indicates the pixel is in a hole with very small size, hard to de-noise.
    float background_sample_weight = 0.0f;  

    // Calculating outlier-resiliant geometric mean
    float y_mean_geometric_acc = 0.0f;
    float y_mean_acc = 0.0f;
    float y_geometric_variance_acc = 0.0f;
    float y_variance_acc = 0.0f;
    float y_max = 0.0f;
    float valid_neightborhood_samples = 0.0f;

    float ray_length_mean_acc = 0.0f;
    float valid_footprint_samples = 0.0f;

    float4 blurred_diffuse_acc = float4(0,0,0,0);
    float2 blurred_diffuse2_acc = float2(0,0);
    float blurred_weight_acc = 0.0f;

    // 24 geometric mean samples are very good and relatively performant
    // 8 geometric samples really let some uglier fireflies in, try to stick to 24.
    //  
    for (int x = -FILTER_WIDTH; x <= FILTER_WIDTH; ++x)
    {
        for (int y = -FILTER_WIDTH; y <= FILTER_WIDTH; ++y)
        {
            const int2 max_index = push.size - 1;
            int2 load_idx = int2(x,y) * FILTER_STRIDE + int2(clamped_index);
            load_idx = flip_oob_index(load_idx, max_index);

            const float sample_ray_length = push.attach.ray_length_image.get()[load_idx];
            float4 sample_diffuse = push.attach.diffuse_raw.get()[load_idx];
            float2 sample_diffuse2 = push.attach.diffuse2_raw.get()[load_idx];
            const float sample_depth = push.attach.view_cam_half_res_depth.get()[load_idx];
            // const float sample_validity = push.attach.half_res_samplecnt.get()[load_idx];
            const bool is_sky = sample_depth == 0.0f;

            if (is_sky)
            {
                continue;
            }

            // sample_diffuse.w = max(sample_diffuse.w, 1e-6f);

            if ((sample_diffuse.w != 0.0f) && !is_part_of_center_blur(int2(x,y)))
            {
                float geometric_mean_acc_value = linear_to_perceptual(sample_diffuse.w);
                y_mean_geometric_acc += geometric_mean_acc_value;
                y_geometric_variance_acc += square(geometric_mean_acc_value);

                valid_neightborhood_samples += 1.0f;
            }

            {
                const float2 sample_ndc_xy = ndc.xy + float2(x,y) * FILTER_STRIDE * inv_half_res_render_target_size * 2;
                const float3 sample_value_ndc = float3(sample_ndc_xy, sample_depth);
                const float4 sample_value_vs_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
                const float3 sample_value_vs = sample_value_vs_pre_div.xyz * rcp(sample_value_vs_pre_div.w);
                
                const float GEO_WEIGHT_THRESHOLD = 1.0f * FILTER_STRIDE;
                {
                    const float plane_distance = planar_surface_distance(inv_half_res_render_target_size, camera.near_plane, depth, vs_position, vs_normal, sample_value_vs);
                    const float geometric_weight_real = abs(plane_distance) < 1.0f ? 1.0f : 0.0f;
                    if (is_part_of_center_blur(int2(x,y)))
                    {
                        blurred_weight_acc += geometric_weight_real;
                        blurred_diffuse_acc += geometric_weight_real * sample_diffuse;
                        blurred_diffuse2_acc += geometric_weight_real * sample_diffuse2;
                    }
                    
                    y_mean_acc += sample_diffuse.w * geometric_weight_real;
                    y_variance_acc += square(sample_diffuse.w) * geometric_weight_real;
                    y_max = max(y_max, sample_diffuse.w) * geometric_weight_real;
                    ray_length_mean_acc += sample_ray_length * geometric_weight_real;

                    foreground_sample_weight += plane_distance > -GEO_WEIGHT_THRESHOLD ? 1 : 0;
                    background_sample_weight += plane_distance < GEO_WEIGHT_THRESHOLD ? 1 : 0;
                    valid_footprint_samples += geometric_weight_real;
                }
            }
        }
    }

    // Foreground footprint quality estimation
    {
        foreground_footprint_quality = foreground_sample_weight * rcp(FILTER_TAPS_TOTAL);
    }

    float y_std_dev = 1000.0f;
    float firefly_energy_factor = 1.0f;
    float footprint_quality = 1.0f;
    // Firefly Filter + Background Pixel Suppression
    if (valid_neightborhood_samples > 1.0f)
    {
        const float BACKGROUND_WEIGHT_MIN = 0.1f;
        const float BACKGROUND_WEIGHT_MAX = 0.5f;
        #if RTGI_FIREFLY_FILTER_TIGHT_AGRESSIVE
            const float tight_neighborgood_factor = (clamp(background_sample_weight * rcp(FILTER_TAPS_TOTAL), BACKGROUND_WEIGHT_MIN, BACKGROUND_WEIGHT_MAX) - BACKGROUND_WEIGHT_MIN) * rcp(BACKGROUND_WEIGHT_MAX - BACKGROUND_WEIGHT_MIN);
        #else
            const float tight_neighborgood_factor = 1.0f;
        #endif

        const float EPSILON = 0.00000001f;

        const float4 blurred_diffuse = blurred_diffuse_acc * rcp(blurred_weight_acc + EPSILON);
        const float2 blurred_diffuse2 = blurred_diffuse2_acc * rcp(blurred_weight_acc + EPSILON);

        filtered_diffuse = blurred_diffuse;
        filtered_diffuse2 = blurred_diffuse2;

        const float y_mean = y_mean_acc * rcp(valid_footprint_samples);
        const float y_variance = (y_variance_acc * rcp(valid_footprint_samples)) - square(y_mean);
        const float y_variance_relative = (y_variance) / (y_mean + EPSILON);
        y_std_dev = sqrt(y_variance);
        const float y_std_dev_relative = y_std_dev / y_mean;

        const float y_mean_perceptual = y_mean_geometric_acc * rcp(valid_neightborhood_samples);
        const float y_mean_geometric = perceptual_to_linear(y_mean_perceptual);
        const float y_variance_geometric_relative = (y_variance) / (y_mean_geometric + EPSILON);

        footprint_quality = (valid_footprint_samples * rcp(FILTER_TAPS_TOTAL));

        float ray_length_mean = ray_length_mean_acc * rcp(valid_footprint_samples);
        if (push.attach.globals.rtgi_settings.pre_blur_ray_length_guiding != 0)
        {
            footprint_quality *= square(square(ray_length_mean));
        }

        const float valid_samples_ceiling_factor = valid_neightborhood_samples / FILTER_TAPS_TOTAL;
        const float footprint_quality_ceiling_factor = foreground_footprint_quality * tight_neighborgood_factor;
        const float CEILING_FACTOR = max(1.0f, push.attach.globals.rtgi_settings.firefly_filter_ceiling * footprint_quality_ceiling_factor * valid_samples_ceiling_factor);
        const float y_center_pixel_perceptual = linear_to_perceptual(filtered_diffuse.w);
        const float y_center_pixel = filtered_diffuse.w;
        const float y_ratio = (y_mean * CEILING_FACTOR) / (EPSILON + y_center_pixel);
        const float y_geometric_ratio = (y_mean_geometric * CEILING_FACTOR) / (EPSILON + y_center_pixel);
        const float y_geometric_perceptual_ratio = (y_mean_perceptual * CEILING_FACTOR) / (EPSILON + y_center_pixel_perceptual);
        const float linear_y_clamp_factor = min(y_ratio, 1.0f);
        const float geometric_y_clamp_factor = min(y_geometric_ratio, 1);

        const float linear_to_geometric_factor = (max(1.0f, y_mean / y_mean_geometric) - 1.0f);

        const float adjustment_factor = geometric_y_clamp_factor;
        if (push.attach.globals.rtgi_settings.firefly_filter_enabled != 0)
        {
            filtered_diffuse *= adjustment_factor;
            filtered_diffuse2 *= adjustment_factor;
            #if RTGI_FIREFLY_ENERGY_HACKS
            firefly_energy_factor = 1.0f / max(1.0f / RTGI_MAX_FIREFLY_FACTOR, adjustment_factor);
            #else
            firefly_energy_factor = 1.0f;
            #endif
        }

        push.attach.debug_image.get()[dtid * 2 + uint2(0,0)] = float4(ray_length_mean_acc * rcp(valid_footprint_samples), 0, 0, 0);
        push.attach.debug_image.get()[dtid * 2 + uint2(0,1)] = float4(ray_length_mean_acc * rcp(valid_footprint_samples), 0, 0, 0);
        push.attach.debug_image.get()[dtid * 2 + uint2(1,0)] = float4(ray_length_mean_acc * rcp(valid_footprint_samples), 0, 0, 0);
        push.attach.debug_image.get()[dtid * 2 + uint2(1,1)] = float4(ray_length_mean_acc * rcp(valid_footprint_samples), 0, 0, 0);
    }

    push.attach.pre_filtered_diffuse_image.get()[dtid] = filtered_diffuse;
    push.attach.pre_filtered_diffuse2_image.get()[dtid] = filtered_diffuse2;
    push.attach.firefly_factor_image.get()[dtid] = min(1.0f, firefly_energy_factor * (1.0f / RTGI_MAX_FIREFLY_FACTOR));
    push.attach.spatial_std_dev_image.get()[dtid] = y_std_dev;
    push.attach.footprint_quality_image.get()[dtid] = footprint_quality;
}