#pragma once

#include "rtgi_pre_blur.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiPreBlurFlattenPush rtgi_pre_blur_flatten_push;
[[vk::push_constant]] RtgiPreblurPreparePush rtgi_pre_blur_prepare_push;
[[vk::push_constant]] RtgiPreBlurApplyPush rtgi_reconstruct_history_apply_diffuse_push;

groupshared float4 gs_diffuse[RTGI_PRE_BLUR_PREPARE_X][RTGI_PRE_BLUR_PREPARE_Y][2];
groupshared float gs_depth[RTGI_PRE_BLUR_PREPARE_X][RTGI_PRE_BLUR_PREPARE_Y][2];
groupshared float2 gs_diffuse2[RTGI_PRE_BLUR_PREPARE_X][RTGI_PRE_BLUR_PREPARE_Y][2];

func downsample_mip_linear(uint2 thread_index, uint2 group_thread_index, uint mip, uint gs_src)
{
    let mip_factor = 1u << mip;
    let push = rtgi_pre_blur_prepare_push;
    let remaining_block_size = RTGI_PRE_BLUR_PREPARE_X/mip_factor;
    if (all(group_thread_index.xy < remaining_block_size))
    {
        float4 depths = {
            gs_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0][gs_src],
            gs_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0][gs_src],
            gs_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1][gs_src],
            gs_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1][gs_src]
        };
        float max_depth = max(max(depths.x, depths.y), max(depths.z, depths.w));
        float min_depth = min(min(depths.x, depths.y), min(depths.z, depths.w));
        // Ignore sky pixels
        // Prefer pixels further away as they are more likely to be disoccluded
        float4 weights = {
            (depths[0] != 0.0f ? 1.0f : 0.0f) * ((depths[0] == max_depth && ((depths[0] - min_depth) > 0.001f)) ? 0 : 1),
            (depths[1] != 0.0f ? 1.0f : 0.0f) * ((depths[1] == max_depth && ((depths[1] - min_depth) > 0.001f)) ? 0 : 1),
            (depths[2] != 0.0f ? 1.0f : 0.0f) * ((depths[2] == max_depth && ((depths[2] - min_depth) > 0.001f)) ? 0 : 1),
            (depths[3] != 0.0f ? 1.0f : 0.0f) * ((depths[3] == max_depth && ((depths[3] - min_depth) > 0.001f)) ? 0 : 1),
        };
        const float weight_sum = (weights.x + weights.y + weights.z + weights.w);
        weights *= rcp(weight_sum + 0.00000001f);
        const float depth = 
            weights.x * gs_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0][gs_src] +
            weights.y * gs_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0][gs_src] +
            weights.z * gs_depth[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1][gs_src] +
            weights.w * gs_depth[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1][gs_src];
        const float4 diffuse = 
            weights.x * gs_diffuse[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0][gs_src] +
            weights.y * gs_diffuse[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0][gs_src] +
            weights.z * gs_diffuse[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1][gs_src] +
            weights.w * gs_diffuse[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1][gs_src];
        const float2 diffuse2 = 
            weights.x * gs_diffuse2[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 0][gs_src] +
            weights.y * gs_diffuse2[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 0][gs_src] +
            weights.z * gs_diffuse2[group_thread_index.x * 2 + 0][group_thread_index.y * 2 + 1][gs_src] +
            weights.w * gs_diffuse2[group_thread_index.x * 2 + 1][group_thread_index.y * 2 + 1][gs_src];
            
        const uint2 group_index = thread_index / RTGI_PRE_BLUR_PREPARE_X;
        const uint2 mip_group_base_index = group_index * remaining_block_size;

        const uint gs_dst = (gs_src + 1u) & 0x1u;

        gs_depth[group_thread_index.x][group_thread_index.y][gs_dst] = depth;
        gs_diffuse[group_thread_index.x][group_thread_index.y][gs_dst] = diffuse;

        gs_diffuse2[group_thread_index.x][group_thread_index.y][gs_dst] = diffuse2;
        push.attach.reconstructed_diffuse_history[mip].get()[mip_group_base_index + group_thread_index] = diffuse;
        push.attach.reconstructed_diffuse2_history[mip].get()[mip_group_base_index + group_thread_index] = float4(diffuse2, depth, 0.0f);
    }
}

[shader("compute")]
[numthreads(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y,1)]
func entry_flatten(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID)
{
    let push = rtgi_pre_blur_flatten_push;

    // Load and precalculate constants
    CameraInfo *camera = &push.attach.globals->view_camera;
    const float2 inv_half_res_render_target_size = rcp(float2(push.size));
    const uint2 half_res_index = dtid.xy;
    const uint2 clamped_index = min( half_res_index, push.size - 1u );      // Can not early out because we perform group shared memory barriers later!

    // Load Pixel Data
    const float depth = push.attach.view_cam_half_res_depth.get()[clamped_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_normals.get()[clamped_index]);

    if (any(dtid.xy >= push.size) || depth == 0.0f)
    {
        return;
    }

    // Reconstruct pixel positions based on depth
    const float pixel_vs_depth = linearise_depth(depth, camera.near_plane);
    const float2 uv = (float2(clamped_index.xy) + 0.5f) * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 vs_position = mul(camera.view, float4(world_position, 1.0f)).xyz;
    const float3 vs_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;

    const float base_weight = 1.0f;
    const float outer_sample_weight = 1.0f / 128.0f;
    float4 diffuse = push.attach.diffuse_raw.get()[clamped_index] * base_weight;
    float2 diffuse2 = push.attach.diffuse2_raw.get()[clamped_index] * base_weight;
    float weight_acc = base_weight;
    int spread = 1;
    if (push.attach.globals.rtgi_settings.firefly_flatten_filter_enabled)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            [unroll]
            for (int y = -1; y <= 1; ++y)
            {
                if (x == 0 && y == 0) { continue; }

                int2 offset = int2(x,y);
                float weight = 1.0f;
                const int2 max_index = push.size - 1;
                int2 load_idx = int2(dtid.xy) + int2(offset);
                load_idx = flip_oob_index(load_idx, max_index);

                const float sample_depth = push.attach.view_cam_half_res_depth.get()[load_idx];
                
                const float2 sample_ndc_xy = ndc.xy + float2(offset) * inv_half_res_render_target_size * 2;
                const float3 sample_value_ndc = float3(sample_ndc_xy, sample_depth);
                const float4 sample_value_vs_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
                const float3 sample_value_vs = sample_value_vs_pre_div.xyz * rcp(sample_value_vs_pre_div.w);
                const float plane_distance = planar_surface_distance(inv_half_res_render_target_size, camera.near_plane, depth, vs_position, vs_normal, sample_value_vs);
                const float geometric_weight_real = abs(plane_distance) < float(spread) ? 1.0f : 0.0f;
                const float sample_weight = geometric_weight_real * outer_sample_weight;

                diffuse += push.attach.diffuse_raw.get()[load_idx] * sample_weight;
                diffuse2 += push.attach.diffuse2_raw.get()[load_idx] * sample_weight;
                weight_acc += sample_weight;
            }
        } 
    }

    push.attach.flattened_diffuse.get()[dtid.xy] = diffuse * rcp(weight_acc);
    push.attach.flattened_diffuse2.get()[dtid.xy] = diffuse2 * rcp(weight_acc);
}

__generic<uint N>
func linear_to_perceptual(vector<float, N> v) -> vector<float, N> 
{
    return log(max(v, 0.01f) + 1.0f);
}

__generic<uint N>
func perceptual_to_linear(vector<float, N> v) -> vector<float, N> 
{
    return exp(v) - 1.0f;
}

[shader("compute")]
[numthreads(RTGI_PRE_BLUR_PREPARE_X, RTGI_PRE_BLUR_PREPARE_Y,1)]
func entry_prepare(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID)
{
    let push = rtgi_pre_blur_prepare_push;

    // Load and precalculate constants
    CameraInfo *camera = &push.attach.globals->view_camera;
    const float2 inv_half_res_render_target_size = rcp(float2(push.size));
    const uint2 half_res_index = dtid.xy;
    const uint2 clamped_index = min( half_res_index, push.size - 1u );      // Can not early out because we perform group shared memory barriers later!

    // Load Pixel Data
    const float depth = push.attach.view_cam_half_res_depth.get()[clamped_index];
    const float4 diffuse = push.attach.diffuse_raw.get()[clamped_index];
    const float2 diffuse2 = push.attach.diffuse2_raw.get()[clamped_index];
    const float pixel_samplecnt = push.attach.half_res_samplecnt.get()[clamped_index];
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
    const int FILTER_WIDTH = 2;
    const int FILTER_STRIDE = 1;
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

            float4 sample_diffuse = push.attach.diffuse_raw.get()[load_idx];
            float2 sample_diffuse2 = push.attach.diffuse2_raw.get()[load_idx];
            const float sample_depth = push.attach.view_cam_half_res_depth.get()[load_idx];
            const float sample_validity = push.attach.half_res_samplecnt.get()[load_idx];
            const bool is_sky = sample_depth == 0.0f;

            if (is_sky || (x == 0 && y == 0))
            {
                continue;
            }

            {
                float geometric_mean_acc_value = linear_to_perceptual(sample_diffuse.w);
                y_mean_geometric_acc += geometric_mean_acc_value;
                valid_neightborhood_samples += 1.0f;
                y_geometric_variance_acc += square(geometric_mean_acc_value);
                y_mean_acc += sample_diffuse.w;
                y_variance_acc += square(sample_diffuse.w);
                y_max = max(y_max, sample_diffuse.w);
            }

            // if (abs(x) < 2 && abs(y) < 2)
            {
                const float2 sample_ndc_xy = ndc.xy + float2(x,y) * FILTER_STRIDE * inv_half_res_render_target_size * 2;
                const float3 sample_value_ndc = float3(sample_ndc_xy, sample_depth);
                const float4 sample_value_vs_pre_div = mul(camera.inv_proj, float4(sample_value_ndc, 1.0f));
                const float3 sample_value_vs = sample_value_vs_pre_div.xyz * rcp(sample_value_vs_pre_div.w);
                
                const float GEO_WEIGHT_THRESHOLD = 1.0f * FILTER_STRIDE;
                // MUST IGNORE POSITIVE PLANE DISTANCE!
                // When ignoring positive plane distance, the sample footprint remains high for enclosed pixels.
                // This MUST happen, because enclosed pixels have no valid mips to sample!
                const bool IGNORE_POSITIVE_PLANE_DISTANCE = true;
                const float geometric_weight = planar_surface_weight(inv_half_res_render_target_size, camera.near_plane, depth, vs_position, vs_normal, sample_value_vs, GEO_WEIGHT_THRESHOLD, IGNORE_POSITIVE_PLANE_DISTANCE);
                const float weight = geometric_weight * (is_sky ? 0.0f : 1.0f);

                foreground_sample_weight += weight;

                {
                    const float plane_distance = planar_surface_distance(inv_half_res_render_target_size, camera.near_plane, depth, vs_position, vs_normal, sample_value_vs);
                    const float geometric_weight_real = abs(plane_distance) < 2.0f ? 1.0f : 0.0f;
                    blurred_weight_acc += geometric_weight_real;
                    blurred_diffuse_acc += geometric_weight_real * sample_diffuse;
                    blurred_diffuse2_acc += geometric_weight_real * sample_diffuse2;
                }
            }

            // Accumulate neighbors that are close to or behind current pixel.
            background_sample_weight += (1.0f - clamp(1000.0f * (sample_depth - depth), 0.0f, 1.0f));
        }
    }

    // Foreground footprint quality estimation
    {
        foreground_footprint_quality = foreground_sample_weight * rcp(FILTER_TAPS_TOTAL);
    }

    // Firefly Filter + Background Pixel Suppression
    if (valid_neightborhood_samples > 1.0f)
    {
        const float BACKGROUND_WEIGHT_MIN = 0.3f;
        const float BACKGROUND_WEIGHT_MAX = 0.5f;
        #if RTGI_FIREFLY_FILTER_TIGHT_AGRESSIVE
            const float tight_neighborgood_factor = (clamp(background_sample_weight * rcp(FILTER_TAPS_TOTAL), BACKGROUND_WEIGHT_MIN, BACKGROUND_WEIGHT_MAX) - BACKGROUND_WEIGHT_MIN) * rcp(BACKGROUND_WEIGHT_MAX - BACKGROUND_WEIGHT_MIN);
        #else
            const float tight_neighborgood_factor = 1.0f;
        #endif

        const float EPSILON = 0.00000001f;

        const float4 blurred_diffuse = blurred_diffuse_acc * rcp(blurred_weight_acc + EPSILON);
        const float2 blurred_diffuse2 = blurred_diffuse2_acc * rcp(blurred_weight_acc + EPSILON);

        const float y_mean = y_mean_acc * rcp(valid_neightborhood_samples);
        const float y_variance = (y_variance_acc * rcp(valid_neightborhood_samples)) - square(y_mean);
        const float y_variance_relative = (y_variance) / (y_mean + EPSILON);

        const float y_mean_perceptual = y_mean_geometric_acc * rcp(valid_neightborhood_samples);
        const float y_mean_geometric = perceptual_to_linear(y_mean_perceptual);
        const float y_variance_geometric_relative = (y_variance) / (y_mean_geometric + EPSILON);

        const float CEILING_TEMPORAL_FACTOR = lerp(0.25f, 1.0f, sqrt(pixel_samplecnt * rcp(push.attach.globals.rtgi_settings.history_frames)));
        const float CEILING_FACTOR = max(1.0f, push.attach.globals.rtgi_settings.firefly_filter_ceiling * tight_neighborgood_factor * foreground_footprint_quality * CEILING_TEMPORAL_FACTOR);
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
        }

        // push.attach.debug_image.get()[dtid * 2 + uint2(0,1)] = float4(spatial_filter_radius_factor, linear_to_geometric_factor, 0, 0);
        // push.attach.debug_image.get()[dtid * 2 + uint2(0,0)] = float4(spatial_filter_radius_factor, linear_to_geometric_factor, 0, 0);
        // push.attach.debug_image.get()[dtid * 2 + uint2(1,1)] = float4(spatial_filter_radius_factor, linear_to_geometric_factor, 0, 0);
        // push.attach.debug_image.get()[dtid * 2 + uint2(1,0)] = float4(spatial_filter_radius_factor, linear_to_geometric_factor, 0, 0);
    }

    gs_depth[gtid.x][gtid.y][0] = depth;
    push.attach.reconstructed_diffuse_history[0].get()[dtid] = filtered_diffuse;
    push.attach.reconstructed_diffuse2_history[0].get()[dtid] = float4(filtered_diffuse2, depth, foreground_footprint_quality);
    gs_diffuse[gtid.x][gtid.y][0] = filtered_diffuse;
    gs_diffuse2[gtid.x][gtid.y][0] = filtered_diffuse2;

    // Mip 0:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid, gtid, 1, 0);

    // Mip 1:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid, gtid, 2, 1);

    // Mip 2:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid, gtid, 3, 0);

    // Mip 3:
    GroupMemoryBarrierWithGroupSync();
    downsample_mip_linear(dtid, gtid, 4, 1);
}

[shader("compute")]
[numthreads(RTGI_PRE_BLUR_APPLY_X,RTGI_PRE_BLUR_APPLY_Y,1)]
func entry_apply(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID)
{
    let push = rtgi_reconstruct_history_apply_diffuse_push;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Calculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const uint2 halfres_pixel_index = dtid.xy;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const float2 sv_xy = float2(halfres_pixel_index) + 0.5f;
    const float2 uv = sv_xy * inv_half_res_render_target_size;
    
    // Load pixel Data
    const float4 fetch0 = push.attach.reconstructed_diffuse_history.get().Load(int3(halfres_pixel_index, 0));
    const float4 fetch1 = push.attach.reconstructed_diffuse2_history.get().Load(int3(halfres_pixel_index, 0));
    const float pixel_depth = fetch1.b;
    const float pixel_samplecnt = push.attach.rtgi_samplecnt.get()[halfres_pixel_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_normals.get()[halfres_pixel_index]);
    float4 diffuse = fetch0;
    float2 diffuse2 = fetch1.xy;

    // Calculate pixel attributes
    const float3 vs_pixel_normal = mul(camera.view, float4(pixel_face_normal, 0.0f)).xyz;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 vs_position_pre_div = mul(camera.inv_proj, float4(ndc, 1.0f));
    const float3 vs_position = -vs_position_pre_div.xyz / vs_position_pre_div.w;

    if (pixel_depth == 0.0f)
    {
        return;
    }

    int mip = 0;
    float plane_pixel_dist_acceptance_threshold = 1.0f;
    {
        const float MIN_MIP = 1.0f;               // Base 2x2 blur helps the stochastic spatial filtering later. 
        const float MAX_FOOTPRINT_MIP = 4.0f;     // Pixels with poor spatial footprint need VERY high mips to become stable.
        const float MAX_MIP_DISOCCLUSIONS = 3.0f; // Disocclusions mips above 3 cause too much blur. mip 3 ~= 8-16px blur, must be roughly same size as spatial filter kernel.

        // A plane pixel dist of 1 is provides good acceptance and sharp edge rejection for usual surfaces.
        // For pixels with poor spatial footprint quality, we drastically increase the threshold to allow a lot more blurring across edges.
        const float MAX_PLANE_PIXEL_DIST_ACCEPTANCE_THRESHOLD = 8.0f;
        const float MIN_PLANE_PIXEL_DIST_ACCEPTANCE_THRESHOLD = 0.75f; 

        const float CONSIDERED_SPATIAL_FOOTPRINT_QUALITY_RANGE = 1.0f; // sample footprints greater than this value are considered full quality
        
        const float MIN_DISOCCLUSION_FIX_FRAMES = push.attach.globals.rtgi_settings.history_frames * 0.5f;
        const float max_disocclusion_fix_frames = push.attach.globals.rtgi_settings.history_frames;

        const float min_relevant_footprint_quality = 0.8f;
        const float foreground_footprint_quality = min(1.0f, square(fetch1.a * rcp(min_relevant_footprint_quality)));

        // Square footprint quality to exaggerate the effect of poor footprints in the mid range of quality.
        const float foreground_footprint_quality_mip = lerp(MAX_FOOTPRINT_MIP, MIN_MIP, foreground_footprint_quality);
        plane_pixel_dist_acceptance_threshold = lerp(MAX_PLANE_PIXEL_DIST_ACCEPTANCE_THRESHOLD, MIN_PLANE_PIXEL_DIST_ACCEPTANCE_THRESHOLD, foreground_footprint_quality);

        const float max_samplecount = push.attach.globals.rtgi_settings.history_frames;
        const float disocclusion_blur_frames = lerp(max_disocclusion_fix_frames, MIN_DISOCCLUSION_FIX_FRAMES, foreground_footprint_quality);
        const float disocclusion_fix_value = min(1.0f, pixel_samplecnt * (1.0f / disocclusion_blur_frames));
        const float disocclusion_fix_strength = pow(disocclusion_fix_value, 1.0f);
        const float disocclusion_blur_mip = lerp(MAX_MIP_DISOCCLUSIONS, MIN_MIP, disocclusion_fix_strength);

        mip = int(ceil(max(disocclusion_blur_mip, foreground_footprint_quality_mip)) + 0.001f);

        // push.attach.debug_image.get()[halfres_pixel_index * 2 + uint2(0,1)] = float4(foreground_footprint_quality,mip,0,0);
        // push.attach.debug_image.get()[halfres_pixel_index * 2 + uint2(0,0)] = float4(foreground_footprint_quality,mip,0,0);
        // push.attach.debug_image.get()[halfres_pixel_index * 2 + uint2(1,1)] = float4(foreground_footprint_quality,mip,0,0);
        // push.attach.debug_image.get()[halfres_pixel_index * 2 + uint2(1,0)] = float4(foreground_footprint_quality,mip,0,0);
    }

    float4 reconstructed_diffuse;
    float2 reconstructed_diffuse2;
    bool reconstruction_valid = false;
    bool wants_reconstruction = mip != 0;
    while (mip > 0)
    {
        // The mip chain base size is round up to 8 to ensure all texels have an exact 2x2 -> 1 match between mip levels.
        // Due to this, the uv is shifted by some amount as there are padding texels on the border of the reconstructed history mip chain.
        // We round to multiple of 16, not 8, as the uv are calculated for the half size. The mip chain is quater size, so going from quater to half means increasing the alignment from 8 to 16.
        const float2 half_size_ru16 = float2(round_up_to_multiple(half_res_render_target_size.x, 16), round_up_to_multiple(half_res_render_target_size.y, 16));
        const float2 corrected_uv = sv_xy * rcp(half_size_ru16);

        const float2 mip_size = float2(uint2(half_size_ru16) >> uint(mip));
        const float2 inv_mip_size = rcp(mip_size);
        const Bilinear bilinear_filter_reconstruct = get_bilinear_filter( saturate( corrected_uv ), mip_size );
        
        const float4 diffuse00 = push.attach.reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,0), mip));
        const float4 diffuse10 = push.attach.reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,0), mip));
        const float4 diffuse01 = push.attach.reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,1), mip));
        const float4 diffuse11 = push.attach.reconstructed_diffuse_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,1), mip));
        const float4 reconstruct00_2 = push.attach.reconstructed_diffuse2_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,0), mip));
        const float4 reconstruct10_2 = push.attach.reconstructed_diffuse2_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,0), mip));
        const float4 reconstruct01_2 = push.attach.reconstructed_diffuse2_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(0,1), mip));
        const float4 reconstruct11_2 = push.attach.reconstructed_diffuse2_history.get().Load(int3(int2(bilinear_filter_reconstruct.origin) + int2(1,1), mip));
        const float4 diffuse00_2 = float4(reconstruct00_2.xy, 0.0f, 0.0f);
        const float4 diffuse10_2 = float4(reconstruct10_2.xy, 0.0f, 0.0f);
        const float4 diffuse01_2 = float4(reconstruct01_2.xy, 0.0f, 0.0f);
        const float4 diffuse11_2 = float4(reconstruct11_2.xy, 0.0f, 0.0f);
        const float depth00 = reconstruct00_2.z;
        const float depth10 = reconstruct10_2.z;
        const float depth01 = reconstruct01_2.z;
        const float depth11 = reconstruct11_2.z;

        const float4 depths = float4(depth00, depth10, depth01, depth11);
        
        // Calculate custom bilinear weight based on plane distance.
        // We want to keep as much as possible so we do NOT use a hard cutoff geometric weighting here.
        // Instead we always take all samples and agressively weigh them by the plane distance.
        // Later, if the sum of weights turns out to be too low, we reject the interpolation.
        const float4 plane_distances_in_pixel_size = depth_distances4(inv_mip_size, camera.near_plane, pixel_depth, vs_position, vs_pixel_normal, depths);
        float4 plane_distance_weights = square(square(1.0f / (1.0f + plane_distances_in_pixel_size)));
        plane_distance_weights *= plane_distances_in_pixel_size < plane_pixel_dist_acceptance_threshold ? 1.0f : 0.0f;

        const float4 weights_reconstruct = get_bilinear_custom_weights( bilinear_filter_reconstruct, plane_distance_weights );
        reconstructed_diffuse = apply_bilinear_custom_weights( diffuse00, diffuse10, diffuse01, diffuse11, weights_reconstruct );
        reconstructed_diffuse2 = apply_bilinear_custom_weights( diffuse00_2, diffuse10_2, diffuse01_2, diffuse11_2, weights_reconstruct ).rg;
        
        if (dot(plane_distance_weights, 1) > 0.0f)
        {
            reconstruction_valid = true;
            break;
        }
        else
        {
            mip -= 1;
        }
    }

    if (reconstruction_valid && push.attach.globals.rtgi_settings.disocclusion_filter_enabled)
    {
        diffuse = reconstructed_diffuse;
        diffuse2 = reconstructed_diffuse2;
    }

    push.attach.diffuse_filtered.get()[halfres_pixel_index] = diffuse;
    push.attach.diffuse2_filtered.get()[halfres_pixel_index] = diffuse2;
}