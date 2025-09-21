#include <daxa/daxa.inl>

#include "clouds.inl"

[[vk::push_constant]] ComposeCloudsPush compose_clouds_push;

#define GS_PRELOAD_WIDTH (COMPOSE_CLOUDS_DISPATCH_X/2+2)
groupshared float4 gs_half_color_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float gs_half_depth_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];

[shader("compute")]
[numthreads(COMPOSE_CLOUDS_DISPATCH_X, COMPOSE_CLOUDS_DISPATCH_Y)]
func entry_compose(uint2 dtid : SV_DispatchThreadID, uint in_group_index : SV_GroupIndex, int2 group_id : SV_GroupID, int2 in_group_id : SV_GroupThreadID)
{
    let push = compose_clouds_push;

    // Precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const uint2 full_res_pixel_index = dtid.xy;
    const float2 full_res_render_target_size = push.attach.globals.settings.render_target_size.xy;
    const float2 inv_half_res_render_target_size = rcp(float2(full_res_render_target_size / 2));
    const float2 inv_full_res_render_target_size = rcp(full_res_render_target_size);

    {
        const int2 group_base_half_index = int2(group_id * int(COMPOSE_CLOUDS_DISPATCH_X/2)) - 1;
        if (all(in_group_id < GS_PRELOAD_WIDTH))
        {
            const int2 preload_index = in_group_id;
            const int2 load_index = clamp(preload_index + group_base_half_index, int2(0,0), int2(push.main_screen_resolution/2-1));
            const float4 half_cloud_color = push.attach.clouds_raymarched_result.get()[load_index];            
            const float4 depths = float4(
                push.attach.view_cam_depth.get()[load_index*2 + uint2(0,0)],
                push.attach.view_cam_depth.get()[load_index*2 + uint2(0,1)],
                push.attach.view_cam_depth.get()[load_index*2 + uint2(1,0)],
                push.attach.view_cam_depth.get()[load_index*2 + uint2(1,1)],
            );
            const float half_depth = max(max(depths.x, depths.y), max(depths.z, depths.w));
            
            gs_half_color_preload[preload_index.x][preload_index.y] = half_cloud_color;
            gs_half_depth_preload[preload_index.x][preload_index.y] = half_depth;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (any(dtid.xy >= push.main_screen_resolution))
    {
        return;
    }

    const float2 sv_xy = float2(full_res_pixel_index) + 0.5f;
    const float2 uv = sv_xy * inv_full_res_render_target_size;
    
    // Load pixel depth and face normal
    const float pixel_depth = push.attach.view_cam_depth.get()[full_res_pixel_index];

    // Calc pixel view attributes
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 position_vs_pre_div = mul(camera.inv_proj, float4(ndc, 1.0f));
    const float3 position_vs = -position_vs_pre_div.xyz / position_vs_pre_div.w;

    // Tent 3x3 filter the preloaded values
    static const uint TENT_WIDTH = 3;
    static const float GAUSS_WEIGHTS_5[5] = { 1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f };
    static const float3 TENT_WEIGHTS_LEFT_3 = { GAUSS_WEIGHTS_5[0] + GAUSS_WEIGHTS_5[1], GAUSS_WEIGHTS_5[2] + GAUSS_WEIGHTS_5[3], GAUSS_WEIGHTS_5[4] + 0.0f };
    const uint2 rtgi_subpixel_index = (full_res_pixel_index & 0x1);
    const float3 tent_weights_x = rtgi_subpixel_index.x == 0 ? TENT_WEIGHTS_LEFT_3 : TENT_WEIGHTS_LEFT_3.zyx;
    const float3 tent_weights_y = rtgi_subpixel_index.y == 0 ? TENT_WEIGHTS_LEFT_3 : TENT_WEIGHTS_LEFT_3.zyx;
    float4 acc_cloud_color = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float acc_weight = 0.0f;
    for (int col = 0; col < TENT_WIDTH; col++)
    {
        for (int row = 0; row < TENT_WIDTH; row++)
        {
            const int2 offset = int2(row - 1, col - 1);

            // Load values from gs
            const int2 sample_gs_index = (in_group_id)/2 + offset + int2(1,1);
            const float4 cloud_color = gs_half_color_preload[sample_gs_index.x][sample_gs_index.y];
            const float cloud_trace_depth = gs_half_depth_preload[sample_gs_index.x][sample_gs_index.y];

            // Calculate weights
            const float tent_weight = tent_weights_x[row] * tent_weights_y[col];
            const float depth_diff = min(abs(cloud_trace_depth - pixel_depth) * 100.0f, 1.0f);  // depth difference is from 0 (identical depth) to 1 (at 1% depth difference)
            const float depth_similarity = pow(1.0f - depth_diff, 512.0f);                      // depth similarity is pow'd to give better samples exponentially stronger weighting
            const float depth_base_weight = 0.00001f;                                           // on full interpolation failure we set a small weight to get at least something :)
            const float geometry_weight = depth_similarity + depth_base_weight;
            const float weight = tent_weight * geometry_weight;

            acc_cloud_color += weight * cloud_color;
            acc_weight += weight;
        }
    }

    // a channel contains transmittance
    const float4 upscaled_cloud_color = acc_cloud_color * rcp(acc_weight + 0.0000001f);
    const float3 pixel_color = push.attach.color_image.get()[dtid];
    const float3 composed_color = pixel_color * (upscaled_cloud_color.a) + (upscaled_cloud_color.rgb);
    push.attach.color_image.get()[dtid] = composed_color;
}