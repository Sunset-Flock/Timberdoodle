#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "pgi_update.inl"
#include "../../shader_lib/pgi.hlsl"
#include "../../shader_lib/misc.hlsl"
#include "../../shader_lib/transform.hlsl"

[[vk::push_constant]] PGIEvalScreenIrradiancePush enty_eval_screen_irradiance_push;

[shader("compute")]
[numthreads(PGI_EVAL_SCREEN_IRRADIANCE_XY, PGI_EVAL_SCREEN_IRRADIANCE_XY, 1)]
func enty_eval_screen_irradiance(uint2 dtid : SV_DispatchThreadID)
{
    let push = enty_eval_screen_irradiance_push;

    let clk_start = clockARB();

    if (any(dtid >= push.irradiance_image_size))
    {
        return;
    }

    const bool is_center_pixel = all(dtid == push.irradiance_image_size/2);
    if (is_center_pixel)
    {
        debug_pixel = true;
    }

    rand_seed(asuint(dtid.x + dtid.y * 13136.1235f) * push.attach.globals.frame_index);

    float depth = push.attach.main_cam_depth.get()[dtid];
    float3 face_normal = uncompress_normal_octahedral_32(push.attach.main_cam_face_normals.get()[dtid]);
    float3 detail_normal = uncompress_normal_octahedral_32(push.attach.main_cam_detail_normals.get()[dtid]);

    let camera = &push.attach.globals.main_camera;
    float3 ws_position = pixel_index_to_world_space(*camera, dtid, depth);
    float3 primary_ray = normalize(ws_position - camera.position);

    PGISampleInfo pgi_sample_info = PGISampleInfo();
    float3 pgi_indirect_irradiance = pgi_sample_probe_volume(
        push.attach.globals, &push.attach.globals.pgi_settings, pgi_sample_info,
        ws_position, camera.position, detail_normal, face_normal, 
        push.attach.probe_radiance.get(), push.attach.probe_visibility.get(), push.attach.probe_info.get(), push.attach.probe_requests.get_formatted()
    );

    push.attach.irradiance_depth.get()[dtid] = float4(pgi_indirect_irradiance, depth);

    let clk_end = clockARB();
    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_PGI_EVAL_CLOCKS)
    {
        push.attach.clocks_image.get()[dtid] = uint(clk_end - clk_start);
    }
}

[[vk::push_constant]] PGIUpscaleScreenIrradiancePush enty_upscale_screen_irradiance_push;

static const uint kRadius = 1;
static const uint kWidth = 1 + 2 * kRadius;
static const float kernel1D[kWidth] = {0.27901, 0.44198, 0.27901};
static const float kernel[kWidth][kWidth] = {
  {kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2]},
  {kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2]},
  {kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2]},
};

float DepthWeight(float linear_depth_a, float linear_depth_b, float3 normalCur, float3 viewDir, float4x4 proj, float phi)
{  
  float angleFactor = max(0.25, -dot(normalCur, viewDir));

  float diff = abs(linear_depth_a - linear_depth_b);
  return exp(-diff * angleFactor / phi);
}

float NormalWeight(float3 normalPrev, float3 normalCur, float phi)
{
  // float d = max(0.05, dot(normalCur, normalPrev));
  // return d * d;
  float3 dd = normalPrev - normalCur;
  float d = dot(dd, dd);
  return exp(-d / phi);
}

// Unused for now.
// Takes ~150us for 720p -> 1440p
groupshared float4 gs_half_irradiance_depth_preload[PGI_EVAL_SCREEN_IRRADIANCE_XY+2][PGI_EVAL_SCREEN_IRRADIANCE_XY+2];
groupshared float4 gs_half_normals_preload[PGI_EVAL_SCREEN_IRRADIANCE_XY+2][PGI_EVAL_SCREEN_IRRADIANCE_XY+2];
[shader("compute")]
[numthreads(PGI_EVAL_SCREEN_IRRADIANCE_XY, PGI_EVAL_SCREEN_IRRADIANCE_XY, 1)]
func entry_upscale_screen_irradiance(uint2 dtid : SV_DispatchThreadID, uint in_group_index : SV_GroupIndex, int2 group_id : SV_GroupID, int2 in_group_id : SV_GroupThreadID)
{
    #if 0
    let push = enty_upscale_screen_irradiance_push;
    let globals = push.attach.globals;

    if (any(dtid >= push.size))
    {
        return;
    }

    Texture2D<float4> half_irradiance_depth = push.attach.half_irradiance_depth.get();
    RWTexture2D<uint> half_normals = push.attach.half_normals.get();

    CameraInfo camera = push.attach.globals.main_camera;

    float full_depth = push.attach.main_cam_depth.get()[dtid];
    float full_depth_linear = linearise_depth(full_depth, camera.near_plane);
    float3 full_normal = uncompress_normal_octahedral_32(push.attach.main_cam_detail_normals.get()[dtid]);

    float3 ws_pos = pixel_index_to_world_space(camera, dtid, full_depth);
    float3 ray = normalize(ws_pos - camera.position);

    float3 acc_irrad = {};
    float acc_weight = {};

    // preload
    const int needed_preload_threads = square(PGI_EVAL_SCREEN_IRRADIANCE_XY+2);
    const int group_threads = square(PGI_EVAL_SCREEN_IRRADIANCE_XY);
    const int group_iters = round_up_div(needed_preload_threads, group_threads);
    const int2 group_base_half_index = (group_id * int(PGI_EVAL_SCREEN_IRRADIANCE_XY/2)) - 1;
    for (int group_iter = 0; group_iter < group_iters; ++group_iter)
    {
        int preload_flat_index = group_iter * group_threads + in_group_index;
        if (preload_flat_index >= needed_preload_threads)
            break;

        int2 preload_index = int2(preload_flat_index % (PGI_EVAL_SCREEN_IRRADIANCE_XY+2), preload_flat_index / (PGI_EVAL_SCREEN_IRRADIANCE_XY+2));
        int2 load_index = clamp(preload_index + group_base_half_index, int2(0,0), int2(push.size/2-1));
        float4 half_irrad_depth = half_irradiance_depth[load_index];
        gs_half_irradiance_depth_preload[preload_index.x][preload_index.y] = half_irrad_depth;
        float3 half_normal = uncompress_normal_octahedral_32(half_normals[load_index]);
        gs_half_normals_preload[preload_index.x][preload_index.y] = float4(half_normal, 0.0f);
    }

    GroupMemoryBarrierWithGroupSync();

    for (int col = 0; col < kWidth; col++)
    {
        for (int row = 0; row < kWidth; row++)
        {
            int2 offset = int2(row - kRadius, col - kRadius);
            int2 pos = int2(dtid/2) + offset;
            
            if (any(pos >= int2(push.size/2)) || any(pos < int2(0,0)))
            {
                continue;
            }

            int2 group_local_pos = in_group_id/2 + offset + int2(1,1);

            float kernelWeight = kernel[row][col];

            float4 irrad_depth = gs_half_irradiance_depth_preload[group_local_pos.x][group_local_pos.y];
            float3 normal = gs_half_normals_preload[group_local_pos.x][group_local_pos.y].xyz;
            
            //float4 irrad_depth = half_irradiance_depth[pos];
            //float3 normal = uncompress_normal_octahedral_32(half_normals[pos]);

            float3 irradiance = irrad_depth.rgb;
            float depth_linear = linearise_depth(irrad_depth.w, camera.near_plane);

            float normalWeight = NormalWeight(normal, full_normal, 0.01f /*TODO*/);
            float depthWeight = DepthWeight(depth_linear, full_depth_linear, full_normal, ray, camera.proj, 0.1f /*TODO*/);
            
            float weight = depthWeight * normalWeight;
            acc_irrad += irradiance * weight * kernelWeight;
            acc_weight += weight * kernelWeight;
        }
    }

    float3 upscaled = acc_irrad * rcp(acc_weight);
    
    push.attach.full_res_pgi_irradiance.get()[dtid].rgb = upscaled;
    #endif
}