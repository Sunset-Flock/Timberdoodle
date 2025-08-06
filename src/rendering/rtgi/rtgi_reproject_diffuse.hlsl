#pragma once

#include "rtgi_reproject_diffuse.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiReprojectDiffusePush rtgi_denoise_diffuse_reproject_push;

[shader("compute")]
[numthreads(RTGI_DENOISE_DIFFUSE_X,RTGI_DENOISE_DIFFUSE_Y,1)]
func entry_reproject(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_denoise_diffuse_reproject_push;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Load and precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const CameraInfo camera_prev_frame = push.attach.globals->view_camera_prev_frame;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);
    const uint2 halfres_pixel_index = dtid;
    const float2 sv_xy = float2(halfres_pixel_index) + 0.5f;

    // Load half res depth and normal
    const float pixel_depth = push.attach.view_cam_half_res_depth.get()[halfres_pixel_index];
    const float pixel_vs_depth = linearise_depth(pixel_depth, camera.near_plane);
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_half_res_face_normals.get()[halfres_pixel_index]);

    if (pixel_depth == 0.0f)
    {
        push.attach.rtgi_samplecnt.get()[halfres_pixel_index] = 0;
        push.attach.rtgi_diffuse_accumulated.get()[dtid] = float4(0,0,0,0);
        return;
    }

    // Calculate pixel positions in cur and prev frame
    const float3 sv_pos = float3(sv_xy, pixel_depth);
    const float2 uv = sv_xy * inv_half_res_render_target_size;
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 world_position_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 world_position = world_position_pre_div.xyz / world_position_pre_div.w;
    const float3 world_position_prev_frame = world_position; // No support for dynamic object motion vectors yet.
    const float4 view_position_prev_frame_pre_div = mul(camera_prev_frame.view, float4(world_position_prev_frame, 1.0f));
    const float3 view_position_prev_frame = -view_position_prev_frame_pre_div.xyz / view_position_prev_frame_pre_div.w;
    const float4 ndc_prev_frame_pre_div = mul(camera_prev_frame.view_proj, float4(world_position_prev_frame, 1.0f));
    const float3 ndc_prev_frame = ndc_prev_frame_pre_div.xyz / ndc_prev_frame_pre_div.w;
    const float2 uv_prev_frame = ndc_prev_frame.xy * 0.5f + 0.5f;
    const float2 sv_xy_prev_frame = ( ndc_prev_frame.xy * 0.5f + 0.5f ) * half_res_render_target_size;
    const float3 sv_pos_prev_frame = float3( sv_xy_prev_frame, ndc_prev_frame.z );

    // Load previous frame half res depth
    const Bilinear bilinear_filter_at_prev_pos = get_bilinear_filter( saturate( uv_prev_frame ), half_res_render_target_size );
    const float2 reproject_gather_uv = ( float2( bilinear_filter_at_prev_pos.origin ) + 1.0 ) * inv_half_res_render_target_size;
    SamplerState nearest_clamp_s = push.attach.globals.samplers.nearest_clamp.get();
    const float4 depth_reprojected4 = push.attach.rtgi_depth_history.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float4 samplecnt_reprojected4 = push.attach.rtgi_samplecnt_history.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const uint4 face_normals_packed_reprojected4 = push.attach.rtgi_face_normal_history.get().GatherRed( nearest_clamp_s, reproject_gather_uv ).wzxy;
    const float3 view_space_normal = mul(camera_prev_frame.view, float4(pixel_face_normal, 0.0f)).xyz;

    // Calculate plane distance based occlusion and normal similarity
    float4 occlusion = float4(1.0f, 1.0f, 1.0f, 1.0f);
    {
        const float in_screen = all(uv_prev_frame > 0.0f && uv_prev_frame < 1.0f) ? 1.0f : 0.0f;
        const float4 normal_similarity = {
            max(0.0f, dot(pixel_face_normal, uncompress_normal_octahedral_32(face_normals_packed_reprojected4.x))),
            max(0.0f, dot(pixel_face_normal, uncompress_normal_octahedral_32(face_normals_packed_reprojected4.y))),
            max(0.0f, dot(pixel_face_normal, uncompress_normal_octahedral_32(face_normals_packed_reprojected4.z))),
            max(0.0f, dot(pixel_face_normal, uncompress_normal_octahedral_32(face_normals_packed_reprojected4.w)))
        };
        const float4 normal_weight = square(square(normal_similarity)); // sqrt as we do want to allow some blurring across very similar normals.
        const float4 geometry_weights = get_geometry_weight4(inv_half_res_render_target_size, camera.near_plane, pixel_depth, view_position_prev_frame, view_space_normal, depth_reprojected4);

        occlusion = geometry_weights * in_screen * normal_weight;
    }

    // Evaluate occlusion, determine disocclusion and sample weights
    const bool disocclusion = dot(1.0f, occlusion) < 0.999f;
    const float4 sample_weights = get_bilinear_custom_weights( bilinear_filter_at_prev_pos, occlusion );

    // Read in diffuse history
    const float4 history_r = push.attach.rtgi_diffuse_history.get().GatherRed( push.attach.globals.samplers.nearest_clamp.get(), reproject_gather_uv ).wzxy;
    const float4 history_g = push.attach.rtgi_diffuse_history.get().GatherGreen( push.attach.globals.samplers.nearest_clamp.get(), reproject_gather_uv ).wzxy;
    const float4 history_b = push.attach.rtgi_diffuse_history.get().GatherBlue( push.attach.globals.samplers.nearest_clamp.get(), reproject_gather_uv ).wzxy;
    const float4 s00 = float4(history_r.x, history_g.x, history_b.x, 0.0f);
    const float4 s10 = float4(history_r.y, history_g.y, history_b.y, 0.0f);
    const float4 s01 = float4(history_r.z, history_g.z, history_b.z, 0.0f);
    const float4 s11 = float4(history_r.w, history_g.w, history_b.w, 0.0f);
    float3 history = apply_bilinear_custom_weights( s00, s10, s01, s11, sample_weights ).rgb;

    if (any(isnan(history)))
    {
        history = float3(0,0,0);
    }

    // Calc new sample count
    float samplecnt = apply_bilinear_custom_weights( samplecnt_reprojected4.x, samplecnt_reprojected4.y, samplecnt_reprojected4.z, samplecnt_reprojected4.w, sample_weights ).x;
    samplecnt = min( samplecnt + 1.0f, push.attach.globals.rtgi_settings.history_frames );
    samplecnt = disocclusion ? 0u : samplecnt;
    const float history_blend = samplecnt / float(push.attach.globals.rtgi_settings.history_frames + 1.0f);
    push.attach.rtgi_samplecnt.get()[halfres_pixel_index] = samplecnt;

    // Read raw traced diffuse
    float4 raw = push.attach.rtgi_diffuse_raw.get()[halfres_pixel_index].rgba;

    // Write accumulated diffuse
    const float3 accumulated = lerp(history, raw.rgb, 0.5f * (1.0f - history_blend ));
    push.attach.rtgi_diffuse_accumulated.get()[dtid] = float4(accumulated, raw.a);
}