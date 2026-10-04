#pragma once

#include "rtgi_upscale.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/pack_unpack.hlsl"
#include "rtgi_shared.hlsl"

[[vk::push_constant]] RtgiUpscaleDiffusePush rtgi_upscale_diffuse_push;

// Hack: where the specular upscale finds almost no matching half-res taps (total weight -> 0), the reflection
// falls back to a LENIENT blend of the same 9 taps: tent x view-space distance falloff x a very soft detail
// normal weight ((dot * 0.5 + 0.5)^2, never 0 for any orientation). No gloss weight, no non-directional cutoff.
// Blends in linearly below RTGI_UPSCALE_SPECULAR_LENIENT_FALLBACK_WEIGHT (tent weights sum to 1 for a full match).
#define RTGI_UPSCALE_SPECULAR_LENIENT_FALLBACK 1
#define RTGI_UPSCALE_SPECULAR_LENIENT_FALLBACK_WEIGHT 0.1f

#define GS_PRELOAD_WIDTH (RTGI_UPSCALE_DIFFUSE_X/2+2)
groupshared float4 gs_half_diffuse_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float2 gs_half_diffuse2_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float4 gs_half_normals_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float4 gs_half_vs_positions[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float gs_half_samplecount[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float4 gs_half_specular_preload[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH];
groupshared float4 gs_half_specular_normal_roughness[GS_PRELOAD_WIDTH][GS_PRELOAD_WIDTH]; // .xyz specular normal, .w roughness

[shader("compute")]
[numthreads(RTGI_UPSCALE_DIFFUSE_X,RTGI_UPSCALE_DIFFUSE_Y,1)]
func entry_upscale_diffuse(uint2 dtid : SV_DispatchThreadID, uint in_group_index : SV_GroupIndex, int2 group_id : SV_GroupID, int2 in_group_id : SV_GroupThreadID)
{
    let push = rtgi_upscale_diffuse_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    // Convergence statistics: the accumulate pass (before this one) summed them.
    if (all(dtid == uint2(0, 0)))
    {
        push.attach.globals.readback.rtgi_convergence_diffuse_sum = push.attach.ray_counters->convergence_diffuse_sum;
        push.attach.globals.readback.rtgi_convergence_specular_sum = push.attach.ray_counters->convergence_specular_sum;
        push.attach.globals.readback.rtgi_convergence_pixels = push.attach.ray_counters->convergence_pixels;
        for (uint b = 0u; b < RTGI_CONVERGENCE_BUCKETS; ++b)
        {
            push.attach.globals.readback.rtgi_convergence_histogram_diffuse[b] = push.attach.ray_counters->convergence_histogram_diffuse[b];
            push.attach.globals.readback.rtgi_convergence_histogram_specular[b] = push.attach.ray_counters->convergence_histogram_specular[b];
        }
    }

    // Precalculate constants
    CameraInfo* camera = &push.attach.globals->view_camera;
    const uint2 full_res_pixel_index = min(dtid.xy, push.size-1);
    const float2 full_res_render_target_size = push.attach.globals.settings.render_target_size.xy;
    const float2 inv_half_res_render_target_size = rcp(float2(full_res_render_target_size / 2));
    const float2 inv_full_res_render_target_size = rcp(full_res_render_target_size);
    const float2 sv_xy = float2(full_res_pixel_index) + 0.5f;
    const float2 uv = sv_xy * inv_full_res_render_target_size;
    
    // Load pixel depth and face normal
    const float pixel_depth = push.attach.view_cam_depth.get()[full_res_pixel_index];
    const float3 pixel_face_normal = uncompress_normal_octahedral_32(push.attach.view_cam_face_normals.get()[full_res_pixel_index]);
    const float3 pixel_detail_normal = uncompress_normal_octahedral_32(push.attach.view_camera_detail_normal_image.get()[full_res_pixel_index]);

    // Calc pixel view attributes
    const float3 ndc = float3(uv * 2.0f - 1.0f, pixel_depth);
    const float4 position_vs_pre_div = mul(camera.inv_proj, float4(ndc, 1.0f));
    const float3 position_vs = -position_vs_pre_div.xyz / position_vs_pre_div.w;
    const float3 pixel_face_normal_vs = mul(camera.view, float4(pixel_face_normal,0.0f)).xyz;

    const float4 position_ws_pre_div = mul(camera.inv_view_proj, float4(ndc, 1.0f));
    const float3 position_ws = -position_ws_pre_div.xyz / position_ws_pre_div.w;

    // Upscale Spatial Result
    float3 upscaled_diffuse = (float3)0;
    {

        // Preload Surrounding half res rtgi values
        // Each Group works on a 8x8 full res tile.
        // To full reconstruct the full res tile we need a (8/2 + 2)^2 section of the half res diffuse.
        {
            Texture2D<float4> half_res_diffuse_tex = push.attach.diffuse_half_res.get();
            Texture2D<float2> half_res_diffuse2_tex = push.attach.diffuse2_half_res.get();
            Texture2D<float> half_res_depth_tex = push.attach.view_cam_half_res_depth.get();
            Texture2D<uint> half_res_face_normal_tex = push.attach.view_cam_half_res_face_normals.get();

            const int2 group_base_half_index = (group_id * int(RTGI_UPSCALE_DIFFUSE_X/2)) - 1;
            const int2 preload_index = in_group_id;
            if (all(preload_index < GS_PRELOAD_WIDTH))
            {
                const int2 load_index = clamp(preload_index + group_base_half_index, int2(0,0), int2(push.size/2-1));
                const float depth = half_res_depth_tex[load_index];
                const float4 sh_y = half_res_diffuse_tex[load_index];
                const float2 cocg = half_res_diffuse2_tex[load_index];
                gs_half_diffuse_preload[preload_index.x][preload_index.y] = sh_y;
                gs_half_diffuse2_preload[preload_index.x][preload_index.y] = cocg;
                gs_half_specular_preload[preload_index.x][preload_index.y] = push.attach.specular_half_res.get()[load_index];
                const uint normal_roughness = push.attach.specular_normal_roughness_half_res.get()[load_index];
                gs_half_specular_normal_roughness[preload_index.x][preload_index.y] = float4(
                    unpack_normal_roughness_normal(normal_roughness), unpack_normal_roughness_roughness(normal_roughness));
                
                const float3 half_normal = uncompress_normal_octahedral_32(half_res_face_normal_tex[load_index]);
                gs_half_normals_preload[preload_index.x][preload_index.y] = float4(half_normal, 0.0f);

                const int2 sample_half_res_idx = load_index;
                const float2 sample_uv = float2(sample_half_res_idx + 0.5f) * inv_half_res_render_target_size;
                const float3 sample_vs = rtgi_half_res_depth_to_neg_view_space(*camera, sample_uv * 2.0f - 1.0f, depth);
                gs_half_vs_positions[preload_index.x][preload_index.y] = float4(sample_vs, depth);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        
        if (any(dtid.xy >= push.size))
        {
            return;
        }

        if (pixel_depth == 0.0f)
        {
            return;
        }

        const float pixel_width_ws_rcp = rcp(calc_pixel_width_ws(inv_half_res_render_target_size, camera.near_plane, pixel_depth));

        // Tent 3x3 filter the preloaded values
        static const uint TENT_WIDTH = 3;
        static const float GAUSS_WEIGHTS_5[5] = { 1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f };
        static const float3 TENT_WEIGHTS_LEFT_3 = { GAUSS_WEIGHTS_5[0] + GAUSS_WEIGHTS_5[1], GAUSS_WEIGHTS_5[2] + GAUSS_WEIGHTS_5[3], GAUSS_WEIGHTS_5[4] + 0.0f };
        // As each pixel is only half the size of a rtgi diffuse pixel, each rtgi diffuse pixel holds exactly 4 screen pixels.
        // We calculate the position of our screen pixel within the rtgi diffuse pixel to get a better sample weighting.
        const uint2 rtgi_subpixel_index = (full_res_pixel_index & 0x1);
        const float3 tent_weights_x = rtgi_subpixel_index.x == 0 ? TENT_WEIGHTS_LEFT_3 : TENT_WEIGHTS_LEFT_3.zyx;
        const float3 tent_weights_y = rtgi_subpixel_index.y == 0 ? TENT_WEIGHTS_LEFT_3 : TENT_WEIGHTS_LEFT_3.zyx;
        float4 acc_diffuse = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        float2 acc_diffuse2 = float2( 0.0f, 0.0f );
        float4 acc_specular = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        float4 fallback_acc_specular = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        // Specular gets its own weights: the full-res DETAIL normal against each tap's half-res specular normal, as
        // sharp as the lobe (roughness of the nearest half-res texel). Keeps normal-map detail in reflections.
        float4 acc_specular_detail = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        float acc_specular_detail_weight = 0.0f;
        float4 lenient_acc_specular = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        float lenient_acc_weight = 0.0f;
        const int2 nearest_gs_index = in_group_id/2 + int2(1,1);
        const float pixel_specular_roughness = gs_half_specular_normal_roughness[nearest_gs_index.x][nearest_gs_index.y].w;
        float acc_weight = 0.0f;
        float4 fallback_acc_diffuse = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        float2 fallback_acc_diffuse2 = float2( 0.0f, 0.0f );
        float fallback_acc_weight = 0.0f;
        float acc_geo_weight = 0.0f;
        for (int col = 0; col < TENT_WIDTH; col++)
        {
            for (int row = 0; row < TENT_WIDTH; row++)
            {
                const int2 offset = int2(row - 1, col - 1);
                const int2 pos = int2(dtid/2) + offset;

                // Load values from gs
                const int2 sample_gs_index = in_group_id/2 + offset + int2(1,1);
                const float4 sample_sh_y = gs_half_diffuse_preload[sample_gs_index.x][sample_gs_index.y];
                const float2 sample_cocg = gs_half_diffuse2_preload[sample_gs_index.x][sample_gs_index.y];
                const float3 sample_face_normal = gs_half_normals_preload[sample_gs_index.x][sample_gs_index.y].xyz;

                // Calculate sample position
                const float3 sample_vs = gs_half_vs_positions[sample_gs_index.x][sample_gs_index.y].xyz;
                const float sample_depth = gs_half_vs_positions[sample_gs_index.x][sample_gs_index.y].w;

                // Calculate weights
                const float tent_weight = tent_weights_x[row] * tent_weights_y[col];
                const float geometry_weight = (abs(calc_plane_distance(position_vs, pixel_face_normal_vs, sample_vs)) * pixel_width_ws_rcp) < 3.0f;
                const float normal_weight = square(square(max(0.0f, dot(sample_face_normal, pixel_face_normal))));
                const float weight = tent_weight * geometry_weight * normal_weight;

                if (sample_depth != 0.0f)
                {
                    acc_diffuse += weight * sample_sh_y;
                    acc_diffuse2 += weight * sample_cocg;
                    const float4 sample_specular = gs_half_specular_preload[sample_gs_index.x][sample_gs_index.y];
                    acc_specular += weight * sample_specular;
                    const float3 sample_specular_normal = gs_half_specular_normal_roughness[sample_gs_index.x][sample_gs_index.y].xyz;
                    const float specular_weight = tent_weight * geometry_weight *
                        rtgi_specular_normal_weight(pixel_detail_normal, sample_specular_normal, pixel_specular_roughness);
                    acc_specular_detail += specular_weight * sample_specular;
                    acc_specular_detail_weight += specular_weight;
                    acc_weight += weight;
                    acc_geo_weight += geometry_weight;

                    // Fallback calculation:
                    const float vs_dst_weight = 1.0f * rcp( 1.0f + square(dot(position_vs - sample_vs, position_vs - sample_vs)));
                    const float fallback_weight = tent_weight * vs_dst_weight * (0.1f + max(0.0f, dot(sample_face_normal, pixel_face_normal)));
                    fallback_acc_diffuse += fallback_weight * sample_sh_y;
                    fallback_acc_diffuse2 += fallback_weight * sample_cocg;
                    fallback_acc_specular += fallback_weight * sample_specular;
                    fallback_acc_weight += fallback_weight;

                    const float lenient_weight = tent_weight * vs_dst_weight * square(saturate(dot(pixel_detail_normal, sample_specular_normal) * 0.5f + 0.5f));
                    lenient_acc_specular += lenient_weight * sample_specular;
                    lenient_acc_weight += lenient_weight;
                }
            }
        }

        // Write upscaled diffuse:
        float4 upscaled_sh_y = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        float2 upscaled_cocg = float2( 0.0f, 0.0f );
        float4 upscaled_specular = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        // Only fall back when NO half-res tap matched this full-res pixel's surface (e.g. thin geometry that
        // does not exist at half res). Normalizing a zero weight sum would otherwise output black.
        if (acc_weight > 1e-5f)
        {
            upscaled_sh_y = acc_diffuse * rcp(acc_weight + 0.0000001f);
            upscaled_cocg = acc_diffuse2 * rcp(acc_weight + 0.0000001f);
            upscaled_specular = acc_specular * rcp(acc_weight + 0.0000001f);
        }
        else
        {
            upscaled_sh_y = fallback_acc_diffuse * rcp(fallback_acc_weight + 0.0000001f);
            upscaled_cocg = fallback_acc_diffuse2 * rcp(fallback_acc_weight + 0.0000001f);
            upscaled_specular = fallback_acc_specular * rcp(fallback_acc_weight + 0.0000001f);
        }

        if (!rtgi_settings.upscale_enabled)
        {
            const int2 sample_gs_index = in_group_id/2 + int2(1,1);
            upscaled_sh_y = gs_half_diffuse_preload[sample_gs_index.x][sample_gs_index.y];
            upscaled_cocg = gs_half_diffuse2_preload[sample_gs_index.x][sample_gs_index.y];
            upscaled_specular = gs_half_specular_preload[sample_gs_index.x][sample_gs_index.y];
        }
        // Prefer the detail-normal weighted specular; keep the diffuse-style weights as the fallback when no
        // half-res tap matches the full-res detail normal (thin details, strong normal maps).
        if (rtgi_settings.upscale_enabled && rtgi_settings.specular_upscale_detail_weighting != 0 && acc_specular_detail_weight > 1e-4f)
        {
            upscaled_specular = acc_specular_detail * rcp(acc_specular_detail_weight);
        }
        float3 specular_radiance = upscaled_specular.rgb;
#if RTGI_UPSCALE_SPECULAR_LENIENT_FALLBACK
        if (rtgi_settings.upscale_enabled && lenient_acc_weight > 1e-6f)
        {
            // Total weight of the taps the specular result was built from (detail weights when they are in use).
            const float specular_total_weight = rtgi_settings.specular_upscale_detail_weighting != 0 ? acc_specular_detail_weight : acc_weight;
            const float specular_trust = saturate(specular_total_weight / RTGI_UPSCALE_SPECULAR_LENIENT_FALLBACK_WEIGHT);
            if (specular_trust < 1.0f)
            {
                const float3 lenient_specular = lenient_acc_specular.rgb / lenient_acc_weight;
                specular_radiance = lerp(lenient_specular, specular_radiance, specular_trust);
            }
        }
#endif
        push.attach.specular_resolved.get()[dtid] = float4(specular_radiance / RTGI_RADIANCE_SCALE, 1.0f);

        if (rtgi_settings.sh_resolve_enabled)
        {
            upscaled_diffuse = sh_resolve_diffuse(upscaled_sh_y, upscaled_cocg, pixel_detail_normal);
        }
        else
        {
            upscaled_diffuse = y_co_cg_to_linear(float3(upscaled_sh_y.w, upscaled_cocg));
        }
    }

    push.attach.diffuse_resolved.get()[dtid] = float4((upscaled_diffuse) / RTGI_RADIANCE_SCALE, 1.0f);
}