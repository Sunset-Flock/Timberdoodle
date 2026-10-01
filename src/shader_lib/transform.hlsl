#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/shared.inl"
#include "../shader_lib/depth_util.glsl"

float3 pixel_index_to_world_space(CameraInfo camera, float2 pixel_index, float depth)
{
    const float2 ndc_xy = ((pixel_index + 0.5f) * camera.inv_screen_size) * 2.0f - 1.0f;
    const float4 unprojected_pos = mul(camera.inv_view_proj, float4(ndc_xy, depth, 1.0f));
    const float3 pixel_pos = (unprojected_pos.xyz / unprojected_pos.w);
    return pixel_pos;
}

// World-space ray from the camera through ndc_xy, scaled so its component along the view direction is 1:
//   world position = camera.position + ray * linear_view_depth.
// Only valid for the infinite reversed-Z perspective (inf_depth_reverse_z_perspective in camera.cpp). There,
// unprojecting (ndc_xy, ndc_z, 1) yields w = ndc_z / near = 1 / linear_depth, so the world position is
// near * col2 + linear_depth * (ndc_x * col0 + ndc_y * col1 + col3), with near * col2 == camera.position
// (cols of inv_view_proj). The ray is affine in ndc_xy: two FMAs per pixel instead of a mat4 multiply + divide.
func camera_view_ray_ws(CameraInfo camera, float2 ndc_xy) -> float3
{
    // Slang indexes matrices row-major: m[row][col].
    const float4x4 m = camera.inv_view_proj;
    return ndc_xy.x * float3(m[0][0], m[1][0], m[2][0]) +
           ndc_xy.y * float3(m[0][1], m[1][1], m[2][1]) +
                      float3(m[0][3], m[1][3], m[2][3]);
}

// Same as camera_view_ray_ws but in view space (camera at the origin), from the columns of inv_proj.
func camera_view_ray_vs(CameraInfo camera, float2 ndc_xy) -> float3
{
    const float4x4 m = camera.inv_proj;
    return ndc_xy.x * float3(m[0][0], m[1][0], m[2][0]) +
           ndc_xy.y * float3(m[0][1], m[1][1], m[2][1]) +
                      float3(m[0][3], m[1][3], m[2][3]);
}

func linear_depth_to_world_space(CameraInfo camera, float2 ndc_xy, float linear_depth) -> float3
{
    return camera.position + camera_view_ray_ws(camera, ndc_xy) * linear_depth;
}

float3 sv_xy_to_world_space(float2 inv_screen_size, float4x4 inv_view_proj, float3 sv_pos)
{
    const float2 ndc_xy = (sv_pos.xy * inv_screen_size) * 2.0f - 1.0f;
    const float4 unprojected_pos = mul(inv_view_proj, float4(ndc_xy, sv_pos.z, 1.0f));
    const float3 pixel_pos = (unprojected_pos.xyz / unprojected_pos.w);
    return pixel_pos;
}