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