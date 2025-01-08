#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/shared.inl"


float3 pixel_index_to_ray_direction(CameraInfo camera, int2 pixel_index)
{
    const float2 ndc_xy = ((float2(pixel_index) + 0.5f) * camera.inv_screen_size) * 2.0f - 1.0f;
    const float4 unprojected_pos = mul(camera.inv_view_proj, float4(ndc_xy, 0.0001f, 1.0f));
    const float3 pixel_ray = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera.position);
    return pixel_ray;
}