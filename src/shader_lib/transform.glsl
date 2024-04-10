#pragma once

#include <daxa/daxa.inl>

#include "../shader_shared/shared.inl"
#include "shader_lib/glsl_to_slang.glsl"

daxa_f32vec3 world_space_to_ndc(CameraInfo camera, daxa_f32vec3 world_space)
{
    daxa_f32vec4 clip_space = mul(camera.view_proj, daxa_f32vec4(world_space, 1));
    return clip_space.xyz / clip_space.w;
}

daxa_f32vec2 ndc_to_uv(daxa_f32vec3 ndc)
{
    return daxa_f32vec2((ndc.xy + 1.0f) * 0.5f);
}

daxa_u32vec2 uv_to_texel_index(CameraInfo camera, daxa_f32vec2 uv)
{
    return daxa_u32vec2(clamp(floor(uv * daxa_f32vec2(camera.screen_size)) ,daxa_f32vec2(0,0), daxa_f32vec2(camera.screen_size - 1)));
}

daxa_u32vec2 ndc_to_texel_index(CameraInfo camera, daxa_f32vec3 ndc)
{
    return uv_to_texel_index(camera, ndc_to_uv(ndc));
}

daxa_u32vec2 world_space_to_texel_index(CameraInfo camera, daxa_f32vec3 world_space)
{
    return ndc_to_texel_index(camera, world_space_to_ndc(camera, world_space));
}