#pragma once

#include <daxa/daxa.inl>

#include "../shader_shared/shared.inl"

vec3 world_space_to_ndc(CameraInfo camera, vec3 world_space)
{
    vec4 clip_space = camera.view_proj * vec4(world_space, 1);
    return clip_space.xyz / clip_space.w;
}

vec2 ndc_to_uv(vec3 ndc)
{
    return vec2((ndc.xy + 1.0f) * 0.5f);
}

uvec2 uv_to_texel_index(CameraInfo camera, vec2 uv)
{
    return uvec2(clamp(floor(uv * vec2(camera.screen_size)) ,vec2(0,0), vec2(camera.screen_size - 1)));
}

uvec2 ndc_to_texel_index(CameraInfo camera, vec3 ndc)
{
    return uv_to_texel_index(camera, ndc_to_uv(ndc));
}

uvec2 world_space_to_texel_index(CameraInfo camera, vec3 world_space)
{
    return ndc_to_texel_index(camera, world_space_to_ndc(camera, world_space));
}