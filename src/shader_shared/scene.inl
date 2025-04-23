#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

#define INVALID_ENTITY_INDEX (~(0u))

struct GPUEntityId
{
    daxa_u32 index;
    daxa_u32 version;
};
DAXA_DECL_BUFFER_PTR(GPUEntityId)

SHARED_FUNCTION bool entity_id_has_value(GPUEntityId id)
{
    return id.version != 0;
}

struct GPUEntityMetaData
{
    daxa_u32 entity_count;
};
DAXA_DECL_BUFFER_PTR(GPUEntityMetaData)

struct GPUPointLight
{
    daxa_f32vec3 position;
    daxa_f32vec3 color;
    daxa_f32 intensity;
    daxa_f32 cutoff;
};
DAXA_DECL_BUFFER_PTR(GPUPointLight);

struct GPUSpotLight
{
    daxa_f32mat4x3 transform;
    daxa_f32vec3 position;
    daxa_f32vec3 direction;
    daxa_f32vec3 color;
    daxa_f32 intensity;
    daxa_f32 cutoff;
    daxa_f32 inner_cone_angle;
    daxa_f32 outer_cone_angle;
};
DAXA_DECL_BUFFER_PTR(GPUSpotLight);