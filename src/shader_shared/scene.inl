#pragma once

#include "daxa/daxa.inl"
#include "vsm_shared.inl"
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
    daxa_f32 constant_falloff;
    daxa_f32 linear_falloff;
    daxa_f32 quadratic_falloff;
    daxa_f32 cutoff;
    daxa_BufferPtr(VSMPointLight) vsm;
};
DAXA_DECL_BUFFER_PTR_ALIGN(GPUPointLight, 8);