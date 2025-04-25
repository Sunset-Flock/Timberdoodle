#pragma once

#include "daxa/daxa.inl"

#include "shader_shared/lights.inl"

func lights_get_mask_volume_cell(LightSettings settings, float3 position) -> int3
{
    return int3(floor((position - settings.mask_volume_min_pos) * rcp(settings.mask_volume_cell_size)));
}

func lights_in_mask_volume(LightSettings settings, int3 cell) -> bool
{
    return all(cell >= int3(0,0,0)) && all(cell < settings.mask_volume_cell_count);
}

func lights_get_mask(LightSettings settings, float3 position, Texture2DArray<uint4> mask_volume) -> uint4
{
    let cell_idx = lights_get_mask_volume_cell(settings, position);
    if (lights_in_mask_volume(settings, cell_idx))
    {
        return mask_volume[cell_idx];
    }
    return uint4(0,0,0,0);
}

func lights_iterate_mask(LightSettings settings, inout uint4 mask) -> uint
{
    uint first_filled_uint = 0;
    uint uint_mask = 0;
    first_filled_uint = mask.w != 0 ? 3 : first_filled_uint;
    first_filled_uint = mask.z != 0 ? 2 : first_filled_uint;
    first_filled_uint = mask.y != 0 ? 1 : first_filled_uint;
    first_filled_uint = mask.x != 0 ? 0 : first_filled_uint;
    uint_mask = mask.w != 0 ? mask.w : uint_mask;
    uint_mask = mask.z != 0 ? mask.z : uint_mask;
    uint_mask = mask.y != 0 ? mask.y : uint_mask;
    uint_mask = mask.x != 0 ? mask.x : uint_mask;

    let first_set_bit = firstbitlow(uint_mask);

    uint_mask = uint_mask & (~(1u << first_set_bit));
    mask[first_filled_uint] = uint_mask;

    return first_filled_uint * 32 + first_set_bit;
}

#define LIGHTS_ENABLE_MASK_ITERATION 1