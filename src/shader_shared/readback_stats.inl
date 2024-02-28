#pragma once

#include <daxa/daxa.inl>

#include "shared.inl"

#include "geometry_pipeline.inl"

struct GPUReadbackData
{
    MeshletInstancesBufferHead meshlet_instances_head;
    daxa_u32 visible_meshlet_list_count;
    daxa_u32 first_pass_meshlet_scratch_offsets_section_size;
    daxa_u32 first_pass_meshlet_scratch_bitfield_section_size;
};