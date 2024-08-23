#pragma once

#include "daxa/daxa.inl"

struct MergedSceneBlasIndirection
{
    daxa_u32 entity_index;
    daxa_u32 mesh_group_index;
    daxa_u32 mesh_index;
    daxa_u32 in_mesh_group_index;
};