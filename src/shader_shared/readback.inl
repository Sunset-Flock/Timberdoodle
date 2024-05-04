#pragma once

#include "daxa/daxa.inl"

#include "geometry_pipeline.inl"
#include "shared.inl"

struct ReadbackValues
{
    daxa_u32 first_pass_meshlet_count[DRAW_LIST_TYPES];
    daxa_u32 second_pass_meshlet_count[DRAW_LIST_TYPES];
};
DAXA_DECL_BUFFER_PTR(ReadbackValues)