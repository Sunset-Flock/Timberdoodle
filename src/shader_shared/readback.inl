#pragma once

#include "daxa/daxa.inl"

#include "shared.inl"

struct ReadbackValues
{
    daxa_u32 drawn_meshes;
    daxa_u32 drawn_meshlets_first_pass;
    daxa_u32 drawn_meshlets_second_pass;
    daxa_u32 visible_meshlets_count;
};
DAXA_DECL_BUFFER_PTR(ReadbackValues)