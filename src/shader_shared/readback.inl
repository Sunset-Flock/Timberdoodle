#pragma once

#include "daxa/daxa.inl"

#include "geometry_pipeline.inl"
#include "shared.inl"

struct ReadbackValues
{
    // written by readback task:
    daxa_u32 first_pass_meshlet_count[PREPASS_DRAW_LIST_TYPE_COUNT];
    daxa_u32 second_pass_meshlet_count[PREPASS_DRAW_LIST_TYPE_COUNT];
    daxa_u32 visible_meshes;    
    // written in command:  
    daxa_u32 sfpm_bitfield_arena_requested;       
    daxa_u32 sfpm_bitfield_arena_allocation_failures_ent_pass;
    daxa_u32 sfpm_bitfield_arena_allocation_failures_mesh_pass;
};
DAXA_DECL_BUFFER_PTR(ReadbackValues)