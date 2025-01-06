#pragma once

#include "daxa/daxa.inl"

#include "geometry_pipeline.inl"
#include "shared.inl"

struct ReadbackValues
{
    // Written by task/compute meshlet cull shader
    daxa_u32 first_pass_meshlet_count_pre_cull[PREPASS_DRAW_LIST_TYPE_COUNT];
    daxa_u32 first_pass_mesh_count_post_cull[PREPASS_DRAW_LIST_TYPE_COUNT];    
    daxa_u32 second_pass_meshlet_count_pre_cull[PREPASS_DRAW_LIST_TYPE_COUNT];
    daxa_u32 second_pass_mesh_count_post_cull[PREPASS_DRAW_LIST_TYPE_COUNT];   
    // Written by shade opaque
    daxa_u32 first_pass_meshlet_count_post_cull;
    daxa_u32 second_pass_meshlet_count_post_cull;
    daxa_u32 hovered_entity;
    // Written in command:  
    daxa_u32 sfpm_bitfield_arena_requested;       
    daxa_u32 sfpm_bitfield_arena_allocation_failures_ent_pass;
    daxa_u32 sfpm_bitfield_arena_allocation_failures_mesh_pass;
    // Written by pgi probe update
    daxa_u32 requested_probes;
};
DAXA_DECL_BUFFER_PTR(ReadbackValues)