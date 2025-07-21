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
    daxa_u32 hovered_mesh_in_meshgroup;
    daxa_u32 hovered_mesh;
    daxa_u32 hovered_meshlet_in_mesh;
    daxa_u32 hovered_triangle_in_meshlet;
    // Written in command:  
    daxa_u32 first_pass_meshlet_bitfield_requested_dynamic_size;       
    // Written by pgi probe update
    daxa_u32 requested_probes;
    
    // Written by vsm debug statictics
    daxa_u32 cached_pages;
    daxa_u32 free_pages;
    daxa_u32 drawn_pages;
    daxa_u32 point_spot_cached_pages;
    daxa_u32 point_spot_cached_visible_pages;
    daxa_u32 directional_cached_pages;
    daxa_u32 directional_cached_visible_pages;
    daxa_u32 drawn_point_spot_pages;
    daxa_u32 drawn_directional_pages;
};
DAXA_DECL_BUFFER_PTR(ReadbackValues)