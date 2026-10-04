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
    
    // Written by vsm debug statistics
    daxa_u32 cached_pages;
    daxa_u32 free_pages;
    daxa_u32 drawn_pages;
    daxa_u32 point_spot_cached_pages;
    daxa_u32 point_spot_cached_visible_pages;
    daxa_u32 directional_cached_pages;
    daxa_u32 directional_cached_visible_pages;
    daxa_u32 drawn_point_spot_pages;
    daxa_u32 drawn_directional_pages;

    // Written by RTGI distribute rays (repacked dispatch) / classic trace: this frame's ray demand and budget
    daxa_u32 rtgi_requested_base_rays;   // 1 per signal per geometry pixel
    daxa_u32 rtgi_requested_extra_rays;  // deficit-driven extras (after the visual impact scaling)
    daxa_u32 rtgi_ray_budget;
    // Written by the RTGI pre-filter from the ray counters: per-signal requested / shot rays this frame.
    daxa_u32 rtgi_requested_diffuse_rays;
    daxa_u32 rtgi_requested_specular_rays;
    daxa_u32 rtgi_shot_diffuse_rays;
    daxa_u32 rtgi_shot_specular_rays;
    // Written by the RTGI upscale from the ray counters: summed per-pixel convergence (fixed point, RTGI_CONVERGENCE_SCALE)
    // and the number of geometry pixels it was summed over.
    daxa_u32 rtgi_convergence_diffuse_sum;
    daxa_u32 rtgi_convergence_specular_sum;
    daxa_u32 rtgi_convergence_pixels;
    daxa_u32 rtgi_convergence_histogram_diffuse[16];  // pixels per history / max history bucket (16 equal buckets)
    daxa_u32 rtgi_convergence_histogram_specular[16];

    // General debug values
    daxa_f32vec4 debug_value;
};
DAXA_DECL_BUFFER_PTR(ReadbackValues)