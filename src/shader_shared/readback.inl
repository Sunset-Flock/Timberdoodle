#pragma once

#include "daxa/daxa.inl"

#include "geometry_pipeline.inl"
#include "shared.inl"

// RT Lab convergence error estimate (rt_lab_convergence.hlsl): signed luminance difference histogram bins and the
// fixed point scale of the error sums.
#define RT_LAB_CONVERGENCE_HISTOGRAM_BINS 64
#define RT_LAB_CONVERGENCE_FIXED_SCALE 16777216.0f // 2^24
#define RT_LAB_CONVERGENCE_PREDICTION_CLAMP 10.0f // keeps the squared / product sums inside u64

struct ReadbackValues
{
    // Written by the RT Lab convergence analysis every frame while a result exists. Fixed point
    // (RT_LAB_CONVERGENCE_FIXED_SCALE) sums over geometry pixels of the squared error and of the positive / negative
    // signed luminance difference (custom - progressive, exposed irradiance). FIRST in the struct: u64 alignment is
    // then the same in C++ and in the shaders.
    daxa_u64 rt_lab_convergence_squared_error_sum;
    daxa_u64 rt_lab_convergence_positive_error_sum;
    daxa_u64 rt_lab_convergence_negative_error_sum;
    // Split history error prediction check: fixed point sums over pixels with a prediction p (Custom |A - B| / 2) and
    // the actual error a (|Custom - Progressive| luminance), both clamped to RT_LAB_CONVERGENCE_PREDICTION_CLAMP.
    daxa_u64 rt_lab_convergence_predicted_sum;
    daxa_u64 rt_lab_convergence_predicted_squared_sum;
    daxa_u64 rt_lab_convergence_actual_sum;
    daxa_u64 rt_lab_convergence_actual_squared_sum;
    daxa_u64 rt_lab_convergence_predicted_actual_sum;
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
    // Written by the RTGI split trace / shade passes (group 0 of each): rays shaded by the sharp (full shading) and
    // coarse (PGI radiance / sky) pass this frame. 0 when split trace / shade is off.
    daxa_u32 rtgi_shade_sharp_rays;
    daxa_u32 rtgi_shade_coarse_rays;
    daxa_u32 rtgi_shade_miss_rays;
    // Written by the RTGI upscale from the ray counters: summed per-pixel convergence (fixed point, RTGI_CONVERGENCE_SCALE)
    // and the number of geometry pixels it was summed over.
    daxa_u32 rtgi_convergence_diffuse_sum;
    daxa_u32 rtgi_convergence_specular_sum;
    daxa_u32 rtgi_convergence_pixels;
    daxa_u32 rtgi_convergence_histogram_diffuse[16];  // pixels per history / max history bucket (16 equal buckets)
    daxa_u32 rtgi_convergence_histogram_specular[16];
    // Written by the RTGI upscale from the ray counters: hash grid revives this frame (pixels, summed written sample
    // counts in 1/RTGI_REVIVE_SAMPLE_SCALE fixed point).
    daxa_u32 rtgi_revived_pixels;
    daxa_u32 rtgi_revived_samples;
    // Written by the RTGI hash grid passes (atomics): occupied cells after the update, cells queried this frame,
    // geometry pixels whose query found no free slot (probe window full).
    daxa_u32 rtgi_hash_grid_live_cells;
    daxa_u32 rtgi_hash_grid_touched_cells;
    daxa_u32 rtgi_hash_grid_failed_queries;
    // Written by the RT Lab convergence analysis: geometry pixels compared, signed luminance difference histogram.
    daxa_u32 rt_lab_convergence_pixels;
    daxa_u32 rt_lab_convergence_histogram[RT_LAB_CONVERGENCE_HISTOGRAM_BINS];
    daxa_u32 rt_lab_convergence_below_range; // pixels with signed difference < -range (not in the histogram)
    daxa_u32 rt_lab_convergence_above_range; // pixels with signed difference > range (not in the histogram)
    daxa_f32 rt_lab_convergence_histogram_range; // the range the histogram was binned with
    daxa_u32 rt_lab_convergence_predicted_pixels; // pixels with a split history error prediction
    daxa_u32 rt_lab_convergence_step; // move estimate step the analysis compared (~0u = static estimate)

    // General debug values
    daxa_f32vec4 debug_value;
};
DAXA_DECL_BUFFER_PTR(ReadbackValues)