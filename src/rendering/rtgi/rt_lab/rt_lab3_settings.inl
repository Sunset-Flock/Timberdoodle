#pragma once

#include "daxa/daxa.inl"

// RT Lab 3 settings (rt_lab3.inl documents the pipeline). Plain shared types: RtLab3State (rt_lab3_state.hpp) holds them on the
// CPU, every pass reads them through a pointer in its push constant.

#define RT_LAB3_MAX_SPATIAL_PASSES 4
#define RT_LAB3_MAX_SPATIAL_TAPS 16
#define RT_LAB3_MAX_FILTER_RADIUS 4

// Spatial reuse surface test (RtLab3Settings::spatial_surface_test).
#define RT_LAB3_SURFACE_PLANE 0    // distance of the neighbor from the receiver's tangent plane, spatial_plane_tolerance pixel widths
#define RT_LAB3_SURFACE_DISTANCE 1 // world space distance between the two points, spatial_distance_factor pixel widths

// RT Lab 4: what a pixel counts as when its estimate goes into the cell grid (the weight of its mean in the cell's blend).
#define RT_LAB3_CELL_WEIGHT_PIXEL 0 // 1 per pixel that has a reservoir
#define RT_LAB3_CELL_WEIGHT_RAYS 1  // the fresh rays the pixel shot this frame (pixels without a fresh ray do not feed the cells)
#define RT_LAB3_CELL_WEIGHT_M 2     // the pixel's accumulated rays shot (ray history, not the reservoir confidence M)

// RT Lab 4: what goes into the cells.
#define RT_LAB3_CELL_SOURCE_RESTIR 0 // the filtered ReSTIR estimate (reuse correlates the pixels: one sample can show up in many pixels)
#define RT_LAB3_CELL_SOURCE_RAYS 1   // the mean of the pixel's FRESH rays (independent samples), each clamped against the same neighborhood ceiling; weight = the ray count


struct RtLab3Settings
{
    // == Temporal reuse ==
    daxa_i32 temporal_enabled;
    daxa_f32 temporal_max_m;            // cap of the previous reservoir's confidence M (limits how long a sample may live)
    daxa_f32 temporal_plane_tolerance;  // plane distance in lab pixel widths
    daxa_f32 temporal_normal_threshold; // min dot of the face normals
    daxa_i32 temporal_visibility;       // shadow ray from the receiver to the reused sample
    daxa_i32 temporal_max_age;          // sample age cap in frames (0 = off): a reservoir whose sample is older than this is not reused, so no sample outlives it
    daxa_f32 temporal_partial_scale;    // M multiplier of a candidate that only partly matches (footprint weight < 1): lerp(1, weight, this)
    // == Spatial reuse ==
    daxa_i32 spatial_passes;            // graph shape (RtLab3State::built_spatial_passes)
    daxa_i32 spatial_taps;              // neighbor reservoirs per pass
    daxa_f32 spatial_radius;            // lab pixels
    daxa_f32 spatial_max_m;             // cap of a neighbor reservoir's M
    daxa_f32 spatial_plane_tolerance;
    daxa_f32 spatial_normal_threshold;
    daxa_i32 spatial_surface_test;      // RT_LAB3_SURFACE_*: how a neighbor is tested to be on the receiver's surface (the normal test applies to both)
    daxa_f32 spatial_distance_factor;   // RT_LAB3_SURFACE_DISTANCE: a neighbor further than this many world space pixel widths (of the receiving pixel) away is rejected
    daxa_i32 spatial_visibility;
    // == Reconnection (both reuse kinds) ==
    daxa_f32 jacobian_max;              // a shifted sample whose Jacobian is outside [1 / max, max] is rejected
    daxa_f32 min_reconnection_distance; // meters: samples closer than this to the receiver are not reused
    // == Frame ==
    daxa_i32 reset; // set by the CPU: ignore the history this frame
    daxa_f32 temporal_max_age_jitter; // 0..1: the age cap of a reuse is cap x (1 - jitter x random), so samples do not all expire on the same frame
};
