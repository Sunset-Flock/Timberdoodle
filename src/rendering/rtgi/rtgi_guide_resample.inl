#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define RTGI_GUIDE_RESAMPLE_X 8
#define RTGI_GUIDE_RESAMPLE_Y 8

// The pioneer trace runs on a grid at 1/RTGI_GUIDE_PIONEER_GRID_DIV resolution of the half-res trace
// grid IN EACH AXIS (so 1/16 the half-res pixel count). Fixed, not a runtime setting -- was a 2-or-4
// UI toggle at one point, but nothing downstream needs it to vary, so it's a compile-time constant
// like RTGI_GUIDE_RESAMPLE_WINDOW below. Which half-res pixel each pioneer cell samples rotates every
// frame (see the `rotation` calc in pioneer_ray_gen / entry_guide_resample_horizontal/vertical -- all
// three MUST stay in sync).
#define RTGI_GUIDE_PIONEER_GRID_DIV 4

// Resample gather window, in DENOISER (HALF-RES) PIXELS -- e.g. 16 means the window reaches 16 half-res
// pixels wide on screen. The actual tap count in pioneer cells is derived as WINDOW /
// RTGI_GUIDE_PIONEER_GRID_DIV (see `guide_resample_taps` in rtgi_guide_resample.hlsl, computed
// identically in both passes).
#define RTGI_GUIDE_RESAMPLE_WINDOW 64

// Spacing, in PIONEER CELLS, between consecutive gathered taps within the window (1 = every cell,
// contiguous -- the old behavior). At the default of 4, RTGI_GUIDE_RESAMPLE_WINDOW taps span
// WINDOW*STRIDE pioneer cells instead of just WINDOW, so the resample reaches much further per axis
// without paying for more taps. Applies to both passes identically (see the `base_x`/`base_y` + `i *
// RTGI_GUIDE_RESAMPLE_STRIDE` indexing in rtgi_guide_resample.hlsl).
#define RTGI_GUIDE_RESAMPLE_STRIDE 2

// 1 = candidate acceptance in all three passes (H, V, resolve) is a hard world-space DISTANCE cutoff
// only (RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD pixel widths, using whichever of the two points is closer
// to camera -- i.e. has the smaller world-space pixel footprint -- as the reference scale, so the
// cutoff can't loosen just because the far point's own footprint is bigger). No coplanarity check, no
// normal check at all -- candidates freely cross between differently oriented surfaces (e.g. around a
// corner) as long as they're spatially close. 0 = the old calc_similar_surface_weight (coplanarity) *
// calc_similar_normal_weight (soft-but-still-a-hard->0-cutoff normal reject) gate. Normals are not used
// to weight the RIS pick either way in mode 1 -- stochastic normal weighting was the original ask but
// isn't wired up yet, so mode 1 just drops normals entirely for now.
#define RTGI_GUIDE_RESAMPLE_DISTANCE_ONLY 1
#define RTGI_GUIDE_RESAMPLE_PX_DIST_THRESHOLD 32.0f

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiGuideResampleHorizontalH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, pioneer_hit_y)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, h_resample_hit_y)
// .x = row's total selection weight (sum of valid candidates' log-brightness), .y = valid candidate count.
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec2>, h_resample_weight)
DAXA_DECL_TASK_HEAD_END

struct RtgiGuideResampleHorizontalPush
{
    RtgiGuideResampleHorizontalH::AttachmentShaderBlob attach;
    daxa_u32vec2 size; // pioneer_grid_size
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtgiGuideResampleVerticalH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, h_resample_hit_y)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, h_resample_weight)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, pioneer_guide_hit_y)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_DECL_TASK_HEAD_END

struct RtgiGuideResampleVerticalPush
{
    RtgiGuideResampleVerticalH::AttachmentShaderBlob attach;
    daxa_u32vec2 size; // pioneer_grid_size
};

