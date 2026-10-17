#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../../shader_shared/shared.inl"
#include "../../../shader_shared/globals.inl"

// Ray redistribution (rt_lab_rays.hlsl): a simplified port of the RTGI ray budget (rtgi_distribute_rays.hlsl), diffuse
// only. Per frame:
//   demand     (per lab pixel) writes the lab geometry, reprojects the history length exactly like the temporal pass
//              (rt_lab_reproject.hlsl: bilinear footprint, occluder aware soft normalize) and turns it into a desired
//              ray count: 1 base ray + calc_desired_extra_rays extras, at most max_rays_per_pixel. Totals the base /
//              extra demand into RtLabRayCounters.
//   distribute (one workgroup per 8x8 tile) drains the totals to the hard budget like RTGI: min_ray_budget base coverage
//              first, the extras get the rest at one global ratio, leftover budget raises the base coverage. Each tile
//              gets exactly floor(fraction x its demand): base rays in rotated Bayer order, extras by a cumulative
//              demand scan. Reserves the tile's rays in the ray list (one atomic), writes ray_alloc (.x offset, .y count).
//   trace      (per ray list entry, rt_lab_trace.hlsl) writes one RtLabRay per entry.
//   gather     (per lab pixel) averages the pixel's rays into lab_raw_diffuse (.rgb mean, .a = ray count = samples).
// Redistribution off: every geometry pixel wants exactly 1 ray and the budget is 1 ray per pixel (= the old trace).
//
// Temporal gradient (RtLabSettings::temporal_gradient_*, A-SVGF style), per frame:
//   gradient trace (rt_lab_trace.inl) re-traces last frame's gradient records -> gradient_samples (per cell)
//   gradient blur  screen space gaussian over the cells, edge aware (bilateral: each tap's cell geometry vs the center
//                  cell's, coarse plane distance + normal tests), separable (horizontal, vertical) -> temporal_gradient (per cell:
//                  lambda + the cell's geometry)
//   demand / temporal read lambda per pixel with an edge aware 2x2 bilinear upscale (rt_lab_temporal_gradient)
//   demand / temporal scale the reprojected history lengths by (1 - lambda) BEFORE the ray redistribution
//   gather picks one traced ray per RT_LAB_GRADIENT_CELL^2 cell (RtLabSettings::temporal_gradient_ris: RIS by brightness
//                  or uniform) and stores its launch data + result as the cell's record for the next frame's re-trace.
#define RT_LAB_RAYS_X 8
#define RT_LAB_RAYS_Y 8
#define RT_LAB_TRACE_THREADS 64
#define RT_LAB_GRADIENT_CELL 4                     // one gradient sample per 4x4 lab pixels
// gradient_record_data.w of a valid record = RT_LAB_GRADIENT_RECORD_MAGIC (top 8 bits) | hit shading seed (low 24 bits)
#define RT_LAB_GRADIENT_RECORD_MAGIC 0xA5000000u
#define RT_LAB_GRADIENT_RECORD_MAGIC_MASK 0xFF000000u
#define RT_LAB_HIT_SEED_MASK 0x00FFFFFFu
#define RT_LAB_GRADIENT_NO_RAY 0xFFFFFFFFu         // gradient selection: the cell has no ray this frame
#define RT_LAB_GRADIENT_RIS_UNIFORM_WEIGHT 0.1f    // RIS target = luminance + this x cell mean luminance
// Edge aware gradient filtering, COARSE tests (rt_lab_gradient_surface_weight). Each cell is represented by the geometry
// of its lab pixel cell x CELL + CELL / 2.
#define RT_LAB_GRADIENT_NORMAL_POWER 1.0f              // normal weight ((1 + dot) / 2)^power
#define RT_LAB_GRADIENT_BLUR_PLANE_TOLERANCE 2.0f      // blur: plane distance falloff in CELL widths (4 pixels)
#define RT_LAB_GRADIENT_UPSCALE_PLANE_TOLERANCE 4.0f   // upscale: plane distance falloff in pixel widths

struct RtLabRayCounters
{
    daxa_u32 base_demand;  // geometry pixels (1 base ray each)
    daxa_u32 extra_demand; // extra rays wanted on top of the base rays
    daxa_u32 ray_count;    // ray list entries reserved by the distribute pass (never above the budget)
    daxa_u32 pad;
};
DAXA_DECL_BUFFER_PTR(RtLabRayCounters)

struct RtLabRay
{
    daxa_f32vec3 radiance; // rgb * RTGI_RADIANCE_SCALE
    daxa_u32 direction;    // octahedral 32
};
DAXA_DECL_BUFFER_PTR(RtLabRay)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabDemandH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)                 // full res NDC depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)        // half res linear depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32>, lab_depth)                     // linear depth, 0 = sky
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, lab_face_normal)               // octahedral 32
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal_history)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec4>, history)                    // last frame's history, null without temporal
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32>, ray_demand)                    // desired rays, 0 = sky
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec4>, temporal_gradient)          // per cell (blur output), null when off
DAXA_DECL_TASK_HEAD_END

struct RtLabDemandPush
{
    daxa_BufferPtr(RtLabDemandH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 history_bound;  // history is bound (temporal on)
    daxa_u32 gradient_bound; // temporal_gradient is bound
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabDistributeH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, ray_demand)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(daxa_u32), ray_list)                             // packed pixel x | y << 16
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec2>, ray_alloc)                 // .x list offset, .y ray count
DAXA_DECL_TASK_HEAD_END

struct RtLabDistributePush
{
    daxa_BufferPtr(RtLabDistributeH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 ray_capacity; // ray list entries
};

// Ray filter (RtLabSettings::ray_filter_*, rt_lab_rays.hlsl entry_rt_lab_ray_filter): one thread per ray list entry.
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabRayFilterH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRayCounters), ray_counters)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), ray_list)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), rays)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(RtLabRay), filtered_rays)
DAXA_DECL_TASK_HEAD_END

struct RtLabRayFilterPush
{
    daxa_BufferPtr(RtLabRayFilterH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // lab resolution
    daxa_u32 ray_capacity; // ray list entries
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabGatherH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32vec2>, ray_alloc)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), rays)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, lab_raw_diffuse)           // .rgb mean * RTGI_RADIANCE_SCALE, .a = rays
// Temporal gradient sample selection (null when off): the corner thread of each cell picks one of the cell's traced rays
// and writes the cell's record (launch data from the lab geometry, direction + result from the ray).
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, lab_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, lab_face_normal)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient_record_origin)    // .xyz origin, .w t min
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec4>, gradient_record_surface)   // .xyz surface, .w luminance
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, gradient_record_data)      // dir, face normal, RIS weight, magic
// Ray filter output (null when off): the pixel means come from these; the gradient selection keeps the raw `rays`.
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(RtLabRay), filtered_rays)
DAXA_DECL_TASK_HEAD_END

struct RtLabGatherPush
{
    daxa_BufferPtr(RtLabGatherH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;
    daxa_u32 gradient_enabled; // the gradient images are bound
    daxa_u32 filtered_bound;   // filtered_rays is bound
};

// Screen space gaussian over the gradient cells, edge aware and separable: tap weight = gaussian x the tap cell's
// geometry vs the center cell's (coarse tests, rt_lab_gradient_surface_weight, plane falloff
// RT_LAB_GRADIENT_BLUR_PLANE_TOLERANCE cell widths), sky cells skipped. lambda = |sum w delta| / sum w max (A-SVGF), 0 without samples
// in reach. Cell geometry comes straight from the g-buffer (the lab geometry of this frame is written later, by the demand).
//   horizontal: gradient_samples (packed) -> blur_sums_dst (.x sum w delta, .y sum w max)
//   vertical:   blur_sums -> temporal_gradient (.x lambda bits, .y the cell's linear depth bits (0 = sky), .z its packed
//               face normal), read by rt_lab_temporal_gradient (rt_lab_shared.hlsl)
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(RtLabGradientBlurH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_depth)                 // full res NDC depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32>, view_cam_half_res_depth)        // half res linear depth
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, view_cam_half_res_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_u32>, gradient_samples)        // horizontal pass input
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_f32vec2>, blur_sums_dst)     // horizontal pass output
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DIndex<daxa_f32vec2>, blur_sums)          // vertical pass input
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DIndex<daxa_u32vec4>, temporal_gradient) // vertical pass output
DAXA_DECL_TASK_HEAD_END

struct RtLabGradientBlurPush
{
    daxa_BufferPtr(RtLabGradientBlurH::AttachmentShaderBlob) attach;
    daxa_u32vec2 size;     // gradient cells
    daxa_u32vec2 lab_size; // lab pixels
    daxa_u32 vertical;     // 0 = horizontal pass, 1 = vertical pass
};
