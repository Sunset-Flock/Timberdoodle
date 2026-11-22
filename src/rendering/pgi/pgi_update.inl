#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"

struct PGIIndirections
{
    DispatchIndirectStruct probe_update_dispatch;
    DispatchIndirectStruct probe_trace_dispatch;
    DispatchIndirectStruct probe_color_update_dispatch;
    DispatchIndirectStruct probe_visibility_update_dispatch;
    DispatchIndirectStruct probe_trace_integrate_dispatch;
    DispatchIndirectStruct probe_shade_rays_dispatch;
    DrawIndexedIndirectStruct probe_debug_draw_dispatch;
    daxa_u32 detailed_probe_count;
    daxa_u32 probe_update_count;
};


DAXA_DECL_RASTER_TASK_HEAD_BEGIN(PGIDrawDebugProbesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ | INDIRECT_COMMAND_READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE(COLOR_ATTACHMENT, REGULAR_2D, color_image) // the debug image: probes draw into it with alpha 3 (exposed blend, replace) so they cover the debug views
DAXA_TH_IMAGE(DEPTH_ATTACHMENT, REGULAR_2D, depth_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, probe_color)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_u32>, probe_requests)
DAXA_TH_TLAS_PTR(READ, tlas)
DAXA_DECL_TASK_HEAD_END

struct PGIDrawDebugProbesPush
{
    PGIDrawDebugProbesH::AttachmentShaderBlob attach;
    daxa_f32vec3* probe_mesh_positions;
};

// Texel update workgroup: 1D, one thread per probe texel.
#define PGI_UPDATE_WG_SIZE 64
// Texel update dispatch is 1D: workgroups are wrapped into rows of this many groups (dispatch limit per dimension).
#define PGI_UPDATE_LINEAR_WG_X 1024


// Integrates the rays of every trace texel of the updated probes into one value per trace texel (the trace summary).
// The probe update passes read the summary as if it were one ray per texel, trace_summary_count holds the number of valid rays
// that were integrated into the texel (0 = no data, the update passes skip the texel).
DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGITraceIntegrateH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ | INDIRECT_COMMAND_READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, trace_summary)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DArrayId<daxa_u32>, trace_summary_count)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), probe_ray_cdf)
DAXA_DECL_TASK_HEAD_END

struct PGITraceIntegratePush
{
    PGITraceIntegrateH::AttachmentShaderBlob attach;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIUpdateProbeTexelsH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ | INDIRECT_COMMAND_READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_color)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_u32>, trace_count)
DAXA_DECL_TASK_HEAD_END

struct PGIUpdateProbeTexelsPush
{
    PGIUpdateProbeTexelsH::AttachmentShaderBlob attach;
    bool update_radiance;
};

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIUpdateProbesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ | INDIRECT_COMMAND_READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info_copy)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, requests)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, probe_color)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(daxa_u32), probe_ray_cdf)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_u32>, trace_count)
DAXA_DECL_TASK_HEAD_END

struct PGIUpdateProbesPush
{
    PGIUpdateProbesH::AttachmentShaderBlob attach;
};

// Probe inspection: the data of one selected probe, host visible, rewritten every frame while the inspection is on.
// Texel arrays are row major with the probe color resolution (at most 48 x 48), rays in trace result order (trace resolution, at most 48 x 48).
#define PGI_INSPECT_MAX_TEXELS (48 * 48)
#define PGI_INSPECT_WG_SIZE 256

struct PGIInspectRay
{
    daxa_f32vec3 direction;
    daxa_f32 distance;      // hit distance, negative = backface hit
    daxa_f32vec3 radiance;
    daxa_f32 _pad;
};

struct PGIInspectData
{
    daxa_i32vec4 grid_coord;        // the probe this data belongs to: absolute probe grid coordinate xyz (stable while the window scrolls), cascade
    daxa_u32 in_window;             // the probe is inside the current probe window (else nothing of it is updated)
    daxa_u32 valid;                 // 1 once the probe was written
    daxa_u32 write_frame;           // frame the texels were written
    daxa_u32 ray_frame;             // frame the rays were traced (the probe is only traced when it is updated, ~u32 = never)
    daxa_u32 ray_count;
    daxa_f32vec3 probe_position;
    daxa_f32 validity;
    daxa_f32vec3 ray_origin;        // probe position at the time of the trace
    daxa_f32 max_visibility_distance;
    daxa_u32 color_resolution;
    daxa_u32 trace_resolution;
    daxa_u32 visibility_resolution;
    // Probe requests. A probe is a vertex of up to 8 cells, each cell keeps a direct and an indirect request timer in its base probe
    // (set to 63 / 255 when the cell is sampled, counted down by one per frame). The probe is deleted (its info reset) once all of its cells have both timers at 0.
    // Index = x + 2 y + 4 z of the cell, the cell base probe is the probe minus (x, y, z). ~0u = no such cell (clamped to the window or not a base probe).
    daxa_u32 request_cell_direct[8];
    daxa_u32 request_cell_indirect[8];
    daxa_u32 request_package;       // raw package at the probe's own entry (bits 16 / 17: the probe is directly / indirectly requested)
    daxa_u32 requested;             // any cell timer is not 0
    daxa_u32 is_new_probe;          // the probe came into existence by the window moving (no history, not requested yet)
    daxa_u32 _pad2;
    daxa_f32vec4 irradiance[PGI_INSPECT_MAX_TEXELS];    // .a = hysteresis
    daxa_f32vec4 radiance[PGI_INSPECT_MAX_TEXELS];
    daxa_u32 cdf[PGI_INSPECT_MAX_TEXELS];               // probe ray cdf, normalized to PGI_RAY_CDF_ONE
    PGIInspectRay rays[PGI_INSPECT_MAX_TEXELS];
    daxa_f32vec2 visibility[PGI_INSPECT_MAX_TEXELS];    // visibility resolution, .x mean ray distance (negative = backface), .y mean deviation
};
DAXA_DECL_BUFFER_PTR(PGIInspectData)

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIInspectH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, probe_color)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_u32>, probe_requests)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, trace_result)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_u32), probe_ray_cdf)
DAXA_TH_BUFFER_PTR(READ_WRITE, daxa_RWBufferPtr(PGIInspectData), inspect)
DAXA_DECL_TASK_HEAD_END

struct PGIInspectPush
{
    daxa_BufferPtr(PGIInspectH::AttachmentShaderBlob) attach;
    daxa_i32 grid_coord[4]; // selected probe: absolute probe grid coordinate xyz (window index + window offset - count / 2 + 1), cascade
    daxa_u32 show_rays;
    daxa_u32 _pad;
};

#define PGI_PRE_UPDATE_XYZ 4

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIPreUpdateProbesH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_f32vec4>, probe_info_copy)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, requests)
DAXA_TH_BUFFER_PTR(WRITE, daxa_RWBufferPtr(PGIIndirections), probe_indirections)
DAXA_DECL_TASK_HEAD_END

struct PGIPreUpdateProbesPush
{
    PGIPreUpdateProbesH::AttachmentShaderBlob attach;
    daxa_u32* workgroups_finished;
    daxa_u32 total_workgroups;
};

#define PGI_EVAL_SCREEN_IRRADIANCE_XY 8
#define PGI_EVAL_SCREEN_IRRADIANCE_Z 1

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIEvalScreenIrradianceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32>, main_cam_depth)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_u32>, main_cam_face_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_u32>, main_cam_detail_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec4>, probe_color)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DArrayId<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayId<daxa_u32>, probe_requests)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, irradiance_depth)
// DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_u32>, normals)
DAXA_DECL_TASK_HEAD_END

struct PGIEvalScreenIrradiancePush
{
    PGIEvalScreenIrradianceH::AttachmentShaderBlob attach;
    daxa_u32vec2 render_target_size;
    daxa_u32vec2 irradiance_image_size;
};

#define PGI_UPSCALE_SCREEN_IRRADIANCE_XY 8
#define PGI_UPSCALE_SCREEN_IRRADIANCE_Z 1

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(PGIUpscaleScreenIrradianceH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32>, main_cam_depth)
DAXA_TH_IMAGE_TYPED(READ, daxa::RWTexture2DId<daxa_u32>, main_cam_detail_normals)
DAXA_TH_IMAGE_TYPED(SAMPLE, daxa::Texture2DId<daxa_f32vec4>, half_irradiance_depth)
DAXA_TH_IMAGE_TYPED(READ, daxa::RWTexture2DId<daxa_u32>, half_normals)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32vec4>, full_res_pgi_irradiance)
DAXA_DECL_TASK_HEAD_END

struct PGIUpscaleScreenIrradiancePush
{
    PGIUpscaleScreenIrradianceH::AttachmentShaderBlob attach;
    daxa_u32vec2 size;
};