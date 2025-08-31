#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../gpu_context.hpp"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/gpu_work_expansion.inl"
#include "../tasks/misc.hpp"
#include "../../scene/scene.hpp"
#include "pgi_update.inl"
#include "pgi_trace_probe_lighting.hlsl"
#include "../../daxa_helper.hpp"

struct PGIState
{
    daxa::BufferId debug_probe_mesh_buffer = {};
    daxa::u32 debug_probe_mesh_triangles = {};
    daxa::u32 debug_probe_mesh_vertices = {};
    daxa_f32vec3* debug_probe_mesh_vertex_positions_addr = {};

    // TODO(pahrens): rename to irradiance
    daxa::TaskImage probe_color = daxa::TaskImage(daxa::TaskImageInfo{.name = "default init pgi probe radiance texture"});
    daxa::TaskImageView probe_color_view = daxa::NullTaskImage;
    daxa::TaskImage probe_visibility = daxa::TaskImage(daxa::TaskImageInfo{.name = "default init pgi probe visibility texture"});
    daxa::TaskImageView probe_visibility_view = daxa::NullTaskImage;
    daxa::TaskImage probe_info = daxa::TaskImage(daxa::TaskImageInfo{.name = "default init pgi probe info texture"});
    daxa::TaskImageView probe_info_view = daxa::NullTaskImage;
    daxa::TaskImage cell_requests = daxa::TaskImage(daxa::TaskImageInfo{.name = "default init pgi cell requests texture"});
    daxa::TaskImageView cell_requests_view = daxa::NullTaskImage;

    void initialize(daxa::Device& device);
    void recreate_and_clear(daxa::Device& device, PGISettings const & settings);
    void cleanup(daxa::Device& device);
};

#include "../scene_renderer_context.hpp"
#include "../../daxa_helper.hpp"

auto pgi_update_probe_texels_pipeline_compile_info() -> daxa::ComputePipelineCompileInfo2 const&;

auto pgi_update_probes_compile_info() -> daxa::ComputePipelineCompileInfo2 const&;

auto pgi_pre_update_probes_compute_compile_info() -> daxa::ComputePipelineCompileInfo2 const&;

auto pgi_eval_screen_irradiance_compute_compile_info() -> daxa::ComputePipelineCompileInfo2 const&;

auto pgi_upscale_screen_irradiance_compute_compile_info() -> daxa::ComputePipelineCompileInfo2 const&;

auto pgi_draw_debug_probes_compile_info() -> daxa::RasterPipelineCompileInfo;

auto pgi_trace_probe_lighting_pipeline_compile_info() -> daxa::RayTracingPipelineCompileInfo2;

struct PGIDrawDebugProbesTask : PGIDrawDebugProbesH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti);
};

struct PGIUpdateProbeTexelsTask : PGIUpdateProbeTexelsH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti);
};

struct PGIUpdateProbesTask : PGIUpdateProbesH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti);
};

struct PGITraceProbeRaysTask : PGITraceProbeLightingH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti);
};

struct PGIPreUpdateProbesTask : PGIPreUpdateProbesH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti);
};

struct PGIEvalScreenIrradianceTask : PGIEvalScreenIrradianceH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti);
};

struct PGIUpscaleScreenIrradianceTask : PGIUpscaleScreenIrradianceH::Task
{
    AttachmentViews views = {};
    RenderContext* render_context = {};
    PGIState* pgi_state = {};
    void callback(daxa::TaskInterface ti);
};

auto pgi_significant_settings_change(PGISettings const & prev, PGISettings const & curr) -> bool;

// Fills any auto calculated setting fields.
// Correctly constrains all input values to valid ones.
void pgi_resolve_settings(PGISettings const & prev_settings, RenderGlobalData & render_data);

auto pgi_create_trace_result_texture(daxa::TaskGraph& tg, PGISettings& settings, PGIState& state) -> daxa::TaskImageView;

auto pgi_create_probe_info_texture(daxa::TaskGraph& tg, PGISettings& settings, PGIState& state) -> daxa::TaskImageView;

auto pgi_create_probe_indirections(daxa::TaskGraph& tg, PGISettings& settings, PGIState& state) -> daxa::TaskBufferView;

auto pgi_create_half_screen_irradiance(daxa::TaskGraph& tg, RenderGlobalData const& render_data) -> daxa::TaskImageView;

auto pgi_create_screen_irradiance(daxa::TaskGraph& tg, RenderGlobalData const& render_data) -> daxa::TaskImageView;

struct TaskPgiUpdateInfo
{
    daxa::TaskGraph& tg;
    RenderContext* render_context = {};
    PGIState& pgi_state;
    daxa::TaskImageView light_mask_volume = {};
    daxa::TaskBufferView mesh_instances = {};
    daxa::TaskTlasView tlas = {};
    daxa::TaskImageView sky_transmittance = {};
    daxa::TaskImageView sky = {};
    // VSM:
    daxa::TaskBufferView vsm_globals = {};
    daxa::TaskBufferView vsm_point_lights = {};
    daxa::TaskBufferView vsm_spot_lights = {};
    daxa::TaskImageView vsm_memory_block = {};
    daxa::TaskImageView vsm_point_spot_page_table = {};
};
struct TaskPGIUpdateOut
{
    daxa::TaskBufferView pgi_indirections = {};
    daxa::TaskImageView pgi_irradiance = {};
    daxa::TaskImageView pgi_visibility = {};
    daxa::TaskImageView pgi_info = {};
    daxa::TaskImageView pgi_requests = {};
};
auto task_pgi_update(TaskPgiUpdateInfo const & info) -> TaskPGIUpdateOut;

struct TaskPGIEvalScreenIrradianceInfo
{
    daxa::TaskGraph& tg;
    RenderContext* render_context = {};
    PGIState& pgi_state;
    daxa::TaskImageView view_camera_depth = {};
    daxa::TaskImageView view_camera_face_normal_image = {};
    daxa::TaskImageView view_camera_detail_normal_image = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView clocks_image = {};
    daxa::TaskImageView pgi_irradiance = {};
    daxa::TaskImageView pgi_visibility = {};
    daxa::TaskImageView pgi_info = {};
    daxa::TaskImageView pgi_requests = {};
};
auto task_pgi_eval_screen_irradiance(TaskPGIEvalScreenIrradianceInfo const & info) -> daxa::TaskImageView;