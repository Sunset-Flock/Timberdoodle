#pragma once
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../gpu_context.hpp"
#include "../../shader_shared/geometry_pipeline.inl"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/gpu_work_expansion.inl"
#include "../tasks/misc.hpp"
#include "../../scene/scene.hpp"

struct DDGIState
{
    daxa::BufferId debug_probe_mesh_buffer = {};
    daxa::u32 debug_probe_mesh_triangles = {};
    daxa::u32 debug_probe_mesh_vertices = {};
    daxa_f32vec3* debug_probe_mesh_vertex_positions_addr = {};

    // Probes are separated into z layers within the texture
    // Probe index (1,2,3) goes into the lyaer with index 3
    // Additionally the previous frames probes are stored in the same texture in layers behind all regular probes
    // So a probes layer is probe_index.z + (probe_count.z * frame)
    daxa::TaskImage probe_radiance = daxa::TaskImage(daxa::TaskImageInfo{.name = "default init ddgi probe radiance texture"});
    daxa::TaskImageView probe_radiance_view = {};

    void initialize(daxa::Device& device);
    void recreate_resources(daxa::Device& device, DDGISettings const & settings);
    void cleanup(daxa::Device& device);
};