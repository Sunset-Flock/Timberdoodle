#pragma once

#include <string>
#include <daxa/utils/imgui.hpp>

#include "../window.hpp"
#include "../scene/scene.hpp"
#include "../scene/asset_processor.hpp"

#include "../shader_shared/geometry.inl"
#include "../shader_shared/geometry_pipeline.inl"

#include "../gpu_context.hpp"

struct DynamicMesh
{
    glm::mat4x4 prev_transform;
    glm::mat4x4 curr_transform;
    std::vector<AABB> meshlet_aabbs;
};
struct SceneDraw
{
    std::array<std::vector<MeshDrawTuple>, 2> opaque_draw_lists = {};
    daxa::TaskBuffer opaque_draw_list_buffer = daxa::TaskBuffer{{.name = "opaque draw lists"}};
    std::vector<DynamicMesh> dynamic_meshes = {};
    // Total maximum entity index.
    // NOT max entity_index of this draw.
    u32 max_entity_index = {};
};

// Used to store all information used only by the renderer.
// Shared with task callbacks.
// Global for a frame within the renderer.
// For reusable sub-components of the renderer, use a new context.
struct RenderContext
{
    RenderContext(GPUContext * ctx) : gpuctx{ctx}
    {
        tgpu_render_data = daxa::TaskBuffer{daxa::TaskBufferInfo{
            .initial_buffers = {
                .buffers = std::array{
                    ctx->device.create_buffer({
                        .size = sizeof(RenderGlobalData),
                        .name = "scene render data",
                    }),
                },
            },
            .name = "scene render data",
        }};
        render_data.samplers = {
            .linear_clamp = gpuctx->device.create_sampler({
                .name = "linear clamp sampler",
            }),
            .linear_repeat = gpuctx->device.create_sampler({
                .address_mode_u = daxa::SamplerAddressMode::REPEAT,
                .address_mode_v = daxa::SamplerAddressMode::REPEAT,
                .address_mode_w = daxa::SamplerAddressMode::REPEAT,
                .name = "linear repeat sampler",
            }),
            .nearest_clamp = gpuctx->device.create_sampler({
                .magnification_filter = daxa::Filter::NEAREST,
                .minification_filter = daxa::Filter::NEAREST,
                .mipmap_filter = daxa::Filter::NEAREST,
                .name = "nearest clamp sampler",
            }),
            .linear_repeat_ani = gpuctx->device.create_sampler({
                .address_mode_u = daxa::SamplerAddressMode::REPEAT,
                .address_mode_v = daxa::SamplerAddressMode::REPEAT,
                .address_mode_w = daxa::SamplerAddressMode::REPEAT,
                .mip_lod_bias = 0.0f,
                .enable_anisotropy = true,
                .max_anisotropy = 16.0f,
                .name = "nearest clamp sampler",
            }),
            .normals = gpuctx->device.create_sampler({
                .address_mode_u = daxa::SamplerAddressMode::REPEAT,
                .address_mode_v = daxa::SamplerAddressMode::REPEAT,
                .address_mode_w = daxa::SamplerAddressMode::REPEAT,
                .mip_lod_bias = 0.0f,
                .enable_anisotropy = true,
                .max_anisotropy = 16.0f,
                .max_lod = 3.0f,
                .name = "normals sampler",
            }),
        };
        render_data.debug = gpuctx->device.get_device_address(gpuctx->shader_debug_context.buffer).value();
    }
    ~RenderContext()
    {
        gpuctx->device.destroy_buffer(tgpu_render_data.get_state().buffers[0]);
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_clamp));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_repeat));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_clamp));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_repeat_ani));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.normals));
    }

    // GPUContext containing all shared global-ish gpu related data.
    GPUContext * gpuctx = {};
    // Passed from scene to renderer each frame to specify what should be drawn.
    SceneDraw scene_draw = {};

    daxa::TaskBuffer tgpu_render_data = {};

    // Data
    Settings prev_settings = {};
    SkySettings prev_sky_settings = {};
    VSMSettings prev_vsm_settings = {};
    RenderGlobalData render_data = {};
    std::array<bool, VSM_CLIP_LEVELS> draw_clip_frustum = {};
    std::array<bool, VSM_CLIP_LEVELS> draw_clip_frustum_pages = {};
    std::vector<u64> vsm_timestamp_results = {};
};