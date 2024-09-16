#pragma once

#include <string>
#include <set>
#include <daxa/utils/imgui.hpp>

#include "../window.hpp"
#include "../scene/scene.hpp"
#include "../scene/asset_processor.hpp"

#include "../shader_shared/geometry.inl"
#include "../shader_shared/geometry_pipeline.inl"
#include "../shader_shared/readback.inl"

#include "../gpu_context.hpp"

struct DynamicMesh
{
    glm::mat4x4 prev_transform = {};
    glm::mat4x4 curr_transform = {};
    std::vector<AABB> meshlet_aabbs = {};
};

union Vec4Union
{
    daxa_f32vec4 _float = { 0, 0, 0, 0 };
    daxa_i32vec4 _int;
    daxa_u32vec4 _uint;
};

struct TgDebugImageInspectorState
{
    // Written by ui:
    f64 min_v = 0.0;
    f64 max_v = 1.0;
    u32 mip = 0u;
    u32 layer = 0u;
    i32 rainbow_ints = false;
    i32 nearest_filtering = true;
    daxa_i32vec4 enabled_channels = { true, true, true, true };
    daxa_i32vec2 mouse_pos_relative_to_display_image = { 0, 0 };    
    daxa_i32vec2 mouse_pos_relative_to_image = { 0, 0 };           
    daxa_i32vec2 display_image_size = { 0, 0 };

    daxa_i32vec2 frozen_mouse_pos_relative_to_display_image = { 0, 0 };    
    daxa_i32vec2 frozen_mouse_pos_relative_to_image = { 0, 0 };   
    Vec4Union frozen_readback_raw = {};
    daxa_f32vec4 frozen_readback_color = { 0, 0, 0, 0 };
    i32 resolution_draw_mode = 0;
    bool fixed_display_mip_sizes = true;
    bool freeze_image = false;
    bool active = false;
    bool display_image_hovered = false;
    bool freeze_image_hover_index = false;
    bool pre_task = false;
    // Written by tg:
    bool slice_valid = true;
    daxa::TaskImageAttachmentInfo attachment_info = {};
    daxa::BufferId readback_buffer = {};
    daxa::ImageInfo runtime_image_info = {};
    daxa::ImageId display_image = {};
    daxa::ImageId frozen_image = {};
    daxa::ImageId stale_image = {};
    daxa::ImageId stale_image1 = {};
};

struct TgDebugContext
{
    std::array<char, 256> search_substr = {};
    std::string task_image_name = "color_image";
    u32 readback_index = 0;

    struct TgDebugTask
    {
        std::string task_name = {};
        std::vector<std::string> attachment_names = {};
    };
    std::unordered_map<std::string, usize> this_frame_duplicate_task_name_counter = {};
    std::vector<TgDebugTask> this_frame_task_attachments = {}; // cleared every frame.
    std::unordered_map<std::string, TgDebugImageInspectorState> inspector_states = {};
    std::set<std::string> active_inspectors = {};

    void cleanup(daxa::Device device)
    {
        for (auto& inspector : inspector_states)
        {
            if (!inspector.second.display_image.is_empty())
                device.destroy_image((inspector.second.display_image));
            if (!inspector.second.frozen_image.is_empty())
                device.destroy_image((inspector.second.frozen_image));
            if (!inspector.second.stale_image.is_empty())
                device.destroy_image((inspector.second.stale_image));
            if (!inspector.second.stale_image1.is_empty())
                device.destroy_image((inspector.second.stale_image1));
            if (!inspector.second.readback_buffer.is_empty())
                device.destroy_buffer((inspector.second.readback_buffer));
        }
    }
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
            .nearest_repeat = gpuctx->device.create_sampler({
                .magnification_filter = daxa::Filter::NEAREST,
                .minification_filter = daxa::Filter::NEAREST,
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
                .name = "linear repeat ani sampler",
            }),
            .nearest_repeat_ani = gpuctx->device.create_sampler({
                .magnification_filter = daxa::Filter::NEAREST,
                .minification_filter = daxa::Filter::NEAREST,
                .mipmap_filter = daxa::Filter::LINEAR,
                .address_mode_u = daxa::SamplerAddressMode::REPEAT,
                .address_mode_v = daxa::SamplerAddressMode::REPEAT,
                .address_mode_w = daxa::SamplerAddressMode::REPEAT,
                .mip_lod_bias = 0.0f,
                .enable_anisotropy = true,
                .max_anisotropy = 16.0f,
                .name = "nearest repeat ani sampler",
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
        render_data.debug = gpuctx->device.buffer_device_address(gpuctx->shader_debug_context.buffer).value();
    }
    ~RenderContext()
    {
        tg_debug.cleanup(gpuctx->device);
        gpuctx->device.destroy_buffer(tgpu_render_data.get_state().buffers[0]);
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_clamp));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_repeat));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_repeat));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_clamp));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_repeat_ani));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_repeat_ani));
        gpuctx->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.normals));
    }

    // GPUContext containing all shared global-ish gpu related data.
    GPUContext * gpuctx = {};
    // Passed from scene to renderer each frame to specify what should be drawn.
    CPUMeshInstanceCounts mesh_instance_counts = {};

    daxa::TaskBuffer tgpu_render_data = {};
    std::vector<daxa::TaskImageView> debug_image_clones = {};

    auto task_debug_clone_image(daxa::TaskGraph tg, daxa::TaskImageView view, std::string new_name) -> daxa::TaskImageView
    {
        auto info = tg.transient_image_info(view);
        info.name = new_name;
        auto clone = tg.create_transient_image(info);
        tg.copy_image_to_image(view, clone);
        debug_image_clones.push_back(clone);
        return clone;
    }

    // Data
    TgDebugContext tg_debug = {};
    ReadbackValues general_readback;
    Settings prev_settings = {};
    SkySettings prev_sky_settings = {};
    VSMSettings prev_vsm_settings = {};
    RenderGlobalData render_data = {};
    std::array<bool, VSM_CLIP_LEVELS> draw_clip_frustum = {};
    std::array<bool, VSM_CLIP_LEVELS> draw_clip_frustum_pages = {};
    std::vector<u64> vsm_timestamp_results = {};
};