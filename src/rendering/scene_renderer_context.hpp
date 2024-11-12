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
    daxa_i32vec2 mouse_pos_relative_to_image_mip0 = { 0, 0 };           
    daxa_i32vec2 display_image_size = { 0, 0 };

    daxa_i32vec2 frozen_mouse_pos_relative_to_image_mip0 = { 0, 0 };   
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
    daxa::ImageId raw_image_copy = {};
    daxa::ImageId stale_image = {};
    daxa::ImageId stale_image1 = {};
};

struct TgDebugContext
{
    daxa_f32vec2 override_mouse_picker_uv = {};
    bool request_mouse_picker_override = {};
    bool override_mouse_picker = {};
    bool override_frozen_state = {};
    std::array<char, 256> search_substr = {};
    std::string task_image_name = "color_image";
    u32 readback_index = 0;

    struct TgDebugTask
    {
        usize task_index = {};
        std::string task_name = {};
        std::vector<daxa::TaskAttachmentInfo> attachments = {};
    };
    std::vector<TgDebugTask> this_frame_task_attachments = {}; // cleared every frame.
    std::unordered_map<std::string, TgDebugImageInspectorState> inspector_states = {};
    std::set<std::string> active_inspectors = {};

    void cleanup(daxa::Device device)
    {
        for (auto& inspector : inspector_states)
        {
            if (!inspector.second.display_image.is_empty())
                device.destroy_image((inspector.second.display_image));
            if (!inspector.second.raw_image_copy.is_empty())
                device.destroy_image((inspector.second.raw_image_copy));
            if (!inspector.second.stale_image.is_empty())
                device.destroy_image((inspector.second.stale_image));
            if (!inspector.second.stale_image1.is_empty())
                device.destroy_image((inspector.second.stale_image1));
            if (!inspector.second.readback_buffer.is_empty())
                device.destroy_buffer((inspector.second.readback_buffer));
        }
    }
};

namespace RenderTimes
{
    static constexpr inline u32 INVALID_RENDER_TIME_INDEX = ~0u;

    enum RenderTimesEnum
    {
        VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_0,
        VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_1,
        VISBUFFER_FIRST_PASS_SELECT_MESHLETS,
        VISBUFFER_FIRST_PASS_GEN_HIZ,
        VISBUFFER_FIRST_PASS_CULL_MESHES,
        VISBUFFER_FIRST_PASS_CULL_MESHLETS_COMPUTE,
        VISBUFFER_FIRST_PASS_CULL_AND_DRAW,
        VISBUFFER_FIRST_PASS_DRAW,
        VISBUFFER_SECOND_PASS_GEN_HIZ,
        VISBUFFER_SECOND_PASS_CULL_MESHES,
        VISBUFFER_SECOND_PASS_CULL_MESHLETS_COMPUTE,
        VISBUFFER_SECOND_PASS_CULL_AND_DRAW,
        VISBUFFER_SECOND_PASS_DRAW,
        VISBUFFER_ANALYZE,
        RAY_TRACED_AMBIENT_OCCLUSION,
        RAY_TRACED_AMBIENT_OCCLUSION_DENOISE,
        VSM_INVALIDATE_PAGES,
        VSM_FREE_WRAPPED_PAGES,
        VSM_MARK_REQUIRED_PAGES,
        VSM_FIND_FREE_PAGES,
        VSM_ALLOCATE_PAGES,
        VSM_CLEAR_PAGES,
        VSM_GEN_DIRY_BIT_HIZ,
        VSM_CULL_AND_DRAW_PAGES,
        VSM_CLEAR_DIRY_BITS,
        SHADE_OPAQUE,
        COUNT,
    };

    static constexpr inline std::array<char const *, RenderTimesEnum::COUNT> NAMES = {
        "VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_0",
        "VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_1",
        "VISBUFFER_FIRST_PASS_SELECT_MESHLETS",
        "VISBUFFER_FIRST_PASS_GEN_HIZ",
        "VISBUFFER_FIRST_PASS_CULL_MESHES",
        "VISBUFFER_FIRST_PASS_CULL_MESHLETS_COMPUTE",
        "VISBUFFER_FIRST_PASS_CULL_AND_DRAW",
        "VISBUFFER_FIRST_PASS_DRAW",
        "VISBUFFER_SECOND_PASS_GEN_HIZ",
        "VISBUFFER_SECOND_PASS_CULL_MESHES",
        "VISBUFFER_SECOND_PASS_CULL_MESHLETS_COMPUTE",
        "VISBUFFER_SECOND_PASS_CULL_AND_DRAW",
        "VISBUFFER_SECOND_PASS_DRAW",
        "VISBUFFER_ANALYZE",
        "RAY_TRACED_AMBIENT_OCCLUSION",
        "RAY_TRACED_AMBIENT_OCCLUSION_DENOISE",
        "VSM_INVALIDATE_PAGES",
        "VSM_FREE_WRAPPED_PAGES",
        "VSM_MARK_REQUIRED_PAGES",
        "VSM_FIND_FREE_PAGES",
        "VSM_ALLOCATE_PAGES",
        "VSM_CLEAR_PAGES",
        "VSM_GEN_DIRY_BIT_HIZ",
        "VSM_CULL_AND_DRAW_PAGES",
        "VSM_CLEAR_DIRY_BITS",
        "SHADE_OPAQUE",
    };

    static constexpr inline auto to_string(RenderTimesEnum index) -> char const *
    {
        return NAMES[index];
    }

    enum RenderGroupTimesEnum
    {
        GROUP_VISBUFFER,
        GROUP_AMBIENT_OCCLUSION,
        GROUP_SHADE_OPAQUE,
        GROUP_VSM_INVALIDATE_STAGES,
        GROUP_VSM_BOOKKEEPING,
        GROUP_VSM_CULL_AND_DRAW,
        GROUP_COUNT
    };

    static constexpr inline std::array<char const *, RenderTimesEnum::COUNT> GROUP_NAMES = {
        "GROUP_VISBUFFER",
        "GROUP_AMBIENT_OCCLUSION",
        "GROUP_SHADE_OPAQUE",
        "GROUP_VSM_INVALIDATE_PAGES",
        "GROUP_VSM_BOOKKEEPING",
        "GROUP_VSM_CULL_AND_DRAW",
    };

    static constexpr inline auto to_string(RenderGroupTimesEnum index) -> char const *
    {
        return GROUP_NAMES[index];
    }

    static constexpr inline std::array GROUP_VISBUFFER_TIMES = std::array{
        VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_0,
        VISBUFFER_FIRST_PASS_ALLOC_BITFIELD_1,
        VISBUFFER_FIRST_PASS_SELECT_MESHLETS,
        VISBUFFER_FIRST_PASS_GEN_HIZ,
        VISBUFFER_FIRST_PASS_CULL_MESHES,
        VISBUFFER_FIRST_PASS_CULL_MESHLETS_COMPUTE,
        VISBUFFER_FIRST_PASS_CULL_AND_DRAW,
        VISBUFFER_FIRST_PASS_DRAW,
        VISBUFFER_SECOND_PASS_GEN_HIZ,
        VISBUFFER_SECOND_PASS_CULL_MESHES,
        VISBUFFER_SECOND_PASS_CULL_MESHLETS_COMPUTE,
        VISBUFFER_SECOND_PASS_CULL_AND_DRAW,
        VISBUFFER_SECOND_PASS_DRAW,
        VISBUFFER_ANALYZE,
    };

    static constexpr inline std::array GROUP_AMBIENT_OCCLUSION_TIMES = std::array{
        RAY_TRACED_AMBIENT_OCCLUSION,
        RAY_TRACED_AMBIENT_OCCLUSION_DENOISE,
    };

    static constexpr inline std::array GROUP_SHADE_OPAQUE_TIMES = std::array{
        SHADE_OPAQUE,
    };

    static constexpr inline std::array GROUP_VSM_INVALIDATE_PAGES = std::array{
        VSM_INVALIDATE_PAGES,
    };

    static constexpr inline std::array GROUP_VSM_BOOKKEEPING_TIMES = std::array{
        VSM_FREE_WRAPPED_PAGES,
        VSM_MARK_REQUIRED_PAGES,
        VSM_FIND_FREE_PAGES,
        VSM_ALLOCATE_PAGES,
        VSM_CLEAR_PAGES,
        VSM_GEN_DIRY_BIT_HIZ,
    };

    static constexpr inline std::array GROUP_VSM_CULL_AND_DRAW_TIMES = std::array{
        VSM_CULL_AND_DRAW_PAGES,
    };

    static constexpr inline std::array<std::span<RenderTimesEnum const>, GROUP_COUNT> GROUP_RENDER_TIMES = {
        GROUP_VISBUFFER_TIMES,
        GROUP_AMBIENT_OCCLUSION_TIMES,
        GROUP_SHADE_OPAQUE_TIMES,
        GROUP_VSM_INVALIDATE_PAGES,
        GROUP_VSM_BOOKKEEPING_TIMES,
        GROUP_VSM_CULL_AND_DRAW_TIMES,
    };

    struct State
    {
        bool enable_render_times = {};
        u32 query_version_index = {};
        u32 query_version_count = {};
        daxa::TimelineQueryPool timeline_query_pool = {};
        std::array<bool, RenderTimes::COUNT> timer_set = {};
        std::array<u64, RenderTimes::COUNT> current_times = {};
        std::array<u64, RenderTimes::COUNT> smooth_current_times = {};

        void init(daxa::Device& device, u32 frames_in_flight)
        {
            query_version_count = frames_in_flight + 1;
            timeline_query_pool = device.create_timeline_query_pool({
                .query_count = 2 * RenderTimes::COUNT * query_version_count,
                .name = "render times query pool",
            });
        }

        void readback_render_times(u32 frame_index)
        {
            query_version_index = frame_index % query_version_count;
            const auto frames_in_flight_query_pool_offset = RenderTimes::COUNT * 2 * query_version_index;
            std::vector<u64> results = timeline_query_pool.get_query_results(frames_in_flight_query_pool_offset, RenderTimes::COUNT * 2);

            for (u32 i = 0; i < RenderTimes::COUNT; ++i)
            {
                if (!timer_set[i])
                {
                    current_times[i] = 0;
                }
                // The query results layout:
                // [0] start timestamp value
                // [1] start timestamp readyness
                // [2] end timestamp value
                // [3] end timestamp readyness
                u64 start = results[i * 4 + 0];
                u64 start_ready = results[i * 4 + 1];
                u64 end = results[i * 4 + 2];
                u64 end_ready = results[i * 4 + 3];
                if (start_ready && end_ready)
                {
                    current_times[i] = end - start;
                }
            }
            for (u32 i = 0; i < RenderTimes::COUNT; ++i)
            {
                if (!timer_set[i])
                {
                    smooth_current_times[i] = 0;
                }
                smooth_current_times[i] = (smooth_current_times[i] * 199 + current_times[i]) / 200;
            }
        }
        
        void write_timestamp(auto & recorder, u32 frame_timestamp)
        {
            if (enable_render_times)
            {
                const auto query_pool_offset = RenderTimes::COUNT * 2 * query_version_index;
                recorder.write_timestamp({
                    .query_pool = timeline_query_pool, 
                    .pipeline_stage = daxa::PipelineStageFlagBits::ALL_COMMANDS, 
                    .query_index = frame_timestamp + query_pool_offset,
                });
            }
        }
        void start_gpu_timer(auto & recorder, u32 render_time_index)
        {
            if (render_time_index == INVALID_RENDER_TIME_INDEX)
            {
                return;
            }
            write_timestamp(recorder, render_time_index * 2);
            timer_set[render_time_index] = true;
            const auto query_pool_offset = RenderTimes::COUNT * 2 * query_version_index;
        }
        void end_gpu_timer(auto & recorder, u32 render_time_index)
        {
            if (render_time_index == INVALID_RENDER_TIME_INDEX)
            {
                return;
            }
            write_timestamp(recorder, render_time_index * 2 + 1);
            const auto query_pool_offset = RenderTimes::COUNT * 2 * query_version_index;
        }
        auto get(u32 render_time_index) -> u64
        {
            return current_times[render_time_index];
        }
        auto get_smooth(u32 render_time_index) -> u64
        {
            return smooth_current_times[render_time_index];
        }
        void reset_timestamps_for_current_frame(auto & recorder)
        {
            for (u32 i = 0; i < COUNT; ++i)
            {
                timer_set[i] = false;
            }
            const auto query_pool_offset = RenderTimes::COUNT * 2 * query_version_index;
            recorder.reset_timestamps({
                .query_pool = timeline_query_pool,
                .start_index = query_pool_offset,
                .count = RenderTimes::COUNT * 2,
            });
        }
    };
};


// Used to store all information used only by the renderer.
// Shared with task callbacks.
// Global for a frame within the renderer.
// For reusable sub-components of the renderer, use a new gpu_context.
struct RenderContext
{
    RenderContext(GPUContext * gpu_context) : gpu_context{gpu_context}
    {
        tgpu_render_data = daxa::TaskBuffer{daxa::TaskBufferInfo{
            .initial_buffers = {
                .buffers = std::array{
                    gpu_context->device.create_buffer({
                        .size = sizeof(RenderGlobalData),
                        .name = "scene render data",
                    }),
                },
            },
            .name = "scene render data",
        }};
        render_data.samplers = {
            .linear_clamp = gpu_context->device.create_sampler({
                .name = "linear clamp sampler",
            }),
            .linear_repeat = gpu_context->device.create_sampler({
                .address_mode_u = daxa::SamplerAddressMode::REPEAT,
                .address_mode_v = daxa::SamplerAddressMode::REPEAT,
                .address_mode_w = daxa::SamplerAddressMode::REPEAT,
                .name = "linear repeat sampler",
            }),
            .nearest_repeat = gpu_context->device.create_sampler({
                .magnification_filter = daxa::Filter::NEAREST,
                .minification_filter = daxa::Filter::NEAREST,
                .address_mode_u = daxa::SamplerAddressMode::REPEAT,
                .address_mode_v = daxa::SamplerAddressMode::REPEAT,
                .address_mode_w = daxa::SamplerAddressMode::REPEAT,
                .name = "linear repeat sampler",
            }),
            .nearest_clamp = gpu_context->device.create_sampler({
                .magnification_filter = daxa::Filter::NEAREST,
                .minification_filter = daxa::Filter::NEAREST,
                .mipmap_filter = daxa::Filter::NEAREST,
                .name = "nearest clamp sampler",
            }),
            .linear_repeat_ani = gpu_context->device.create_sampler({
                .address_mode_u = daxa::SamplerAddressMode::REPEAT,
                .address_mode_v = daxa::SamplerAddressMode::REPEAT,
                .address_mode_w = daxa::SamplerAddressMode::REPEAT,
                .mip_lod_bias = 0.0f,
                .enable_anisotropy = true,
                .max_anisotropy = 16.0f,
                .name = "linear repeat ani sampler",
            }),
            .nearest_repeat_ani = gpu_context->device.create_sampler({
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
            .normals = gpu_context->device.create_sampler({
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
        render_data.debug = gpu_context->device.buffer_device_address(gpu_context->shader_debug_context.buffer).value();
        render_times.init(gpu_context->device, gpu_context->swapchain.info().max_allowed_frames_in_flight);
    }
    ~RenderContext()
    {
        tg_debug.cleanup(gpu_context->device);
        gpu_context->device.destroy_buffer(tgpu_render_data.get_state().buffers[0]);
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_clamp));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_repeat));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_repeat));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_clamp));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_repeat_ani));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_repeat_ani));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.normals));
    }

    // GPUContext containing all shared global-ish gpu related data.
    GPUContext * gpu_context = {};
    // Passed from scene to renderer each frame to specify what should be drawn.
    CPUMeshInstanceCounts mesh_instance_counts = {};

    daxa::TaskBuffer tgpu_render_data = {};

    // Data
    TgDebugContext tg_debug = {};
    ReadbackValues general_readback;
    Settings prev_settings = {};
    SkySettings prev_sky_settings = {};
    VSMSettings prev_vsm_settings = {};
    RenderGlobalData render_data = {};
    std::array<bool, VSM_CLIP_LEVELS> draw_clip_frustum = {};
    std::vector<u64> vsm_timestamp_results = {};

    // Timing code:
    RenderTimes::State render_times = {};
};