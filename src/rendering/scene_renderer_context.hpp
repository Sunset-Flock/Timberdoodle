#pragma once

#include <string>
#include <set>
#include <daxa/utils/imgui.hpp>

#include "../scene/scene.hpp"
#include "../daxa_helper.hpp"

#include "../shader_shared/geometry.inl"
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
    daxa_f32vec4 _float = {0, 0, 0, 0};
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
    daxa_i32vec4 enabled_channels = {true, true, true, true};
    daxa_i32vec2 mouse_pos_relative_to_display_image = {0, 0};
    daxa_i32vec2 mouse_pos_relative_to_image_mip0 = {0, 0};
    daxa_i32vec2 display_image_size = {0, 0};

    daxa_i32vec2 frozen_mouse_pos_relative_to_image_mip0 = {0, 0};
    Vec4Union frozen_readback_raw = {};
    daxa_f32vec4 frozen_readback_color = {0, 0, 0, 0};
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
        for (auto & inspector : inspector_states)
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

    static constexpr inline u32 GROUP_SIZE_MAX = 16;
    static constexpr inline u32 GROUP_COUNT_MAX = 16;
    using TimingName = std::string_view;
    struct GroupNames
    {
        std::string_view name = {};
        std::array<TimingName, GROUP_SIZE_MAX> timing_names = {};
    };
    static constexpr std::array<GroupNames, GROUP_COUNT_MAX> GROUPS = {
        GroupNames{
            "VISBUFFER",
            {
                "FIRST_PASS_ALLOC_BITFIELD_0",
                "FIRST_PASS_ALLOC_BITFIELD_1",
                "FIRST_PASS_SELECT_MESHLETS",
                "FIRST_PASS_GEN_HIZ",
                "FIRST_PASS_CULL_MESHES",
                "FIRST_PASS_CULL_MESHLETS_COMPUTE",
                "FIRST_PASS_CULL_AND_DRAW",
                "FIRST_PASS_DRAW",
                "SECOND_PASS_GEN_HIZ",
                "SECOND_PASS_CULL_MESHES",
                "SECOND_PASS_CULL_MESHLETS_COMPUTE",
                "SECOND_PASS_CULL_AND_DRAW",
                "SECOND_PASS_DRAW",
                "ANALYZE",
            },
        },
        GroupNames{
            "RTAO",
            {
                "TRACE",
                "DENOISE",
            },
        },
        GroupNames{
            "SHADE_OPAQUE",
            {
                "SHADE_OPAQUE",
            },
        },
        GroupNames{
            "SHADE_GBUFFER",
            {
                "SHADE_GBUFFER",
            },
        },
        GroupNames{
            "VSM",
            {
                "INVALIDATE_PAGES",
                "FREE_WRAPPED_PAGES",
                "MARK_REQUIRED_PAGES",
                "FIND_FREE_PAGES",
                "ALLOCATE_PAGES",
                "CLEAR_PAGES",
                "GEN_DIRY_BIT_HIZ",
                "CULL_AND_DRAW_PAGES",
                "CLEAR_DIRY_BITS",
            },
        },
        GroupNames{
            "PGI",
            {
                "TRACE_SHADE_RAYS",
                "PRE_UPDATE_PROBES",
                "UPDATE_PROBES",
                "UPDATE_PROBE_TEXELS",
                "EVAL_SCREEN_IRRADIANCE",
            },
        },
        GroupNames{
            "MISC",
            {
                "CULL_LIGHTS",
            },
        },
    };

    template <daxa::StringLiteral NAME>
    static consteval auto group_index() -> u32
    {
        for (u32 g = 0; g < GROUPS.size(); ++g)
        {
            if (std::string_view(NAME.value, NAME.SIZE).compare(GROUPS[g].name) == 0)
            {
                return g;
            }
        }
        return std::numeric_limits<u32>::max();
    }

    template <daxa::StringLiteral GROUP>
    static consteval auto in_group_timing_count() -> u32
    {
        constexpr u32 gidx = group_index<GROUP>();
        if constexpr (gidx != std::numeric_limits<u32>::max())
        {
            constexpr GroupNames group = GROUPS[gidx];
            for (u32 t = 0; t < group.timing_names.size(); ++t)
            {
                if (group.timing_names[t].name.size() == 0)
                {
                    return t;
                }
            }
        }
        return GROUP_SIZE_MAX;
    }

    static consteval auto group_count() -> u32
    {
        for (u32 g = 0; g < GROUPS.size(); ++g)
        {
            if (GROUPS[g].name.size() == 0)
            {
                return g;
            }
        }
        return GROUP_COUNT_MAX;
    }
    static constexpr inline u32 GROUP_COUNT = group_count();

    static consteval auto group_sizes() -> std::array<u32, GROUP_COUNT>
    {
        std::array<u32, GROUP_COUNT> ret = {};
        for (u32 g = 0; g < GROUP_COUNT; ++g)
        {
            u32 count = {};
            for (u32 t = 0; t < GROUPS[g].timing_names.size(); ++t)
            {
                if (GROUPS[g].timing_names[t].size() == 0)
                {
                    break;
                }
                count += 1;
            }
            ret[g] = count;
        }
        return ret;
    }
    static constexpr inline std::array<u32, GROUP_COUNT> GROUP_SIZES = group_sizes();

    template <daxa::StringLiteral GROUP>
    static consteval auto group_size() -> u32
    {
        constexpr u32 gidx = group_index<GROUP>();
        if constexpr (gidx != std::numeric_limits<u32>::max())
        {
            return GROUP_SIZES[gidx];
        }
        return std::numeric_limits<u32>::max();
    }

    static inline auto group_size(u32 gidx) -> u32
    {
        if (gidx < GROUP_COUNT)
        {
            return GROUP_SIZES[gidx];
        }
        return std::numeric_limits<u32>::max();
    }

    static consteval auto group_sizes_prefix_sum() -> std::array<u32, GROUP_COUNT>
    {
        std::array<u32, GROUP_COUNT> ret = {};
        u32 prefix_sum = 0;
        for (u32 g = 0; g < GROUP_COUNT; ++g)
        {
            ret[g] = prefix_sum;
            prefix_sum += GROUP_SIZES[g];
        }
        return ret;
    }
    static constexpr inline std::array<u32, GROUP_COUNT> GROUP_FLAT_INDEX_START = group_sizes_prefix_sum();

    static constexpr inline u32 FLAT_TIMINGS_COUNT = GROUP_FLAT_INDEX_START[GROUP_COUNT - 1] + GROUP_SIZES[GROUP_COUNT - 1];

    static consteval auto flat_timing_names() -> std::array<TimingName, FLAT_TIMINGS_COUNT>
    {
        std::array<TimingName, FLAT_TIMINGS_COUNT> ret = {};
        for (u32 g = 0; g < GROUP_COUNT; ++g)
        {
            u32 offset = GROUP_FLAT_INDEX_START[g];
            for (u32 t = 0; t < GROUP_SIZES[g]; ++t)
            {
                ret[offset + t] = GROUPS[g].timing_names[t];
            }
        }
        return ret;
    }
    static constexpr std::array<TimingName, FLAT_TIMINGS_COUNT> FLAT_TIMING_NAMES = flat_timing_names();

    static constexpr auto timing_name(u32 flat_index) -> TimingName
    {
        return FLAT_TIMING_NAMES[flat_index];
    }

    template <daxa::StringLiteral GROUP, daxa::StringLiteral NAME>
    static consteval auto in_group_timing_index() -> u32
    {
        constexpr u32 gidx = group_index<GROUP>();
        if constexpr (gidx != std::numeric_limits<u32>::max())
        {
            constexpr GroupNames group = GROUPS[gidx];
            for (u32 t = 0; t < group.timing_names.size(); ++t)
            {
                if (std::string_view(NAME.value, NAME.SIZE).compare(group.timing_names[t]) == 0)
                {
                    return t;
                }
            }
        }
        return std::numeric_limits<u32>::max();
    }

    template <daxa::StringLiteral GROUP, daxa::StringLiteral NAME>
    static consteval auto flat_timing_index() -> u32
    {
        constexpr u32 gidx = group_index<GROUP>();
        if constexpr (gidx != std::numeric_limits<u32>::max())
        {
            constexpr u32 tidx = in_group_timing_index<GROUP, NAME>();
            if constexpr (tidx != std::numeric_limits<u32>::max())
            {
                return GROUP_FLAT_INDEX_START[gidx] + tidx;
            }
        }
        return std::numeric_limits<u32>::max();
    }

    template <daxa::StringLiteral GROUP, daxa::StringLiteral NAME>
    static consteval auto index() -> u32
    {
        constexpr u32 i = flat_timing_index<GROUP, NAME>();
        return i;
    }

    template <daxa::StringLiteral GROUP>
    static consteval auto group_first_flat_index() -> u32
    {
        constexpr u32 gidx = group_index<GROUP>();
        if constexpr (gidx != std::numeric_limits<u32>::max())
        {
            return GROUP_FLAT_INDEX_START[gidx];
        }
        return std::numeric_limits<u32>::max();
    }

    static constexpr auto group_first_flat_index(u32 gidx) -> u32
    {
        if (gidx < GROUP_COUNT)
        {
            return GROUP_FLAT_INDEX_START[gidx];
        }
        return std::numeric_limits<u32>::max();
    }

    template <daxa::StringLiteral GROUP>
    static constexpr auto in_group_timing_name(u32 in_group_index) -> std::string_view
    {
        constexpr u32 gidx = group_index<GROUP>();
        if constexpr (gidx != std::numeric_limits<u32>::max())
        {
            return GROUPS[gidx].name;
        }
        return std::numeric_limits<u32>::max();
    }

    static constexpr auto in_group_timing_name(u32 group_index, u32 in_group_index) -> std::string_view
    {
        if (group_index < GROUP_COUNT)
        {
            if (in_group_index < GROUP_SIZES[group_index])
            {
                return GROUPS[group_index].timing_names[in_group_index];
            }
        }
        return std::string_view{};
    }

    template <daxa::StringLiteral GROUP>
    static consteval auto group_names() -> std::span<TimingName const>
    {
        constexpr u32 gidx = group_index<GROUP>();
        if constexpr (gidx != std::numeric_limits<u32>::max())
        {
            return std::span{GROUPS[gidx].timing_names.data(), GROUPS[gidx].timing_names.size()};
        }
        return std::span<TimingName>{};
    }

    static constexpr auto group_names(u32 gidx) -> std::span<TimingName const>
    {
        if (gidx < GROUP_COUNT)
        {
            return std::span{GROUPS[gidx].timing_names.data(), GROUPS[gidx].timing_names.size()};
        }
        return std::span<TimingName>{};
    }

    static constexpr auto group_name(u32 gidx) -> std::string_view
    {
        if (gidx < GROUP_COUNT)
        {
            return GROUPS[gidx].name;
        }
        return std::string_view{};
    }

    static constexpr inline u32 test = in_group_timing_index<"VISBUFFER", "FIRST_PASS_GEN_HIZ">();
    static constexpr inline u32 test2 = flat_timing_index<"VISBUFFER", "FIRST_PASS_GEN_HIZ">();

    struct State
    {
        bool enable_render_times = {};
        u32 query_version_index = {};
        u32 query_version_count = {};
        daxa::TimelineQueryPool timeline_query_pool = {};
        std::array<bool, FLAT_TIMINGS_COUNT> timer_set = {};
        std::array<u64, FLAT_TIMINGS_COUNT> current_times = {};
        std::array<u64, FLAT_TIMINGS_COUNT> smooth_current_times = {};

        void init(daxa::Device & device, u32 frames_in_flight)
        {
            query_version_count = frames_in_flight;
            timeline_query_pool = device.create_timeline_query_pool({
                .query_count = 2 * FLAT_TIMINGS_COUNT * query_version_count,
                .name = "render times query pool",
            });
        }

        void readback_render_times(u32 frame_index)
        {
            query_version_index = frame_index % query_version_count;
            auto const frames_in_flight_query_pool_offset = FLAT_TIMINGS_COUNT * 2 * query_version_index;
            std::vector<u64> results = timeline_query_pool.get_query_results(frames_in_flight_query_pool_offset, FLAT_TIMINGS_COUNT * 2);

            for (u32 i = 0; i < FLAT_TIMINGS_COUNT; ++i)
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
            for (u32 i = 0; i < FLAT_TIMINGS_COUNT; ++i)
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
                auto const query_pool_offset = FLAT_TIMINGS_COUNT * 2 * query_version_index;
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
        }
        void end_gpu_timer(auto & recorder, u32 render_time_index)
        {
            if (render_time_index == INVALID_RENDER_TIME_INDEX)
            {
                return;
            }
            write_timestamp(recorder, render_time_index * 2 + 1);
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
            for (u32 i = 0; i < FLAT_TIMINGS_COUNT; ++i)
            {
                timer_set[i] = false;
            }
            auto const query_pool_offset = FLAT_TIMINGS_COUNT * 2 * query_version_index;
            recorder.reset_timestamps({
                .query_pool = timeline_query_pool,
                .start_index = query_pool_offset,
                .count = FLAT_TIMINGS_COUNT * 2,
            });
        }
    };
}; // namespace RenderTimes

// Used to store all information used only by the renderer.
// Shared with task callbacks.
// Global for a frame within the renderer.
// For reusable sub-components of the renderer, use a new gpu_context.
struct RenderContext
{
    RenderContext(GPUContext * gpu_context)
        : gpu_context{gpu_context}
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
        lighting_phase_wait = gpu_context->device.create_binary_semaphore({.name = "as build to shade phase sema"});
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
    PGISettings prev_pgi_settings = {};
    RenderGlobalData render_data = {};
    std::array<bool, VSM_CLIP_LEVELS> draw_clip_frustum = {};
    i32 debug_frustum = {-1};
    bool visualize_point_frustum = {};
    bool visualize_spot_frustum = {};

    daxa::BinarySemaphore lighting_phase_wait = {};

    // TimingName code:
    RenderTimes::State render_times = {};
};