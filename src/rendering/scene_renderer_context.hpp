#pragma once

#include <string>
#include <set>
#include <daxa/utils/imgui.hpp>

#include "../scene/scene.hpp"
#include "../daxa_helper.hpp"

#include "../shader_shared/geometry.inl"
#include "../shader_shared/readback.inl"
#include "../shader_shared/ao.inl"

#include "../gpu_context.hpp"

struct DynamicMesh
{
    glm::mat4x4 prev_transform = {};
    glm::mat4x4 curr_transform = {};
    std::vector<AABB> meshlet_aabbs = {};
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
                "GEN_DIRY_BIT_HIZ_DIRECTIONAL",
                "GEN_DIRY_BIT_HIZ_POINT_SPOT",
                "CULL_MESHES_DIRECTIONAL",
                "CULL_MESHES_POINT_SPOT",
                "CULL_AND_DRAW_PAGES_DIRECTIONAL",
                "CULL_AND_DRAW_PAGES_POINT_SPOT",
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
            "RTGI",
            {
                "TRACE",
                "PRE_FILTER",
                "PRE_BLUR",
                "TEMPORAL_ACCUMULATION",
                "POST_BLUR_VERTICAL",
                "POST_BLUR_HORIZONTAL",
                "UPSCALE",
            },
        },
        GroupNames{
            "CLOUDS",
            {
                "VOLUME_SHADOW_MAP",
                "RAYMARCH",
                "COMPOSE",
            }},
        GroupNames{
            "MISC",
            {"CULL_LIGHTS",
                "AUTO_EXPOSURE_GEN_HIST",
                "AUTO_EXPOSURE_AVERAGE",
                "BUILD_TLAS"},
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
                if (group.timing_names[t].size() == 0)
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
        return std::string_view{"INVALID IDX"};
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
        return std::string_view{"INVALID IDX"};
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
        f64 time_scale = {};
        daxa::TimelineQueryPool timeline_query_pool = {};
        std::array<bool, FLAT_TIMINGS_COUNT> timer_set = {};
        std::array<f64, FLAT_TIMINGS_COUNT> current_times = {};
        std::array<f64, FLAT_TIMINGS_COUNT> smooth_times = {};
        std::array<f64, FLAT_TIMINGS_COUNT> smooth_variances = {};
        std::array<f64, GROUP_COUNT> current_group_times = {};
        std::array<f64, GROUP_COUNT> smooth_group_times = {};
        std::array<f64, GROUP_COUNT> smooth_group_variances = {};

        void init(daxa::Device & device, u32 frames_in_flight)
        {
            query_version_count = frames_in_flight;
            timeline_query_pool = device.create_timeline_query_pool({
                .query_count = 2 * FLAT_TIMINGS_COUNT * query_version_count,
                .name = "render times query pool",
            });
            time_scale = device.properties().limits.timestamp_period;
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
                f64 start = s_cast<f64>(results[i * 4 + 0]);
                f64 start_ready = s_cast<f64>(results[i * 4 + 1]);
                f64 end = s_cast<f64>(results[i * 4 + 2]);
                f64 end_ready = s_cast<f64>(results[i * 4 + 3]);
                if (start_ready && end_ready)
                {
                    current_times[i] = (end - start) * time_scale;
                }
            }
            for (u32 i = 0; i < FLAT_TIMINGS_COUNT; ++i)
            {
                if (!timer_set[i])
                {
                    smooth_times[i] = 0;
                }
                smooth_times[i] = (smooth_times[i] * 99.0 + current_times[i]) / 100.0;
                f64 current_diff_to_mean = std::abs(current_times[i] - smooth_times[i]);
                current_diff_to_mean = std::min(smooth_times[i] * 4, current_diff_to_mean); // clamp outliers down
                f64 const variance = current_diff_to_mean * current_diff_to_mean;
                smooth_variances[i] = (smooth_variances[i] * 19 + variance) / 20.0;
            }
            for (u32 group_i = 0; group_i < GROUP_COUNT; ++group_i)
            {
                u32 const group_size = GROUP_SIZES[group_i];
                u32 const group_first_index = group_first_flat_index(group_i);

                f64 raw_sum = 0;
                f64 smooth_sum = 0.0;
                for (u32 timer_i = 0; timer_i < group_size; ++timer_i)
                {
                    u32 const timer_index = group_first_index + timer_i;
                    raw_sum += current_times[timer_index];
                    smooth_sum += smooth_times[timer_index];
                }

                current_group_times[group_i] = raw_sum;
                smooth_group_times[group_i] = smooth_sum;
                f64 current_diff_to_mean = std::abs(static_cast<f64>(current_group_times[group_i]) - smooth_group_times[group_i]);
                current_diff_to_mean = std::min(smooth_group_times[group_i] * 4, current_diff_to_mean); // clamp outliers down
                f64 const variance = current_diff_to_mean * current_diff_to_mean;
                smooth_group_variances[group_i] = (smooth_group_variances[group_i] * 19 + variance) / 20.0;
            }
        }

        void write_timestamp(daxa::CommandRecorder & recorder, u32 frame_timestamp)
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
        void start_gpu_timer(daxa::CommandRecorder & recorder, u32 render_time_index)
        {
            if (render_time_index == INVALID_RENDER_TIME_INDEX)
            {
                return;
            }
            write_timestamp(recorder, render_time_index * 2);
            timer_set[render_time_index] = true;
        }
        void end_gpu_timer(daxa::CommandRecorder & recorder, u32 render_time_index)
        {
            if (render_time_index == INVALID_RENDER_TIME_INDEX)
            {
                return;
            }
            write_timestamp(recorder, render_time_index * 2 + 1);
        }
        struct ScopedGPUTimer
        {
            daxa::CommandRecorder * recorder = {};
            State * state = {};
            u32 render_time_index = {};

            ScopedGPUTimer(ScopedGPUTimer const &) = delete;
            ScopedGPUTimer(ScopedGPUTimer &&) = delete;
            ScopedGPUTimer(daxa::CommandRecorder * recorder, u32 render_time_index, State * state)
                : recorder{recorder}, render_time_index{render_time_index}, state{state}
            {
                state->start_gpu_timer(*recorder, render_time_index);
            }
            ~ScopedGPUTimer()
            {
                state->end_gpu_timer(*recorder, render_time_index);
            }
        };
        auto scoped_gpu_timer(daxa::CommandRecorder & recorder, u32 render_time_index) -> ScopedGPUTimer
        {
            return ScopedGPUTimer{&recorder, render_time_index, this};
        }
        auto get(u32 render_time_index) -> u64
        {
            return static_cast<u64>(current_times[render_time_index]);
        }
        auto get_average(u32 render_time_index) -> u64
        {
            return static_cast<u64>(smooth_times[render_time_index]);
        }
        auto get_variance(u32 render_time_index) -> u64
        {
            return static_cast<u64>(smooth_variances[render_time_index]);
        }
        auto get_group(u32 group_index) -> u64
        {
            return static_cast<u64>(current_group_times[group_index]);
        }
        auto get_group_average(u32 group_index) -> u64
        {
            return static_cast<u64>(smooth_group_times[group_index]);
        }
        auto get_group_variance(u32 group_index) -> u64
        {
            return static_cast<u64>(smooth_group_variances[group_index]);
        }
        void reset_timestamps_for_current_frame(daxa::CommandRecorder & recorder)
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
        tgpu_render_data = daxa::TaskBuffer{{
            .buffer = gpu_context->device.create_buffer({
                .size = sizeof(RenderGlobalData),
                .name = "scene render data",
            }),
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
            .clouds_noise_sampler = gpu_context->device.create_sampler({
                .magnification_filter = daxa::Filter::NEAREST,
                .minification_filter = daxa::Filter::NEAREST,
                .mipmap_filter = daxa::Filter::NEAREST,
                .address_mode_u = daxa::SamplerAddressMode::REPEAT,
                .address_mode_v = daxa::SamplerAddressMode::REPEAT,
                .address_mode_w = daxa::SamplerAddressMode::REPEAT,
                .mip_lod_bias = 0.5f,
                .name = "Clouds noise sampler",
            }),
        };
        render_data.debug = gpu_context->device.buffer_device_address(gpu_context->shader_debug_context.buffer).value();
        render_times.init(gpu_context->device, s_cast<u32>(gpu_context->swapchain.info().max_allowed_frames_in_flight));
    }
    ~RenderContext()
    {
        gpu_context->device.destroy_buffer(tgpu_render_data.id());
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_clamp));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_repeat));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_repeat));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_clamp));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.linear_repeat_ani));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.nearest_repeat_ani));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.normals));
        gpu_context->device.destroy_sampler(std::bit_cast<daxa::SamplerId>(render_data.samplers.clouds_noise_sampler));
    }

    // GPUContext containing all shared global-ish gpu related data.
    GPUContext * gpu_context = {};
    // Passed from scene to renderer each frame to specify what should be drawn.
    CPUMeshInstanceCounts mesh_instance_counts = {};

    daxa::TaskBuffer tgpu_render_data = {};

    // Data
    ReadbackValues general_readback;

    // Prev Settings
    Settings prev_settings = {};
    SkySettings prev_sky_settings = {};
    VSMSettings prev_vsm_settings = {};
    PGISettings prev_pgi_settings = {};
    LightSettings prev_light_settings = {};
    AoSettings prev_ao_settings = {};
    RtgiSettings prev_rtgi_settings = {};
    VolumetricSettings prev_volumetric_settings = {};

    // Settings
    RenderGlobalData render_data = {};

    std::array<bool, VSM_CLIP_LEVELS> draw_clip_frustum = {};
    i32 debug_frustum = {-1};
    bool visualize_point_frustum = {};
    bool visualize_spot_frustum = {};
    bool visualize_clouds_bounds = {};

    // TimingName code:
    RenderTimes::State render_times = {};
};