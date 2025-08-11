#pragma once

#include <set>
#include <unordered_map>
#include <string>

#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/imgui.hpp>

// Please do not include project specific files here!

auto channel_count_of_format(daxa::Format format) -> daxa::u32;

enum struct ScalarKind
{
    FLOAT,
    INT,
    UINT
};
auto scalar_kind_of_format(daxa::Format format) -> ScalarKind;

auto is_format_depth_stencil(daxa::Format format) -> bool;

union Vec4Union
{
    daxa_f32vec4 _float = {0, 0, 0, 0};
    daxa_i32vec4 _int;
    daxa_u32vec4 _uint;
};

struct DaxaTgDebugImageInspectorState
{
    // Written by ui:
    daxa::f64 min_v = 0.0;
    daxa::f64 max_v = 1.0;
    daxa::u32 mip = 0u;
    daxa::u32 layer = 0u;
    daxa::i32 rainbow_ints = false;
    daxa::i32 nearest_filtering = true;
    daxa_i32vec4 enabled_channels = {true, true, true, true};
    daxa_i32vec2 mouse_pos_relative_to_display_image = {0, 0};
    daxa_i32vec2 mouse_pos_relative_to_image_mip0 = {0, 0};
    daxa_i32vec2 display_image_size = {0, 0};

    daxa_i32vec2 frozen_mouse_pos_relative_to_image_mip0 = {0, 0};
    Vec4Union frozen_readback_raw = {};
    daxa_f32vec4 frozen_readback_color = {0, 0, 0, 0};
    daxa::f32 inspector_image_draw_scale = -1.0f;
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
};

struct DaxaTgDebugContext
{
    daxa::u32 frames_in_flight = 2;
    daxa_f32vec2 override_mouse_picker_uv = {};
    bool ui_open = {};
    bool request_mouse_picker_override = {};
    bool override_mouse_picker = {};
    bool override_frozen_state = {};
    std::array<char, 256> search_substr = {};
    std::string task_image_name = "color_image";
    daxa::u32 readback_index = 0;

    struct TgDebugTask
    {
        daxa::usize task_index = {};
        std::string task_name = {};
        std::vector<daxa::TaskAttachmentInfo> attachments = {};
    };
    std::vector<TgDebugTask> this_frame_debug_tasks = {}; // cleared every frame.
    std::unordered_map<std::string, DaxaTgDebugImageInspectorState> inspector_states = {};
    std::set<std::string> active_inspectors = {};
    std::unordered_map<std::string, daxa::u32> task_name_counters = {};

    void cleanup(daxa::Device)
    {
    }
};

inline auto debug_task_draw_display_image_pipeline_info() -> daxa::ComputePipelineCompileInfo2 const &
{
    static const auto PATH = "./src/rendering/daxa_tg_debugger/daxa_tg_debugger.hlsl";
    static const auto ENTRY = "entry_draw_debug_display";
    static daxa::ComputePipelineCompileInfo2 const info = []()
    {
        return daxa::ComputePipelineCompileInfo2{
            .source = daxa::ShaderSource{daxa::ShaderFile{PATH}},
            .entry_point = std::string(ENTRY),
            .name = (std::filesystem::path(PATH).filename().string() + "::") + std::string(ENTRY),
        };
    }();
    return info;
}

void debug_task(daxa::TaskInterface ti, DaxaTgDebugContext & tg_debug, daxa::ComputePipeline & pipeline, bool pre_task);

void tg_resource_debug_ui(
    daxa::ImGuiRenderer & imgui_renderer, 
    daxa::Device & device, 
    daxa::SamplerId lin_clamp_sampler, 
    daxa::SamplerId nearest_clamp_sampler, 
    DaxaTgDebugContext & tg_debug, 
    daxa::u32 frame_index, 
    bool ui_open);