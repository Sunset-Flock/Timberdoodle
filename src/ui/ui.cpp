#include "ui.hpp"
#include <filesystem>
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include "widgets/helpers.hpp"
#include "../daxa_helper.hpp"

void setup_colors()
{
    ImVec4 * colors = ImGui::GetStyle().Colors;
    ImGuiStyle & style = ImGui::GetStyle();
    // clang-format off
    colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = bg_0;
    colors[ImGuiCol_ChildBg]                = bg_0;
    colors[ImGuiCol_PopupBg]                = bg_1;
    colors[ImGuiCol_Border]                 = {alt_2.x, alt_2.y, alt_2.z, 0.5};
    colors[ImGuiCol_BorderShadow]           = bg_1;
    colors[ImGuiCol_FrameBg]                = bg_2;
    colors[ImGuiCol_FrameBgHovered]         = hovered_1;
    colors[ImGuiCol_FrameBgActive]          = hovered_1;
    colors[ImGuiCol_TitleBg]                = bg_0;
    colors[ImGuiCol_TitleBgActive]          = bg_0;
    colors[ImGuiCol_TitleBgCollapsed]       = bg_0;
    colors[ImGuiCol_MenuBarBg]              = bg_0;
    colors[ImGuiCol_ScrollbarBg]            = bg_0;
    colors[ImGuiCol_ScrollbarGrab]          = bg_2;
    colors[ImGuiCol_ScrollbarGrabHovered]   = hovered_1;
    colors[ImGuiCol_ScrollbarGrabActive]    = select_blue_1;
    colors[ImGuiCol_CheckMark]              = select_blue_1;
    colors[ImGuiCol_SliderGrab]             = alt_1;
    colors[ImGuiCol_SliderGrabActive]       = select_blue_1;
    colors[ImGuiCol_Button]                 = bg_2;
    colors[ImGuiCol_ButtonHovered]          = hovered_1;
    colors[ImGuiCol_ButtonActive]           = select_blue_1;
    colors[ImGuiCol_Header]                 = bg_2;
    colors[ImGuiCol_HeaderHovered]          = hovered_1;
    colors[ImGuiCol_HeaderActive]           = select_blue_1;
    colors[ImGuiCol_Separator]              = alt_1;
    colors[ImGuiCol_SeparatorHovered]       = hovered_1;
    colors[ImGuiCol_SeparatorActive]        = select_blue_1;
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_Tab]                    = bg_1;
    colors[ImGuiCol_TabHovered]             = hovered_1;
    colors[ImGuiCol_TabActive]              = alt_1;
    colors[ImGuiCol_TabUnfocused]           = bg_1;
    colors[ImGuiCol_TabUnfocusedActive]     = bg_1;
    colors[ImGuiCol_PlotLines]              = select_blue_1;
    colors[ImGuiCol_PlotLinesHovered]       = select_blue_1;
    colors[ImGuiCol_PlotHistogram]          = select_blue_1;
    colors[ImGuiCol_PlotHistogramHovered]   = select_blue_1;
    colors[ImGuiCol_TableHeaderBg]          = bg_0;
    colors[ImGuiCol_TableBorderStrong]      = {alt_1.x, alt_1.y, alt_1.z, 0.5};
    colors[ImGuiCol_TableBorderLight]       = {alt_2.x, alt_2.y, alt_2.z, 0.5};
    colors[ImGuiCol_TableRowBg]             = bg_0;
    colors[ImGuiCol_TableRowBgAlt]          = bg_1;
    colors[ImGuiCol_TextSelectedBg]         = bg_3;
    colors[ImGuiCol_DragDropTarget]         = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_NavHighlight]           = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);

    style.WindowPadding                     = ImVec2(8.00f, 8.00f);
    style.FramePadding                      = ImVec2(5.00f, 2.00f);
    style.CellPadding                       = ImVec2(6.00f, 6.00f);
    style.ItemSpacing                       = ImVec2(6.00f, 6.00f);
    style.ItemInnerSpacing                  = ImVec2(6.00f, 6.00f);
    style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
    style.IndentSpacing                     = 25;
    style.ScrollbarSize                     = 15;
    style.GrabMinSize                       = 10;
    style.WindowBorderSize                  = 1;
    style.ChildBorderSize                   = 0;
    style.PopupBorderSize                   = 1;
    style.FrameBorderSize                   = 0;
    style.TabBorderSize                     = 0;
    style.WindowRounding                    = 5;
    style.ChildRounding                     = 4;
    style.FrameRounding                     = 3;
    style.PopupRounding                     = 4;
    style.ScrollbarRounding                 = 2;
    style.GrabRounding                      = 3;
    style.LogSliderDeadzone                 = 4;
    style.TabRounding                       = 4;
    style.WindowMenuButtonPosition = ImGuiDir_None;
    // clang-format on
};

UIEngine::UIEngine(Window & window, AssetProcessor & asset_processor, GPUContext * gpu_context)
    : scene_graph(&imgui_renderer, &icons, gpu_context->lin_clamp_sampler),
      property_viewer(&imgui_renderer, &icons, gpu_context->lin_clamp_sampler),
      gpu_context{gpu_context}
{
    auto * imgui_context = ImGui::CreateContext();
    auto * implot_context = ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(window.glfw_handle, true);
    ImGuiIO & io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    icons.reserve(s_cast<u32>(ICONS::SIZE));
    for (u32 icon_idx = 0; icon_idx < s_cast<u32>(ICONS::SIZE); icon_idx++)
    {
        AssetProcessor::NonmanifestLoadRet ret = asset_processor.load_nonmanifest_texture(ICON_TO_PATH.at(icon_idx));
        if (auto const * err = std::get_if<AssetProcessor::AssetLoadResultCode>(&ret))
        {
            DEBUG_MSG(fmt::format("[UIEngine::UIEngine] ERROR failed to load icon from path {}", ICON_TO_PATH.at(icon_idx)));
            icons.push_back({});
            continue;
        }
        icons.push_back(std::get<daxa::ImageId>(ret));
    }
    constexpr static std::string_view text_font_path = "deps\\timberdoodle_assets\\ui\\fonts\\sarasa-term-k-regular.ttf";
    if (std::filesystem::exists(text_font_path))
    {
        io.Fonts->AddFontFromFileTTF(text_font_path.data(), text_font_size, nullptr, io.Fonts->GetGlyphRangesDefault());
    }
    /// NOTE: Needs to after all the init functions
    imgui_renderer = daxa::ImGuiRenderer({gpu_context->device, gpu_context->swapchain.get_format(), imgui_context, false});
    setup_colors();
}

void UIEngine::main_update(GPUContext const & gpu_context, RenderContext & render_context, Scene const & scene, ApplicationState & app_state)
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground |
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGuiViewport const * viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");

    ImGui::Begin("DockSpace Demo", nullptr, window_flags);
    ImGui::PopStyleVar(3);
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    ImGui::End();

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Widgets"))
        {
            ImGui::MenuItem("Settings", NULL, &renderer_settings);
            ImGui::MenuItem("Widget Settings", NULL, &widget_settings);
            ImGui::MenuItem("Renderer Statistics", NULL, &widget_renderer_statistics);
            ImGui::MenuItem("Scene Hierarchy", NULL, &widget_scene_hierarchy);
            ImGui::MenuItem("Shader Debug Menu", NULL, &shader_debug_menu);
            ImGui::MenuItem("VSM Debug Menu", NULL, &vsm_debug_menu);
            ImGui::MenuItem("Widget Property Viewer", NULL, &widget_property_viewer);
            ImGui::MenuItem("TaskGraphDebugUi", NULL, &tg_debug_ui);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (vsm_debug_menu)
    {
        if (ImGui::Begin("VSM Debug Menu", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            bool enable = s_cast<bool>(render_context.render_data.vsm_settings.enable);
            bool force_clip_level = s_cast<bool>(render_context.render_data.vsm_settings.force_clip_level);
            bool enable_caching = s_cast<bool>(render_context.render_data.vsm_settings.enable_caching);
            ImGui::BeginChild("Checkboxes", ImVec2(0, ImGui::CalcTextSize("a").y * 6.0f));
            {
                ImGui::Text("Draw cascade frustum");
                ImGui::SetWindowFontScale(0.5);
                for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
                {
                    ImGui::Checkbox(fmt::format("##clips{}", clip).c_str(), &render_context.draw_clip_frustum.at(clip));
                    ImGui::SameLine();
                }
                ImGui::SetWindowFontScale(1.0);
                ImGui::Dummy({});
                ImGui::Text("Draw cascade frustum pages");
                ImGui::SetWindowFontScale(0.5);
                for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
                {
                    ImGui::BeginDisabled(!render_context.draw_clip_frustum.at(clip));
                    ImGui::Checkbox(fmt::format("##pages{}", clip).c_str(), &render_context.draw_clip_frustum_pages.at(clip));
                    ImGui::EndDisabled();
                    ImGui::SameLine();
                }
                ImGui::SetWindowFontScale(1.0);
            }
            ImGui::EndChild();

            ImGui::Checkbox("Enable VSM", &enable);
            ImGui::Checkbox("Force clip level", &force_clip_level);
            ImGui::Checkbox("Enable caching", &enable_caching);
            auto use_simplified_light_matrix = s_cast<bool>(render_context.render_data.vsm_settings.use_simplified_light_matrix);
            ImGui::Checkbox("Use simplified light matrix", &use_simplified_light_matrix);
            render_context.render_data.vsm_settings.use_simplified_light_matrix = use_simplified_light_matrix;
            ImGui::SliderFloat("Clip 0 scale", &render_context.render_data.vsm_settings.clip_0_frustum_scale, 0.1f, 20.0f);
            ImGui::SliderFloat("Clip selection bias", &render_context.render_data.vsm_settings.clip_selection_bias, -0.5f, 2.0f);
            ImGui::SliderFloat("Slope bias", &render_context.render_data.vsm_settings.slope_bias, 0.0, 10.0);
            ImGui::SliderFloat("Constant bias", &render_context.render_data.vsm_settings.constant_bias, 0.0, 20.0);
            ImGui::BeginDisabled(!force_clip_level);
            i32 forced_clip_level = render_context.render_data.vsm_settings.forced_clip_level;
            ImGui::SliderInt("Forced clip level", &forced_clip_level, 0, VSM_CLIP_LEVELS - 1);
            ImGui::EndDisabled();
            render_context.render_data.vsm_settings.enable = enable;
            render_context.render_data.vsm_settings.force_clip_level = force_clip_level;
            render_context.render_data.vsm_settings.forced_clip_level = force_clip_level ? forced_clip_level : -1;
            render_context.render_data.vsm_settings.enable_caching = enable_caching;

            ImGui::Image(
                imgui_renderer.create_texture_id({
                    .image_view_id = render_context.gpu_context->shader_debug_context.vsm_debug_page_table.get_state().images[0].default_view(),
                    .sampler_id = std::bit_cast<daxa::SamplerId>(render_context.render_data.samplers.nearest_clamp),
                }),
                ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x));
            ImGui::Image(
                imgui_renderer.create_texture_id({
                    .image_view_id = render_context.gpu_context->shader_debug_context.vsm_debug_meta_memory_table.get_state().images[0].default_view(),
                    .sampler_id = std::bit_cast<daxa::SamplerId>(render_context.render_data.samplers.nearest_clamp),
                }),
                ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x));
        }
        ImGui::End();
    }
    if (widget_settings)
    {
        if (ImGui::Begin("Widget Settings", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            ImGui::SeparatorText("Scene graph widget settings");
            ImGui::SliderFloat("icon size", &scene_graph.icon_size, 1.0f, 50.0f);
            ImGui::SliderFloat("spacing", &scene_graph.icon_text_spacing, 1.0f, 50.0f);
            ImGui::SliderFloat("indent", &scene_graph.indent, 1.0f, 50.0f);
            ImGui::Separator();
            ImGui::InputScalarN("resolution", ImGuiDataType_U32, &render_context.render_data.settings.render_target_size, 2);
        }
        ImGui::End();
    }
    if (widget_renderer_statistics)
    {
        perf_sample_count += 1;
        auto get_exec_time_from_timestamp = [&render_context](u32 timestamp_start_index) -> u64
        {
            // Timestamps ready
            if (render_context.vsm_timestamp_results.at(timestamp_start_index + 1) != 0u &&
                render_context.vsm_timestamp_results.at(timestamp_start_index + 3) != 0u)
            {
                auto const end_timestamp = render_context.vsm_timestamp_results.at(timestamp_start_index + 2);
                auto const start_timestamp = render_context.vsm_timestamp_results.at(timestamp_start_index);
                return end_timestamp - start_timestamp;
            }
            else
            {
                DEBUG_MSG(fmt::format("[WARN] Unwritten timestamp {}", timestamp_start_index));
                return 0ull;
            }
        };
        f32 const weight = 0.99;
        static constexpr std::array task_names{
            "Bookkeeping",
            "VSM draw",
            "Sampling"};

        static float t = 0;
        t += gather_perm_measurements ? render_context.render_data.delta_time : 0.0f;
        bool auto_reset_timings = false;
        if (ImGui::Begin("Render statistics", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            // Calculate Statistics:
            u32 mesh_instance_count = render_context.mesh_instance_counts.mesh_instance_count;
            u32 first_pass_meshlets = render_context.general_readback.first_pass_meshlet_count[0] + render_context.general_readback.first_pass_meshlet_count[1];
            u32 second_pass_meshlets = render_context.general_readback.second_pass_meshlet_count[0] + render_context.general_readback.second_pass_meshlet_count[1];
            u32 total_meshlets_drawn = first_pass_meshlets + second_pass_meshlets;
            
            u32 used_dynamic_section_sfpm_bitfield = (render_context.general_readback.sfpm_bitfield_arena_requested * 4) / 1000;
            u32 dynamic_section_sfpm_bitfield_size = (FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE * 4  - (4 * MAX_ENTITIES)) / 1000;
            struct VisbufferPipelineStat
            {
                char const * name = {};
                char const * unit = {};
                u32 value = {};
                u32 max_value = {};
            };
            std::array visbuffer_pipeline_stats = {
                VisbufferPipelineStat{"Mesh Instances (Unculled)", "", mesh_instance_count, MAX_MESH_INSTANCES},
                VisbufferPipelineStat{"First Pass Meshlets", "", first_pass_meshlets, MAX_MESHLET_INSTANCES},
                VisbufferPipelineStat{"Second Pass Meshlets", "", second_pass_meshlets, MAX_MESHLET_INSTANCES},
                VisbufferPipelineStat{"Total Meshlet Instances", "", total_meshlets_drawn, MAX_MESHLET_INSTANCES},
                VisbufferPipelineStat{"First Pass Bitfield Use", "kb", used_dynamic_section_sfpm_bitfield, dynamic_section_sfpm_bitfield_size},
            };

            ImGui::SeparatorText("Visbuffer Pipeline Statistics");
            {
                if (ImGui::BeginTable("Test", 4, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                {
                    ImGui::TableSetupColumn("Value Name", {});
                    ImGui::TableSetupColumn("Value", {});
                    ImGui::TableSetupColumn("Value Max", {});
                    ImGui::TableSetupColumn("Value %", {});
                    ImGui::TableHeadersRow();
                    for (auto const& stat : visbuffer_pipeline_stats)
                    {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text(stat.name);
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%i%s", stat.value, stat.unit);
                        ImGui::TableSetColumnIndex(2);
                        ImGui::Text("%i%s", stat.max_value, stat.unit);
                        ImGui::TableSetColumnIndex(3);
                        ImGui::Text("%f%%", static_cast<f32>(stat.value) / static_cast<f32>(stat.max_value) * 100.0f);
                    }
                    ImGui::EndTable();
                }
            }
            ImGui::SeparatorText("Timings");
            if (gather_perm_measurements)
            {
                if (ImGui::Button("Stop gathering")) { gather_perm_measurements = false; }
            }
            else
            {
                if (ImGui::Button("Start gathering"))
                {
                    gather_perm_measurements = true;
                    auto_reset_timings = true;
                }
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset timings") || auto_reset_timings)
            {
                t = 0;
                for (i32 i = 0; i < 10; i++)
                {
                    measurements.vsm_timings_ewa.at(i) = 0.0f;
                    measurements.vsm_timings_mean.at(i) = 0.0f;
                    measurements.mean_sample_count = 0;
                }
                for (i32 i = 0; i < 3; i++)
                {
                    measurements.scrolling_ewa.at(i).erase();
                    measurements.scrolling_mean.at(i).erase();
                    measurements.scrolling_raw.at(i).erase();
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Show entire measured interval", &show_entire_interval);
            f32 rolling_mean_weight = s_cast<f32>(measurements.mean_sample_count) / s_cast<f32>(measurements.mean_sample_count + 1);
            for (u32 i = 0; i < 11; i++)
            {
                u64 const timestamp_value = s_cast<f32>(get_exec_time_from_timestamp(i * 4)) / 1'000.0f;
                measurements.vsm_timings_raw.at(i) = timestamp_value;
                if (timestamp_value != 0)
                {
                    measurements.vsm_timings_ewa.at(i) = measurements.vsm_timings_ewa.at(i) * weight + (1.0f - weight) * timestamp_value;
                    measurements.vsm_timings_mean.at(i) = measurements.vsm_timings_mean.at(i) * rolling_mean_weight + timestamp_value * (1.0f - rolling_mean_weight);
                }
            }
            measurements.mean_sample_count += 1;
            //  ========================= BOOKKEEPING =======================================
            f32 bookkeeping_ewa = 0.0;
            f32 bookkeeping_average = 0.0;
            f32 bookkeeping_raw = 0.0;
            for (u32 i = 0; i < 6; i++)
            {
                bookkeeping_raw += measurements.vsm_timings_raw.at(i);
                bookkeeping_ewa += measurements.vsm_timings_ewa.at(i);
                bookkeeping_average += measurements.vsm_timings_mean.at(i);
            }
            if (gather_perm_measurements)
            {
                measurements.scrolling_raw.at(0).add_point(ImVec2(t, bookkeeping_raw == 0.0 ? measurements.scrolling_raw.at(0).back().y : bookkeeping_raw));
                measurements.scrolling_ewa.at(0).add_point(ImVec2(t, bookkeeping_ewa));
                measurements.scrolling_mean.at(0).add_point(ImVec2(t, bookkeeping_average));
            }
            // ========================== DRAW ==============================================
            f32 draw_raw = measurements.vsm_timings_raw.at(6);
            f32 draw_ewa = measurements.vsm_timings_ewa.at(6);
            f32 draw_average = measurements.vsm_timings_mean.at(6);
            if (gather_perm_measurements)
            {
                measurements.scrolling_raw.at(1).add_point(ImVec2(t, draw_raw == 0.0 ? measurements.scrolling_raw.at(1).back().y : draw_raw));
                measurements.scrolling_ewa.at(1).add_point(ImVec2(t, draw_ewa));
                measurements.scrolling_mean.at(1).add_point(ImVec2(t, draw_average));
            }
            // ========================== SAMPLE =============================================
            f32 sample_raw = measurements.vsm_timings_raw.at(10);
            f32 sample_ewa = measurements.vsm_timings_ewa.at(10);
            f32 sample_average = measurements.vsm_timings_mean.at(10);
            if (gather_perm_measurements)
            {
                measurements.scrolling_raw.at(2).add_point(ImVec2(t, sample_raw == 0.0 ? measurements.scrolling_raw.at(2).back().y : sample_raw));
                measurements.scrolling_ewa.at(2).add_point(ImVec2(t, sample_ewa));
                measurements.scrolling_mean.at(2).add_point(ImVec2(t, sample_average));
            }
            // ================================================================================

            static i32 selected_item = 0;
            std::array<char const *, 3> items = {"exp. weight average", "rolling average", "raw values"};
            if (ImGui::BeginCombo("##combo", items.at(selected_item)))
            {
                for (int i = 0; i < 3; i++)
                {
                    bool is_selected = (selected_item == i);
                    if (ImGui::Selectable(items[i], is_selected))
                    {
                        selected_item = i;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            static bool calculate_percentiles = false;
            if (selected_item == 2)
            {
                ImGui::SameLine();
                ImGui::Checkbox("calculate percentiles", &calculate_percentiles);
            }
            else
            {
                calculate_percentiles = false;
            }

            for (i32 stat = 0; stat < 3; stat++)
            {
                if (selected_item == 0) { ImGui::TextUnformatted(fmt::format("{:<30} {:>10.2f} us", fmt::format("{} ewa: ", task_names.at(stat)).c_str(), measurements.scrolling_ewa.at(stat).back().y).c_str()); }
                else if (selected_item == 1) { ImGui::TextUnformatted(fmt::format("{:<30} {:>10.2f} us", fmt::format("{} average: ", task_names.at(stat)).c_str(), measurements.scrolling_mean.at(stat).back().y).c_str()); }
                else
                {
                    ImGui::TextUnformatted(fmt::format("{:<30} {:>10.2f} us", fmt::format("{} raw: ", task_names.at(stat)).c_str(), measurements.scrolling_raw.at(stat).back().y).c_str());
                    if (calculate_percentiles)
                    {
                        auto sorted_values = measurements.scrolling_raw.at(stat).data;
                        std::sort(sorted_values.begin(), sorted_values.end(),
                            [](auto const & a, auto const & b) -> bool
                            { return a.y < b.y; });
                        ImGui::TextUnformatted(fmt::format("{:<30} {:>10.2f} us", "\t 95th percentile: ", sorted_values.at(sorted_values.size() * 0.95f).y).c_str());
                        ImGui::TextUnformatted(fmt::format("{:<30} {:>10.2f} us", "\t 99th percentile: ", sorted_values.at(sorted_values.size() * 0.99f).y).c_str());
                    }
                }
            }

            static float history = 20.0f;
            if (ImPlot::BeginPlot("##Scrolling"))
            {
                decltype(measurements.scrolling_ewa) * measurements_scrolling_selected;
                if (selected_item == 0) { measurements_scrolling_selected = &measurements.scrolling_ewa; }
                else if (selected_item == 1) { measurements_scrolling_selected = &measurements.scrolling_mean; }
                else { measurements_scrolling_selected = &measurements.scrolling_raw; }
                ImPlot::SetupAxes("Timeline", "Execution time", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
                if (!show_entire_interval)
                {
                    ImPlot::SetupAxisLimits(ImAxis_X1, t - history, t, ImGuiCond_Always);
                }
                else
                {
                    ImPlot::SetupAxisLimits(ImAxis_X1, measurements_scrolling_selected->at(0).front().x, measurements_scrolling_selected->at(0).back().x, ImGuiCond_Always);
                }
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 5000);
                ImPlot::SetupAxisFormat(ImAxis_Y1, "%.0f us");
                ImPlot::SetupAxisFormat(ImAxis_X1, "%.0fs");
                for (i32 i = 0; i < 3; i++)
                {
                    ImPlot::PlotLine(
                        task_names.at(i),
                        &measurements_scrolling_selected->at(i).data[0].x,
                        &measurements_scrolling_selected->at(i).data[0].y,
                        measurements_scrolling_selected->at(i).data.size(),
                        0,
                        measurements_scrolling_selected->at(i).offset,
                        2 * sizeof(f32));
                }
                ImPlot::EndPlot();
            }
        }
        ImGui::End();
    }
    if (widget_scene_hierarchy)
    {
        ui_scenegraph(scene);
    }
    if (renderer_settings)
    {
        ui_renderer_settings(scene, render_context.render_data.settings);
    }
    if (widget_property_viewer)
    {
        property_viewer.render({
            .sky_settings = &render_context.render_data.sky_settings,
            .post_settings = &render_context.render_data.postprocess_settings,
        });
    }
    if (true)
    {
        ImGui::ShowDemoWindow();
    }
    if (shader_debug_menu)
    {
        if (ImGui::Begin("Shader Debug Menu", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            auto & cinematic_camera = app_state.cinematic_camera;
            ImGui::SeparatorText("Observer Camera");
            {
                IMGUI_UINT_CHECKBOX2("draw from observer (H)", render_context.render_data.settings.draw_from_observer);
                ImGui::Checkbox("control observer   (J)", &app_state.control_observer);
                app_state.reset_observer = app_state.reset_observer || (ImGui::Button("snap observer to main camera (K)"));
                std::array<char const * const, 3> modes = {
                    "redraw meshlets visible last frame",
                    "redraw meshlet post cull",
                    "redraw all drawn meshlets",
                };
                ImGui::Combo("observer draw pass mode", &render_context.render_data.settings.observer_show_pass, modes.data(), modes.size());
                auto const view_quat = glm::quat_cast(app_state.observer_camera_controller.make_camera_info(render_context.render_data.settings).view);
                ImGui::Text("%s", fmt::format("observer view quat {} {} {} {}", view_quat.w, view_quat.x, view_quat.y, view_quat.z).c_str());
            }
            ImGui::SeparatorText("Cinematic Camera");
            {
                ImGui::BeginDisabled(!app_state.use_preset_camera);
                ImGui::Checkbox("Override keyframe (I)", &cinematic_camera.override_keyframe);
                ImGui::EndDisabled();
                cinematic_camera.override_keyframe &= app_state.use_preset_camera;
                ImGui::BeginDisabled(!cinematic_camera.override_keyframe);
                i32 current_keyframe = cinematic_camera.current_keyframe_index;
                f32 keyframe_progress = cinematic_camera.current_keyframe_time / cinematic_camera.path_keyframes.at(current_keyframe).transition_time;
                ImGui::SliderInt("keyframe", &current_keyframe, 0, cinematic_camera.path_keyframes.size() - 1);
                ImGui::SliderFloat("keyframe progress", &keyframe_progress, 0.0f, 1.0f);
                if (cinematic_camera.override_keyframe) { cinematic_camera.set_keyframe(current_keyframe, keyframe_progress); }
                ImGui::EndDisabled();
                ImGui::Checkbox("use preset camera", &app_state.use_preset_camera);
                if (ImGui::Button("snap observer to cinematic"))
                {
                    app_state.observer_camera_controller.position = cinematic_camera.position;
                }
            }
            ImGui::SeparatorText("Debug Shader Interface");
            {
                ImGui::InputFloat("debug f32vec4 drag speed", &debug_f32vec4_drag_speed);
                ImGui::DragFloat4(
                    "debug f32vec4",
                    reinterpret_cast<f32 *>(&render_context.gpu_context->shader_debug_context.shader_debug_input.debug_fvec4), debug_f32vec4_drag_speed);
                ImGui::DragInt4(
                    "debug i32vec4",
                    reinterpret_cast<i32 *>(&render_context.gpu_context->shader_debug_context.shader_debug_input.debug_ivec4));
                ImGui::Text(
                    "out debug f32vec4: (%f,%f,%f,%f)",
                    render_context.gpu_context->shader_debug_context.shader_debug_output.debug_fvec4.x,
                    render_context.gpu_context->shader_debug_context.shader_debug_output.debug_fvec4.y,
                    render_context.gpu_context->shader_debug_context.shader_debug_output.debug_fvec4.z,
                    render_context.gpu_context->shader_debug_context.shader_debug_output.debug_fvec4.w);
                ImGui::Text(
                    "out debug i32vec4: (%i,%i,%i,%i)",
                    render_context.gpu_context->shader_debug_context.shader_debug_output.debug_ivec4.x,
                    render_context.gpu_context->shader_debug_context.shader_debug_output.debug_ivec4.y,
                    render_context.gpu_context->shader_debug_context.shader_debug_output.debug_ivec4.z,
                    render_context.gpu_context->shader_debug_context.shader_debug_output.debug_ivec4.w);
            }
            ImGui::SeparatorText("Debug Shader Lens");
            {
                ImGui::Text("Press ALT + LEFT_CLICK to set the detector to the cursor position");
                ImGui::Text("Press ALT + Keyboard arrow keys to move detector");
                ImGui::Checkbox("draw_magnified_area_rect", &render_context.gpu_context->shader_debug_context.draw_magnified_area_rect);
                ImGui::InputInt("detector window size", &render_context.gpu_context->shader_debug_context.detector_window_size, 2);
                ImGui::Text(
                    "detector texel position: (%i,%i)",
                    render_context.gpu_context->shader_debug_context.detector_window_position.x,
                    render_context.gpu_context->shader_debug_context.detector_window_position.y);
                ImGui::Text(
                    "detector center value: (%f,%f,%f,%f)",
                    render_context.gpu_context->shader_debug_context.shader_debug_output.texel_detector_center_value.x,
                    render_context.gpu_context->shader_debug_context.shader_debug_output.texel_detector_center_value.y,
                    render_context.gpu_context->shader_debug_context.shader_debug_output.texel_detector_center_value.z,
                    render_context.gpu_context->shader_debug_context.shader_debug_output.texel_detector_center_value.w);
                auto debug_lens_image_view_id = imgui_renderer.create_texture_id({
                    .image_view_id = render_context.gpu_context->shader_debug_context.debug_lens_image.default_view(),
                    .sampler_id = std::bit_cast<daxa::SamplerId>(render_context.render_data.samplers.nearest_clamp),
                });
                auto const width = ImGui::GetContentRegionMax().x;
                ImGui::Image(debug_lens_image_view_id, ImVec2(width, width));
            }
        }
        ImGui::End();
    }
    tg_resource_debug_ui(render_context);
}

void UIEngine::tg_resource_debug_ui(RenderContext & render_context)
{
    if (tg_debug_ui && ImGui::Begin("TG Debug Clones", nullptr, ImGuiWindowFlags_NoCollapse))
    {
        bool const clear_search = ImGui::Button("clear");
        if (clear_search)
            render_context.tg_debug.search_substr = {};
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200);
        ImGui::InputText("Search for Task", render_context.tg_debug.search_substr.data(), render_context.tg_debug.search_substr.size());
        for (auto & c : render_context.tg_debug.search_substr)
            c = std::tolower(c);

        bool const search_used = render_context.tg_debug.search_substr[0] != '\0';

        ImGui::BeginChild("Tasks");
        for (auto task : render_context.tg_debug.this_frame_task_attachments)
        {
            if (task.task_name.size() == 0 || task.task_name.c_str()[0] == 0)
                continue;

            if (search_used)
            {
                std::string compare_string = task.task_name;
                for (auto & c : compare_string)
                    c = std::tolower(c);
                if (!strstr(compare_string.c_str(), render_context.tg_debug.search_substr.data()))
                    continue;
            }

            if (ImGui::CollapsingHeader(task.task_name.c_str()))
            {
                for (auto attach : task.attachments)
                {
                    if (ImGui::Button(attach.name()))
                    {
                        std::string inspector_key = task.task_name + "::AT." + attach.name();
                        bool already_active = render_context.tg_debug.inspector_states[inspector_key].active;
                        if (already_active)
                        {
                            auto iter = render_context.tg_debug.active_inspectors.find(inspector_key);
                            if (iter != render_context.tg_debug.active_inspectors.end())
                            {
                                render_context.tg_debug.active_inspectors.erase(iter);
                            }
                            render_context.tg_debug.inspector_states[inspector_key].active = false;
                        }
                        else
                        {
                            render_context.tg_debug.active_inspectors.emplace(inspector_key);
                            render_context.tg_debug.inspector_states[inspector_key].active = true;
                        }
                    }
                    ImGui::SameLine();
                    switch (attach.type)
                    {
                        case daxa::TaskAttachmentType::IMAGE:
                            ImGui::Text("| view: %s", daxa::to_string(attach.value.image.view.slice).c_str());
                            ImGui::SameLine();
                            ImGui::Text((fmt::format("| task access: {}", daxa::to_string(attach.value.image.task_access))).c_str());
                            break;
                        default: break;
                    }
                }
            }
        }
        ImGui::EndChild();

        ImGui::End();
        for (auto active_inspector_key : render_context.tg_debug.active_inspectors)
        {
            tg_debug_image_inspector(render_context, active_inspector_key);
        }
    }
    if (render_context.tg_debug.request_mouse_picker_override)
    {
        render_context.tg_debug.override_mouse_picker = true;
    }
    else
    {
        render_context.tg_debug.override_mouse_picker = false;
    }
    render_context.tg_debug.request_mouse_picker_override = false;
    render_context.tg_debug.this_frame_task_attachments.clear();
    render_context.tg_debug.this_frame_duplicate_task_name_counter.clear();
}

auto format_vec4_rows_float(daxa_f32vec4 vec) -> std::string
{
    return fmt::format("R: {:10.7}\nG: {:10.7}\nB: {:10.7}\nA: {:10.7}",
        vec.x,
        vec.y,
        vec.z,
        vec.w);
}

auto format_vec4_rows(Vec4Union vec_union, tido::ScalarKind scalar_kind) -> std::string
{
    switch (scalar_kind)
    {
        case tido::ScalarKind::FLOAT:
            return format_vec4_rows_float(vec_union._float);
        case tido::ScalarKind::INT:
            return fmt::format("R: {:11}\nG: {:11}\nB: {:11}\nA: {:11}",
                vec_union._int.x,
                vec_union._int.y,
                vec_union._int.z,
                vec_union._int.w);
        case tido::ScalarKind::UINT:
            return fmt::format("R: {:11}\nG: {:11}\nB: {:11}\nA: {:11}",
                vec_union._uint.x,
                vec_union._uint.y,
                vec_union._uint.z,
                vec_union._uint.w);
    }
    return std::string();
}

void UIEngine::tg_debug_image_inspector(RenderContext & render_context, std::string active_inspector_key)
{
    render_context.tg_debug.readback_index = (render_context.tg_debug.readback_index + 1) % 3;
    auto & state = render_context.tg_debug.inspector_states[active_inspector_key];
    if (ImGui::Begin(fmt::format("Inspector for {}", active_inspector_key.c_str()).c_str(), nullptr, {}))
    {
        // The ui update is staggered a frame.
        // This is because the ui gets information from the task graph with a delay of one frame.
        // Because of this we first shedule a draw for the previous frames debug image canvas.
        ImTextureID tex_id = {};
        daxa::ImageInfo clone_image_info = {};
        daxa::ImageInfo const & image_info = state.runtime_image_info;
        if (!state.display_image.is_empty())
        {
            clone_image_info = render_context.gpu_context->device.image_info(state.display_image).value();

            daxa::SamplerId sampler = gpu_context->lin_clamp_sampler;
            if (state.nearest_filtering)
            {
                sampler = gpu_context->nearest_clamp_sampler;
            }
            tex_id = imgui_renderer.create_texture_id({
                .image_view_id = state.display_image.default_view(),
                .sampler_id = sampler,
            });
            u32 active_channels_of_format = tido::channel_count_of_format(image_info.format);

            // Now we actually process the ui.
            daxa::TaskImageAttachmentInfo const & attachment_info = state.attachment_info;
            auto slice = attachment_info.view.slice;

            if (ImGui::BeginTable("Some Inspected Image", 2, ImGuiTableFlags_NoHostExtendX | ImGuiTableFlags_SizingFixedFit))
            {
                ImGui::TableSetupColumn("Inspector settings", ImGuiTableFlags_NoHostExtendX | ImGuiTableFlags_SizingFixedFit);
                ImGui::TableSetupColumn("Image view", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableNextColumn();
                ImGui::SeparatorText("Inspector settings");
                ImGui::Checkbox("pre task", &state.pre_task);
                ImGui::SameLine();
                ImGui::Checkbox("freeze image", &state.freeze_image);
                ImGui::SetItemTooltip("make sure to NOT have freeze image set when switching this setting. The frozen image is either pre or post.");
                ImGui::PushItemWidth(80);
                i32 imip = state.mip;
                ImGui::InputInt("mip", &imip, 1);
                state.mip = imip;
                i32 ilayer = state.layer;
                ImGui::SameLine();
                ImGui::InputInt("layer", &ilayer, 1);
                state.layer = ilayer;
                ImGui::PopItemWidth();
                ImGui::PushItemWidth(180);
                ImGui::Text("selected mip size: (%i,%i,%i)", std::max(image_info.size.x >> state.mip, 1u), std::max(image_info.size.y >> state.mip, 1u), std::max(image_info.size.z >> state.mip, 1u));
                if (!state.slice_valid)
                    ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
                ImGui::Text(state.slice_valid ? "" : "SELECTED SLICE INVALID");
                if (!state.slice_valid)
                    ImGui::PopStyleColor();
                auto modes = std::array{
                    "Linear",
                    "Nearest",
                };
                ImGui::Combo("sampler", &state.nearest_filtering, modes.data(), modes.size());

                if (ImGui::BeginTable("Channels", 4, ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_SizingFixedFit))
                {
                    std::array channel_names = {"r", "g", "b", "a"};
                    std::array<bool, 4> channels = {};

                    u32 active_channel_count = 0;
                    i32 last_active_channel = -1;
                    for (u32 channel = 0; channel < 4; ++channel)
                    {
                        channels[channel] = std::bit_cast<std::array<i32, 4>>(state.enabled_channels)[channel];
                        active_channel_count += channels[channel] ? 1u : 0u;
                        last_active_channel = channels[channel] ? channel : channels[channel];
                    }

                    for (u32 channel = 0; channel < 4; ++channel)
                    {
                        auto const disabled = channel >= active_channels_of_format;
                        ImGui::BeginDisabled(disabled);
                        if (disabled)
                            channels[channel] = false;
                        ImGui::TableNextColumn();
                        bool const clicked = ImGui::Checkbox(channel_names[channel], channels.data() + channel);
                        ImGui::EndDisabled();
                        if (disabled)
                            ImGui::SetItemTooltip("image format does not have this channel");
                    }

                    state.enabled_channels.x = channels[0];
                    state.enabled_channels.y = channels[1];
                    state.enabled_channels.z = channels[2];
                    state.enabled_channels.w = channels[3];
                    ImGui::EndTable();
                }
                ImGui::PopItemWidth();
                ImGui::PushItemWidth(90);
                ImGui::InputDouble("min", &state.min_v);
                ImGui::SetItemTooltip("min value only effects rgb, not alpha");
                ImGui::SameLine();
                ImGui::InputDouble("max", &state.max_v);
                ImGui::SetItemTooltip("max value only effects rgb, not alpha");

                Vec4Union readback_raw = {};
                daxa_f32vec4 readback_color = {};
                daxa_f32vec4 readback_color_min = {};
                daxa_f32vec4 readback_color_max = {};

                tido::ScalarKind scalar_kind = tido::scalar_kind_of_format(image_info.format);
                if (!state.readback_buffer.is_empty())
                {
                    switch (scalar_kind)
                    {
                        case tido::ScalarKind::FLOAT: readback_raw._float = gpu_context->device.buffer_host_address_as<daxa_f32vec4>(state.readback_buffer).value()[render_context.tg_debug.readback_index * 2]; break;
                        case tido::ScalarKind::INT:   readback_raw._int = gpu_context->device.buffer_host_address_as<daxa_i32vec4>(state.readback_buffer).value()[render_context.tg_debug.readback_index * 2]; break;
                        case tido::ScalarKind::UINT:  readback_raw._uint = gpu_context->device.buffer_host_address_as<daxa_u32vec4>(state.readback_buffer).value()[render_context.tg_debug.readback_index * 2]; break;
                    }
                    auto floatvec_readback = gpu_context->device.buffer_host_address_as<daxa_f32vec4>(state.readback_buffer).value();
                    auto flt_min = std::numeric_limits<f32>::min();
                    auto flt_max = std::numeric_limits<f32>::max();
                    readback_color = floatvec_readback[render_context.tg_debug.readback_index * 2 + 1];
                }

                constexpr auto MOUSE_PICKER_FREEZE_COLOR = 0xFFBBFFFF;
                auto mouse_picker = [&](daxa_i32vec2 image_idx, bool frozen, Vec4Union readback_union)
                {
                    if (frozen)
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, MOUSE_PICKER_FREEZE_COLOR);
                    }
                    // ImGui::Dummy({0, 2});
                    constexpr auto MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH = 7;
                    constexpr auto MOUSE_PICKER_MAGNIFIER_DISPLAY_SIZE = ImVec2{70.0f, 70.0f};
                    daxa_i32vec2 image_idx_at_mip = {
                        image_idx.x >> state.mip,
                        image_idx.y >> state.mip,
                    };
                    ImVec2 magnify_start_uv = {
                        float(image_idx_at_mip.x - (MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH / 2)) * (1.0f / float(clone_image_info.size.x)),
                        float(image_idx_at_mip.y - (MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH / 2)) * (1.0f / float(clone_image_info.size.y)),
                    };
                    ImVec2 magnify_end_uv = {
                        float(image_idx_at_mip.x + MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH / 2 + 1) * (1.0f / float(clone_image_info.size.x)),
                        float(image_idx_at_mip.y + MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH / 2 + 1) * (1.0f / float(clone_image_info.size.y)),
                    };
                    if (tex_id && image_idx.x >= 0 && image_idx.y >= 0 && image_idx.x < image_info.size.x && image_idx.y < image_info.size.y)
                    {
                        ImGui::Image(tex_id, MOUSE_PICKER_MAGNIFIER_DISPLAY_SIZE, magnify_start_uv, magnify_end_uv);
                    }
                    if (frozen)
                    {
                        ImGui::PopStyleColor(1);
                    }
                    ImGui::SameLine();
                    daxa_i32vec2 index_at_mip = {
                        image_idx.x >> state.mip,
                        image_idx.y >> state.mip,
                    };
                    ImGui::Text("(%5d,%5d) %s\n%s",
                        index_at_mip.x,
                        index_at_mip.y,
                        frozen ? "FROZEN" : "      ",
                        format_vec4_rows(readback_union, scalar_kind).c_str());
                };

                ImGui::SeparatorText("Mouse Picker (?)\n");
                ImGui::SetItemTooltip(
                    "Usage:\n"
                    "  * left click on image to freeze selection, left click again to unfreeze\n"
                    "  * hold shift to replicate the selection on all other open inspector mouse pickers (also replicates freezes)\n"
                    "  * use middle mouse button to grab and move zoomed in image");
                if (state.display_image_hovered || state.freeze_image_hover_index || render_context.tg_debug.override_mouse_picker)
                {
                    mouse_picker(state.frozen_mouse_pos_relative_to_image_mip0, state.freeze_image_hover_index, state.frozen_readback_raw);
                }
                else
                {
                    mouse_picker(daxa_i32vec2{0, 0}, false, {});
                }

                ImGui::PopItemWidth();

                ImGui::TableNextColumn();
                ImGui::Text("slice used in task: %s", daxa::to_string(slice).c_str());
                ImGui::Text("size: (%i,%i,%i), mips: %i, layers: %i, format: %s",
                    image_info.size.x,
                    image_info.size.y,
                    image_info.size.z,
                    image_info.mip_level_count,
                    image_info.array_layer_count,
                    daxa::to_string(image_info.format).data());

                auto resolution_draw_modes = std::array{
                    "auto size",
                    "1x",
                    "1/2x",
                    "1/4x",
                    "1/8x",
                    "1/16x",
                    "2x",
                    "4x",
                    "8x",
                    "16x",
                };
                auto resolution_draw_mode_factors = std::array{
                    -1.0f,
                    1.0f,
                    1.0f / 2.0f,
                    1.0f / 4.0f,
                    1.0f / 8.0f,
                    1.0f / 16.0f,
                    2.0f,
                    4.0f,
                    8.0f,
                    16.0f,
                };
                ImGui::SetNextItemWidth(100.0f);
                ImGui::Combo("draw resolution mode", &state.resolution_draw_mode, resolution_draw_modes.data(), resolution_draw_modes.size());
                ImGui::SameLine();
                ImGui::Checkbox("fix mip sizes", &state.fixed_display_mip_sizes);
                ImGui::SetItemTooltip("fixes all displayed mip sizes to be the scaled size of mip 0");
                if (tex_id)
                {
                    float const aspect = static_cast<float>(clone_image_info.size.y) / static_cast<float>(clone_image_info.size.x);
                    ImVec2 const auto_sized_draw_size_x_based = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x * aspect);
                    ImVec2 const auto_sized_draw_size_y_based = ImVec2(ImGui::GetContentRegionAvail().y / aspect, ImGui::GetContentRegionAvail().y);
                    ImVec2 const auto_sized_draw_size = auto_sized_draw_size_x_based.x < auto_sized_draw_size_y_based.x ? auto_sized_draw_size_x_based : auto_sized_draw_size_y_based;
                    ImVec2 image_display_size = auto_sized_draw_size;
                    if (state.resolution_draw_mode != 0)
                    {
                        ImVec2 fixed_size_draw_size = {};
                        if (state.fixed_display_mip_sizes)
                        {
                            fixed_size_draw_size.x = static_cast<float>(image_info.size.x) * resolution_draw_mode_factors[state.resolution_draw_mode];
                            fixed_size_draw_size.y = static_cast<float>(image_info.size.y) * resolution_draw_mode_factors[state.resolution_draw_mode];
                        }
                        else
                        {
                            fixed_size_draw_size.x = static_cast<float>(clone_image_info.size.x) * resolution_draw_mode_factors[state.resolution_draw_mode];
                            fixed_size_draw_size.y = static_cast<float>(clone_image_info.size.y) * resolution_draw_mode_factors[state.resolution_draw_mode];
                        };

                        image_display_size = fixed_size_draw_size;
                    }

                    ImVec2 start_pos = ImGui::GetCursorScreenPos();
                    state.display_image_size = daxa_i32vec2(image_display_size.x, image_display_size.y);
                    ImGui::BeginChild("scrollable image", ImVec2(0, 0), {}, ImGuiWindowFlags_HorizontalScrollbar);
                    ImVec2 scroll_offset = ImVec2{ImGui::GetScrollX(), ImGui::GetScrollY()};
                    if (state.display_image_hovered && ImGui::IsKeyDown(ImGuiKey_MouseMiddle))
                    {
                        ImGui::SetScrollX(ImGui::GetScrollX() - ImGui::GetIO().MouseDelta.x);
                        ImGui::SetScrollY(ImGui::GetScrollY() - ImGui::GetIO().MouseDelta.y);
                    }
                    ImGui::Image(tex_id, image_display_size);
                    ImVec2 const mouse_pos = ImGui::GetMousePos();
                    ImVec2 const end_pos = ImVec2{start_pos.x + image_display_size.x, start_pos.y + image_display_size.y};

                    ImVec2 const clipped_display_image_size = {
                        end_pos.x - start_pos.x,
                        end_pos.y - start_pos.y,
                    };
                    state.display_image_hovered = ImGui::IsMouseHoveringRect(start_pos, end_pos) && (ImGui::IsItemHovered() || ImGui::IsItemClicked());
                    state.freeze_image_hover_index = state.freeze_image_hover_index ^ (state.display_image_hovered && ImGui::IsItemClicked());
                    state.mouse_pos_relative_to_display_image = daxa_i32vec2(mouse_pos.x - start_pos.x, mouse_pos.y - start_pos.y);
                    render_context.tg_debug.request_mouse_picker_override |= state.display_image_hovered && ImGui::IsKeyDown(ImGuiKey_LeftShift);

                    bool const override_other_inspectors = render_context.tg_debug.override_mouse_picker && state.display_image_hovered;
                    bool const get_overriden = render_context.tg_debug.override_mouse_picker && !state.display_image_hovered;
                    if (override_other_inspectors)
                    {
                        render_context.tg_debug.override_frozen_state = state.freeze_image_hover_index;
                        render_context.tg_debug.override_mouse_picker_uv = {
                            float(state.mouse_pos_relative_to_display_image.x) / clipped_display_image_size.x,
                            float(state.mouse_pos_relative_to_display_image.y) / clipped_display_image_size.y,
                        };
                    }
                    if (get_overriden)
                    {
                        state.freeze_image_hover_index = render_context.tg_debug.override_frozen_state;
                        state.mouse_pos_relative_to_display_image = {
                            i32(render_context.tg_debug.override_mouse_picker_uv.x * clipped_display_image_size.x),
                            i32(render_context.tg_debug.override_mouse_picker_uv.y * clipped_display_image_size.y),
                        };
                    }

                    state.mouse_pos_relative_to_image_mip0 = daxa_i32vec2(
                        ((state.mouse_pos_relative_to_display_image.x + scroll_offset.x) / static_cast<float>(state.display_image_size.x)) * static_cast<float>(image_info.size.x),
                        ((state.mouse_pos_relative_to_display_image.y + scroll_offset.y) / static_cast<float>(state.display_image_size.y)) * static_cast<float>(image_info.size.y));

                    float x = ImGui::GetScrollMaxX();
                    float y = ImGui::GetScrollMaxY();

                    if (!state.freeze_image_hover_index)
                    {
                        state.frozen_mouse_pos_relative_to_image_mip0 = state.mouse_pos_relative_to_image_mip0;
                        state.frozen_readback_raw = readback_raw;
                        state.frozen_readback_color = readback_color;
                    }
                    if (state.display_image_hovered)
                    {
                        ImGui::BeginTooltip();
                        mouse_picker(state.mouse_pos_relative_to_image_mip0, false, readback_raw);
                        ImGui::EndTooltip();
                    }
                    if (state.display_image_hovered || render_context.tg_debug.override_mouse_picker)
                    {
                        ImVec2 const frozen_mouse_pos_relative_to_display_image = {
                            float(state.mouse_pos_relative_to_image_mip0.x) / float(image_info.size.x) * state.display_image_size.x - scroll_offset.x,
                            float(state.mouse_pos_relative_to_image_mip0.y) / float(image_info.size.y) * state.display_image_size.y - scroll_offset.y,
                        };
                        ImVec2 const window_marker_pos = {
                            start_pos.x + frozen_mouse_pos_relative_to_display_image.x,
                            start_pos.y + frozen_mouse_pos_relative_to_display_image.y,
                        };
                        ImGui::GetWindowDrawList()->AddCircle(window_marker_pos, 5.0f, ImGui::GetColorU32(ImVec4{
                                                                                           readback_color.x > 0.5f ? 0.0f : 1.0f,
                                                                                           readback_color.y > 0.5f ? 0.0f : 1.0f,
                                                                                           readback_color.z > 0.5f ? 0.0f : 1.0f,
                                                                                           1.0f,
                                                                                       }));
                    }
                    if (state.freeze_image_hover_index)
                    {
                        ImVec2 const frozen_mouse_pos_relative_to_display_image = {
                            float(state.frozen_mouse_pos_relative_to_image_mip0.x) / float(image_info.size.x) * state.display_image_size.x - scroll_offset.x,
                            float(state.frozen_mouse_pos_relative_to_image_mip0.y) / float(image_info.size.y) * state.display_image_size.y - scroll_offset.y,
                        };
                        ImVec2 const window_marker_pos = {
                            start_pos.x + frozen_mouse_pos_relative_to_display_image.x,
                            start_pos.y + frozen_mouse_pos_relative_to_display_image.y,
                        };
                        auto inv_color = ImVec4{
                            state.frozen_readback_color.x > 0.5f ? 0.0f : 1.0f,
                            state.frozen_readback_color.y > 0.5f ? 0.0f : 1.0f,
                            state.frozen_readback_color.z > 0.5f ? 0.0f : 1.0f,
                            1.0f,
                        };
                        ImGui::GetWindowDrawList()->AddCircle(window_marker_pos, 5.0f, ImGui::GetColorU32(inv_color));
                    }
                    ImGui::EndChild();
                }
                ImGui::EndTable();
            }
        }
        ImGui::End();
    }
}

void UIEngine::ui_scenegraph(Scene const & scene)
{
    if (scene._gltf_asset_manifest.empty())
    {
        return;
    }
    bool began = false;
    if (scene_graph.begin())
    {
        began = true;
        struct StackEntry
        {
            RenderEntityId id;
            bool is_first_child;
            bool is_last_child;
            i32 accumulated_indent;
        };
        // ROOT - first/last - indent and accumulate
        //  INNER - first - indent
        //    LEAF - first - indent and unindent
        //    INNER - last - indent and accumulate
        //      LEAF - first/last - indent once and unindent twice (one accumulated from parent INNER )
        //  INNER - middle - indent and unindent
        //  INNER - last - indent and unindent twice (one accumulated from ROOT)
        std::vector<StackEntry> entity_graph_stack;
        entity_graph_stack.push_back({scene._gltf_asset_manifest.at(0).root_render_entity, true, true, -1});

        while (!entity_graph_stack.empty())
        {
            auto const top_stack_entry = entity_graph_stack.back();
            entity_graph_stack.pop_back();

            auto const & entity = *scene._render_entities.slot(top_stack_entry.id);
            // Indent if this is a first child and is not root - we don't want to indent root
            if (top_stack_entry.is_first_child && entity.type != EntityType::ROOT)
            {
                scene_graph.add_level();
            }
            RetNodeState const result = scene_graph.add_node(entity, scene);

            // Check if we are both the last child and we will add no more of our own children to the stack
            bool const should_remove_indentation = top_stack_entry.is_last_child && result == RetNodeState::CLOSED;
            // If that is the case remove our own indentation + the accumulated indentation from our parents
            if (should_remove_indentation)
            {
                // Remove all the previous indentations that were accumulated as well as our own
                for (u32 pop_count = 0; pop_count < top_stack_entry.accumulated_indent + 1; pop_count++)
                {
                    scene_graph.remove_level();
                }
            }
            // If this node was not opened on previous frame or it is a leaf (aka has no more children)
            // it is CLOSED (maybe COLLAPSED is a better name) and we don't want to continue to the next entry on stack
            if (result != RetNodeState::OPEN) { continue; }

            // Otherwise we find collect all of our children into a vector
            /// NOTE: We collect into a vector because we want our children to be processed in the order
            //        of first -> last, but if we just pushed them directly to the stack we would have the inverse order
            //        as the first child will be pushed first and latter children would be stacked on top of it
            RenderEntityId curr_child_index = entity.first_child.value();
            std::vector<StackEntry> child_entries = {};
            while (true)
            {
                RenderEntity const * curr_child = scene._render_entities.slot(curr_child_index);
                bool const is_first_child = child_entries.empty();
                bool const is_last_child = !curr_child->next_sibling.has_value();
                // If the curenntly processed (pop from stack) node was a last child and we are adding
                // it's own last child we should accumulate the indentation so that the leaf then knows
                // how many levels of indentation it should remove
                bool const should_accumulate_remove_indent = top_stack_entry.is_last_child && is_last_child;
                i32 const accumulated_indent = should_accumulate_remove_indent ? top_stack_entry.accumulated_indent + 1 : 0;
                child_entries.push_back({.id = curr_child_index,
                    .is_first_child = is_first_child,
                    .is_last_child = is_last_child,
                    .accumulated_indent = accumulated_indent});
                if (is_last_child) { break; }
                // Move on to the next child
                curr_child_index = curr_child->next_sibling.value();
            }
            entity_graph_stack.reserve(entity_graph_stack.size() + child_entries.size());
            entity_graph_stack.insert(entity_graph_stack.end(), child_entries.rbegin(), child_entries.rend());
        }
    }
    scene_graph.end(began);
}

void UIEngine::ui_renderer_settings(Scene const & scene, Settings & settings)
{
    if (ImGui::Begin("Renderer Settings", nullptr, ImGuiWindowFlags_NoCollapse))
    {
        ImGui::SeparatorText("General settings");
        std::array<char const * const, 3> aa_modes = {
            "AA_MODE_NONE",
            "AA_MODE_SUPER_SAMPLE",
            "AA_MODE_DVM",
        };
        ImGui::Combo("anti_aliasing_mode", &settings.anti_aliasing_mode, aa_modes.data(), aa_modes.size());
        auto ao_modes = std::array{
            "None",
            "RT"};
        ImGui::Combo("ao mode", &settings.ao_mode, ao_modes.data(), ao_modes.size());
        ImGui::InputInt("ao samples", &settings.ao_samples);
        ImGui::SeparatorText("Debug Visualizations");
        {
            auto modes = std::array{
                "None",
                "Overdraw",
                "Triangle Id",
                "Meshlet Id",
                "Entity Id",
                "VSM Overdraw",
                "VSM Clip Level",
                "DEPTH",
                "ALBEDO",
                "NORMAL",
                "LIGHT",
                "AO",
            };
            ImGui::Combo("debug visualization", &settings.debug_draw_mode, modes.data(), modes.size());
            ImGui::InputFloat("debug visualization overdraw scale", &settings.debug_overdraw_scale);
            ImGui::Checkbox("enable_mesh_cull", reinterpret_cast<bool *>(&settings.enable_mesh_cull));
            ImGui::Checkbox("enable_meshlet_cull", reinterpret_cast<bool *>(&settings.enable_meshlet_cull));
            ImGui::Checkbox("enable_triangle_cull", reinterpret_cast<bool *>(&settings.enable_triangle_cull));
            ImGui::Checkbox("enable_atomic_visbuffer", reinterpret_cast<bool *>(&settings.enable_atomic_visbuffer));
            ImGui::Checkbox("enable_merged_scene_blas", reinterpret_cast<bool *>(&settings.enable_merged_scene_blas));
            ImGui::Checkbox("use_rt_pipeline_for_ao", reinterpret_cast<bool *>(&settings.use_rt_pipeline_for_ao));
        }
    }
    ImGui::End();
}

UIEngine::~UIEngine()
{
    for (u32 icon_idx = 0; icon_idx < s_cast<u32>(ICONS::SIZE); icon_idx++)
    {
        if (!icons.at(icon_idx).is_empty())
        {
            gpu_context->device.destroy_image(icons.at(icon_idx));
        }
    }
    ImPlot::DestroyContext();
}