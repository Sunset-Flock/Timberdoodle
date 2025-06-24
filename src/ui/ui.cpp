#include "ui.hpp"
#include <filesystem>
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include "widgets/helpers.hpp"
#include "../daxa_helper.hpp"
#include "../shader_shared/gpu_work_expansion.inl"

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
      gpu_context{gpu_context},
      window{&window}
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

void UIEngine::main_update(GPUContext const & gpu_context, RenderContext & render_context, Scene & scene, ApplicationState & app_state, Window & window)
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Clear Select Render Data
    render_context.render_data.cursor_uv = {
        std::clamp(s_cast<f32>(window.get_cursor_x()) / s_cast<f32>(window.get_width()), 0.0f, 1.0f),
        std::clamp(s_cast<f32>(window.get_cursor_y()) / s_cast<f32>(window.get_height()), 0.0f, 1.0f),
    };
    if (window.is_cursor_captured())
    {
        render_context.render_data.cursor_uv = { 0.5f, 0.5f };
    }
    render_context.render_data.hovered_entity_index = ~0u;
    render_context.render_data.selected_entity_index = ~0u;

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
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open"))
            {
                app_state.desired_scene_path = open_file_dialog();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Widgets"))
        {
            ImGui::MenuItem("Settings", NULL, &renderer_settings);
            ImGui::MenuItem("Widget Settings", NULL, &widget_settings);
            ImGui::MenuItem("Renderer Statistics", NULL, &widget_renderer_statistics);
            ImGui::MenuItem("Scene Hierarchy", NULL, &widget_scene_interface);
            ImGui::MenuItem("Shader Debug Menu", NULL, &shader_debug_menu);
            ImGui::MenuItem("Widget Property Viewer", NULL, &widget_property_viewer);
            ImGui::MenuItem("TaskGraphDebugUi", NULL, &tg_debug_ui);
            ImGui::MenuItem("Imgui Demo", NULL, &demo_window);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
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
        ui_render_statistics(scene, render_context, app_state);
    }
    if (widget_scene_interface)
    {
        ui_scene_graph(scene);
    }
    if (renderer_settings)
    {
        ui_renderer_settings(scene, render_context, app_state);
    }
    if (widget_property_viewer)
    {
        property_viewer.render(scene_interface, scene, render_context);
    }
    if (demo_window)
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
                ImGui::Checkbox("observer draw first pass", reinterpret_cast<bool *>(&render_context.render_data.settings.observer_draw_first_pass));
                ImGui::Checkbox("observer draw second pass", reinterpret_cast<bool *>(&render_context.render_data.settings.observer_draw_second_pass));
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
    render_context.tg_debug.ui_open = tg_debug_ui;
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
                    std::string inspector_key = task.task_name + "::AT." + attach.name();
                    ImGui::PushID(inspector_key.c_str());
                    if (ImGui::Button(attach.name()))
                    {
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
                    ImGui::PopID();
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

void UIEngine::ui_scene_graph(Scene const & scene)
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

void UIEngine::ui_renderer_settings(Scene const & scene, RenderContext & render_context, ApplicationState & app_state)
{
    RenderGlobalData & render_data = render_context.render_data;
    debug_visualization_index_override = 0;
    if (ImGui::Begin("Renderer Settings", nullptr, ImGuiWindowFlags_NoCollapse))
    {
        ImGui::SeparatorText("General settings");
        {
            ImGui::Checkbox("enable reference path trace", reinterpret_cast<bool *>(&render_data.settings.enable_reference_path_trace));
            ImGui::Checkbox("decompose scene", r_cast<bool *>(&app_state.decompose_bistro));
            ImGui::Checkbox("enable async compute", r_cast<bool *>(&render_data.settings.enable_async_compute));
            ImGui::Checkbox("enable vsync", r_cast<bool *>(&render_data.settings.enable_vsync));
            std::array<char const * const, 2> aa_modes = {
                "NONE",
                "SUPER_SAMPLE",
            };
            ImGui::Combo("anti_aliasing_mode", &render_data.settings.anti_aliasing_mode, aa_modes.data(), aa_modes.size());
            if (ImGui::CollapsingHeader("Lights Settings"))
            {
                {
                    auto modes = std::array{
                        "NONE",                 // DEBUG_DRAW_MODE_NONE
                        "ALBEDO",               // DEBUG_DRAW_MODE_ALBEDO
                        "SMOOTH_NORMAL",        // DEBUG_DRAW_MODE_SMOOTH_NORMAL
                        "DIRECT_DIFFUSE",       // DEBUG_DRAW_MODE_DIRECT_DIFFUSE
                        "INDIRECT_DIFFUSE",     // DEBUG_DRAW_MODE_INDIRECT_DIFFUSE
                        "ALL_DIFFUSE",          // DEBUG_DRAW_MODE_ALL_DIFFUSE
                        "SHADE_OPAQUE_CLOCKS",  // DEBUG_DRAW_MODE_SHADE_OPAQUE_CLOCKS
                        "LIGHT_MASK_VOLUME",    // DEBUG_DRAW_MODE_LIGHT_MASK_VOLUME
                    };
                    auto mode_mappings = std::array{
                        DEBUG_DRAW_MODE_NONE,
                        DEBUG_DRAW_MODE_ALBEDO,
                        DEBUG_DRAW_MODE_SMOOTH_NORMAL,
                        DEBUG_DRAW_MODE_DIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_ALL_DIFFUSE,
                        DEBUG_DRAW_MODE_SHADE_OPAQUE_CLOCKS,
                        DEBUG_DRAW_MODE_LIGHT_MASK_VOLUME,
                    };
                    ImGui::Combo("lights debug visualization", &lights_debug_visualization, modes.data(), modes.size());
                    if (lights_debug_visualization != 0)
                    {
                        debug_visualization_index_override = mode_mappings[lights_debug_visualization];
                    }
                }
                ImGui::InputFloat3("Mask Volume Size", &render_data.light_settings.mask_volume_size.x);
                ImGui::InputInt3("Mask Volume Cell Count", &render_data.light_settings.mask_volume_cell_count.x);
                ImGui::Checkbox("Cull All Point Lights", r_cast<bool*>(&render_data.light_settings.cull_all_point_lights));
                ImGui::Checkbox("Cull All Spot Lights", r_cast<bool*>(&render_data.light_settings.cull_all_spot_lights));
                ImGui::Checkbox("Draw Point Light Influence", r_cast<bool*>(&render_data.light_settings.debug_draw_point_influence));
                ImGui::Checkbox("Draw Spot Light Influence", r_cast<bool*>(&render_data.light_settings.debug_draw_spot_influence));
                ImGui::Checkbox("Mark Influence", r_cast<bool*>(&render_data.light_settings.debug_mark_influence));
                ImGui::Checkbox("Mark Influence Shadowed", r_cast<bool*>(&render_data.light_settings.debug_mark_influence_shadowed));
                ImGui::SliderInt("Debug Point Light Idx", &render_data.light_settings.selected_debug_point_light, -1, render_data.light_settings.point_light_count-1);
                ImGui::SliderInt("Debug Spot Light Idx", &render_data.light_settings.selected_debug_spot_light, -1, render_data.light_settings.spot_light_count-1);
            }
            
            if (ImGui::CollapsingHeader("Debug Visualizations"))
            {
                auto modes = std::array{
                    "NONE",                        // DEBUG_DRAW_MODE_NONE
                    "OVERDRAW",                    // DEBUG_DRAW_MODE_OVERDRAW
                    "TRIANGLE_CONNECTIVITY",       // DEBUG_DRAW_MODE_TRIANGLE_CONNECTIVITY
                    "TRIANGLE_ID",                 // DEBUG_DRAW_MODE_TRIANGLE_ID
                    "MESHLET_ID",                  // DEBUG_DRAW_MODE_MESHLET_ID
                    "MESH_ID",                     // DEBUG_DRAW_MODE_MESH_ID
                    "MESH_GROUP_ID",               // DEBUG_DRAW_MODE_MESH_GROUP_ID
                    "ENTITY_ID",                   // DEBUG_DRAW_MODE_ENTITY_ID
                    "MESH_LOD",                    // DEBUG_DRAW_MODE_MESH_LOD
                    "VSM_OVERDRAW",                // DEBUG_DRAW_MODE_VSM_OVERDRAW
                    "VSM_CLIP_LEVEL",              // DEBUG_DRAW_MODE_VSM_CLIP_LEVEL
                    "VSM_SPOT_LEVEL",              // DEBUG_DRAW_MODE_VSM_SPOT_LEVEL
                    "VSM_POINT_LEVEL",             // DEBUG_DRAW_MODE_VSM_POINT_LEVEL
                    "DEPTH",                       // DEBUG_DRAW_MODE_DEPTH
                    "ALBEDO",                      // DEBUG_DRAW_MODE_ALBEDO
                    "FACE_NORMAL",                 // DEBUG_DRAW_MODE_FACE_NORMAL
                    "SMOOTH_NORMAL",               // DEBUG_DRAW_MODE_SMOOTH_NORMAL
                    "MAPPED_NORMAL",               // DEBUG_DRAW_MODE_MAPPED_NORMAL
                    "FACE_TANGENT",                // DEBUG_DRAW_MODE_FACE_TANGENT
                    "SMOOTH_TANGENT",              // DEBUG_DRAW_MODE_SMOOTH_TANGENT
                    "DIRECT_DIFFUSE",              // DEBUG_DRAW_MODE_DIRECT_DIFFUSE
                    "INDIRECT_DIFFUSE",            // DEBUG_DRAW_MODE_INDIRECT_DIFFUSE
                    "PER_PIXEL_DIFFUSE",           // DEBUG_DRAW_MODE_PER_PIXEL_DIFFUSE
                    "INDIRECT_DIFFUSE_AO",         // DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO
                    "ALL_DIFFUSE",                 // DEBUG_DRAW_MODE_ALL_DIFFUSE
                    "SHADE_OPAQUE_CLOCKS",         // DEBUG_DRAW_MODE_SHADE_OPAQUE_CLOCKS
                    "PGI_EVAL_CLOCKS",             // DEBUG_DRAW_MODE_PGI_EVAL_CLOCKS
                    "RTAO_TRACE_CLOCKS",           // DEBUG_DRAW_MODE_RTAO_TRACE_CLOCKS
                    "PGI_CASCADE_SMOOTH",          // DEBUG_DRAW_MODE_PGI_CASCADE_SMOOTH
                    "PGI_CASCADE_ABSOLUTE",        // DEBUG_DRAW_MODE_PGI_CASCADE_ABSOLUTE
                    "PGI_CASCADE_SMOOTH_ABS_DIFF", // DEBUG_DRAW_MODE_PGI_CASCADE_SMOOTH_ABS_DIFF
                    "UV",                          // DEBUG_DRAW_MODE_UV
                    "LIGHT_MASK_VOLUME",           // DEBUG_DRAW_MODE_LIGHT_MASK_VOLUME
                };
                ImGui::Combo("debug visualization", &debug_visualization_index, modes.data(), modes.size());
                ImGui::InputFloat("debug visualization scale", &render_data.settings.debug_visualization_scale);
                ImGui::InputInt("override_lod", &render_data.settings.lod_override);
                ImGui::InputFloat("lod_acceptable_pixel_error", &render_data.settings.lod_acceptable_pixel_error);
                ImGui::SetItemTooltip("Pixel errors below one are necessary to avoid shading issues as normals are more sensitive to lodding then positions");
            }
        }
        ImGui::SeparatorText("Features");
        {
            if (ImGui::CollapsingHeader("Visbuffer Pipeline Settings"))
            {
                {
                    auto modes = std::array{
                        "NONE", // DEBUG_DRAW_MODE_NONE
                        "OVERDRAW", // DEBUG_DRAW_MODE_OVERDRAW
                        "TRIANGLE_CONNECTIVITY", // DEBUG_DRAW_MODE_TRIANGLE_CONNECTIVITY
                        "TRIANGLE_ID", // DEBUG_DRAW_MODE_TRIANGLE_ID
                        "MESHLET_ID", // DEBUG_DRAW_MODE_MESHLET_ID
                        "MESH_ID", // DEBUG_DRAW_MODE_MESH_ID
                        "MESH_GROUP_ID", // DEBUG_DRAW_MODE_MESH_GROUP_ID
                        "ENTITY_ID", // DEBUG_DRAW_MODE_ENTITY_ID
                        "MESH_LOD", // DEBUG_DRAW_MODE_MESH_LOD
                        "DEPTH", // DEBUG_DRAW_MODE_DEPTH
                        "ALBEDO", // DEBUG_DRAW_MODE_ALBEDO
                        "FACE_NORMAL", // DEBUG_DRAW_MODE_FACE_NORMAL
                        "SMOOTH_NORMAL", // DEBUG_DRAW_MODE_SMOOTH_NORMAL
                        "MAPPED_NORMAL", // DEBUG_DRAW_MODE_MAPPED_NORMAL
                        "FACE_TANGENT", // DEBUG_DRAW_MODE_FACE_TANGENT
                        "SMOOTH_TANGENT", // DEBUG_DRAW_MODE_SMOOTH_TANGENT
                        "UV", // DEBUG_DRAW_MODE_UV
                    };
                    auto mode_mappings = std::array{
                        DEBUG_DRAW_MODE_NONE,
                        DEBUG_DRAW_MODE_OVERDRAW,
                        DEBUG_DRAW_MODE_TRIANGLE_CONNECTIVITY,
                        DEBUG_DRAW_MODE_TRIANGLE_ID,
                        DEBUG_DRAW_MODE_MESHLET_ID,
                        DEBUG_DRAW_MODE_MESH_ID,
                        DEBUG_DRAW_MODE_MESH_GROUP_ID,
                        DEBUG_DRAW_MODE_ENTITY_ID,
                        DEBUG_DRAW_MODE_MESH_LOD,
                        DEBUG_DRAW_MODE_DEPTH,
                        DEBUG_DRAW_MODE_ALBEDO,
                        DEBUG_DRAW_MODE_FACE_NORMAL,
                        DEBUG_DRAW_MODE_SMOOTH_NORMAL,
                        DEBUG_DRAW_MODE_MAPPED_NORMAL,
                        DEBUG_DRAW_MODE_FACE_TANGENT,
                        DEBUG_DRAW_MODE_SMOOTH_TANGENT,
                        DEBUG_DRAW_MODE_UV,
                    };
                    ImGui::Combo("visbuffer debug visualization", &visbuffer_debug_visualization, modes.data(), modes.size());
                    if (visbuffer_debug_visualization != 0)
                    {
                        debug_visualization_index_override = mode_mappings[visbuffer_debug_visualization];
                    }
                }
                ImGui::Checkbox("enable_mesh_cull", reinterpret_cast<bool *>(&render_data.settings.enable_mesh_cull));
                ImGui::Checkbox("enable_meshlet_cull", reinterpret_cast<bool *>(&render_data.settings.enable_meshlet_cull));
                ImGui::Checkbox("enable_triangle_cull", reinterpret_cast<bool *>(&render_data.settings.enable_triangle_cull));
                ImGui::Checkbox("enable_separate_compute_meshlet_culling", reinterpret_cast<bool *>(&render_data.settings.enable_separate_compute_meshlet_culling));
                ImGui::Checkbox("enable_prefix_sum_work_expansion", reinterpret_cast<bool *>(&render_data.settings.enable_prefix_sum_work_expansion));
            }
            if (ImGui::CollapsingHeader("Per Pixel Diffuse (SSAO/RTAO/RTGI)"))
            {
                {
                    auto modes = std::array{
                        "NONE",                        // DEBUG_DRAW_MODE_NONE
                        "PER_PIXEL_DIFFUSE",           // DEBUG_DRAW_MODE_PER_PIXEL_DIFFUSE
                        "INDIRECT_DIFFUSE",            // DEBUG_DRAW_MODE_INDIRECT_DIFFUSE
                        "INDIRECT_DIFFUSE_AO",         // DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO
                        "ALL_DIFFUSE",                 // DEBUG_DRAW_MODE_ALL_DIFFUSE
                        "TRACE_CLOCKS",                // DEBUG_DRAW_MODE_RTAO_TRACE_CLOCKS
                    };
                    auto mode_mappings = std::array{
                        DEBUG_DRAW_MODE_NONE,
                        DEBUG_DRAW_MODE_PER_PIXEL_DIFFUSE,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO,
                        DEBUG_DRAW_MODE_ALL_DIFFUSE,
                        DEBUG_DRAW_MODE_RTAO_TRACE_CLOCKS,
                    };
                    ImGui::Combo("ppd debug visualization", &ppd_debug_visualization, modes.data(), modes.size());
                    if (ppd_debug_visualization != 0)
                    {
                        debug_visualization_index_override = mode_mappings[ppd_debug_visualization];
                    }
                }
                auto const modes = std::array{
                    "NONE",                                         // PER_PIXEL_DIFFUSE_MODE_NONE
                    "RAY_TRACED_AMBIENT_OCCLUSION",                 // PER_PIXEL_DIFFUSE_MODE_RTAO
                    "SHORT_RANGE_RAY_TRACED_GLOBAL_ILLUMINATION",   // PER_PIXEL_DIFFUSE_MODE_SHORT_RANGE_RTGI
                    "FULL_RAY_TRACED_GLOBAL_ILLUMINATION",          // PER_PIXEL_DIFFUSE_MODE_FULL_RTGI
                };
                ImGui::Combo("Mode", &render_context.render_data.ppd_settings.mode, modes.data(), modes.size());
                ImGui::InputInt("Sample count", &render_context.render_data.ppd_settings.sample_count);
                ImGui::SliderFloat("RTAO Range            ", &render_context.render_data.ppd_settings.ao_range, 0.01f, 10.0f);
                ImGui::SliderFloat("Short Range RTGI Range", &render_context.render_data.ppd_settings.short_range_rtgi_range, 0.01f, 10.0f);
                ImGui::SliderFloat("Denoiser Epsilon      ", &render_context.render_data.ppd_settings.denoiser_accumulation_max_epsi, 0.75f, 0.999f);
                ImGui::Checkbox("Debug Primary Trace", reinterpret_cast<bool *>(&render_context.render_data.ppd_settings.debug_primary_trace));
            }
            if (ImGui::CollapsingHeader("PGI Settings"))
            {
                {
                    auto modes = std::array{
                        "NONE",                        // DEBUG_DRAW_MODE_NONE
                        "INDIRECT_DIFFUSE",            // DEBUG_DRAW_MODE_INDIRECT_DIFFUSE
                        "INDIRECT_DIFFUSE_AO",         // DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO
                        "ALL_DIFFUSE",                 // DEBUG_DRAW_MODE_ALL_DIFFUSE
                        "PGI_EVAL_CLOCKS",             // DEBUG_DRAW_MODE_PGI_EVAL_CLOCKS
                        "PGI_CASCADE_SMOOTH",          // DEBUG_DRAW_MODE_PGI_CASCADE_SMOOTH
                        "PGI_CASCADE_ABSOLUTE",        // DEBUG_DRAW_MODE_PGI_CASCADE_ABSOLUTE
                        "PGI_CASCADE_SMOOTH_ABS_DIFF", // DEBUG_DRAW_MODE_PGI_CASCADE_SMOOTH_ABS_DIFF
                    };
                    auto mode_mappings = std::array{
                        DEBUG_DRAW_MODE_NONE,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO,
                        DEBUG_DRAW_MODE_ALL_DIFFUSE,
                        DEBUG_DRAW_MODE_PGI_EVAL_CLOCKS,
                        DEBUG_DRAW_MODE_PGI_CASCADE_SMOOTH,
                        DEBUG_DRAW_MODE_PGI_CASCADE_ABSOLUTE,
                        DEBUG_DRAW_MODE_PGI_CASCADE_SMOOTH_ABS_DIFF,
                    };
                    ImGui::Combo("pgi debug visualization", &pgi_debug_visualization, modes.data(), modes.size());
                    if (pgi_debug_visualization != 0)
                    {
                        debug_visualization_index_override = mode_mappings[pgi_debug_visualization];
                    }
                }
                ImGui::Checkbox("Enable", reinterpret_cast<bool *>(&render_data.pgi_settings.enabled));
                ImGui::Checkbox("Enable Probe Repositioning", reinterpret_cast<bool *>(&render_data.pgi_settings.probe_repositioning));
                ImGui::Checkbox("Enable Probe Repositioning Spring", reinterpret_cast<bool *>(&render_data.pgi_settings.probe_repositioning_spring_force));
                ImGui::InputInt("Cascade Count", &render_data.pgi_settings.cascade_count);
                ImGui::SliderFloat("Cascade Blend", &render_data.pgi_settings.cascade_blend, 0.0f, 1.0f);
                auto update_rates = std::array{
                    "FULL",    // PGI_UPDATE_RATE_FULL
                    "1_OF_2",  // PGI_UPDATE_RATE_1_OF_2
                    "1_OF_8",  // PGI_UPDATE_RATE_1_OF_8
                    "1_OF_16", // PGI_UPDATE_RATE_1_OF_16
                    "1_OF_32", // PGI_UPDATE_RATE_1_OF_32
                    "1_OF_64", // PGI_UPDATE_RATE_1_OF_64
                };
                ImGui::Combo("Update Rate", &render_data.pgi_settings.update_rate, update_rates.data(), update_rates.size());
                ImGui::InputInt("Probe Surface Resolution", &render_data.pgi_settings.probe_irradiance_resolution);
                ImGui::InputInt("Probe Trace Resolution  ", &render_data.pgi_settings.probe_trace_resolution);
                ImGui::InputInt("Probe Visibility Resolution  ", &render_data.pgi_settings.probe_visibility_resolution);
                ImGui::InputFloat("Probe cos wrap around", &render_data.pgi_settings.cos_wrap_around);
                ImGui::InputFloat3("Probe range", &render_data.pgi_settings.probe_range.x);
                ImGui::InputInt3("Probe Count", &render_data.pgi_settings.probe_count.x);
                ImGui::SeparatorText("Debug");
                ImGui::Checkbox("Debug Draw Probe Influence", reinterpret_cast<bool *>(&render_data.pgi_settings.debug_probe_influence));
                ImGui::Checkbox("Debug Draw Probe Repositioning", reinterpret_cast<bool *>(&render_data.pgi_settings.debug_draw_repositioning));
                ImGui::Checkbox("Debug Draw Probe Repositioning Forces", reinterpret_cast<bool *>(&render_data.pgi_settings.debug_draw_repositioning_forces));
                ImGui::Checkbox("Debug Draw Probe Grid", reinterpret_cast<bool *>(&render_data.pgi_settings.debug_draw_grid));
                auto debug_daw_modes = std::array{
                    "OFF",         // PGI_DEBUG_PROBE_DRAW_MODE_OFF
                    "IRRADIANCE",  // PGI_DEBUG_PROBE_DRAW_MODE_IRRADIANCE
                    "DISTANCE",    // PGI_DEBUG_PROBE_DRAW_MODE_DISTANCE
                    "UNCERTAINTY", // PGI_DEBUG_PROBE_DRAW_MODE_UNCERTAINTY
                    "TEXEL",       // PGI_DEBUG_PROBE_DRAW_MODE_TEXEL
                    "UV",          // PGI_DEBUG_PROBE_DRAW_MODE_UV
                    "NORMAL",      // PGI_DEBUG_PROBE_DRAW_MODE_NORMAL
                    "HYSTERESIS",  // PGI_DEBUG_PROBE_DRAW_MODE_HYSTERESIS
                };
                ImGui::Combo("Debug Probe Draw", &render_data.pgi_settings.debug_probe_draw_mode, debug_daw_modes.data(), debug_daw_modes.size());
                ImGui::InputInt("Debug Force Cascade", &render_data.pgi_settings.debug_force_cascade);
                ImGui::InputInt3("Debug Probe Index", &render_data.pgi_settings.debug_probe_index.x);
            }
            if (ImGui::CollapsingHeader("VSM Settings"))
            {
                {
                    auto modes = std::array{
                        "NONE", // DEBUG_DRAW_MODE_NONE
                        "MESHLET_ID", // DEBUG_DRAW_MODE_MESHLET_ID
                        "ENTITY_ID", // DEBUG_DRAW_MODE_ENTITY_ID
                        "MESH_LOD", // DEBUG_DRAW_MODE_MESH_LOD
                        "VSM_OVERDRAW", // DEBUG_DRAW_MODE_VSM_OVERDRAW
                        "VSM_CLIP_LEVEL", // DEBUG_DRAW_MODE_VSM_CLIP_LEVEL
                        "VSM_POINT_LEVEL", // DEBUG_DRAW_MODE_VSM_POINT_LEVEL
                        "DIRECT_DIFFUSE", // DEBUG_DRAW_MODE_DIRECT_DIFFUSE
                        "INDIRECT_DIFFUSE", // DEBUG_DRAW_MODE_INDIRECT_DIFFUSE
                        "ALL_DIFFUSE", // DEBUG_DRAW_MODE_ALL_DIFFUSE
                        "SHADE_OPAQUE_CLOCKS", // DEBUG_DRAW_MODE_SHADE_OPAQUE_CLOCKS
                        "LIGHT_MASK_VOLUME", // DEBUG_DRAW_MODE_LIGHT_MASK_VOLUME
                    };
                    auto mode_mappings = std::array{
                        DEBUG_DRAW_MODE_NONE,
                        DEBUG_DRAW_MODE_MESHLET_ID,
                        DEBUG_DRAW_MODE_ENTITY_ID,
                        DEBUG_DRAW_MODE_MESH_LOD,
                        DEBUG_DRAW_MODE_VSM_OVERDRAW,
                        DEBUG_DRAW_MODE_VSM_CLIP_LEVEL,
                        DEBUG_DRAW_MODE_VSM_POINT_LEVEL,
                        DEBUG_DRAW_MODE_DIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_ALL_DIFFUSE,
                        DEBUG_DRAW_MODE_SHADE_OPAQUE_CLOCKS,
                        DEBUG_DRAW_MODE_LIGHT_MASK_VOLUME,
                    };
                    ImGui::Combo("vsm debug visualization", &vsm_debug_visualization, modes.data(), modes.size());
                    if (vsm_debug_visualization != 0)
                    {
                        debug_visualization_index_override = mode_mappings[vsm_debug_visualization];
                    }
                }
                bool enable = s_cast<bool>(render_context.render_data.vsm_settings.enable);
                bool shadow_everything = s_cast<bool>(render_context.render_data.vsm_settings.shadow_everything);
                bool force_clip_level = s_cast<bool>(render_context.render_data.vsm_settings.force_clip_level);
                bool enable_directional_caching = s_cast<bool>(render_context.render_data.vsm_settings.enable_directional_caching);
                bool enable_point_caching = s_cast<bool>(render_context.render_data.vsm_settings.enable_point_caching);
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
                }
                ImGui::EndChild();

                ImGui::Checkbox("Enable VSM", &enable);
                ImGui::Checkbox("Shadow everything", &shadow_everything);
                ImGui::Checkbox("Force clip level", &force_clip_level);
                ImGui::SliderInt("Vis point light idx", &render_context.render_data.vsm_settings.force_point_light_idx, -1, render_context.render_data.vsm_settings.point_light_count - 1);
                ImGui::SliderInt("Vis spot light idx", &render_context.render_data.vsm_settings.force_spot_light_idx, -1, render_context.render_data.vsm_settings.spot_light_count - 1);
                ImGui::Checkbox("Enable directional caching", &enable_directional_caching);
                ImGui::Checkbox("Enable point caching", &enable_point_caching);
                auto use_fixed_near_far = s_cast<bool>(render_context.render_data.vsm_settings.fixed_near_far);
                ImGui::Checkbox("Use fixed near far", &use_fixed_near_far);
                render_context.render_data.vsm_settings.fixed_near_far = use_fixed_near_far;
                ImGui::SliderFloat("Clip 0 scale", &render_context.render_data.vsm_settings.clip_0_frustum_scale, 0.1f, 20.0f);
                ImGui::SliderFloat("Clip selection bias", &render_context.render_data.vsm_settings.clip_selection_bias, -0.5f, 2.0f);
                ImGui::SliderFloat("Slope bias", &render_context.render_data.vsm_settings.slope_bias, 0.0, 10.0);
                ImGui::SliderFloat("Constant bias", &render_context.render_data.vsm_settings.constant_bias, 0.0, 20.0);
                ImGui::BeginDisabled(!force_clip_level);
                i32 forced_clip_level = render_context.render_data.vsm_settings.forced_clip_level;
                ImGui::SliderInt("Forced clip level", &forced_clip_level, 0, VSM_CLIP_LEVELS - 1);
                ImGui::EndDisabled();

                i32 forced_lower_point_mip_level = render_context.render_data.vsm_settings.forced_lower_point_mip_level;
                i32 forced_upper_point_mip_level = render_context.render_data.vsm_settings.forced_upper_point_mip_level;
                ImGui::SliderInt("Forced lower point mip level", &forced_lower_point_mip_level, 0, 6);
                ImGui::SliderInt("Forced upper point mip level", &forced_upper_point_mip_level, 0, 6);

                ImGui::InputInt("Debug cubemap face", &render_context.debug_frustum);
                ImGui::Checkbox("Visualize point frustum", &render_context.visualize_point_frustum);
                ImGui::Checkbox("Visualize spot frustum", &render_context.visualize_spot_frustum);
                render_context.debug_frustum = glm::clamp(render_context.debug_frustum, -1, 5);

                render_context.render_data.vsm_settings.enable = enable;
                render_context.render_data.vsm_settings.shadow_everything = shadow_everything;
                render_context.render_data.vsm_settings.force_clip_level = force_clip_level;
                render_context.render_data.vsm_settings.forced_clip_level = force_clip_level ? forced_clip_level : -1;
                render_context.render_data.vsm_settings.forced_lower_point_mip_level = std::min(forced_upper_point_mip_level, forced_lower_point_mip_level);
                render_context.render_data.vsm_settings.forced_upper_point_mip_level = std::max(forced_upper_point_mip_level, forced_lower_point_mip_level);
                render_context.render_data.vsm_settings.enable_directional_caching = enable_directional_caching;
                render_context.render_data.vsm_settings.enable_point_caching = enable_point_caching;

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
        }
    }
    if (debug_visualization_index_override != 0)
    {
        render_data.settings.debug_draw_mode = debug_visualization_index_override;
    }
    else
    {
        render_data.settings.debug_draw_mode = debug_visualization_index;
    }
    ImGui::End();
}

void UIEngine::ui_visbuffer_pipeline_statistics(Scene const & scene, RenderContext & render_context, ApplicationState & app_state)
{
    // Calculate Statistics:
    u32 mesh_instance_count = render_context.mesh_instance_counts.mesh_instance_count;
    u32 first_pass_meshes_post_cull = render_context.general_readback.first_pass_mesh_count_post_cull[0] + render_context.general_readback.first_pass_mesh_count_post_cull[1];
    u32 first_pass_meshlets_pre_cull = render_context.general_readback.first_pass_meshlet_count_pre_cull[0] + render_context.general_readback.first_pass_meshlet_count_pre_cull[1];
    u32 first_pass_meshlets_post_cull = render_context.general_readback.first_pass_meshlet_count_post_cull;
    u32 second_pass_meshes_post_cull = render_context.general_readback.second_pass_mesh_count_post_cull[0] + render_context.general_readback.second_pass_mesh_count_post_cull[1];
    u32 second_pass_meshlets_pre_cull = render_context.general_readback.second_pass_meshlet_count_pre_cull[0] + render_context.general_readback.second_pass_meshlet_count_pre_cull[1];
    u32 second_pass_meshlets_post_cull = render_context.general_readback.second_pass_meshlet_count_post_cull;
    u32 total_meshlets_drawn = first_pass_meshlets_post_cull + second_pass_meshlets_post_cull;

    u32 meshlet_bitfield_used_size = ((FIRST_PASS_MESHLET_BITFIELD_U32_OFFSETS_SIZE + render_context.general_readback.first_pass_meshlet_bitfield_requested_dynamic_size) * 4u) / 1000u;
    u32 meshlet_bitfield_total_size = ((FIRST_PASS_MESHLET_BITFIELD_U32_SIZE) * 4u) / 1000u;
    struct VisbufferPipelineStat
    {
        char const * name = {};
        char const * unit = {};
        u32 value = {};
        u32 max_value = {};
    };
    std::array visbuffer_pipeline_stats = {
        VisbufferPipelineStat{"Mesh Instances Pre Cull", "", mesh_instance_count, MAX_MESH_INSTANCES},
        VisbufferPipelineStat{"First Pass Meshes Post Cull", "", first_pass_meshes_post_cull, MAX_MESH_INSTANCES},
        VisbufferPipelineStat{"Second Pass Meshes Post Cull", "", second_pass_meshes_post_cull, MAX_MESH_INSTANCES},
        VisbufferPipelineStat{"First Pass Meshlets Pre Cull", "", first_pass_meshlets_pre_cull, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS},
        VisbufferPipelineStat{"First Pass Meshlets Post Cull", "", first_pass_meshlets_post_cull, MAX_MESHLET_INSTANCES},
        VisbufferPipelineStat{"Second Pass Meshlets Pre Cull", "", second_pass_meshlets_pre_cull, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS},
        VisbufferPipelineStat{"Second Pass Meshlets Post Cull", "", second_pass_meshlets_post_cull, MAX_MESHLET_INSTANCES},
        VisbufferPipelineStat{"Total Meshlet Instances Post Cull", "", total_meshlets_drawn, MAX_MESHLET_INSTANCES},
        VisbufferPipelineStat{"First Pass Bitfield Use", "kb", meshlet_bitfield_used_size, meshlet_bitfield_total_size},
    };
    if (ImGui::CollapsingHeader("Visbuffer Pipeline Statistics"))
    {
        if (ImGui::BeginTable("Visbuffer GPU Buffer Metrics", 4, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
        {
            ImGui::TableSetupColumn("Value Name", {});
            ImGui::TableSetupColumn("Value", {});
            ImGui::TableSetupColumn("Value Max", {});
            ImGui::TableSetupColumn("Value %", {});
            ImGui::TableHeadersRow();
            for (auto const & stat : visbuffer_pipeline_stats)
            {
                f32 const percentage = static_cast<f32>(stat.value) / static_cast<f32>(stat.max_value) * 100.0f;
                if (percentage > 100.0f)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, 1.0f)));
                }
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text(stat.name);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%i%s", stat.value, stat.unit);
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%i%s", stat.max_value, stat.unit);
                ImGui::TableSetColumnIndex(3);
                ImGui::Text("%f%%", percentage);
                if (percentage > 100.0f)
                {
                    ImGui::PopStyleColor();
                }
            }
            ImGui::EndTable();
        }
    }
}

void UIEngine::ui_pgi_statistics(Scene const & scene, RenderContext & render_context, ApplicationState & app_state)
{
    if (ImGui::CollapsingHeader("Probe Global Illumination Statistics"))
    {
        u32 heavy_ray_count = 512000;
        u32 super_heavy_ray_count = 1024000;
        auto & render_data = render_context.render_data;
        u32 total_probes = render_data.pgi_settings.probe_count.x * render_data.pgi_settings.probe_count.y * render_data.pgi_settings.probe_count.z * render_data.pgi_settings.cascade_count;
        static u32 updated_probes = 0;
        static u32 ray_count = 0;
        if ((render_data.frame_index % 11) == 0)
        {
            updated_probes = render_context.general_readback.requested_probes;
            ray_count = render_data.pgi_settings.probe_trace_resolution * render_data.pgi_settings.probe_trace_resolution * render_context.general_readback.requested_probes;
        }
        ImGui::Text(fmt::format("Total Probe Count {}", total_probes).c_str());
        ImGui::Text(fmt::format("PGI Updated Probes {:>7} / {:>7} = {}%%",
            updated_probes,
            total_probes,
            float(updated_probes) / float(total_probes) * 100.0f)
                .c_str());
        if (ray_count > super_heavy_ray_count)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, 1.0f)));
        }
        else if (ray_count > heavy_ray_count)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 1.0f)));
        }
        ImGui::Text(fmt::format("Rays Shoot this frame {}", ray_count).c_str());
        if (ray_count > heavy_ray_count)
        {
            ImGui::PopStyleColor(1);
        }
    }
}

void ui_light_statistics(Scene const & scene, RenderContext & render_context, ApplicationState & app_state)
{
    if (ImGui::CollapsingHeader("Lights Statistics"))
    {
        ImGui::Text("Max Point Lights: %i", MAX_POINT_LIGHTS);
        ImGui::Text("Max Spot Lights: %i", MAX_SPOT_LIGHTS);
        ImGui::Text("Max Light Instances: %i", MAX_LIGHT_INSTANCES_PER_FRAME);
        ImGui::Text("Point Lights: %i", render_context.render_data.light_settings.point_light_count);
        ImGui::Text("Spot Lights: %i", render_context.render_data.light_settings.spot_light_count);
    }
}

void UIEngine::ui_render_statistics(Scene const & scene, RenderContext & render_context, ApplicationState & app_state)
{
    static float t = 0;
    t += gather_perm_measurements ? render_context.render_data.delta_time : 0.0f;
    bool auto_reset_timings = false;
    if (ImGui::Begin("Render statistics", nullptr, ImGuiWindowFlags_NoCollapse))
    {
        ImGui::SeparatorText("CPU Group Timings");
        {
            ImGui::Text("Delta Time:                   %fms", app_state.delta_time * 1000.0f);
            ImGui::Text("Frames Per Second:            %f", 1.0f / app_state.delta_time);
            ImGui::Text("CPU Windowing:                %fms", app_state.time_taken_cpu_windowing * 1000.0f);
            ImGui::Text("CPU Application:              %fms", app_state.time_taken_cpu_application * 1000.0f);
            ImGui::Text("CPU Wait For GPU:             %fms", app_state.time_taken_cpu_wait_for_gpu * 1000.0f);
            ImGui::Text("CPU Renderer Prepare:         %fms", app_state.time_taken_cpu_renderer_prepare * 1000.0f);
            ImGui::Text("CPU Renderer Record + Submit: %fms", app_state.time_taken_cpu_renderer_record * 1000.0f);
        }
        ImGui::SeparatorText("GPU Group Timings");
        {
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
                for (i32 i = 0; i < render_times_history.scrolling_ewa.size(); i++)
                {
                    render_times_history.scrolling_ewa.at(i).erase();
                    render_times_history.scrolling_mean.at(i).erase();
                    render_times_history.scrolling_raw.at(i).erase();
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Show entire measured interval", &show_entire_interval);

            // ==== update scrolling times for render time groups =====

            f32 const rolling_mean_weight = s_cast<f32>(render_times_history.mean_sample_count) / s_cast<f32>(render_times_history.mean_sample_count + 1);
            f32 const ewa_weight = 0.95f;
            for (u32 group_i = 0; group_i < RenderTimes::GROUP_COUNT; ++group_i)
            {
                u32 group_first_idx = RenderTimes::group_first_flat_index(group_i);
                u32 group_size = RenderTimes::group_size(group_i);

                f32 raw = 0.0f;
                for (u32 time = group_first_idx; time < (group_first_idx + group_size); ++time)
                {
                    raw += static_cast<u32>(render_context.render_times.get(time)) * 0.001f;
                }
                render_times_history.scrolling_raw.at(group_i).add_point(ImVec2(t, raw));

                f32 const new_rolling_average = render_times_history.scrolling_mean.at(group_i).back().y * rolling_mean_weight + raw * (1.0f - rolling_mean_weight);
                render_times_history.scrolling_mean.at(group_i).add_point(ImVec2(t, new_rolling_average));

                f32 const new_ewa = render_times_history.scrolling_ewa.at(group_i).back().y * ewa_weight + (1.0f - ewa_weight) * raw;
                render_times_history.scrolling_ewa.at(group_i).add_point(ImVec2(t, new_ewa));
            }
            render_times_history.mean_sample_count += 1;

            // ========================================================

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

            static float history = 20.0f;
            if (ImPlot::BeginPlot("##Scrolling"))
            {
                decltype(render_times_history.scrolling_ewa) * measurements_scrolling_selected = {};
                if (selected_item == 0) { measurements_scrolling_selected = &render_times_history.scrolling_ewa; }
                else if (selected_item == 1) { measurements_scrolling_selected = &render_times_history.scrolling_mean; }
                else { measurements_scrolling_selected = &render_times_history.scrolling_raw; }
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
                for (i32 i = 0; i < measurements_scrolling_selected->size(); i++)
                {
                    ImPlot::PlotLine(
                        std::string(RenderTimes::group_name(i)).c_str(),
                        &measurements_scrolling_selected->at(i).data[0].x,
                        &measurements_scrolling_selected->at(i).data[0].y,
                        measurements_scrolling_selected->at(i).data.size(),
                        0,
                        measurements_scrolling_selected->at(i).offset,
                        2 * sizeof(f32));
                }
                ImPlot::EndPlot();
            }

            for (i32 group_i = 0; group_i < RenderTimes::GROUP_COUNT; group_i++)
            {
                std::string_view timing_unit = {};
                decltype(render_times_history.scrolling_ewa) * data = {};
                if (selected_item == 0)
                {
                    timing_unit = "ewa";
                    data = &render_times_history.scrolling_ewa;
                }
                else if (selected_item == 1)
                {
                    timing_unit = "average";
                    data = &render_times_history.scrolling_mean;
                }
                else
                {
                    timing_unit = "raw";
                    data = &render_times_history.scrolling_raw;
                }
                bool open = ImGui::CollapsingHeader(fmt::format("{:<30}",
                    fmt::format("{} {}: ",
                        RenderTimes::group_name(group_i), timing_unit.data())
                        .c_str())
                        .c_str());
                ImGui::SameLine();
                ImGui::Text(fmt::format("{:>10.2f} us", data->at(group_i).back().y).c_str());
                if (open)
                {
                    if (ImGui::BeginTable("Detail Timings", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Timing", {});
                        ImGui::TableSetupColumn("Value", {});
                        ImGui::TableSetupColumn("Value Smooth", {});
                        ImGui::TableHeadersRow();
                        for (auto in_group_i = 0; in_group_i < RenderTimes::group_size(group_i); ++in_group_i)
                        {
                            u32 timing_index = RenderTimes::group_first_flat_index(group_i) + in_group_i;
                            std::string_view name = RenderTimes::group_names(group_i)[in_group_i];

                            bool skip = render_context.render_times.get(timing_index) == 0;
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name.data());
                            ImGui::TableSetColumnIndex(1);
                            if (skip)
                                ImGui::Text("-");
                            else
                                ImGui::Text("%fus", static_cast<f32>(render_context.render_times.get(timing_index)) * 0.001f);
                            ImGui::TableSetColumnIndex(2);
                            if (skip)
                                ImGui::Text("-");
                            else
                                ImGui::Text("%fus", static_cast<f32>(render_context.render_times.get_smooth(timing_index)) * 0.001f);
                        }
                        ImGui::EndTable();
                    }
                }
                if (calculate_percentiles)
                {
                    auto sorted_values = render_times_history.scrolling_raw.at(group_i).data;
                    std::sort(sorted_values.begin(), sorted_values.end(),
                        [](auto const & a, auto const & b) -> bool
                        { return a.y < b.y; });
                    ImGui::TextUnformatted(fmt::format("{:<30} {:>10.2f} us", "\t 95th percentile: ", sorted_values.at(sorted_values.size() * 0.95f).y).c_str());
                    ImGui::TextUnformatted(fmt::format("{:<30} {:>10.2f} us", "\t 99th percentile: ", sorted_values.at(sorted_values.size() * 0.99f).y).c_str());
                }
            }
        }
        ImGui::SeparatorText("Utalizations");
        {
            ui_visbuffer_pipeline_statistics(scene, render_context, app_state);
            ui_pgi_statistics(scene, render_context, app_state);
            ui_light_statistics(scene, render_context, app_state);
        }
        ImGui::SeparatorText("Device Memory Use");
        {
            auto mem_report = render_context.gpu_context->device.device_memory_report_convenient();

            {
                bool open = (ImGui::CollapsingHeader("Total Buffer VRAM Use:        "));
                ImGui::SameLine();
                ImGui::Text(fmt::format("{:>10.2f} mb", s_cast<f32>(mem_report.total_buffer_device_memory_use) / 1024.0f  / 1024.0f).c_str());
                if (open){
                    std::stable_sort(mem_report.buffer_list.begin(), mem_report.buffer_list.end(), [](auto& a, auto& b){ return a.size > b.size; });
                    usize const list_len_max = 30;
                    usize const list_len = std::min(list_len_max, mem_report.buffer_list.size());
                    if (ImGui::BeginTable("Buffer Mem Use", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Name", {});
                        ImGui::TableSetupColumn("Size", {});
                        ImGui::TableSetupColumn("Block Allocated?", {});
                        ImGui::TableHeadersRow();
                        for (auto i = 0; i < list_len; ++i)
                        {
                            auto buf = mem_report.buffer_list[i];
                            char const * name = render_context.gpu_context->device.buffer_info(buf.id).value().name.data();
    
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name);
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%fmb", s_cast<f32>(buf.size) / 1024.0f / 1024.0f);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text(buf.block_allocated ? "Yes" : "No");
                        }
                    }
                    ImGui::EndTable();
                }
            }
            {
                bool open = (ImGui::CollapsingHeader("Total Image VRAM Use:         "));
                ImGui::SameLine();
                ImGui::Text(fmt::format("{:>10.2f} mb", s_cast<f32>(mem_report.total_image_device_memory_use) / 1024.0f  / 1024.0f).c_str());
                if (open){
                    std::stable_sort(mem_report.image_list.begin(), mem_report.image_list.end(), [](auto& a, auto& b){ return a.size > b.size; });
                    usize const list_len_max = 30;
                    usize const list_len = std::min(list_len_max, mem_report.image_list.size());
                    if (ImGui::BeginTable("Image Mem Use", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Name", {});
                        ImGui::TableSetupColumn("Size", {});
                        ImGui::TableSetupColumn("Block Allocated?", {});
                        ImGui::TableHeadersRow();
                        for (auto i = 0; i < list_len; ++i)
                        {
                            auto img = mem_report.image_list[i];
                            char const * name = render_context.gpu_context->device.image_info(img.id).value().name.data();
    
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name);
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%fmb", s_cast<f32>(img.size) / 1024.0f / 1024.0f);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text(img.block_allocated ? "Yes" : "No");
                        }
                    }
                    ImGui::EndTable();
                }
            }
            {
                bool open = (ImGui::CollapsingHeader("Total Aliased Tlas VRAM Use:   "));
                ImGui::SameLine();
                ImGui::Text(fmt::format("{:>10.2f} mb", s_cast<f32>(mem_report.total_aliased_tlas_device_memory_use) / 1024.0f  / 1024.0f).c_str());
                if (open){
                    std::stable_sort(mem_report.tlas_list.begin(), mem_report.tlas_list.end(), [](auto& a, auto& b){ return a.size > b.size; });
                    usize const list_len_max = 30;
                    usize const list_len = std::min(list_len_max, mem_report.tlas_list.size());
                    if (ImGui::BeginTable("Tlas Aliased Mem Use", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Name", {});
                        ImGui::TableSetupColumn("Size", {});
                        ImGui::TableSetupColumn("Buffer Allocated?", {});
                        ImGui::TableHeadersRow();
                        for (auto i = 0; i < list_len; ++i)
                        {
                            auto tlas = mem_report.tlas_list[i];
                            char const * name = render_context.gpu_context->device.tlas_info(tlas.id).value().name.data();
    
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name);
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%fmb", s_cast<f32>(tlas.size) / 1024.0f / 1024.0f);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("Yes");
                        }
                    }
                    ImGui::EndTable();
                }
            }
            {
                bool open = (ImGui::CollapsingHeader("Total Aliased Blas VRAM Use:   "));
                ImGui::SameLine();
                ImGui::Text(fmt::format("{:>10.2f} mb", s_cast<f32>(mem_report.total_aliased_blas_device_memory_use) / 1024.0f  / 1024.0f).c_str());
                if (open){
                    std::stable_sort(mem_report.blas_list.begin(), mem_report.blas_list.end(), [](auto& a, auto& b){ return a.size > b.size; });
                    usize const list_len_max = 30;
                    usize const list_len = std::min(list_len_max, mem_report.blas_list.size());
                    if (ImGui::BeginTable("Blas Aliased Mem Use", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Name", {});
                        ImGui::TableSetupColumn("Size", {});
                        ImGui::TableSetupColumn("Buffer Allocated?", {});
                        ImGui::TableHeadersRow();
                        for (auto i = 0; i < list_len; ++i)
                        {
                            auto blas = mem_report.blas_list[i];
                            char const * name = render_context.gpu_context->device.blas_info(blas.id).value().name.data();
    
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name);
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%fmb", s_cast<f32>(blas.size) / 1024.0f / 1024.0f);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("Yes");
                        }
                    }
                    ImGui::EndTable();
                }
            }
            if (ImGui::CollapsingHeader(fmt::format("Total  Block VRAM Use:        {:>10.2f} mb", s_cast<f32>(mem_report.total_memory_block_device_memory_use) / 1024.0f / 1024.0f ).c_str()));
            if (ImGui::CollapsingHeader(fmt::format("Total  VRAM Use:              {:>10.2f} mb", s_cast<f32>(mem_report.total_device_memory_use) / 1024.0f / 1024.0f ).c_str()));
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