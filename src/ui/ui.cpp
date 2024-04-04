#include "ui.hpp"
#include <filesystem>
#include <imgui.h>
#include <implot.h>
#include "widgets/helpers.hpp"

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

UIEngine::UIEngine(Window & window, AssetProcessor & asset_processor, GPUContext * context)
    : scene_graph(&imgui_renderer, &icons, context->lin_clamp_sampler),
      property_viewer(&imgui_renderer, &icons, context->lin_clamp_sampler),
      context{context}
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
    imgui_renderer = daxa::ImGuiRenderer({context->device, context->swapchain.get_format(), imgui_context, false});
    setup_colors();
}

void UIEngine::main_update(RenderContext & render_ctx, Scene const & scene)
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
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (vsm_debug_menu)
    {
        if(ImGui::Begin("VSM Debug Menu", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            bool visualize_clip_levels = s_cast<bool>(render_ctx.render_data.vsm_settings.visualize_clip_levels);
            bool force_clip_level = s_cast<bool>(render_ctx.render_data.vsm_settings.force_clip_level);
            bool disable_caching = s_cast<bool>(render_ctx.render_data.vsm_settings.disable_caching);
            ImGui::Checkbox("Visualize clip levels", &visualize_clip_levels);
            ImGui::Checkbox("Force clip level",&force_clip_level);
            ImGui::Checkbox("Disable caching",&disable_caching);
            ImGui::SliderFloat("Clip 0 scale", &render_ctx.render_data.vsm_settings.clip_0_frustum_scale, 0.1f, 20.f);
            ImGui::BeginDisabled(!force_clip_level);
            i32 forced_clip_level = render_ctx.render_data.vsm_settings.forced_clip_level;
            ImGui::SliderInt("Forced clip level", &forced_clip_level, 0, VSM_CLIP_LEVELS - 1);
            ImGui::EndDisabled();
            render_ctx.render_data.vsm_settings.visualize_clip_levels = visualize_clip_levels;
            render_ctx.render_data.vsm_settings.force_clip_level = force_clip_level;
            render_ctx.render_data.vsm_settings.forced_clip_level = force_clip_level ? forced_clip_level : -1;
            render_ctx.render_data.vsm_settings.disable_caching = disable_caching;

            ImGui::Image(
                imgui_renderer.create_texture_id({
                    .image_view_id = render_ctx.gpuctx->shader_debug_context.vsm_debug_page_table.get_state().images[0].default_view(),
                    .sampler_id = std::bit_cast<daxa::SamplerId>(render_ctx.render_data.samplers.nearest_clamp),
                }),
                ImVec2(VSM_DEBUG_PAGE_TABLE_RESOLUTION, VSM_DEBUG_PAGE_TABLE_RESOLUTION)
            );
            ImGui::Image(
                imgui_renderer.create_texture_id({
                    .image_view_id = render_ctx.gpuctx->shader_debug_context.vsm_debug_meta_memory_table.get_state().images[0].default_view(),
                    .sampler_id = std::bit_cast<daxa::SamplerId>(render_ctx.render_data.samplers.nearest_clamp),
                }),
                ImVec2(VSM_DEBUG_META_MEMORY_TABLE_RESOLUTION, VSM_DEBUG_META_MEMORY_TABLE_RESOLUTION)
            );
            ImGui::End();
        }
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
            ImGui::InputScalarN("resolution", ImGuiDataType_U32, &render_ctx.render_data.settings.render_target_size, 2);
            ImGui::End();
        }
    }
    if (widget_renderer_statistics)
    {
        if (ImGui::Begin("Renderer Statistics", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            ImGui::Text("fps: 100");
            ImGui::End();
        }
    }
    if (widget_scene_hierarchy)
    {
        ui_scenegraph(scene);
    }
    if (renderer_settings)
    {
        ui_renderer_settings(scene, render_ctx.render_data.settings);
    }
    if (widget_property_viewer)
    {
        property_viewer.render({
            .sky_settings = &render_ctx.render_data.sky_settings,
            .post_settings = &render_ctx.render_data.postprocess_settings,
        });
    }
    if (demo_window)
    {
        ImGui::ShowDemoWindow();
    }
}

void UIEngine::ui_scenegraph(Scene const & scene)
{
    if (scene._gltf_asset_manifest.empty())
    {
        return;
    }
    scene_graph.begin();

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

    scene_graph.end();
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
        ImGui::End();
    }
}

UIEngine::~UIEngine()
{
    for (u32 icon_idx = 0; icon_idx < s_cast<u32>(ICONS::SIZE); icon_idx++)
    {
        if (!icons.at(icon_idx).is_empty())
        {
            context->device.destroy_image(icons.at(icon_idx));
        }
    }
    ImPlot::DestroyContext();
}