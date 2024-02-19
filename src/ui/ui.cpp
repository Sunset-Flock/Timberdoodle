#include "ui.hpp"
#include <filesystem>
#include <imgui.h>
#include "widgets/helpers.hpp"

UIEngine::UIEngine(Window & window, AssetProcessor & asset_processor, GPUContext * context)
    : scene_graph(&imgui_renderer, &icons, std::bit_cast<daxa::SamplerId>(context->shader_globals.samplers.linear_clamp)),
      context{context}
{
    auto * imgui_context = ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(window.glfw_handle, true);
    ImGuiIO & io = ImGui::GetIO();

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
    imgui_renderer = daxa::ImGuiRenderer({context->device, context->swapchain.get_format(), imgui_context});
}

void UIEngine::main_update(Settings & settings, SkySettings & sky_settings, Scene const & scene)
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Widgets"))
        {
            ImGui::MenuItem("Settings", NULL, &widget_settings);
            ImGui::MenuItem("Renderer Statistics", NULL, &widget_renderer_statistics);
            ImGui::MenuItem("Scene Hierarchy", NULL, &widget_scene_hierarchy);
            ImGui::MenuItem("Camera Settings", NULL, &camera_settings);
            ImGui::MenuItem("Shader Debug Menu", NULL, &shader_debug_menu);
            ImGui::MenuItem("Widget Property Viewer", NULL, &widget_property_viewer);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (widget_settings)
    {
        if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            ImGui::SeparatorText("Scene graph widget settings");
            ImGui::SliderFloat("icon size", &scene_graph.icon_size, 1.0f, 50.0f);
            ImGui::SliderFloat("spacing", &scene_graph.icon_text_spacing, 1.0f, 50.0f);
            ImGui::SliderFloat("indent", &scene_graph.indent, 1.0f, 50.0f);
            ImGui::Separator();
            ImGui::InputScalarN("resolution", ImGuiDataType_U32, &settings.render_target_size, 2);
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
        draw_scenegraph(scene);
    }
    if (widget_property_viewer)
    {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowTitleAlign, {0.5f, 0.5f});
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 6});
        ImGui::Begin("Selector widget", nullptr, ImGuiWindowFlags_NoCollapse);
        auto flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
        ImGui::BeginChild("property selector", ImVec2(26, 0), false, flags);

        auto * window = ImGui::GetCurrentWindow();
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {2, 2});
        ImGui::PushID("Selector Icons");
        ImGui::Dummy({0, 3});
        std::array const selector_icons = {ICONS::SUN, ICONS::CAMERA, ICONS::MESH};
        for (i32 i = 0; i < selector_icons.size(); i++)
        {
            if (i == selected)
            {
                ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
                ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
            }
            else { ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.0f, 0.0f, 0.0f, 0.0f)); }
            bool got_selected = {};
            if (ImGui::ImageButton(
                    std::to_string(i).c_str(),
                    imgui_renderer.create_texture_id({
                        .image_view_id = icons.at(s_cast<u32>(selector_icons.at(i))).default_view(),
                        .sampler_id = std::bit_cast<daxa::SamplerId>(context->shader_globals.samplers.linear_clamp),
                    }),
                    ImVec2(26.0f, 22.0f),
                    ImVec2(0.0f, 1.0f),
                    ImVec2(1.2f, 0.0f),
                    ImVec4(0.0f, 0.0f, 0.0f, 0.0f)))
            {
                got_selected = true;
            };
            ImGui::PopStyleColor(selected == i ? 3 : 1);
            selected = got_selected ? i : selected;
        }
        ImGui::PopID();
        ImGui::PopStyleVar();
        ImGui::EndChild();
        ImGui::SameLine();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.5f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {2, 2});
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {2, 6});
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyleColorVec4(ImGuiCol_FrameBgHovered));
        // Sun settings
        if (selected == 0)
        {
            ImGui::BeginChild("Sun settings", {0, 0}, false, ImGuiWindowFlags_NoScrollbar);
            {
                // ImGui::Dummy({2, 1});
                // {
                //     ImGui::Dummy({2, 1});
                //     ImGui::SameLine();
                //     auto const start_pos = ImGui::GetCurrentWindow()->DC.CursorPos;
                //     auto const pos_x = ImGui::GetCursorPosX();
                //     ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.0, 0.0, 0.0, 0.0});
                //     ImGui::PushStyleColor(ImGuiCol_HeaderActive, {0.0, 0.0, 0.0, 0.0});
                //     if (ImGui::TreeNode("Settings"))
                //     {
                //         ImGui::Text("text 1");
                //         ImGui::Text("text 2");
                //         ImGui::Text("text 3");
                //         ImGui::Text("text 4");
                //         ImGui::TreePop();
                //     }
                //     auto const end_pos = ImGui::GetCurrentWindow()->DC.CursorPos;
                //     auto const size = ImVec2(ImGui::GetContentRegionAvail().x - pos_x - 5, end_pos.y - start_pos.y);
                //     auto draw_list = ImGui::GetWindowDrawList();
                //     draw_list->AddRectFilled(
                //         start_pos,
                //         {start_pos.x + size.x, start_pos.y + size.y},
                //         ImGui::GetColorU32(ImGui::GetStyleColorVec4(ImGuiCol_FrameBgActive)),
                //         3.0f
                //     );
                //     ImGui::SetCursorScreenPos(start_pos);
                //     if (ImGui::TreeNode("Settings"))
                //     {
                //         ImGui::Text("text 1");
                //         ImGui::Text("text 2");
                //         ImGui::Text("text 3");
                //         ImGui::Text("text 4");
                //         ImGui::TreePop();
                //     }
                //     ImGui::PopStyleColor(2);
                // }

                ImGui::Dummy({2, 1});
                {
                    ImGui::Dummy({2, 1});
                    ImGui::SameLine();
                    auto const start_pos = ImGui::GetCurrentWindow()->DC.CursorPos;
                    auto const pos_x = ImGui::GetCursorPosX();
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.0, 0.0, 0.0, 0.0});
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, {0.0, 0.0, 0.0, 0.0});

                    auto * draw_list = ImGui::GetWindowDrawList();
                    auto const idx_start_count = draw_list->IdxBuffer.size();
                    if (ImGui::TreeNode("Sun Settings"))
                    {
                        auto & sky_settings = context->shader_globals.sky_settings;
                        f32 const angle_y_rad = glm::acos(sky_settings.sun_direction.z);
                        f32 const angle_x_rad = glm::acos(sky_settings.sun_direction.x / sin(angle_y_rad));
                        f32 angle_y_deg = glm::degrees(angle_y_rad);
                        f32 angle_x_deg = glm::degrees(angle_x_rad);
                        ImGui::SliderFloat("Angle X", &angle_x_deg, 0.0f, 360.0f, "%.1f°");
                        ImGui::SliderFloat("Angle Y", &angle_y_deg, 0.0f, 180.0f, "%.1f°");
                        sky_settings.sun_direction =
                            {
                                daxa_f32(glm::cos(glm::radians(angle_x_deg)) * glm::sin(glm::radians(angle_y_deg))),
                                daxa_f32(glm::sin(glm::radians(angle_x_deg)) * glm::sin(glm::radians(angle_y_deg))),
                                daxa_f32(glm::cos(glm::radians(angle_y_deg))),
                            };
                        ImGui::TreePop();
                    }
                    auto const end_pos = ImGui::GetCurrentWindow()->DC.CursorPos;
                    auto const size = ImVec2(ImGui::GetContentRegionAvail().x - pos_x - 5, end_pos.y - start_pos.y);
                    auto const idx_end_count = draw_list->IdxBuffer.size();
                    draw_list->AddRectFilled(
                        start_pos,
                        {start_pos.x + size.x, start_pos.y + size.y},
                        ImGui::GetColorU32(ImGui::GetStyleColorVec4(ImGuiCol_FrameBgActive)),
                        3.0f
                    );
                    auto const idx_rect_end_count = draw_list->IdxBuffer.size();

                    ImVector<ImDrawIdx> new_indices = {};
                    auto const prefix_idx_count = idx_start_count;
                    auto const context_idx_count = idx_end_count - idx_start_count;
                    auto const rect_idx_count = idx_rect_end_count - idx_end_count;
                    new_indices.resize(idx_rect_end_count);
                    std::memcpy(new_indices.Data, draw_list->IdxBuffer.Data, sizeof(ImDrawIdx) * prefix_idx_count);
                    std::memcpy(new_indices.Data + prefix_idx_count, draw_list->IdxBuffer.Data + idx_end_count, sizeof(ImDrawIdx) * rect_idx_count);
                    std::memcpy(new_indices.Data + prefix_idx_count + rect_idx_count, draw_list->IdxBuffer.Data + prefix_idx_count, sizeof(ImDrawIdx) * context_idx_count);
                    draw_list->IdxBuffer = new_indices;
                    draw_list->_IdxWritePtr = draw_list->IdxBuffer.end();
                    ImGui::PopStyleColor(2);
                }
            }
            ImGui::EndChild();
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleVar(3);

        ImGui::End();
        ImGui::PopStyleVar(3);
    }
}

void UIEngine::draw_scenegraph(Scene const & scene)
{
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

    ImGui::ShowDemoWindow();
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
}