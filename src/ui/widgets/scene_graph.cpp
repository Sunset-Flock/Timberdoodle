#include "scene_graph.hpp"
#include <daxa/gpu_resources.hpp>
#include <imgui.h>
#include "helpers.hpp"
namespace tido
{
    namespace ui
    {
        static constexpr u32 stylevar_change_count = 4;

        SceneGraph::SceneGraph(daxa::ImGuiRenderer * renderer, std::vector<daxa::ImageId> const * icons, daxa::SamplerId linear_sampler) : 
            renderer{renderer},
            icons{icons},
            linear_sampler(linear_sampler),
            per_level_indent{10.0f}
        {
        }

        void SceneGraph::begin()
        {
            /// NOTE: For now we count the number of stylevars changed with a constexpr value
            //        this MUST match the number of pushes we do here otherwise we get style leaking
            //        into other windows which might be undesireable
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
            ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, {4, 5});
            ImGui::PushStyleVar(ImGuiStyleVar_WindowTitleAlign, {0.5f, 0.5f});
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
            ImGui::Begin("Scene Hierarchy", nullptr, ImGuiWindowFlags_NoCollapse);

            static ImGuiTableFlags flags = 
                ImGuiTableFlags_BordersOuterV |
                ImGuiTableFlags_BordersOuterH |
                ImGuiTableFlags_Resizable |
                ImGuiTableFlags_RowBg | 
                ImGuiTableFlags_NoBordersInBody;
            ImGui::BeginTable("Scene Hierarchy", 1, flags);
            context = ImGui::GetCurrentContext();
            table = context->CurrentTable;
            window = context->CurrentWindow;
            ImGui::TableSetupColumn("Name");
        }

        auto SceneGraph::get_cell_bounds() -> ImRect
        {
            // True bounds of the cell
            ImRect cell_row_bb = ImGui::TableGetCellBgRect(table, 0);
            f32 label_height = std::max(
                ImGui::CalcTextSize("x").y,
                table->RowMinHeight - table->RowCellPaddingY * 2.0f
            );
            return ImRect(
                cell_row_bb.Min.x,
                cell_row_bb.Min.y,
                cell_row_bb.Max.x,
                std::max(
                    cell_row_bb.Max.y,
                    cell_row_bb.Min.y + label_height + context->Style.CellPadding.y * 2.0f 
                )
            );
        }

        void SceneGraph::end()
        {
            /// NOTE: Make sure this value matches the number of stylevars we pushed in begin()
            ImGui::EndTable();
            ImGui::End(); // Scene graph widget window
            ImGui::PopStyleVar(stylevar_change_count);
        }

        void SceneGraph::add_level()
        {
            ImGui::Indent(per_level_indent);
            window->DC.TreeDepth++;
        }

        void SceneGraph::remove_level()
        {
            ImGui::Unindent(per_level_indent);
            window->DC.TreeDepth--;
        }

        auto SceneGraph::add_leaf_node(std::string uuid, NodeType type) -> RetNodeState
        {
            ImRect cell_bounds = get_cell_bounds();

            ImVec2 font_size = ImGui::CalcTextSize("x");
            ImGuiID const elem_id = window->GetID(fmt::format("bounds_elem_{}", uuid).c_str());

            State const elem_state = button_like_behavior(cell_bounds, elem_id);
            if(elem_state.pressed)
            {
                DEBUG_MSG(fmt::format("elem pressed {}", uuid));
                selected_id = elem_id;
            }
            const bool selected = selected_id == elem_id;
            const bool hovered = elem_state.hovered;
            if(hovered || selected) 
            {
                ImGuiCol_ new_color = selected ? ImGuiCol_ButtonActive : ImGuiCol_ButtonHovered;
                ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(ImGuiCol(new_color)), table->CurrentColumn);
            }
            ICONS icon = ICONS::MESH;
            ImGui::Image(renderer->create_texture_id({
                    .image_view_id = icons->at(s_cast<u32>(icon)).default_view(),
                    .sampler_id = linear_sampler
                }), 
                ImVec2(font_size.x * 2.0f, font_size.y),
                ImVec2(0.0, 1.1), ImVec2(1.0, -0.1),
                ImGui::GetStyleColorVec4(ImGuiCol_Text)
            );
            ImGui::SameLine();
            ImGui::TextUnformatted(fmt::format("{}", uuid).c_str());
            return RetNodeState::CLOSED;
        }

        auto SceneGraph::add_inner_node(std::string uuid) -> RetNodeState
        {
            ImRect cell_bounds = get_cell_bounds();

            ImVec2 font_size = ImGui::CalcTextSize("x");
            ImRect icon_bounds = ImRect(
                ImVec2( window->DC.CursorPos.x, cell_bounds.Min.y),
                ImVec2( window->DC.CursorPos.x + font_size.x * 2.0f, cell_bounds.Max.y)
            );
            ImRect elem_bounds = ImRect(
                ImVec2(icon_bounds.Max.x, cell_bounds.Min.y),
                cell_bounds.Max
            );
            ImGuiID const icon_id = window->GetID(fmt::format("bounds_icon_{}", uuid).c_str());
            ImGuiID const elem_id = window->GetID(fmt::format("bounds_elem_{}", uuid).c_str());

            State const icon_state = button_like_behavior(icon_bounds, icon_id);
            State const elem_state = button_like_behavior(elem_bounds, elem_id);
            bool component_state = window->DC.StateStorage->GetBool(elem_id);
            if(icon_state.pressed)
            {
                window->DC.StateStorage->SetBool(elem_id, !component_state);
                DEBUG_MSG(fmt::format("icon pressed {}", uuid));
                component_state = !component_state;
            }
            if(elem_state.pressed)
            {
                DEBUG_MSG(fmt::format("elem pressed {}", uuid));
                selected_id = elem_id;
            }
            const bool selected = selected_id == elem_id;
            const bool hovered = icon_state.hovered || elem_state.hovered;
            if(hovered || selected) 
            {
                ImGuiCol_ new_color = selected ? ImGuiCol_ButtonActive : ImGuiCol_ButtonHovered;
                ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(ImGuiCol(new_color)), table->CurrentColumn);
            }
            ICONS arrow_icon = component_state ? ICONS::CHEVRON_UP : ICONS::CHEVRON_DOWN;
            ImGui::Image(renderer->create_texture_id({
                    .image_view_id = icons->at(s_cast<u32>(arrow_icon)).default_view(),
                    .sampler_id = linear_sampler
                }), 
                ImVec2(font_size.x * 2.0f, font_size.y),
                ImVec2(0.2, 0.8), ImVec2(0.8, 0.2),
                ImGui::GetStyleColorVec4(ImGuiCol_Text)
            );
            ImGui::SameLine();
            ImGui::TextUnformatted(fmt::format("{}", uuid).c_str());
            return component_state ? RetNodeState::OPEN : RetNodeState::CLOSED;
        }

        auto SceneGraph::add_node(NodeType type, std::string uuid) -> RetNodeState
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();

            switch(type)
            {
                case NodeType::INNER: 
                    return add_inner_node(uuid);
                case NodeType::MESH:
                    return add_leaf_node(uuid, type);
                default:
                    return RetNodeState::ERROR;
            }
        }
    }
}