#include "scene_graph.hpp"
#include <imgui.h>

namespace tido
{
    namespace ui
    {
        void SceneGraph::begin()
        {
            static ImGuiTableFlags flags = 
                ImGuiTableFlags_BordersV |
                ImGuiTableFlags_BordersOuterH |
                ImGuiTableFlags_Resizable |
                ImGuiTableFlags_RowBg | 
                ImGuiTableFlags_NoBordersInBody;
            ImGui::BeginTable("Scene Hierarchy", 1, flags);
            context = ImGui::GetCurrentContext();
            table = context->CurrentTable;
            window = context->CurrentWindow;
            ImGui::TableSetupColumn("Name");
            ImGui::TableHeadersRow();

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
            ImGui::EndTable();
        }

        void SceneGraph::add_level()
        {
            ImGui::Indent();
            window->DC.TreeDepth++;
        }

        void SceneGraph::remove_level()
        {
            ImGui::Unindent();
            window->DC.TreeDepth--;
        }

        auto SceneGraph::add_leaf_node(std::string uuid) -> RetNodeState
        {
            ImGuiID id = window->GetID(fmt::format("bounds_leaf_{}", uuid).c_str());
            ImRect cell_bounds = get_cell_bounds();
            ImGui::ItemAdd(cell_bounds, id);
            bool hovered, held, pressed;
            pressed = ImGui::ButtonBehavior(cell_bounds, id, &hovered, &held, ImGuiButtonFlags_AllowOverlap | ImGuiButtonFlags_PressedOnClick);
            if(hovered) 
            {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(ImGuiCol(ImGuiCol_ButtonHovered)), table->CurrentColumn);
            }
            if(pressed)
            {
                DEBUG_MSG(fmt::format("Leaf with uuid {} pressed", uuid).c_str());
            }
            window->DC.CursorPos.x += context->FontSize * 2;
            ImGui::TextUnformatted(fmt::format("Leaf {}", uuid).c_str());
            return RetNodeState::CLOSED;
        }

        auto SceneGraph::add_inner_node(std::string uuid) -> RetNodeState
        {
            f32 ARROW_SCALE = 1.0f;
            ImRect cell_bounds = get_cell_bounds();
            ImVec2 arrow_pos = ImVec2(
                window->DC.CursorPos.x + context->Style.CellPadding.x,
                window->DC.CursorPos.y //+ context->Style.CellPadding.y
            );
            ImGuiID id = window->GetID(fmt::format("bounds_inner_{}", uuid).c_str());
            bool hovered, held, pressed;
            pressed = ImGui::ButtonBehavior(cell_bounds, id, &hovered, &held, ImGuiButtonFlags_AllowOverlap | ImGuiButtonFlags_PressedOnClick);

            auto const elem_id = window->GetID(uuid.c_str());
            bool elem_state = window->DC.StateStorage->GetBool(elem_id);
            if(hovered)
            {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(ImGuiCol(ImGuiCol_ButtonHovered)), table->CurrentColumn);
            }
            if(pressed)
            {
                window->DC.StateStorage->SetBool(elem_id, !elem_state);
                DEBUG_MSG(fmt::format("Switching inner node {} states {} -> {}", uuid, elem_state, !elem_state));
                elem_state = !elem_state;
            }
            ImGuiDir arrow_dir = elem_state ? ImGuiDir_Down : ImGuiDir_Right;
            ImGui::RenderArrow(window->DrawList, arrow_pos, ImGui::GetColorU32(ImGuiCol(ImGuiCol_Text)), arrow_dir, ARROW_SCALE);
            window->DC.CursorPos.x += context->FontSize * 2;
            ImGui::TextUnformatted(fmt::format("Inner {}", uuid).c_str());
            return elem_state ? RetNodeState::OPEN : RetNodeState::CLOSED;
        }

        auto SceneGraph::add_node(NodeType type, std::string uuid) -> RetNodeState
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();

            switch(type)
            {
                case NodeType::INNER: 
                    return add_inner_node(uuid);
                case NodeType::LEAF:
                    return add_leaf_node(uuid);
                default:
                    return RetNodeState::ERROR;
            }
        }
    }
}