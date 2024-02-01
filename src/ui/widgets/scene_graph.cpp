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
            icon_size{16.0f},
            icon_text_spacing{3.0f},
            indent{16.0f},
            renderer{renderer},
            icons{icons},
            linear_sampler(linear_sampler)
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
                table->RowMinHeight
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
            ImGui::Indent(indent);
            window->DC.TreeDepth++;
        }

        void SceneGraph::remove_level()
        {
            ImGui::Unindent(indent);
            window->DC.TreeDepth--;
        }

        auto SceneGraph::add_leaf_node(std::string uuid, ICONS icon) -> RetNodeState
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();

            ImRect cell_bounds = get_cell_bounds();
            f32 const real_cell_max_y = std::max(cell_bounds.Max.y, cell_bounds.Min.y + icon_size + context->Style.CellPadding.y * 2.0f);
            ImRect real_cell_bounds = ImRect(cell_bounds.Min, ImVec2(cell_bounds.Max.x,real_cell_max_y));

            ImVec2 font_size = ImGui::CalcTextSize("x");
            ImGuiID const elem_id = window->GetID(fmt::format("bounds_elem_{}", uuid).c_str());

            State const elem_state = button_like_behavior(real_cell_bounds, elem_id);
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
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + icon_size);
            ImGui::Image(renderer->create_texture_id({
                    .image_view_id = icons->at(s_cast<u32>(icon)).default_view(),
                    .sampler_id = linear_sampler
                }), 
                ImVec2(icon_size, icon_size),
                ImVec2(0.0, 1.0), ImVec2(1.0, 0.0),
                icon_to_color(icon)
            );
            const f32 prev_line_window_rel_pos_x = window->DC.CursorPosPrevLine.x - window->Pos.x + window->Scroll.x;
            ImGui::SameLine(prev_line_window_rel_pos_x + icon_text_spacing);
            ImGui::TextUnformatted(fmt::format("{}", uuid).c_str());
            return RetNodeState::CLOSED;
        }

        auto SceneGraph::add_inner_node(std::string uuid, ICONS icon) -> RetNodeState
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();

            ImRect cell_bounds = get_cell_bounds();

            ImVec2 font_size = ImGui::CalcTextSize("x");
            ImRect icon_bounds = ImRect(
                ImVec2( window->DC.CursorPos.x, cell_bounds.Min.y),
                ImVec2( 
                    window->DC.CursorPos.x + icon_size, 
                    std::max(cell_bounds.Max.y, cell_bounds.Min.y + icon_size)
                )
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
                ImVec2(icon_size, icon_size),
                ImVec2(0.0, 1.0), ImVec2(1.0, 0.0),
                icon_to_color(arrow_icon)
            );
            if(icon != ICONS::SIZE)
            {
                ImGui::SameLine();
                ImGui::Image(renderer->create_texture_id({
                        .image_view_id = icons->at(s_cast<u32>(icon)).default_view(),
                        .sampler_id = linear_sampler
                    }), 
                    ImVec2(icon_size, icon_size),
                    ImVec2(0.0, 1.0), ImVec2(1.0, 0.0),
                    icon_to_color(icon)
                );
            }
            const f32 prev_line_window_rel_pos_x = window->DC.CursorPosPrevLine.x - window->Pos.x + window->Scroll.x;
            ImGui::SameLine(prev_line_window_rel_pos_x + icon_text_spacing);
            ImGui::TextUnformatted(fmt::format("{}", uuid).c_str());
            return component_state ? RetNodeState::OPEN : RetNodeState::CLOSED;
        }

        auto SceneGraph::add_meshgroup_node(RenderEntity const & entity, Scene const & scene) -> RetNodeState
        {
            std::string const meshgroup_uuid = fmt::format("{}", entity.name);
            RetNodeState state = add_inner_node(meshgroup_uuid, ICONS::MESHGROUP);
            if(state != RetNodeState::OPEN) { return RetNodeState::CLOSED; }
            MeshGroupManifestEntry const & meshgroup_manifest_entry = scene._mesh_group_manifest.at(entity.mesh_group_manifest_index.value());

            add_level();
            for(u32 mesh_idx = 0; mesh_idx < meshgroup_manifest_entry.mesh_count; mesh_idx++)
            {
                std::string const mesh_uuid = fmt::format("{} - mesh {}", entity.name, mesh_idx);
                RetNodeState inner_node_state = add_inner_node(mesh_uuid, ICONS::MESH);
                MeshManifestEntry const & mesh_manifest_entry = 
                    scene._mesh_manifest.at(meshgroup_manifest_entry.mesh_manifest_indices.at(mesh_idx));
                ImGui::SameLine();
                const u32 meshlet_count = mesh_manifest_entry.runtime->meshlet_count;
                const char plural_ending = meshlet_count > 1 ? 's' : ' ';
                ImGui::TextColored(ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled), "( %d meshlet%c )", meshlet_count, plural_ending);
                if(inner_node_state == RetNodeState::OPEN)
                {
                    MaterialManifestEntry const & material_manifest_entry = scene._material_manifest.at(mesh_manifest_entry.runtime.value().material_index);
                    std::string const material_uuid = fmt::format("{}", material_manifest_entry.name);
                    add_level();
                    add_leaf_node(material_uuid, ICONS::MATERIAL);
                    remove_level();
                }
            }
            remove_level();
            return RetNodeState::CLOSED;
        }

        auto SceneGraph::add_node(RenderEntity const & entity, Scene const & scene) -> RetNodeState
        {

            std::string const uuid = fmt::format("{}", entity.name);
            switch(entity.type)
            {
                case EntityType::ROOT: [[fallthrough]];
                case EntityType::TRANSFORM:
                    return add_inner_node(uuid, ICONS::COLLECTION);
                case EntityType::MESHGROUP: 
                    return add_meshgroup_node(entity, scene);
                case EntityType::CAMERA:
                    return add_leaf_node(uuid, ICONS::CAMERA);
                case EntityType::LIGHT:
                    return add_leaf_node(uuid, ICONS::LIGHT);
                default:
                    return RetNodeState::ERROR;
            }
        }
    }
}