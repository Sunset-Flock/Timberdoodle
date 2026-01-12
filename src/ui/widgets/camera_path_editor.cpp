#include "camera_path_editor.hpp"
#include "../../json_handler.hpp"

namespace tido
{
    namespace ui
    {

        CameraPathEditor::CameraPathEditor(daxa::ImGuiRenderer * renderer)
            : renderer{renderer}
        {

        }


        void CameraPathEditor::render(RenderContext & render_context, CinematicCamera & camera, CameraController & main_camera)
        {

            if (ImGui::Begin("Camera Path Editor", nullptr, ImGuiWindowFlags_NoCollapse))
            {
                static ImGuiTableFlags flags =
                    ImGuiTableFlags_BordersOuterV |
                    ImGuiTableFlags_BordersOuterH |
                    ImGuiTableFlags_Resizable |
                    ImGuiTableFlags_RowBg |
                    ImGuiTableFlags_NoBordersInBody | 
                    ImGuiTableFlags_ScrollY;

                ImGui::BeginTable("Current keyframes", 2, flags, ImVec2(0.0f, 600.0f));
                ImGui::TableSetupColumn("Index", ImGuiTableColumnFlags_WidthFixed);
                ImGui::TableSetupColumn("Info", ImGuiTableColumnFlags_WidthStretch);

                ImGuiListClipper clipper;
                clipper.Begin(s_cast<i32>(camera.path_keyframes.size()));

                while (clipper.Step())
                {
                    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i)
                    {
                        ImGui::TableNextRow(ImGuiTableRowFlags_None, 10);
                        ImGui::TableNextColumn();

                        ImGui::PushStyleColor(ImGuiCol_Header, ImGuiCol_TableRowBg);
                        ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGuiCol_TableRowBg);
                        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImGuiCol_TableRowBg);
                        if(ImGui::Selectable(fmt::format("{}", i).c_str(), (i == selected_index), ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowItemOverlap))
                        {
                            selected_index = i;
                        }
                        ImGui::PopStyleColor(3);
                        ImGui::TableNextColumn();

                        if(selected_index == i)
                        {
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(select_blue_1), 0);
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, ImGui::GetColorU32(select_blue_1), 1);
                        }

                        CameraAnimationKeyframe & current_keyframe = camera.path_keyframes.at(i);
                        f32vec3 & keyframe_pos = current_keyframe.position;

                        auto pos_format_text = fmt::format("x : {}, y : {}, z : {}", keyframe_pos.x, keyframe_pos.y, keyframe_pos.z);
                        ImGui::PushID(i);
                        ImGui::Text("Position");
                        ImGui::SameLine();
                        ImGui::DragFloat3("##", &keyframe_pos.x);

                        ImGui::Text("Keyframe time");
                        ImGui::SameLine();
                        ImGui::InputFloat("##", &current_keyframe.transition_time);
                        ImGui::PopID();
                    }
                }
                clipper.End();
                ImGui::EndTable();
            }

            auto const camera_info = main_camera.make_camera_info(render_context.render_data.settings);
            auto const view_quat = glm::quat_cast(camera_info.view);
            auto const position = camera_info.position;

            ImGui::InputFloat("Transition time", &current_transition_time);

            ImGui::Text("From current Camera");
            if(ImGui::Button("Add keypoint"))
            {
                camera.path_keyframes.push_back({ view_quat, position, current_transition_time });
            }
            ImGui::BeginDisabled(selected_index == -1);
            if(ImGui::Button("Append Keypoint"))
            {
                camera.path_keyframes.insert(
                    camera.path_keyframes.begin() + selected_index + 1,
                    { view_quat, position, current_transition_time }
                );
            }
            ImGui::SameLine();
            if(ImGui::Button("Prepend Keypoint"))
            {
                camera.path_keyframes.insert(
                    camera.path_keyframes.begin() + selected_index,
                    { view_quat, position, current_transition_time }
                );
            }
            if(ImGui::Button("Replace selected")) 
            {
                camera.path_keyframes.at(selected_index) = { view_quat, position, current_transition_time };
            }
            if(ImGui::Button("Remove selected"))
            {
                camera.path_keyframes.erase(camera.path_keyframes.begin() + selected_index);
                selected_index = -1;
            }
            ImGui::EndDisabled();

            if(ImGui::Button("Export Path"))
            {
                export_camera_animation(std::filesystem::path("settings/camera/exported_path.json"), camera.path_keyframes);
            }

            ImGui::Checkbox("Debug draw path", &debug_draw_path);

            if (debug_draw_path && camera.path_keyframes.size() > 1) {
                for(i32 keyframe_index = 0; keyframe_index < camera.path_keyframes.size(); ++keyframe_index)
                {
                    static constexpr i32 DEBUG_DRAW_LINE_COUNT = 20;

                    // std::array<u32, 4> keyframe_indices = {
                    //     (keyframe_index + (camera.path_keyframes.size()) - 1u) % camera.path_keyframes.size(),
                    //     (keyframe_index + (camera.path_keyframes.size())) % camera.path_keyframes.size(),
                    //     (keyframe_index + 1) % camera.path_keyframes.size(),
                    //     (keyframe_index + 2) % camera.path_keyframes.size(),
                    // };

                    std::array<u32, 4> keyframe_indices = {
                        keyframe_index > 1 ? s_cast<u32>(keyframe_index - 1) : 0,
                        s_cast<u32>(keyframe_index),
                        std::min(s_cast<u32>(keyframe_index + 1), s_cast<u32>(camera.path_keyframes.size() - 1)),
                        std::min(s_cast<u32>(keyframe_index + 2), s_cast<u32>(camera.path_keyframes.size() - 1)),
                    };

                    f32vec3 last_position = camera.path_keyframes.at(keyframe_index).position;
                    render_context.gpu_context->shader_debug_context.sphere_draws.cpu_draws.push_back(
                        ShaderDebugSphereDraw{
                            .position = daxa_f32vec3(last_position.x, last_position.y, last_position.z),
                            .radius = 0.2f,
                            .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
                            .color = keyframe_index == selected_index ? daxa_f32vec3(1.0f, 1.0f, 0.0f) : daxa_f32vec3(0.0f, 1.0f, 0.0f),
                        }
                    );

                    for (i32 i = 1; i <= DEBUG_DRAW_LINE_COUNT; i++)
                    {

                        auto t = float(i) / DEBUG_DRAW_LINE_COUNT;
                        f32vec3 line_position;
                        f32vec3 velocity;

                        catmull_rom(
                            line_position,
                            velocity,
                            t,
                            camera.path_keyframes.at(keyframe_indices[0]).position,
                            camera.path_keyframes.at(keyframe_indices[1]).position,
                            camera.path_keyframes.at(keyframe_indices[2]).position,
                            camera.path_keyframes.at(keyframe_indices[3]).position
                        );

                        render_context.gpu_context->shader_debug_context.line_draws.cpu_draws.push_back(
                            ShaderDebugLineDraw{
                                .start = daxa_f32vec3(last_position.x, last_position.y, last_position.z),
                                .end = daxa_f32vec3(position.x, position.y, position.z),
                                .color = daxa_f32vec3(0.0, 0.0, 1.0),
                                .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,
                            });
                        last_position = line_position;
                    }
                }
            }

            ImGui::End();
        }
    }
}