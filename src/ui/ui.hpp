#pragma once

#include <cmath>
#include <iostream>
#include <string>
#include <stack>
#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>

#include "../window.hpp"
#include "../scene/scene.hpp"
#include "../timberdoodle.hpp"

#include "scene_graph.hpp"
using namespace tido::types;
using namespace tido::ui;

struct UIEngine
{
    bool widget_settings = false;
    bool widget_renderer_statistics = false;
    bool widget_scene_hierarchy = false;
    tido::ui::SceneGraph scene_graph{};

    UIEngine(Window &window)
    {
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(window.glfw_handle, true);
    }

    void main_update(Settings &settings, Scene const & scene)
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
                ImGui::EndMenu();
            }
            ImGui::EndMainMenuBar();
        }
        if (widget_settings)
        {
            if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoCollapse))
            {
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
        if(widget_scene_hierarchy)
        {
            static constexpr u32 max_child_count = 6;
            static constexpr u32 max_parent_count = 10;
            struct StackEntry
            {
                RenderEntityId id;
                bool last_child;
                bool first_child;
                u32 last_in_row;
            };
            // ROOT - first - indent
            //  INNER - first - indent
            //    LEAF - first - don't indent
            //    INNER - last  - pop indent 
            //      LEAF - first/last - indent, pop_indent
            //  INNER
            //  INNER
            //  INNER - last
            std::stack<StackEntry> entity_graph_stack;
            u32 counter = 0;
            if(ImGui::Begin("Scene Hierarchy", nullptr, ImGuiWindowFlags_NoCollapse))
            {
                scene_graph.begin();
                entity_graph_stack.push({scene._scene_file_manifest.at(0).root_render_entity, true, false});
                while(!entity_graph_stack.empty())
                {
                    auto const top_stack_entry = entity_graph_stack.top();
                    entity_graph_stack.pop();

                    auto const &entity = *scene._render_entities.slot(top_stack_entry.id);
                    if(top_stack_entry.first_child) 
                    {
                        scene_graph.add_level();
                    }
                    NodeType type = entity.first_child.has_value() ? NodeType::INNER : NodeType::LEAF;
                    RetNodeState const result = scene_graph.add_node(type, entity.name);
                    counter++;
                    if(result != RetNodeState::OPEN) { continue; }
                    // Last child but we have no more children
                    if(top_stack_entry.last_child && type == NodeType::LEAF) 
                    {
                        for(u32 pop_count = 0; pop_count < top_stack_entry.last_in_row; pop_count++)
                        {
                            scene_graph.remove_level();
                        }
                    } 
                    // Last child but we have more children
                    else if(top_stack_entry.last_child)
                    {
                        RenderEntityId curr_child_index = entity.first_child.value();
                        bool first_child = true;
                        while(true)
                        {
                            auto const * curr_child = scene._render_entities.slot(curr_child_index);
                            bool is_last_child = !curr_child->next_sibling.has_value();
                            entity_graph_stack.push({curr_child_index, first_child, is_last_child, is_last_child ? top_stack_entry.last_in_row + 2 : 0});
                            if(is_last_child) { break; }
                            else { curr_child_index = curr_child->next_sibling.value(); }
                            first_child = false;
                        }
                    }
                    // Normal child
                    else
                    {
                        if(type == NodeType::LEAF) { continue; }
                        RenderEntityId curr_child_index = entity.first_child.value();
                        bool first_child = true;
                        while(true)
                        {
                            auto const * curr_child = scene._render_entities.slot(curr_child_index);
                            bool is_last_child = curr_child->next_sibling.has_value();
                            entity_graph_stack.push({curr_child_index, first_child, is_last_child, 0});
                            if(is_last_child) { break; }
                            else { curr_child_index = curr_child->next_sibling.value(); }
                            first_child = false;
                        }
                    }
                }
                scene_graph.end();

                ImGui::End();
            }
            ImGui::ShowDemoWindow();
        }
        ImGui::Render();
    }
};