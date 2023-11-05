#pragma once

#include <string>
#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui.h>

#include "window.hpp"
#include "sandbox.hpp"

namespace tido
{
    struct UIEngine
    {
        bool widget_settings = false;
        bool widget_renderer_statistics = false;
        UIEngine(Window &window)
        {
            ImGui::CreateContext();
            ImGui_ImplGlfw_InitForVulkan(window.glfw_handle, true);
        }
        void main_update(Settings &settings)
        {
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            if (ImGui::BeginMainMenuBar())
            {
                if (ImGui::BeginMenu("Widgets"))
                {
                    ImGui::MenuItem("Settings", NULL, &widget_settings);
                    ImGui::MenuItem("Renderer Statistics", NULL, &widget_renderer_statistics);
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
            ImGui::Render();
        }
    };
}