#include "property_viewer.hpp"
#include <implot.h>

#include "helpers.hpp"

namespace tido
{
    namespace ui
    {
        PropertyViewer::PropertyViewer(daxa::ImGuiRenderer * renderer, std::vector<daxa::ImageId> const * icons, daxa::SamplerId linear_sampler)
            : renderer{renderer},
              icons{icons},
              linear_sampler{linear_sampler}
        {
        }

        void PropertyViewer::render(RenderInfo const & info)
        {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowTitleAlign, {0.5f, 0.5f});
            ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0, 6});
            ImGui::SetNextWindowSizeConstraints(ImVec2(250, 0), ImVec2(FLT_MAX, FLT_MAX));
            ImGui::Begin("Selector widget", nullptr, ImGuiWindowFlags_NoCollapse);
            auto flags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
            // Left Icon menu
            ImGui::BeginChild("property selector", ImVec2(26, 0), false, flags);
            auto * window = ImGui::GetCurrentWindow();
            {
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, {2, 2});
                ImGui::PushID("Selector Icons");
                // Dummy is to introduce the vertical offset to the left icon bar
                ImGui::Dummy({0, 3});
                for (i32 i = 0; i < selector_icons.size(); i++)
                {
                    if (i == selected)
                    {
                        ImGui::PushStyleColor(ImGuiCol_Button, bg_1);
                        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, bg_1);
                        ImGui::PushStyleColor(ImGuiCol_ButtonActive, bg_1);
                    }
                    else { ImGui::PushStyleColor(ImGuiCol_Button, bg_0); }
                    bool got_selected = {};
                    if (ImGui::ImageButton(
                            std::to_string(i).c_str(),
                            renderer->create_texture_id({
                                .image_view_id = icons->at(s_cast<u32>(selector_icons.at(i))).default_view(),
                                .sampler_id = std::bit_cast<daxa::SamplerId>(linear_sampler),
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
            }
            ImGui::SameLine();

            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.5f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {2, 2});
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {2, 6});
            ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_1);
            // Sun settings
            if (selected == 0)
            {
                ImGui::BeginChild("Sun settings", {0, 0}, false, ImGuiWindowFlags_NoScrollbar);
                {
                    auto sun_settings_content = [&info]()
                    {
                        if (ImGui::CollapsingHeader("Sun Settings"))
                        {
                            ImGui::Indent(12);
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_3);

                            f32vec3 const sun_dir = {
                                info.sky_settings->sun_direction.x,
                                info.sky_settings->sun_direction.y,
                                info.sky_settings->sun_direction.z};
                            f32 const angle_y_rad = glm::acos(sun_dir.z);
                            f32 const angle_x_rad = glm::atan(sun_dir.y / sun_dir.x);
                            f32 offset = 0.0f;
                            if (sun_dir.x < 0.0f)
                            {
                                offset += sun_dir.y < 0 ? -180.0f : 180.0f;
                            }
                            f32 angle_y_deg = glm::degrees(angle_y_rad);
                            f32 angle_x_deg = glm::degrees(angle_x_rad) + offset;
                            angle_x_deg += angle_x_deg < 0 ? 360.0f : 0.0f;
                            auto const horizontal_max_width = ImGui::GetContentRegionAvail().x / 4;
                            better_drag_float({"Angle X", &angle_x_deg, 0.5f, 0.0f, 360.0f, "%.1f°", horizontal_max_width, -10, true});
                            better_drag_float({"Angle Y", &angle_y_deg, 0.5f, 0.0f, 180.0f, "%.1f°", horizontal_max_width, -10, true});
                            info.sky_settings->sun_direction =
                                {
                                    daxa_f32(glm::cos(glm::radians(angle_x_deg)) * glm::sin(glm::radians(angle_y_deg))),
                                    daxa_f32(glm::sin(glm::radians(angle_x_deg)) * glm::sin(glm::radians(angle_y_deg))),
                                    daxa_f32(glm::cos(glm::radians(angle_y_deg))),
                                };
                            ImGui::PopStyleColor();
                            ImGui::Unindent(12);
                        }
                    };

                    auto atmsphere_settings_content = [&info]
                    {
                        struct LayerSettingsContentCallable
                        {
                            i32 layer_idx = {};
                            DensityProfileLayer * layers = {};
                            LayerSettingsContentCallable(i32 layer_idx, DensityProfileLayer * layers)
                                : layer_idx{layer_idx},
                                  layers{layers}
                            {
                            }
                            void operator()()
                            {
                                ImGui::PushID(layer_idx);
                                if (ImGui::CollapsingHeader(std::format("Layer {}", std::to_string(layer_idx)).c_str()))
                                {
                                    ImGui::Indent(12);
                                    auto const horizontal_max_width = ImGui::GetContentRegionAvail().x / 4;
                                    auto & mie_layer = layers[layer_idx];
                                    ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_5);
                                    better_drag_float({"width", &mie_layer.layer_width, 0.3f, 0.0f, 100.0f, "%.1f", horizontal_max_width, -20});
                                    better_drag_float({"const", &mie_layer.const_term, 0.1f, -10.0f, 10.0f, "%.3f", horizontal_max_width, -20});
                                    better_drag_float({"lin", &mie_layer.lin_term, 0.1f, -10.0f, 0.0f, "%.3f", horizontal_max_width, -20});
                                    better_drag_float({"exp", &mie_layer.exp_term, 0.01f, 0.0f, 2.0f, "%.3f", horizontal_max_width, -20});
                                    better_drag_float({"scale", &mie_layer.exp_scale, 0.01f, -2.0f, 2.0f, "%.3f", horizontal_max_width, -20});
                                    ImGui::PopStyleColor();
                                    ImGui::Unindent(12);
                                }
                                ImGui::PopID();
                            };
                        };

                        auto plot_layer = [](DensityProfileLayer * layers)
                        {
                            ImPlot::PushStyleColor(ImPlotCol_FrameBg, bg_4);
                            if (ImPlot::BeginPlot("##lines", {ImGui::GetContentRegionAvail().x - 20, 0}))
                            {
                                auto plot_layer_info = [](DensityProfileLayer const & layer, f32 layer_offset, ImVec4 const & col)
                                {
                                    const i32 samples = 1000;
                                    std::array<f32, samples> x_values = {};
                                    std::array<f32, samples> y_values = {};
                                    for (i32 i = 0; i < x_values.size(); i++)
                                    {
                                        auto const h = layer_offset + (s_cast<f32>(i) / samples) * layer.layer_width;
                                        auto const val = glm::max(layer.exp_term * std::exp(layer.exp_scale * h) + layer.lin_term * h + layer.const_term, 0.0f);
                                        x_values[i] = val;
                                        y_values[i] = h;
                                    }
                                    ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::ColorConvertFloat4ToU32(col));
                                    ImPlot::PlotLine("", x_values.data(), y_values.data(), samples);
                                    ImPlot::PopStyleColor();
                                };
                                plot_layer_info(layers[0], 0.0f, select_blue_1);
                                plot_layer_info(layers[1], layers[0].layer_width, select_orange_1);
                                ImPlot::EndPlot();
                            }
                            ImPlot::PopStyleColor();
                        };
                        if (ImGui::CollapsingHeader("Atmosphere Settings"))
                        {
                            ImGui::Indent(4);
                            std::string layer_name = {};
                            DensityProfileLayer * layers = {};
                            auto layer_settings_content = [&info, &plot_layer, &layer_name, &layers]
                            {
                                ImGui::PushID(layer_name.c_str());
                                if (ImGui::CollapsingHeader(std::format("{} Settings", layer_name).c_str()))
                                {
                                    ImGui::Indent(8);
                                    plot_layer(layers);
                                    draw_with_bg_rect(LayerSettingsContentCallable(0, layers), 12, bg_5);
                                    ImGui::Dummy({0, 1});
                                    draw_with_bg_rect(LayerSettingsContentCallable(1, layers), 12, bg_5);
                                    ImGui::Dummy({0, 1});
                                    ImGui::Unindent(8);
                                }
                                ImGui::PopID();
                            };

                            ImGui::Dummy({2, 0});
                            ImGui::SameLine();
                            layer_name = "Rayleigh";
                            layers = info.sky_settings->rayleigh_density;
                            draw_with_bg_rect(layer_settings_content, 12, bg_4);
                            ImGui::Dummy({0, 1});

                            ImGui::Dummy({2, 0});
                            ImGui::SameLine();
                            layer_name = "Mie";
                            layers = info.sky_settings->mie_density;
                            draw_with_bg_rect(layer_settings_content, 12, bg_4);
                            ImGui::Dummy({0, 1});

                            ImGui::Dummy({2, 0});
                            ImGui::SameLine();
                            layer_name = "Absorption";
                            layers = info.sky_settings->absorption_density;
                            draw_with_bg_rect(layer_settings_content, 12, bg_4);
                            ImGui::Dummy({0, 1});

                            ImGui::Unindent(4);
                        }
                    };

                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.0, 0.0, 0.0, 0.0});
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, {0.0, 0.0, 0.0, 0.0});
                    ImGui::PushStyleColor(ImGuiCol_Header, {0.0, 0.0, 0.0, 0.0});
                    ImGui::Dummy({0, 1});
                    ImGui::Dummy({2, 0});
                    ImGui::SameLine();
                    draw_with_bg_rect(sun_settings_content, 8, bg_3);

                    ImGui::Dummy({0, 1});
                    ImGui::Dummy({2, 0});
                    ImGui::SameLine();
                    draw_with_bg_rect(atmsphere_settings_content, 8, bg_3);

                    ImGui::PopStyleColor(3);
                }
                ImGui::EndChild();
            }
            if (selected == 1)
            {
                ImGui::BeginChild("Sun settings", {0, 0}, false, ImGuiWindowFlags_NoScrollbar);
                {
                    auto camera_settings = [&info]()
                    {
                        if (ImGui::CollapsingHeader("Camera"))
                        {
                            ImGui::Indent(12);
                            auto const horizontal_max_width = ImGui::GetContentRegionAvail().x / 2.5f;
                            auto & post = info.post_settings;
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_3);
                            better_drag_float({"exposure bias", &post->exposure_bias, 0.1f, 0.01f, 10.0f, "%.1f", horizontal_max_width, -20});
                            better_drag_float({"sensor sensitivity", &post->sensor_sensitivity, 100.0f, 100.0f, 7000.0f, "%.0f", horizontal_max_width, -20});
                            better_drag_float({"calibration", &post->calibration, 0.1f, 1.0f, 30.0f, "%.1f", horizontal_max_width, -20});
                            better_drag_float({"adaption speed", &post->luminance_adaption_tau, 0.05f, 0.1f, 10.0f, "%.2f", horizontal_max_width, -20});
                            ImGui::PopStyleColor();
                            ImGui::Unindent(12);
                        }
                    };
                    auto histogram_settings = [&info]()
                    {
                        if (ImGui::CollapsingHeader("Histogram"))
                        {
                            ImGui::Indent(12);
                            auto const horizontal_max_width = ImGui::GetContentRegionAvail().x / 2.5f;
                            auto & post = info.post_settings;
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_3);
                            better_drag_float({"min lum log2", &post->min_luminance_log2, 0.1f, -12.0f, 12.0f, "%.1f", horizontal_max_width, -20});
                            better_drag_float({"max lum log2", &post->max_luminance_log2, 0.1f, -12.0f, 12.0f, "%.1f", horizontal_max_width, -20});
                            post->max_luminance_log2 = glm::max(post->max_luminance_log2, post->min_luminance_log2 + 0.1f);
                            ImGui::PopStyleColor();
                            ImGui::Unindent(12);
                        }
                    };
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, {0.0, 0.0, 0.0, 0.0});
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive, {0.0, 0.0, 0.0, 0.0});
                    ImGui::PushStyleColor(ImGuiCol_Header, {0.0, 0.0, 0.0, 0.0});
                    ImGui::Dummy({0, 1});
                    ImGui::Dummy({2, 0});
                    ImGui::SameLine();
                    draw_with_bg_rect(camera_settings, 8, bg_3);

                    ImGui::Dummy({0, 1});
                    ImGui::Dummy({2, 0});
                    ImGui::SameLine();
                    draw_with_bg_rect(histogram_settings, 8, bg_3);
                    ImGui::PopStyleColor(3);
                }
                ImGui::EndChild();
            }
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(3);

            ImGui::End();
            ImGui::PopStyleVar(3);
        }
    } // namespace ui
} // namespace tido