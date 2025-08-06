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

        void PropertyViewer::render(SceneInterfaceState & scene_interface, Scene & scene, RenderContext & render_context)
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
                    auto sun_settings_content = [&render_context]()
                    {
                        if (ImGui::CollapsingHeader("Sun Settings"))
                        {
                            ImGui::Indent(12);
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_3);

                            f32vec3 const sun_dir = {
                                render_context.render_data.sky_settings.sun_direction.x,
                                render_context.render_data.sky_settings.sun_direction.y,
                                render_context.render_data.sky_settings.sun_direction.z};
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
                            better_drag_float({"Angle X", &angle_x_deg, 0.5f, 0.1f, 360.0f, "%.1f°", horizontal_max_width, -10, true});
                            better_drag_float({"Angle Y", &angle_y_deg, 0.5f, 0.1f, 180.0f, "%.1f°", horizontal_max_width, -10, true});
                            static float sun_speed = 0.0f;
                            ImGui::DragFloat("sun speed", &sun_speed);
                            angle_y_deg += sun_speed;
                            render_context.render_data.sky_settings.sun_direction =
                                {
                                    daxa_f32(glm::cos(glm::radians(angle_x_deg)) * glm::sin(glm::radians(angle_y_deg))),
                                    daxa_f32(glm::sin(glm::radians(angle_x_deg)) * glm::sin(glm::radians(angle_y_deg))),
                                    daxa_f32(glm::cos(glm::radians(angle_y_deg))),
                                };
                            ImGui::PopStyleColor();
                            ImGui::Unindent(12);
                        }
                    };

                    auto atmsphere_settings_content = [&render_context]
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
                            auto layer_settings_content = [&render_context, &plot_layer, &layer_name, &layers]
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
                            layers = render_context.render_data.sky_settings.rayleigh_density;
                            draw_with_bg_rect(layer_settings_content, 12, bg_4);
                            ImGui::Dummy({0, 1});

                            ImGui::Dummy({2, 0});
                            ImGui::SameLine();
                            layer_name = "Mie";
                            layers = render_context.render_data.sky_settings.mie_density;
                            draw_with_bg_rect(layer_settings_content, 12, bg_4);
                            ImGui::Dummy({0, 1});

                            ImGui::Dummy({2, 0});
                            ImGui::SameLine();
                            layer_name = "Absorption";
                            layers = render_context.render_data.sky_settings.absorption_density;
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
                    auto camera_settings = [&render_context]()
                    {
                        if (ImGui::CollapsingHeader("Camera"))
                        {
                            ImGui::Indent(12);
                            auto const horizontal_max_width = ImGui::GetContentRegionAvail().x / 2.5f;
                            auto & post = render_context.render_data.postprocess_settings;
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_3);
                            better_drag_float({"exposure bias", &post.exposure_bias, 0.1f, 0.01f, 10.0f, "%.1f", horizontal_max_width, -20});
                            better_drag_float({"sensor sensitivity", &post.sensor_sensitivity, 100.0f, 100.0f, 7000.0f, "%.0f", horizontal_max_width, -20});
                            better_drag_float({"calibration", &post.calibration, 0.1f, 1.0f, 30.0f, "%.1f", horizontal_max_width, -20});
                            better_drag_float({"adaption speed", &post.luminance_adaption_tau, 0.05f, 0.1f, 10.0f, "%.2f", horizontal_max_width, -20});
                            ImGui::PopStyleColor();
                            ImGui::Unindent(12);
                        }
                    };
                    auto histogram_settings = [&render_context]()
                    {
                        if (ImGui::CollapsingHeader("Histogram"))
                        {
                            ImGui::Indent(12);
                            auto const horizontal_max_width = ImGui::GetContentRegionAvail().x / 2.5f;
                            auto & post = render_context.render_data.postprocess_settings;
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_3);
                            better_drag_float({"min lum log2", &post.min_luminance_log2, 0.1f, -12.0f, 12.0f, "%.1f", horizontal_max_width, -20});
                            better_drag_float({"max lum log2", &post.max_luminance_log2, 0.1f, -12.0f, 12.0f, "%.1f", horizontal_max_width, -20});
                            better_drag_float({"clip low", &post.auto_exposure_histogram_clip_lo, 0.1f, 0.0f, 1.0f, "%.2f", horizontal_max_width, -20});
                            better_drag_float({"clip high", &post.auto_exposure_histogram_clip_hi, 0.1f, 0.0f, 1.0f, "%.2f", horizontal_max_width, -20});
                            post.auto_exposure_histogram_clip_lo = glm::min(post.auto_exposure_histogram_clip_lo, post.auto_exposure_histogram_clip_hi);
                            post.auto_exposure_histogram_clip_hi = glm::max(post.auto_exposure_histogram_clip_lo, post.auto_exposure_histogram_clip_hi);
                            post.max_luminance_log2 = glm::max(post.max_luminance_log2, post.min_luminance_log2 + 0.1f);
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
            if (selected == 2)
            {
                if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && ImGui::IsKeyDown(ImGuiKey_LeftShift) && !ImGui::IsAnyItemHovered())
                {
                    scene_interface.picked_entity = render_context.general_readback.hovered_entity;
                    scene_interface.picked_mesh_in_meshgroup = render_context.general_readback.hovered_mesh_in_meshgroup;
                    scene_interface.picked_mesh = render_context.general_readback.hovered_mesh;
                    scene_interface.picked_meshlet_in_mesh = render_context.general_readback.hovered_meshlet_in_mesh;
                    scene_interface.picked_triangle_in_meshlet = render_context.general_readback.hovered_triangle_in_meshlet;
                }
                ImGui::BeginChild("Selected Geometry Info", {0, 0}, false, ImGuiWindowFlags_NoScrollbar);
                {
                    auto camera_settings = [&]()
                    {
                        ImGui::SeparatorText("Picked Geometry");
                        {
                            ImGui::Indent(12);
                            auto const horizontal_max_width = ImGui::GetContentRegionAvail().x / 2.5f;
                            auto & post = render_context.render_data.postprocess_settings;
                            ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_3);

                            auto modes = std::array{
                                "ENTITY", // MARK_SELECTED_MODE_ENTITY
                                "MESH", // MARK_SELECTED_MODE_MESH
                                "MESHLET", // MARK_SELECTED_MODE_MESHLET
                                "TRIANGLE", // MARK_SELECTED_MODE_TRIANGLE
                            };
                            ImGui::Combo("Selected Mark Mode", &render_context.render_data.selected_mark_mode, modes.data(), modes.size());

                            ImGui::Text(">>>Left Click + Left Shift< To Select<<<");
                            ImGui::Text("Entity: idx:           %i", scene_interface.picked_entity);
                            if (scene_interface.picked_entity != ~0)
                            {

                                auto ent_slot = scene._render_entities.slot_by_index(scene_interface.picked_entity);
    
                                auto mesh_group_manifest_index = scene._render_entities.slot_by_index(scene_interface.picked_entity)->mesh_group_manifest_index.value();
                                auto const & mesh_group = scene._mesh_group_manifest.at(mesh_group_manifest_index);
                                auto const mesh_lod_group_manifest_index = mesh_group.mesh_lod_group_manifest_indices_array_offset + scene_interface.picked_mesh_in_meshgroup;
                                MeshLodGroupManifestEntry const & mesh_lod_group_manifest = scene._mesh_lod_group_manifest[mesh_lod_group_manifest_index];
                                auto const material_idx = mesh_lod_group_manifest.material_index.value_or(0);
                                MaterialManifestEntry const & material_manifest = scene._material_manifest.at(material_idx);
                                ImGui::Text(fmt::format("MeshGroup: idx:        {} \"{}\"", mesh_group_manifest_index, mesh_group.name).c_str());

                                auto const & mesh = scene._mesh_lod_group_manifest[scene_interface.picked_mesh/MAX_MESHES_PER_LOD_GROUP];
                                ImGui::Text(fmt::format("Entiy Position:     X: {}\n"
                                                        "                    Y: {}\n"
                                                        "                    Z: {}", ent_slot->combined_transform[3][0], ent_slot->combined_transform[3][1], ent_slot->combined_transform[3][2]).c_str());
                                ImGui::Text(fmt::format("Mesh: idx:             {}", scene_interface.picked_mesh).c_str());
                                ImGui::Text(fmt::format("Mesh In Meshgroup Idx: {}", scene_interface.picked_mesh_in_meshgroup).c_str());
                                ImGui::Text(fmt::format("Meshlet: idx:          {}", scene_interface.picked_meshlet_in_mesh).c_str());
                                ImGui::Text(fmt::format("Triangle: idx:         {}", scene_interface.picked_triangle_in_meshlet).c_str());
                                ImGui::Text(fmt::format("Material:              {}", material_manifest.name).c_str());
                                ImGui::Text(fmt::format("Material idx:          {}", material_idx).c_str());
                                ImGui::Text(fmt::format("  * alpha_discard_enabled    {}", material_manifest.alpha_discard_enabled).c_str());                    
                                ImGui::Text(fmt::format("  * double_sided             {}", material_manifest.double_sided).c_str());              
                                ImGui::Text(fmt::format("  * blend_enabled            {}", material_manifest.blend_enabled).c_str());              
                                ImGui::Text(fmt::format("  * normal_compressed_bc5_rg {}", material_manifest.normal_compressed_bc5_rg).c_str());                        
                                ImGui::Text(fmt::format("  * is_metal                 {}", material_manifest.is_metal).c_str());          
                            }
                            
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

                    ImGui::PopStyleColor(3);
                }

                // Set Render Data
                render_context.render_data.hovered_entity_index = render_context.general_readback.hovered_entity;
                render_context.render_data.selected_entity_index = scene_interface.picked_entity;
                render_context.render_data.selected_mesh_index = scene_interface.picked_mesh;
                render_context.render_data.selected_meshlet_in_mesh_index = scene_interface.picked_meshlet_in_mesh;
                render_context.render_data.selected_triangle_in_meshlet_index = scene_interface.picked_triangle_in_meshlet;

                ImGui::EndChild();
            }
            ImGui::PopStyleColor();
            ImGui::PopStyleVar(3);

            ImGui::End();
            ImGui::PopStyleVar(3);
        }
    } // namespace ui
} // namespace tido