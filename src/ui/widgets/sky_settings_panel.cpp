#include "sky_settings_panel.hpp"
#include "helpers.hpp"
#include <cmath>
#include <cstdio>
#include <array>
#include <algorithm>
#include <glm/glm.hpp>

// Compact, table-based sky/atmosphere block.
// ImGui::BeginTable carries the density-layer grids, ImDrawList draws the sun dial,
// compass and density-profile charts. No ImPlot dependency.

namespace tido
{
    namespace ui
    {
        namespace
        {
            constexpr ImU32 accent_rayleigh = IM_COL32(94, 165, 214, 255);
            constexpr ImU32 accent_mie = IM_COL32(214, 156, 74, 255);
            constexpr ImU32 accent_absorption = IM_COL32(180, 130, 214, 255);
            constexpr ImU32 accent_sun = IM_COL32(235, 205, 130, 255);

            // ---- density profile: small filled chart, density on X, altitude on Y ----
            void draw_density_profile_chart(DensityProfileLayer const (&layers)[PROFILE_LAYER_COUNT], ImVec2 size, ImU32 color)
            {
                ImVec2 const p0 = ImGui::GetCursorScreenPos();
                ImGui::Dummy(size);
                auto * draw_list = ImGui::GetWindowDrawList();
                f32 const total_width = layers[0].layer_width + layers[1].layer_width;
                if (total_width <= 0.0f) { return; }

                constexpr i32 SAMPLES = 48;
                std::array<f32, SAMPLES + 1> vals{};
                f32 max_val = 1e-5f;
                for (i32 i = 0; i <= SAMPLES; ++i)
                {
                    f32 const h = (s_cast<f32>(i) / SAMPLES) * total_width;
                    auto const & l = (h <= layers[0].layer_width) ? layers[0] : layers[1];
                    vals[i] = std::max(0.0f, l.exp_term * std::exp(l.exp_scale * h) + l.lin_term * h + l.const_term);
                    max_val = std::max(max_val, vals[i]);
                }
                std::array<ImVec2, SAMPLES + 1> poly{};
                for (i32 i = 0; i <= SAMPLES; ++i)
                {
                    f32 const h = (s_cast<f32>(i) / SAMPLES) * total_width;
                    poly[i] = {
                        p0.x + (vals[i] / max_val) * size.x,
                        p0.y + size.y - (h / total_width) * size.y,
                    };
                }
                // One trapezoid per sample: the convex filler fans across the concave curve, and ear
                // clipping trips on the near-coincident vertices a fast decay piles up against the axis.
                ImU32 const fill_color = (color & 0x00FFFFFFu) | 0x22000000u;
                ImDrawListFlags const draw_flags_backup = draw_list->Flags;
                // without this the trapezoids feather their shared edges and seam
                draw_list->Flags &= ~ImDrawListFlags_AntiAliasedFill;
                for (i32 sample = 0; sample < SAMPLES; ++sample)
                {
                    draw_list->AddQuadFilled(
                        {p0.x, poly[sample].y},
                        poly[sample],
                        poly[sample + 1],
                        {p0.x, poly[sample + 1].y},
                        fill_color);
                }
                draw_list->Flags = draw_flags_backup;
                draw_list->AddPolyline(poly.data(), SAMPLES + 1, color, ImDrawFlags_None, 1.6f);
                draw_list->AddLine(p0, {p0.x, p0.y + size.y}, ImGui::GetColorU32(alt_2));
                draw_list->AddLine({p0.x, p0.y + size.y}, {p0.x + size.x, p0.y + size.y}, ImGui::GetColorU32(alt_2));
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.0f km", total_width);
                draw_list->AddText({p0.x + 2.0f, p0.y - 2.0f}, ImGui::GetColorU32(ImGuiCol_TextDisabled), buf);
                draw_list->AddText({p0.x + 2.0f, p0.y + size.y + 2.0f}, ImGui::GetColorU32(ImGuiCol_TextDisabled), "0");
            }

            // ---- one row per density layer: width / const / lin / exp / scale ----
            void draw_density_layer_row(char const * label, DensityProfileLayer & layer, ImU32 accent)
            {
                ImGui::PushID(label);
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(accent), "%s", label);
                ImGui::TableSetColumnIndex(1);
                ImGui::SetNextItemWidth(-1);
                ImGui::DragFloat("##width", &layer.layer_width, 0.3f, 0.0f, 100.0f, "%.1f");
                ImGui::TableSetColumnIndex(2);
                ImGui::SetNextItemWidth(-1);
                ImGui::DragFloat("##const", &layer.const_term, 0.05f, -10.0f, 10.0f, "%.3f");
                ImGui::TableSetColumnIndex(3);
                ImGui::SetNextItemWidth(-1);
                ImGui::DragFloat("##lin", &layer.lin_term, 0.05f, -10.0f, 0.0f, "%.3f");
                ImGui::TableSetColumnIndex(4);
                ImGui::SetNextItemWidth(-1);
                ImGui::DragFloat("##exp", &layer.exp_term, 0.01f, 0.0f, 2.0f, "%.3f");
                ImGui::TableSetColumnIndex(5);
                ImGui::SetNextItemWidth(-1);
                ImGui::DragFloat("##scale", &layer.exp_scale, 0.01f, -2.0f, 2.0f, "%.3f");
                ImGui::PopID();
            }

            void draw_density_layers_table(char const * table_id, DensityProfileLayer (&layers)[PROFILE_LAYER_COUNT], ImU32 accent)
            {
                ImGuiTableFlags const flags = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerV;
                if (ImGui::BeginTable(table_id, 6, flags))
                {
                    ImGui::TableSetupColumn("layer", ImGuiTableColumnFlags_WidthFixed, 54.0f);
                    ImGui::TableSetupColumn("width", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                    ImGui::TableSetupColumn("const", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                    ImGui::TableSetupColumn("lin", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                    ImGui::TableSetupColumn("exp", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                    ImGui::TableSetupColumn("scale", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                    ImGui::TableHeadersRow();
                    draw_density_layer_row("Layer 0", layers[0], accent);
                    draw_density_layer_row("Layer 1", layers[1], accent);
                    ImGui::EndTable();
                }
            }

            // ---- one collapsible card per profile: swatch/scalar fields, chart + layer table ----
            void draw_profile_section(
                char const * name,
                char const * subtitle,
                f32 * scattering_rgb, // nullable, 3 contiguous floats
                f32 * extinction_rgb, // nullable, 3 contiguous floats
                f32 * scale_height,   // nullable
                f32 * phase_g,        // nullable
                DensityProfileLayer (&layers)[PROFILE_LAYER_COUNT],
                ImU32 accent)
            {
                ImGui::PushID(name);
                if (ImGui::CollapsingHeader(name, ImGuiTreeNodeFlags_DefaultOpen))
                {
                    ImGui::Indent(8);
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, bg_3);
                    ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(accent), "%s", subtitle);

                    if (scattering_rgb)
                    {
                        ImGui::SetNextItemWidth(200.0f);
                        ImGui::ColorEdit3("scattering (km-1)", scattering_rgb, ImGuiColorEditFlags_Float);
                    }
                    if (extinction_rgb)
                    {
                        ImGui::SameLine(0, 16);
                        ImGui::SetNextItemWidth(200.0f);
                        ImGui::ColorEdit3("extinction (km-1)", extinction_rgb, ImGuiColorEditFlags_Float);
                    }
                    if (scale_height)
                    {
                        ImGui::SetNextItemWidth(90.0f);
                        ImGui::DragFloat("scale height (km)", scale_height, 0.02f, 0.1f, 50.0f, "%.3f");
                    }
                    if (phase_g)
                    {
                        ImGui::SameLine(0, 16);
                        ImGui::SetNextItemWidth(90.0f);
                        ImGui::DragFloat("phase g (HG)", phase_g, 0.005f, -1.0f, 1.0f, "%.3f");
                    }

                    ImGui::Spacing();
                    ImGui::BeginGroup();
                    draw_density_profile_chart(layers, {130.0f, 84.0f}, accent);
                    ImGui::EndGroup();
                    ImGui::SameLine(0, 14);
                    ImGui::BeginGroup();
                    draw_density_layers_table(name, layers, accent);
                    ImGui::EndGroup();

                    ImGui::PopStyleColor();
                    ImGui::Unindent(8);
                }
                ImGui::PopID();
            }

            // ---- sun elevation dial: horizon-to-zenith arc, drag the marker to set elevation ----
            auto draw_sun_elevation_dial(ImVec2 size, f32 * elevation_deg) -> bool
            {
                ImVec2 const p0 = ImGui::GetCursorScreenPos();
                ImGui::InvisibleButton("##sun_elevation_dial", size);
                bool const dragging = ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) { ImGui::SetMouseCursor(ImGuiMouseCursor_Hand); }
                auto * draw_list = ImGui::GetWindowDrawList();
                ImVec2 const center{p0.x + size.x * 0.5f, p0.y + size.y - 4.0f};
                f32 const radius = std::min(size.x * 0.5f - 4.0f, size.y - 12.0f);
                if (dragging)
                {
                    ImVec2 const mouse = ImGui::GetIO().MousePos;
                    f32 const picked = std::atan2(center.y - mouse.y, mouse.x - center.x) * (180.0f / IM_PI);
                    *elevation_deg = std::clamp(picked, -90.0f, 90.0f);
                }
                // ImGui arc angles run clockwise on screen, so PI -> 2PI is the half above the center
                draw_list->PathArcTo(center, radius, IM_PI, IM_PI * 2.0f, 32);
                draw_list->PathStroke(ImGui::GetColorU32(alt_2), ImDrawFlags_None, 2.0f);
                draw_list->AddLine({center.x - radius, center.y}, {center.x + radius, center.y}, ImGui::GetColorU32(alt_1));
                f32 const theta = *elevation_deg * (IM_PI / 180.0f);
                ImVec2 const sun_pos{center.x + radius * std::cos(theta), center.y - radius * std::sin(theta)};
                draw_list->AddLine(center, sun_pos, ImGui::GetColorU32(alt_2), 1.0f);
                draw_list->AddCircleFilled(sun_pos, 5.0f, accent_sun, 16);
                return dragging;
            }

            // ---- azimuth compass: ring with a marker line, drag it to set the bearing ----
            auto draw_azimuth_compass(ImVec2 size, f32 * azimuth_deg) -> bool
            {
                ImVec2 const p0 = ImGui::GetCursorScreenPos();
                ImGui::InvisibleButton("##azimuth_compass", size);
                bool const dragging = ImGui::IsItemActive();
                if (ImGui::IsItemHovered()) { ImGui::SetMouseCursor(ImGuiMouseCursor_Hand); }
                auto * draw_list = ImGui::GetWindowDrawList();
                ImVec2 const center{p0.x + size.x * 0.5f, p0.y + size.y * 0.5f};
                f32 const radius = std::min(size.x, size.y) * 0.5f - 3.0f;
                if (dragging)
                {
                    ImVec2 const mouse = ImGui::GetIO().MousePos;
                    f32 const picked = std::atan2(mouse.x - center.x, center.y - mouse.y) * (180.0f / IM_PI);
                    *azimuth_deg = picked < 0.0f ? picked + 360.0f : picked;
                }
                draw_list->AddCircle(center, radius, ImGui::GetColorU32(alt_2), 32, 2.0f);
                f32 const theta = *azimuth_deg * (IM_PI / 180.0f);
                ImVec2 const tip{center.x + radius * std::sin(theta), center.y - radius * std::cos(theta)};
                draw_list->AddLine(center, tip, accent_rayleigh, 2.0f);
                draw_list->AddCircleFilled(center, 2.5f, accent_rayleigh, 12);
                return dragging;
            }

            // ---- atmosphere cross-section: ground disc + translucent shell ring ----
            void draw_atmosphere_cross_section(ImVec2 size)
            {
                ImVec2 const p0 = ImGui::GetCursorScreenPos();
                ImGui::Dummy(size);
                auto * draw_list = ImGui::GetWindowDrawList();
                ImVec2 const center{p0.x + size.y * 0.5f, p0.y + size.y * 0.5f};
                f32 const shell_r = size.y * 0.5f - 2.0f;
                f32 const ground_r = shell_r * 0.72f;
                draw_list->AddCircleFilled(center, shell_r, (accent_rayleigh & 0x00FFFFFFu) | 0x38000000u, 48);
                draw_list->AddCircle(center, shell_r, accent_rayleigh, 48, 1.5f);
                draw_list->AddCircleFilled(center, ground_r, ImGui::GetColorU32(alt_2), 48);
                draw_list->AddLine({center.x + ground_r, center.y}, {p0.x + size.y + 14.0f, center.y + 18.0f}, ImGui::GetColorU32(ImGuiCol_TextDisabled));
                draw_list->AddText({p0.x + size.y + 16.0f, center.y + 12.0f}, ImGui::GetColorU32(ImGuiCol_Text), "6360 km  surface");
                draw_list->AddLine({center.x + shell_r * 0.9f, center.y - shell_r * 0.45f}, {p0.x + size.y + 14.0f, center.y - 18.0f}, ImGui::GetColorU32(ImGuiCol_TextDisabled));
                draw_list->AddText({p0.x + size.y + 16.0f, center.y - 24.0f}, accent_rayleigh, "6460 km  top");
            }
        } // namespace

        void draw_sky_settings_panel(RenderContext & render_context)
        {
            auto & sky = render_context.render_data.sky_settings;

            // ---- presets ----
            {
                ImGui::TextDisabled("schema v1");
                if (ImGui::Button("Default")) { /* load_sky_settings("settings/sky/default.json") */ }
                ImGui::SameLine();
                if (ImGui::Button("Golden Hour")) { /* load_sky_settings("settings/sky/golden_hour.json") */ }
                ImGui::SameLine();
                if (ImGui::Button("Overcast")) { /* load_sky_settings("settings/sky/overcast.json") */ }
                ImGui::SameLine();
                if (ImGui::Button("Night")) { /* load_sky_settings("settings/sky/night.json") */ }
                ImGui::SameLine();
                if (ImGui::Button("+ Save preset")) { ImGui::OpenPopup("save_sky_preset"); }
                if (ImGui::BeginPopup("save_sky_preset"))
                {
                    static std::array<char, 64> name_buf = {};
                    ImGui::InputTextWithHint("##name", "preset name", name_buf.data(), name_buf.size());
                    if (ImGui::Button("Save")) { /* save_sky_settings(sky, "settings/sky/" + name + ".json") */ ImGui::CloseCurrentPopup(); }
                    ImGui::EndPopup();
                }
            }
            ImGui::Separator();

            // ---- sun ----
            if (ImGui::CollapsingHeader("Sun", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Indent(8);
                glm::vec3 const sun_dir = {sky.sun_direction.x, sky.sun_direction.y, sky.sun_direction.z};
                f32 const angle_y_rad = glm::acos(glm::clamp(sun_dir.z, -1.0f, 1.0f));
                f32 angle_x_deg = glm::degrees(glm::atan(sun_dir.y, sun_dir.x));
                angle_x_deg += angle_x_deg < 0.0f ? 360.0f : 0.0f;
                f32 angle_y_deg = glm::degrees(angle_y_rad);

                ImGui::BeginGroup();
                ImGui::SetNextItemWidth(90.0f);
                ImGui::DragFloat("Angle X (azimuth)", &angle_x_deg, 0.5f, 0.1f, 360.0f, "%.1f°");
                ImGui::SetNextItemWidth(90.0f);
                ImGui::DragFloat("Angle Y (zenith)", &angle_y_deg, 0.5f, 0.1f, 180.0f, "%.1f°");
                static bool animate_sun = false;
                static f32 sun_speed = 0.0f;
                ImGui::Checkbox("Animate sun", &animate_sun);
                if (animate_sun)
                {
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(80.0f);
                    ImGui::DragFloat("speed (deg/s)", &sun_speed, 0.05f, -10.0f, 10.0f, "%.2f");
                }
                ImGui::EndGroup();

                ImGui::SameLine(0, 24);
                ImGui::BeginGroup();
                ImGui::TextDisabled("horizon -> zenith");
                f32 elevation_deg = 90.0f - angle_y_deg;
                if (draw_sun_elevation_dial({140.0f, 78.0f}, &elevation_deg)) { angle_y_deg = 90.0f - elevation_deg; }
                ImGui::EndGroup();

                ImGui::SameLine(0, 20);
                ImGui::BeginGroup();
                ImGui::TextDisabled("azimuth");
                draw_azimuth_compass({64.0f, 64.0f}, &angle_x_deg);
                ImGui::EndGroup();

                sky.sun_direction = {
                    std::cos(glm::radians(angle_x_deg)) * std::sin(glm::radians(angle_y_deg)),
                    std::sin(glm::radians(angle_x_deg)) * std::sin(glm::radians(angle_y_deg)),
                    std::cos(glm::radians(angle_y_deg)),
                };
                ImGui::Unindent(8);
            }

            // ---- atmosphere shell ----
            if (ImGui::CollapsingHeader("Atmosphere Shell", ImGuiTreeNodeFlags_DefaultOpen))
            {
                ImGui::Indent(8);
                ImGui::BeginGroup();
                ImGui::SetNextItemWidth(90.0f);
                ImGui::DragFloat("Bottom radius (km)", &sky.atmosphere_bottom, 1.0f, 5000.0f, sky.atmosphere_top - 1.0f, "%.0f");
                ImGui::SetNextItemWidth(90.0f);
                ImGui::DragFloat("Top radius (km)", &sky.atmosphere_top, 1.0f, sky.atmosphere_bottom + 1.0f, 8000.0f, "%.0f");
                ImGui::TextDisabled("thickness: %.0f km", sky.atmosphere_top - sky.atmosphere_bottom);
                ImGui::EndGroup();
                ImGui::SameLine(0, 20);
                draw_atmosphere_cross_section({140.0f, 84.0f});
                ImGui::Unindent(8);
            }

            // ---- rayleigh / mie / absorption ----
            draw_profile_section("Rayleigh", "air molecules", &sky.rayleigh_scattering.x, nullptr,
                &sky.rayleigh_scale_height, nullptr, sky.rayleigh_density, accent_rayleigh);
            draw_profile_section("Mie", "aerosols", &sky.mie_scattering.x, &sky.mie_extinction.x,
                &sky.mie_scale_height, &sky.mie_phase_function_g, sky.mie_density, accent_mie);
            draw_profile_section("Absorption", "ozone", nullptr, &sky.absorption_extinction.x,
                nullptr, nullptr, sky.absorption_density, accent_absorption);

            // ---- resolution & quality ----
            if (ImGui::CollapsingHeader("Resolution & Quality"))
            {
                ImGui::Indent(8);
                ImGuiTableFlags const flags = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_BordersInnerV;
                if (ImGui::BeginTable("lut_res", 3, flags))
                {
                    ImGui::TableSetupColumn("LUT", ImGuiTableColumnFlags_WidthFixed, 130.0f);
                    ImGui::TableSetupColumn("dimensions", ImGuiTableColumnFlags_WidthFixed, 140.0f);
                    ImGui::TableSetupColumn("steps", ImGuiTableColumnFlags_WidthFixed, 90.0f);
                    ImGui::TableHeadersRow();

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::Text("Transmittance");
                    ImGui::TableSetColumnIndex(1); ImGui::SetNextItemWidth(-1);
                    ImGui::DragScalarN("##t_dim", ImGuiDataType_U32, &sky.transmittance_dimensions.x, 2, 1.0f);
                    ImGui::TableSetColumnIndex(2); ImGui::SetNextItemWidth(-1);
                    ImGui::DragScalar("##t_steps", ImGuiDataType_U32, &sky.transmittance_step_count, 1.0f);

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::Text("Multiscattering");
                    ImGui::TableSetColumnIndex(1); ImGui::SetNextItemWidth(-1);
                    ImGui::DragScalarN("##m_dim", ImGuiDataType_U32, &sky.multiscattering_dimensions.x, 2, 1.0f);
                    ImGui::TableSetColumnIndex(2); ImGui::SetNextItemWidth(-1);
                    ImGui::DragScalar("##m_steps", ImGuiDataType_U32, &sky.multiscattering_step_count, 1.0f);

                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0); ImGui::Text("Sky-view");
                    ImGui::TableSetColumnIndex(1); ImGui::SetNextItemWidth(-1);
                    ImGui::DragScalarN("##s_dim", ImGuiDataType_U32, &sky.sky_dimensions.x, 2, 1.0f);
                    ImGui::TableSetColumnIndex(2); ImGui::SetNextItemWidth(-1);
                    ImGui::DragScalar("##s_steps", ImGuiDataType_U32, &sky.sky_step_count, 1.0f);

                    ImGui::EndTable();
                }
                ImGui::Unindent(8);
            }
        }
    } // namespace ui
} // namespace tido
