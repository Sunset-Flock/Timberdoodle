#include "ui.hpp"
#include <filesystem>
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <vector>
#include <array>
#include <limits>
#include "widgets/helpers.hpp"
#include "../daxa_helper.hpp"
#include "../rendering/rtgi/rt_lab/rt_lab.hpp"
#include "../rendering/rtgi/rt_lab/rt_lab_resolve.inl" // RT_LAB_STABILIZE_MAX_RADIUS
#include "../shader_shared/gpu_work_expansion.inl"
#include "../shader_lib/volumetric.hlsl"

namespace
{
    // Index == DEBUG_DRAW_MODE_* value (shader_shared/shared.inl).
    constexpr std::array<char const *, DEBUG_DRAW_MODE_COUNT> DEBUG_DRAW_MODE_NAMES = {
        "NONE",
        "OVERDRAW",
        "TRIANGLE_CONNECTIVITY",
        "TRIANGLE_ID",
        "MESHLET_ID",
        "MESH_ID",
        "MESH_GROUP_ID",
        "ENTITY_ID",
        "MESH_LOD",
        "DEPTH",
        "ALBEDO",
        "ROUGHNESS",
        "METALNESS",
        "UV",
        "FACE_NORMAL",
        "SMOOTH_NORMAL",
        "MAPPED_NORMAL",
        "FACE_TANGENT",
        "SMOOTH_TANGENT",
        "DIRECT_DIFFUSE",
        "DIRECT_SPECULAR",
        "INDIRECT_DIFFUSE",
        "INDIRECT_DIFFUSE_AO",
        "AO",
        "INDIRECT_SPECULAR",
        "ALL_DIFFUSE",
        "ALL_SPECULAR",
        "ALL_LIGHTING",
        "SHADE_OPAQUE_CLOCKS",
        "LIGHT_MASK_VOLUME",
        "VSM_OVERDRAW",
        "VSM_CLIP_LEVEL",
        "VSM_SPOT_LEVEL",
        "VSM_POINT_LEVEL",
        "PGI_EVAL_CLOCKS",
        "PGI_CASCADE_SMOOTH",
        "PGI_CASCADE_ABSOLUTE",
        "PGI_LOW_QUALITY_SAMPLING",
        "PGI_IRRADIANCE",
        "PGI_RADIANCE",
        "RTGI_DEBUG_PRIMARY_TRACE",
        "RTGI_DIFFUSE_TRACE_CLOCKS",
        "RTGI_SPECULAR_TRACE_CLOCKS",
        "RTGI_TOTAL_RAYS_SHOT",
        "RTGI_DIFFUSE_RAYS_SHOT",
        "RTGI_SPECULAR_RAYS_SHOT",
        "RTGI_TOTAL_RAYS_PER_TILE",
        "RTGI_DIFFUSE_RAYS_PER_TILE",
        "RTGI_SPECULAR_RAYS_PER_TILE",
        "RTGI_RAY_SHARE",
        "RTGI_RAY_MATERIAL",
        "RTGI_GUIDE_DIRECTION",
        "RTGI_GUIDE_LOW_RES_DIRECTION",
        "RTGI_DIFFUSE_AO_GUIDE",
        "RTGI_SPECULAR_HIT_DISTANCE",
        "RTGI_DIFFUSE_PERCEPTUAL_MEAN",
        "RTGI_SPECULAR_PERCEPTUAL_MEAN",
        "RTGI_DIFFUSE_AO_GUIDE_TEMPORAL",
        "RTGI_SPECULAR_HIT_DISTANCE_TEMPORAL",
        "RTGI_DIFFUSE_PERCEPTUAL_MEAN_TEMPORAL",
        "RTGI_SPECULAR_PERCEPTUAL_MEAN_TEMPORAL",
        "RTGI_DIFFUSE_HISTORY_LENGTH",
        "RTGI_SPECULAR_HISTORY_LENGTH",
        "RTGI_DIFFUSE_TEMPORAL_REACTIVITY",
        "RTGI_SPECULAR_TEMPORAL_REACTIVITY",
        "RTGI_SPECULAR_VIRTUAL_AMOUNT",
        "RTGI_HASH_GRID_CELLS",
        "RTGI_HASH_GRID_RADIANCE",
        "RTGI_HASH_GRID_SAMPLE_COUNT",
        "RTGI_HASH_GRID_RADIANCE_JITTERED",
        "RT_LAB_CONVERGENCE_PROGRESSIVE",
        "RT_LAB_CONVERGENCE_CUSTOM",
        "RT_LAB_CONVERGENCE_DIFFERENCE",
        "RT_LAB_CONVERGENCE_DIFFERENCE_SQUARED",
        "RT_LAB_CONVERGENCE_RELATIVE_DIFFERENCE",
        "RT_LAB_CONVERGENCE_PREDICTED_ERROR",
        "RT_LAB_CONVERGENCE_PREDICTION_RATIO",
        "RT_LAB_HISTORY_LENGTH",
        "RT_LAB_VARIANCE",
        "RT_LAB_SPATIAL_SAMPLE_COUNT",
        "RT_LAB_COARSE_BLEND",
        "RT_LAB_COARSE_SAMPLE_COUNT",
        "RT_LAB_SPLIT_ERROR",
        "RT_LAB_SPLIT_RELATIVE_ERROR",
        "RT_LAB_TEMPORAL_GRADIENT",
        "RT_LAB_COARSE_ANTI_LAG",
        "RT_LAB_COARSE_ANTI_LAG_MEAN",
        "RT_LAB_COARSE_ANTI_LAG_FRAME_MEAN",
        "RT_LAB_COARSE_ANTI_LAG_RELATIVE_STD",
        "RT_LAB_COARSE_ANTI_LAG_FRAME_RELATIVE_STD",
    };

    static_assert(DEBUG_DRAW_MODE_NAMES.back() != nullptr, "DEBUG_DRAW_MODE_NAMES is missing entries for DEBUG_DRAW_MODE_COUNT");

    // Per-feature debug view selector: "NONE" + the feature's modes (its own first, then related ones). Anything
    // but NONE overrides the main debug visualization.
    void debug_mode_sub_selector(char const * label, i32 * selected, std::initializer_list<i32> modes, i32 & override_mode)
    {
        std::vector<char const *> labels = {"NONE"};
        for (i32 const mode : modes) { labels.push_back(DEBUG_DRAW_MODE_NAMES[mode]); }
        tido::ui::filter_combo(label, selected, labels.data(), s_cast<i32>(labels.size()));
        if (*selected > 0 && *selected <= s_cast<i32>(modes.size()))
        {
            override_mode = *(modes.begin() + (*selected - 1));
        }
    }
} // namespace

namespace
{
    // RT Lab convergence measurement analytics (RtLabMeasurement): distribution heatmap (frames x metric) with the
    // thresholds and the "> X % of tiles" lines, the distribution at one frame, and optionally a comparison result B
    // (its lines on A's heatmap + a table). The viewer only shows the first view_max_frames frames. Metric values are
    // fractions.
    void rt_lab_measure_analytics(RtLabMeasurement const & m, RtLabMeasurement const * compare, RtLabMeasureUi & ui)
    {
        char const * const metric_names[RT_LAB_MEASURE_METRICS] = {
            "Convergence Error (split |A-B| / min(A,B))",
            "Temporal Instability (output std / mean)",
        };
        ImGui::Combo("Metric", &ui.metric, metric_names, RT_LAB_MEASURE_METRICS);
        ImGui::SetItemTooltip(
            "Convergence error: post-spatial split history relative error, reference free.\n"
            "Temporal instability: relative temporal std of the final output luma (flicker), running mean / variance over\n"
            "the instability window. Both per 4x4 tile (RMS), binned into 512 log bins (range in the axis).");
        i32 const metric = std::clamp(ui.metric, 0, RT_LAB_MEASURE_METRICS - 1);
        char const * const preset_names[RT_LAB_PRESET_COUNT] = {"Progressive", "Custom"};
        ImGui::TextDisabled("%i frames, %s preset, %u tiles, instability window %i frames",
            m.frames, preset_names[std::clamp(m.preset, 0, RT_LAB_PRESET_COUNT - 1)], m.denominator[metric], m.window);
        ImGui::SliderInt("Max Frames (viewer)", &ui.view_max_frames, 0, std::max(m.frames, compare != nullptr ? compare->frames : 0));
        ImGui::SetItemTooltip("Only show the first N frames of the results. 0 = all.");
        i32 const frames = ui.view_max_frames > 0 ? std::min(ui.view_max_frames, m.frames) : m.frames;
        i32 const compare_frames = compare == nullptr ? 0 : (ui.view_max_frames > 0 ? std::min(ui.view_max_frames, compare->frames) : compare->frames);
        ImGui::InputFloat("Threshold (%)", &ui.threshold_percent, 0.0f, 0.0f, "%.3g");
        ImGui::SetItemTooltip("Drawn as a line on the heatmap (Threshold Line) and the frame distribution. 0 = off.");

        // == Distribution heatmap (x = frame, y = metric, color = tiles) ==
        ImGui::InputScalarN("Overlay Lines: > X% of Tiles", ImGuiDataType_Float, ui.overlay_above_percent.data(), static_cast<i32>(ui.overlay_above_percent.size()), nullptr, nullptr, "%.3g");
        ImGui::SetItemTooltip("Per frame line at the metric value that X%% of the tiles are still above (the 100 - X percentile).\n50 = median, 10 = p90, 1 = p99. 0 = off.");
        char const * const color_modes[] = {"log10(1 + tiles)", "% of tiles per frame"};
        ImGui::Combo("Heatmap Color", &ui.heatmap_color_mode, color_modes, 2);
        ImGui::SetItemTooltip("log10(1 + tiles): shows sparse tails. %% of tiles per frame: every frame column is its own distribution.");
        if (ImGui::BeginCombo("Colormap", ImPlot::GetColormapName(ui.heatmap_colormap)))
        {
            for (i32 c = 0; c < ImPlot::GetColormapCount(); ++c)
            {
                if (ImGui::Selectable(ImPlot::GetColormapName(c), c == ui.heatmap_colormap))
                {
                    ui.heatmap_colormap = c;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::Checkbox("Threshold Line", &ui.show_thresholds);
        ImGui::SameLine();
        ImGui::Checkbox("Overlay Lines", &ui.show_overlay_lines);
        ImGui::SameLine();
        ImGui::Checkbox("Fit Y To Data", &ui.fit_y_to_data);
        ImGui::SliderFloat("Heatmap Height", &ui.heatmap_height, 150.0f, 1000.0f, "%.0f px");

        constexpr i32 HEATMAP_ROWS = 128;
        constexpr i32 BINS_PER_ROW = RT_LAB_MEASURE_BINS / HEATMAP_ROWS;
        f64 const log_min = static_cast<f64>(m.log_min);
        f64 const log_max = static_cast<f64>(m.log_max);
        f64 const row_height = (log_max - log_min) / static_cast<f64>(HEATMAP_ROWS);
        i32 const columns = std::min(frames, 256);
        f64 const denominator = static_cast<f64>(std::max(m.denominator[metric], 1u));
        if (ui.heatmap_generation != m.generation || ui.heatmap_metric != metric || ui.heatmap_built_color_mode != ui.heatmap_color_mode || ui.heatmap_frames != frames)
        {
            ui.heatmap_frames = frames;
            ui.heatmap_generation = m.generation;
            ui.heatmap_metric = metric;
            ui.heatmap_built_color_mode = ui.heatmap_color_mode;
            ui.heatmap.assign(static_cast<size_t>(HEATMAP_ROWS * columns), 0.0f);
            std::vector<i32> frames_per_column(static_cast<size_t>(columns), 0);
            ui.heatmap_max = 0.0f;
            ui.heatmap_lowest_row = HEATMAP_ROWS;
            ui.heatmap_highest_row = -1;
            for (i32 f = 0; f < frames; ++f)
            {
                i32 const column = f * columns / frames;
                frames_per_column[column] += 1;
                u32 const * row = m.rows.data() + static_cast<size_t>(f) * RT_LAB_MEASURE_ROW_SIZE + metric * RT_LAB_MEASURE_METRIC_STRIDE;
                for (i32 b = 0; b < RT_LAB_MEASURE_BINS; ++b)
                {
                    if (row[b] == 0)
                    {
                        continue;
                    }
                    i32 const value_row = b / BINS_PER_ROW; // 0 = lowest values
                    ui.heatmap_lowest_row = std::min(ui.heatmap_lowest_row, value_row);
                    ui.heatmap_highest_row = std::max(ui.heatmap_highest_row, value_row);
                    i32 const heatmap_row = HEATMAP_ROWS - 1 - value_row; // row 0 is drawn at the top: highest values
                    ui.heatmap[heatmap_row * columns + column] += static_cast<f32>(row[b]);
                }
            }
            for (i32 r = 0; r < HEATMAP_ROWS; ++r)
            {
                for (i32 c = 0; c < columns; ++c)
                {
                    f32 & v = ui.heatmap[r * columns + c];
                    if (ui.heatmap_color_mode == 0)
                    {
                        v = std::log10(1.0f + v);
                    }
                    else
                    {
                        v = 100.0f * v / (static_cast<f32>(denominator) * static_cast<f32>(std::max(frames_per_column[c], 1)));
                    }
                    ui.heatmap_max = std::max(ui.heatmap_max, v);
                }
            }
        }
        i32 const slice = (ui.slice_frame <= 0 ? frames : std::min(ui.slice_frame, frames)) - 1;
        f64 const heatmap_scale_max = static_cast<f64>(std::max(ui.heatmap_max, ui.heatmap_color_mode == 0 ? 1.0f : 1e-3f));
        constexpr f32 COLOR_SCALE_WIDTH = 100.0f;
        ImPlot::PushColormap(ui.heatmap_colormap);
        if (ImPlot::BeginPlot("Distribution Over Time##rt_lab_measure", ImVec2(ImGui::GetContentRegionAvail().x - COLOR_SCALE_WIDTH, ui.heatmap_height)))
        {
            bool const fit = ui.fit_y_to_data && ui.heatmap_highest_row >= ui.heatmap_lowest_row;
            ImPlot::SetupAxes("frames", "log10(metric %)", ImPlotAxisFlags_AutoFit, fit ? ImPlotAxisFlags_None : ImPlotAxisFlags_AutoFit);
            if (fit)
            {
                f64 const y_min = log_min + 2.0 + static_cast<f64>(ui.heatmap_lowest_row) * row_height;
                f64 const y_max = log_min + 2.0 + static_cast<f64>(ui.heatmap_highest_row + 1) * row_height;
                ImPlot::SetupAxisLimits(ImAxis_Y1, y_min - row_height, y_max + row_height, ImPlotCond_Always);
            }
            ImPlot::SetupLegend(ImPlotLocation_NorthEast);
            ImPlot::PlotHeatmap("##tiles", ui.heatmap.data(), HEATMAP_ROWS, columns, 0.0, heatmap_scale_max, nullptr,
                ImPlotPoint(1.0, log_min + 2.0), ImPlotPoint(static_cast<f64>(frames), log_max + 2.0));
            if (ui.show_thresholds && ui.threshold_percent > 0.0f)
            {
                f64 const y = std::log10(static_cast<f64>(ui.threshold_percent));
                ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 0.5f), 1.0f);
                ImPlot::PlotInfLines(fmt::format("threshold {:.3g}%", ui.threshold_percent).c_str(), &y, 1, ImPlotInfLinesFlags_Horizontal);
            }
            if (ui.show_overlay_lines)
            {
                f64 const top = log_max + 2.0;
                auto plot_overlay = [&](RtLabMeasurement const & source, i32 source_frames, char const * prefix, f32 weight, f32 alpha)
                {
                    std::vector<f64> xs(static_cast<size_t>(source_frames));
                    std::vector<f64> ys(static_cast<size_t>(source_frames));
                    for (u32 l = 0; l < ui.overlay_above_percent.size(); ++l)
                    {
                        f32 const above = ui.overlay_above_percent[l];
                        if (above <= 0.0f || above >= 100.0f)
                        {
                            continue;
                        }
                        for (i32 f = 0; f < source_frames; ++f)
                        {
                            xs[f] = static_cast<f64>(f + 1);
                            f64 const v = rt_lab_measure_percentile(source, metric, f, 1.0 - static_cast<f64>(above) * 0.01);
                            ys[f] = std::isfinite(v) ? std::log10(100.0 * v) : top; // above the histogram range: top
                        }
                        // Same color for the same line in A and B.
                        ImVec4 color = ImPlot::GetColormapColor(static_cast<i32>(l), ImPlotColormap_Dark);
                        color.w = alpha;
                        ImPlot::SetNextLineStyle(color, weight);
                        ImPlot::PlotLine(fmt::format("{}> {:.3g}% of tiles", prefix, above).c_str(), xs.data(), ys.data(), source_frames);
                    }
                };
                plot_overlay(m, frames, compare != nullptr ? "A: " : "", 2.5f, 1.0f);
                if (compare != nullptr)
                {
                    plot_overlay(*compare, compare_frames, "B: ", 1.0f, 0.7f);
                }
            }
            f64 const slice_x = static_cast<f64>(slice + 1);
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 0.3f, 0.3f, 0.8f), 1.0f);
            ImPlot::PlotInfLines("slice frame", &slice_x, 1);
            ImPlot::EndPlot();
        }
        ImGui::SameLine();
        ImPlot::ColormapScale(ui.heatmap_color_mode == 0 ? "log10(1 + tiles)" : "% of tiles", 0.0, heatmap_scale_max,
            ImVec2(COLOR_SCALE_WIDTH - 10.0f, ui.heatmap_height));
        ImPlot::PopColormap();

        // == A vs B ==
        if (compare != nullptr)
        {
            ImGui::SeparatorText("A vs B (last shown frame)");
            if (ImGui::BeginTable("##rt_lab_compare", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame))
            {
                ImGui::TableSetupColumn("Metric");
                ImGui::TableSetupColumn("Tiles above");
                ImGui::TableSetupColumn("A");
                ImGui::TableSetupColumn("B");
                ImGui::TableSetupColumn("B / A");
                ImGui::TableHeadersRow();
                char const * const short_metric_names[RT_LAB_MEASURE_METRICS] = {"Convergence error", "Temporal instability"};
                for (i32 mi = 0; mi < RT_LAB_MEASURE_METRICS; ++mi)
                {
                    for (f32 const above : ui.overlay_above_percent)
                    {
                        if (above <= 0.0f || above >= 100.0f)
                        {
                            continue;
                        }
                        f64 const fraction = 1.0 - static_cast<f64>(above) * 0.01;
                        f64 const a = rt_lab_measure_percentile(m, mi, frames - 1, fraction);
                        f64 const b = rt_lab_measure_percentile(*compare, mi, compare_frames - 1, fraction);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(short_metric_names[mi]);
                        ImGui::TableNextColumn();
                        ImGui::Text("> %.3g%%", above);
                        auto value_cell = [](f64 v)
                        {
                            ImGui::TableNextColumn();
                            if (std::isfinite(v))
                            {
                                ImGui::Text("%.3g%%", 100.0 * v);
                            }
                            else
                            {
                                ImGui::TextDisabled("out of range");
                            }
                        };
                        value_cell(a);
                        value_cell(b);
                        ImGui::TableNextColumn();
                        if (std::isfinite(a) && std::isfinite(b) && a > 0.0)
                        {
                            f64 const ratio = b / a;
                            ImGui::TextColored(ratio <= 1.0 ? ImVec4(0.5f, 1.0f, 0.5f, 1.0f) : ImVec4(1.0f, 0.6f, 0.5f, 1.0f), "%.3f", ratio);
                        }
                        else
                        {
                            ImGui::TextDisabled("-");
                        }
                    }
                }
                ImGui::EndTable();
            }
            ImGui::TextDisabled("Values: the metric that this share of tiles is still above (lower = better). A: %i, B: %i frames.", frames, compare_frames);
        }

        // == Distribution at one frame ==
        ImGui::SliderInt("Slice Frame (0 = last)", &ui.slice_frame, 0, frames);
        {
            u32 const * row = m.rows.data() + static_cast<size_t>(slice) * RT_LAB_MEASURE_ROW_SIZE + metric * RT_LAB_MEASURE_METRIC_STRIDE;
            f64 const denominator = static_cast<f64>(std::max(m.denominator[metric], 1u));
            ImGui::TextDisabled("Frame %i: %.2f%% of tiles below %.3g%%, %.2f%% above %.3g%% (or no value)", slice + 1,
                100.0 * row[RT_LAB_MEASURE_UNDER] / denominator, 100.0 * rt_lab_measure_bin_edge(m, 0),
                100.0 * (denominator - static_cast<f64>(m.cumulative[metric][static_cast<size_t>(slice) * (RT_LAB_MEASURE_BINS + 1) + RT_LAB_MEASURE_BINS])) / denominator,
                100.0 * rt_lab_measure_bin_edge(m, RT_LAB_MEASURE_BINS));
            constexpr i32 SLICE_POINTS = RT_LAB_MEASURE_BINS / BINS_PER_ROW;
            std::array<f64, SLICE_POINTS> slice_x = {};
            std::array<f64, SLICE_POINTS> slice_y = {};
            for (i32 i = 0; i < SLICE_POINTS; ++i)
            {
                u32 count = 0;
                for (i32 b = i * BINS_PER_ROW; b < (i + 1) * BINS_PER_ROW; ++b)
                {
                    count += row[b];
                }
                slice_x[i] = 100.0 * std::sqrt(rt_lab_measure_bin_edge(m, static_cast<u32>(i * BINS_PER_ROW)) * rt_lab_measure_bin_edge(m, static_cast<u32>((i + 1) * BINS_PER_ROW)));
                slice_y[i] = 100.0 * static_cast<f64>(count) / denominator;
            }
            if (ImPlot::BeginPlot("Distribution At Frame##rt_lab_measure", ImVec2(-1, 220)))
            {
                ImPlot::SetupAxes("metric (%)", "% of tiles", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
                ImPlot::PlotLine("tiles", slice_x.data(), slice_y.data(), SLICE_POINTS);
                if (ui.threshold_percent > 0.0f)
                {
                    f64 const x = static_cast<f64>(ui.threshold_percent);
                    ImPlot::PlotInfLines(fmt::format("threshold {:.3g}%", ui.threshold_percent).c_str(), &x, 1);
                }
                ImPlot::EndPlot();
            }
        }
    }
} // namespace

void setup_colors()
{
    ImVec4 * colors = ImGui::GetStyle().Colors;
    ImGuiStyle & style = ImGui::GetStyle();
    // clang-format off
    colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = bg_0;
    colors[ImGuiCol_ChildBg]                = bg_0;
    colors[ImGuiCol_PopupBg]                = bg_1;
    colors[ImGuiCol_Border]                 = {alt_2.x, alt_2.y, alt_2.z, 0.5};
    colors[ImGuiCol_BorderShadow]           = bg_1;
    colors[ImGuiCol_FrameBg]                = bg_2;
    colors[ImGuiCol_FrameBgHovered]         = hovered_1;
    colors[ImGuiCol_FrameBgActive]          = hovered_1;
    colors[ImGuiCol_TitleBg]                = bg_0;
    colors[ImGuiCol_TitleBgActive]          = bg_0;
    colors[ImGuiCol_TitleBgCollapsed]       = bg_0;
    colors[ImGuiCol_MenuBarBg]              = bg_0;
    colors[ImGuiCol_ScrollbarBg]            = bg_0;
    colors[ImGuiCol_ScrollbarGrab]          = bg_2;
    colors[ImGuiCol_ScrollbarGrabHovered]   = hovered_1;
    colors[ImGuiCol_ScrollbarGrabActive]    = select_blue_1;
    colors[ImGuiCol_CheckMark]              = select_blue_1;
    colors[ImGuiCol_SliderGrab]             = alt_1;
    colors[ImGuiCol_SliderGrabActive]       = select_blue_1;
    colors[ImGuiCol_Button]                 = bg_2;
    colors[ImGuiCol_ButtonHovered]          = hovered_1;
    colors[ImGuiCol_ButtonActive]           = select_blue_1;
    colors[ImGuiCol_Header]                 = bg_2;
    colors[ImGuiCol_HeaderHovered]          = hovered_1;
    colors[ImGuiCol_HeaderActive]           = select_blue_1;
    colors[ImGuiCol_Separator]              = alt_1;
    colors[ImGuiCol_SeparatorHovered]       = hovered_1;
    colors[ImGuiCol_SeparatorActive]        = select_blue_1;
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_Tab]                    = bg_1;
    colors[ImGuiCol_TabHovered]             = hovered_1;
    colors[ImGuiCol_TabSelected]            = alt_1;
    colors[ImGuiCol_TabDimmed]              = bg_1;
    colors[ImGuiCol_TabDimmedSelected]      = bg_1;
    colors[ImGuiCol_PlotLines]              = select_blue_1;
    colors[ImGuiCol_PlotLinesHovered]       = select_blue_1;
    colors[ImGuiCol_PlotHistogram]          = select_blue_1;
    colors[ImGuiCol_PlotHistogramHovered]   = select_blue_1;
    colors[ImGuiCol_TableHeaderBg]          = bg_2;
    colors[ImGuiCol_TableBorderStrong]      = {alt_1.x, alt_1.y, alt_1.z, 0.5};
    colors[ImGuiCol_TableBorderLight]       = {alt_2.x, alt_2.y, alt_2.z, 0.5};
    colors[ImGuiCol_TableRowBg]             = bg_0;
    colors[ImGuiCol_TableRowBgAlt]          = bg_1;
    colors[ImGuiCol_TextSelectedBg]         = bg_5;
    colors[ImGuiCol_DragDropTarget]         = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_NavCursor]              = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);

    style.WindowPadding                     = ImVec2(8.00f, 8.00f);
    style.FramePadding                      = ImVec2(5.00f, 2.00f);
    style.CellPadding                       = ImVec2(6.00f, 6.00f);
    style.ItemSpacing                       = ImVec2(6.00f, 6.00f);
    style.ItemInnerSpacing                  = ImVec2(6.00f, 6.00f);
    style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
    style.IndentSpacing                     = 25;
    style.ScrollbarSize                     = 15;
    style.GrabMinSize                       = 10;
    style.WindowBorderSize                  = 1;
    style.ChildBorderSize                   = 0;
    style.PopupBorderSize                   = 1;
    style.FrameBorderSize                   = 0;
    style.TabBorderSize                     = 0;
    style.WindowRounding                    = 5;
    style.ChildRounding                     = 4;
    style.FrameRounding                     = 3;
    style.PopupRounding                     = 4;
    style.ScrollbarRounding                 = 2;
    style.GrabRounding                      = 3;
    style.LogSliderDeadzone                 = 4;
    style.TabRounding                       = 4;
    style.WindowMenuButtonPosition = ImGuiDir_None;
    // clang-format on
};

UIEngine::UIEngine(Window & window, AssetProcessor & asset_processor, GPUContext * gpu_context)
    : scene_graph(&imgui_renderer, &icons, gpu_context->lin_clamp_sampler),
      property_viewer(&imgui_renderer, &icons, gpu_context->lin_clamp_sampler),
      path_editor(&imgui_renderer),
      gpu_context{gpu_context},
      window{&window}
{
    auto * imgui_context = ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(window.glfw_handle, true);
    ImGuiIO & io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    icons.reserve(s_cast<u32>(ICONS::SIZE));
    for (u32 icon_idx = 0; icon_idx < s_cast<u32>(ICONS::SIZE); icon_idx++)
    {
        AssetProcessor::NonmanifestLoadRet ret = asset_processor.load_nonmanifest_texture({ICON_TO_PATH.at(icon_idx)});
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
    imgui_renderer = daxa::ImGuiRenderer({gpu_context->device, gpu_context->swapchain.get_format(), imgui_context, false});
    setup_colors();
    
    main_task_graph_debug_ui = daxa::TaskGraphDebugUi({
        .device = gpu_context->device,
        .imgui_renderer = imgui_renderer,
        .buffer_layout_cache_folder = "./tg_dbg_cache",
    });
}

void UIEngine::main_update(RenderContext & render_context, Scene & scene, ApplicationState & app_state, ThreadPool & threadpool)
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    if (app_state.screenshot_writing)
    {
        ImVec2 const display = ImGui::GetIO().DisplaySize;
        ImGui::SetNextWindowPos(ImVec2(display.x * 0.5f, display.y * 0.35f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowBgAlpha(0.75f);
        ImGui::Begin("##screenshot_writing", nullptr,
            ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
            ImGuiWindowFlags_NoNav | ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
            ImGuiWindowFlags_NoMove);
        ImGui::TextUnformatted("Writing screenshot...");
        ImGui::End();
    }

    // Clear Select Render Data
    render_context.render_data.cursor_uv = {
        std::clamp(s_cast<f32>(window->get_cursor_x()) / s_cast<f32>(window->get_width()), 0.0f, 1.0f),
        std::clamp(s_cast<f32>(window->get_cursor_y()) / s_cast<f32>(window->get_height()), 0.0f, 1.0f),
    };
    if (window->is_cursor_captured())
    {
        render_context.render_data.cursor_uv = { 0.5f, 0.5f };
    }
    render_context.render_data.hovered_entity_index = ~0u;
    render_context.render_data.selected_entity_index = ~0u;

    ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;
    ImGuiWindowFlags window_flags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground |
        ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
        ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBringToFrontOnFocus;
    ImGuiViewport const * viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");

    ImGui::Begin("DockSpace Demo", nullptr, window_flags);
    ImGui::PopStyleVar(3);
    ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
    ImGui::End();

    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Open"))
            {
                app_state.desired_scene_path = open_file_dialog("GLTF\0*.gltf\0");
            }
            ImGui::MenuItem("Load and convert VDB", NULL, &convert_vdb_window);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Widgets"))
        {
            ImGui::MenuItem("Settings", NULL, &renderer_settings);
            ImGui::MenuItem("RT Lab", NULL, &widget_rt_lab);
            ImGui::MenuItem("Widget Settings", NULL, &widget_settings);
            ImGui::MenuItem("Renderer Statistics", NULL, &widget_renderer_statistics);
            ImGui::MenuItem("Scene Hierarchy", NULL, &widget_scene_interface);
            ImGui::MenuItem("Camera Path Editor", NULL, &widget_camera_path_editor);
            ImGui::MenuItem("Widget Property Viewer", NULL, &widget_property_viewer);
            ImGui::MenuItem("TaskGraphDebugUi", NULL, &tg_debug_ui);
            ImGui::MenuItem("Imgui Demo", NULL, &demo_window);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (widget_settings)
    {
        if (ImGui::Begin("Widget Settings", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            ImGui::SeparatorText("Scene graph widget settings");
            ImGui::SliderFloat("icon size", &scene_graph.icon_size, 1.0f, 50.0f);
            ImGui::SliderFloat("spacing", &scene_graph.icon_text_spacing, 1.0f, 50.0f);
            ImGui::SliderFloat("indent", &scene_graph.indent, 1.0f, 50.0f);
            ImGui::Separator();
            ImGui::InputScalarN("resolution", ImGuiDataType_U32, &render_context.render_data.settings.render_target_size, 2);
        }
        ImGui::End();
    }
    if (widget_renderer_statistics)
    {
        ui_render_statistics(render_context, app_state);
    }
    if (widget_scene_interface)
    {
        ui_scene_graph(scene);
    }
    // Per-feature debug view selectors (renderer settings, RT Lab) override the main debug visualization for this frame.
    debug_visualization_index_override = 0;
    if (renderer_settings)
    {
        ui_renderer_settings(render_context, app_state);
    }
    if (widget_rt_lab)
    {
        ui_rt_lab(render_context, app_state);
    }
    render_context.render_data.settings.debug_draw_mode = debug_visualization_index_override != 0 ? debug_visualization_index_override : debug_visualization_index;
    if (vsm_windows.view_meta_memory || vsm_windows.view_page_table || vsm_windows.view_reconstructed_shadow_map)
    {
        ui_vsm_textures(render_context);
    }
    if (widget_camera_path_editor)
    {
        path_editor.render(render_context, app_state.cinematic_camera, app_state.camera_controller);
    }
    if (widget_property_viewer)
    {
        property_viewer.render(scene_interface, scene, render_context, app_state);
    }
    if (demo_window)
    {
        ImGui::ShowDemoWindow();
    }
    if (convert_vdb_window)
    {
        ui_convert_vdb_windw(app_state, &threadpool);
    }
    // tg_resource_debug_ui(
    //     imgui_renderer, 
    //     render_context.gpu_context->device, 
    //     render_context.gpu_context->lin_clamp_sampler, 
    //     render_context.gpu_context->nearest_clamp_sampler, 
    //     render_context.tg_debug, 
    //     app_state.frame_index, 
    //     tg_debug_ui);
}

void UIEngine::ui_scene_graph(Scene const & scene)
{
    if (scene._gltf_asset_manifest.empty())
    {
        return;
    }
    bool began = false;
    if (scene_graph.begin())
    {
        began = true;
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
                for (u32 pop_count = 0; pop_count < s_cast<u32>(top_stack_entry.accumulated_indent + 1); pop_count++)
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
    }
    scene_graph.end(began);
}

void UIEngine::ui_renderer_settings(RenderContext & render_context, ApplicationState & app_state)
{
    RenderGlobalData & render_data = render_context.render_data;
    if (ImGui::Begin("Renderer Settings", nullptr, ImGuiWindowFlags_NoCollapse))
    {
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.35f);
        ImGui::SeparatorText("General settings");
        {
            if (ImGui::Button("Take Screenshot (F1)")) { app_state.request_screenshot = true; }
            ImGui::Checkbox("enable reference path trace", reinterpret_cast<bool *>(&render_data.settings.enable_reference_path_trace));
            ImGui::Checkbox("decompose scene", r_cast<bool *>(&app_state.decompose_bistro));
            ImGui::Checkbox("enable vsync", r_cast<bool *>(&render_data.settings.enable_vsync));
            {
                // GPU selection: saved to GPU_PREFERENCE_PATH, applied on the next start.
                static std::string pending = load_gpu_preference();
                auto const devices = gpu_context->instance.list_devices_properties();
                auto device_name = [&](u32 i) { return std::string(reinterpret_cast<char const *>(devices[i].device_name)); };
                std::string const current = device_name(gpu_context->physical_device_index);
                std::string const shown = pending.empty() ? current : pending;
                if (ImGui::BeginCombo("GPU (next restart)", shown.c_str()))
                {
                    for (u32 i = 0; i < devices.size(); ++i)
                    {
                        std::string const name = device_name(i);
                        bool const selected = name == shown;
                        if (ImGui::Selectable(fmt::format("{}{}##gpu{}", name, i == gpu_context->physical_device_index ? "  (running)" : "", i).c_str(), selected))
                        {
                            pending = name;
                            save_gpu_preference(name);
                        }
                        if (selected) { ImGui::SetItemDefaultFocus(); }
                    }
                    ImGui::EndCombo();
                }
                ImGui::SetItemTooltip("Running on: %s. The choice is saved to %s and used when the app starts next.", current.c_str(), GPU_PREFERENCE_PATH);
                if (!pending.empty() && pending != current)
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "Restart to switch to %s", pending.c_str());
                }
            }
            std::array<char const * const, 2> aa_modes = {
                "NONE",
                "SUPER_SAMPLE",
            };
            ImGui::Combo("anti_aliasing_mode", &render_data.settings.anti_aliasing_mode, aa_modes.data(), s_cast<i32>(aa_modes.size()));
            if (ImGui::CollapsingHeader("Lights Settings"))
            {
                {
                    auto & flashlight = render_context.flashlight;
                    ImGui::SeparatorText("Flashlight");
                    ImGui::Checkbox("Flashlight (F)", &flashlight.enabled);
                    if (flashlight.enabled && !render_context.flashlight_spot_index.has_value())
                    {
                        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f), "No free spot light slot (max %i)", MAX_SPOT_LIGHTS);
                    }
                    ImGui::ColorEdit3("Flashlight Color", &flashlight.color.x);
                    ImGui::DragFloat("Flashlight Intensity (cd)", &flashlight.intensity, 10.0f, 0.0f, 1'000'000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
                    ImGui::DragFloat("Flashlight Range", &flashlight.range, 0.5f, 0.1f, 1000.0f);
                    ImGui::SliderAngle("Flashlight Inner Angle", &flashlight.inner_cone_angle, 0.0f, 89.0f);
                    ImGui::SliderAngle("Flashlight Outer Angle", &flashlight.outer_cone_angle, 0.5f, 89.0f);
                    ImGui::DragFloat3("Flashlight Offset", &flashlight.offset.x, 0.01f, -2.0f, 2.0f);
                    ImGui::SetItemTooltip("Camera space: x right, y up, z backwards");
                    ImGui::Separator();
                }
                {
                    debug_mode_sub_selector("lights debug visualization", &lights_debug_visualization, {
                        DEBUG_DRAW_MODE_DIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_DIRECT_SPECULAR,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO,
                        DEBUG_DRAW_MODE_AO,
                        DEBUG_DRAW_MODE_INDIRECT_SPECULAR,
                        DEBUG_DRAW_MODE_ALL_DIFFUSE,
                        DEBUG_DRAW_MODE_ALL_SPECULAR,
                        DEBUG_DRAW_MODE_ALL_LIGHTING,
                        DEBUG_DRAW_MODE_SHADE_OPAQUE_CLOCKS,
                        DEBUG_DRAW_MODE_LIGHT_MASK_VOLUME,
                        DEBUG_DRAW_MODE_ALBEDO,
                        DEBUG_DRAW_MODE_ROUGHNESS,
                        DEBUG_DRAW_MODE_METALNESS,
                        DEBUG_DRAW_MODE_MAPPED_NORMAL,
                    }, debug_visualization_index_override);
                }
                ImGui::InputFloat3("Mask Volume Size", &render_data.light_settings.mask_volume_size.x);
                ImGui::InputInt3("Mask Volume Cell Count", &render_data.light_settings.mask_volume_cell_count.x);
                ImGui::Checkbox("Cull All Point Lights", r_cast<bool*>(&render_data.light_settings.cull_all_point_lights));
                ImGui::Checkbox("Cull All Spot Lights", r_cast<bool*>(&render_data.light_settings.cull_all_spot_lights));
                ImGui::Checkbox("Draw Point Light Influence", r_cast<bool*>(&render_data.light_settings.debug_draw_point_influence));
                ImGui::Checkbox("Draw Spot Light Influence", r_cast<bool*>(&render_data.light_settings.debug_draw_spot_influence));
                ImGui::Checkbox("Mark Influence", r_cast<bool*>(&render_data.light_settings.debug_mark_influence));
                ImGui::Checkbox("Mark Influence Shadowed", r_cast<bool*>(&render_data.light_settings.debug_mark_influence_shadowed));
                ImGui::SliderInt("Debug Point Light Idx", &render_data.light_settings.selected_debug_point_light, -1, render_data.light_settings.point_light_count-1);
                ImGui::SliderInt("Debug Spot Light Idx", &render_data.light_settings.selected_debug_spot_light, -1, render_data.light_settings.spot_light_count-1);
            }
            
            if (ImGui::CollapsingHeader("Debug"))
            {
                auto const & modes = DEBUG_DRAW_MODE_NAMES;
                tido::ui::filter_combo("debug visualization", &debug_visualization_index, modes.data(), s_cast<i32>(modes.size()));
                ImGui::InputFloat("debug visualization scale", &render_data.settings.debug_visualization_scale);
                ImGui::SliderFloat("debug visualization blend", &render_data.settings.debug_visualization_blend, 0.0f, 1.0f);
                ImGui::SetItemTooltip("0 = debug overlay blended max with normal image, 1 = full debug overlay");
                ImGui::SliderInt("debug visualization tile", &render_data.settings.debug_visualization_tile, -1, 15);
                ImGui::SetItemTooltip("-1 = full screen, 0-15 = draw into that tile slot (4x4 grid)");

                ImGui::SeparatorText("Debug Tape");
                ImGui::TextUnformatted("Graphs the per-frame general readback debug_value float4 (x=red, y=green, z=blue, w=yellow).");
                {
                    auto & debug_ctx = render_context.gpu_context->shader_debug_context;

                    daxa_f32vec4 const latest = render_context.general_readback.debug_value;
                    daxa_f32vec4 const smoothed = debug_ctx.tape_smoothed;
                    ImGui::Text("latest:   (% 10.4f, % 10.4f, % 10.4f, % 10.4f)", latest.x, latest.y, latest.z, latest.w);
                    ImGui::Text("smoothed: (% 10.4f, % 10.4f, % 10.4f, % 10.4f)", smoothed.x, smoothed.y, smoothed.z, smoothed.w);
                    ImGui::SetItemTooltip("Exponential moving average, blend factor 1/64 per frame.");

                    static int tape_history = 512;
                    static bool tape_auto_fit = true;
                    static float tape_y_scale = 1.0f;
                    ImGui::SliderInt("tape history (frames)", &tape_history, 16, SHADER_DEBUG_TAPE_SIZE);
                    ImGui::Checkbox("tape auto fit Y", &tape_auto_fit);
                    if (!tape_auto_fit)
                    {
                        ImGui::SameLine();
                        ImGui::SetNextItemWidth(200.0f);
                        ImGui::SliderFloat("Y scale", &tape_y_scale, 0.001f, 1000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                    }

                    const i32 count = std::min<i32>(tape_history, SHADER_DEBUG_TAPE_SIZE);
                    const u32 frame_index = render_context.render_data.frame_index;
                    static std::vector<float> xs = {};
                    static std::array<std::vector<float>, 4> ys = {};
                    xs.resize(count);
                    for (auto & y : ys) { y.resize(count); }
                    // Unwrap the ring oldest -> newest. x is "frames ago" (negative), 0 = latest.
                    for (i32 i = 0; i < count; ++i)
                    {
                        const u32 slot = (frame_index - u32(count - 1 - i)) & (SHADER_DEBUG_TAPE_SIZE - 1);
                        const daxa_f32vec4 v = debug_ctx.tape[slot];
                        xs[i] = static_cast<float>(i - (count - 1));
                        ys[0][i] = v.x; ys[1][i] = v.y; ys[2][i] = v.z; ys[3][i] = v.w;
                    }

                    const ImVec4 line_colors[4] = {
                        ImVec4{1.0f, 0.0f, 0.0f, 1.0f}, // x -> red
                        ImVec4{0.0f, 1.0f, 0.0f, 1.0f}, // y -> green
                        ImVec4{0.2f, 0.4f, 1.0f, 1.0f}, // z -> blue
                        ImVec4{1.0f, 1.0f, 0.0f, 1.0f}, // w -> yellow
                    };
                    const char * const channel_names[4] = {"x", "y", "z", "w"};
                    if (ImPlot::BeginPlot("##debug_tape", {ImGui::GetContentRegionAvail().x, 200.0f}))
                    {
                        const ImPlotAxisFlags y_flags = tape_auto_fit ? ImPlotAxisFlags_AutoFit : ImPlotAxisFlags_None;
                        ImPlot::SetupAxes("frames ago", "value", ImPlotAxisFlags_AutoFit, y_flags);
                        if (!tape_auto_fit)
                        {
                            ImPlot::SetupAxisLimits(ImAxis_Y1, -tape_y_scale, tape_y_scale, ImGuiCond_Always);
                        }
                        for (i32 c = 0; c < 4; ++c)
                        {
                            ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::ColorConvertFloat4ToU32(line_colors[c]));
                            ImPlot::PlotLine(channel_names[c], xs.data(), ys[c].data(), count);
                            ImPlot::PopStyleColor();
                        }
                        ImPlot::EndPlot();
                    }
                }
            }
            if (ImGui::CollapsingHeader("Shading Settings"))
            {
                auto debug_material_quality = std::array{
                    "NONE",           // SHADING_QUALITY_NONE
                    "LOW",            // SHADING_QUALITY_LOW
                    "HIGH",           // SHADING_QUALITY_HIGH
                };
                ImGui::Combo("debug material quality", &render_data.settings.debug_material_quality, debug_material_quality.data(), s_cast<i32>(debug_material_quality.size()));
                ImGui::Checkbox("enable emissives", r_cast<bool *>(&render_data.settings.enable_emissives));
                ImGui::SetItemTooltip("When off, emissive material contribution is dropped everywhere.");
            }
            if (ImGui::CollapsingHeader("Geometry"))
            {
                ImGui::InputInt("override_lod", &render_data.settings.lod_override);
                ImGui::InputFloat("lod_acceptable_pixel_error", &render_data.settings.lod_acceptable_pixel_error);
                ImGui::SetItemTooltip("Pixel errors below one are necessary to avoid shading issues as normals are more sensitive to lodding then positions");
                ImGui::Checkbox("auto lod bias", r_cast<bool *>(&render_data.settings.enable_auto_lod_bias));
                ImGui::SetItemTooltip("Raises the lod bias when the visbuffer mesh/meshlet buffers run above the usage soft limit (sharply above 100%%), slowly lowers it again otherwise.");
                ImGui::BeginDisabled(render_data.settings.enable_auto_lod_bias);
                ImGui::SliderFloat("lod bias", &render_data.settings.lod_bias, 0.0f, RenderContext::AUTO_LOD_BIAS_MAX);
                ImGui::SetItemTooltip("log2 scale on the acceptable pixel error.");
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::SmallButton("reset##lod_bias"))
                {
                    render_data.settings.lod_bias = 0.0f;
                }
            }
            if (ImGui::CollapsingHeader("Task Graph"))
            {
                ImGui::Checkbox("enable async compute", r_cast<bool *>(&render_data.settings.enable_async_compute));
                ImGui::Checkbox("enable memory aliasing", r_cast<bool *>(&render_data.settings.enable_memory_aliasing));
                ImGui::Checkbox("enable task reordering", r_cast<bool *>(&render_data.settings.enable_task_reordering));
                ImGui::Checkbox("optimize transient lifetimes", r_cast<bool *>(&render_data.settings.optimize_transient_lifetimes));
            }
        }
        ImGui::SeparatorText("Features");
        {
            if (ImGui::CollapsingHeader("Visbuffer Pipeline Settings"))
            {
                {
                    debug_mode_sub_selector("visbuffer debug visualization", &visbuffer_debug_visualization, {
                        DEBUG_DRAW_MODE_OVERDRAW,
                        DEBUG_DRAW_MODE_TRIANGLE_CONNECTIVITY,
                        DEBUG_DRAW_MODE_TRIANGLE_ID,
                        DEBUG_DRAW_MODE_MESHLET_ID,
                        DEBUG_DRAW_MODE_MESH_ID,
                        DEBUG_DRAW_MODE_MESH_GROUP_ID,
                        DEBUG_DRAW_MODE_ENTITY_ID,
                        DEBUG_DRAW_MODE_MESH_LOD,
                        DEBUG_DRAW_MODE_DEPTH,
                        DEBUG_DRAW_MODE_ALBEDO,
                        DEBUG_DRAW_MODE_ROUGHNESS,
                        DEBUG_DRAW_MODE_METALNESS,
                        DEBUG_DRAW_MODE_UV,
                        DEBUG_DRAW_MODE_FACE_NORMAL,
                        DEBUG_DRAW_MODE_SMOOTH_NORMAL,
                        DEBUG_DRAW_MODE_MAPPED_NORMAL,
                        DEBUG_DRAW_MODE_FACE_TANGENT,
                        DEBUG_DRAW_MODE_SMOOTH_TANGENT,
                    }, debug_visualization_index_override);
                }
                ImGui::Checkbox("enable_mesh_cull", reinterpret_cast<bool *>(&render_data.settings.enable_mesh_cull));
                ImGui::Checkbox("enable_meshlet_cull", reinterpret_cast<bool *>(&render_data.settings.enable_meshlet_cull));
                ImGui::Checkbox("enable_triangle_cull", reinterpret_cast<bool *>(&render_data.settings.enable_triangle_cull));
                ImGui::Checkbox("enable_separate_compute_meshlet_culling", reinterpret_cast<bool *>(&render_data.settings.enable_separate_compute_meshlet_culling));
                ImGui::Checkbox("enable_prefix_sum_work_expansion", reinterpret_cast<bool *>(&render_data.settings.enable_prefix_sum_work_expansion));
                ImGui::SliderFloat("buffer usage soft limit", &render_data.settings.visbuffer_usage_soft_limit, 0.1f, 1.0f, "%.2f");
                ImGui::SetItemTooltip("Relative mesh/meshlet buffer usage above which the auto lod bias starts rising.\nThe bias descends again once usage is 0.1 below this limit.");
            }
            if (ImGui::CollapsingHeader("PGI Settings"))
            {
                {
                    debug_mode_sub_selector("pgi debug visualization", &pgi_debug_visualization, {
                        DEBUG_DRAW_MODE_PGI_EVAL_CLOCKS,
                        DEBUG_DRAW_MODE_PGI_CASCADE_SMOOTH,
                        DEBUG_DRAW_MODE_PGI_CASCADE_ABSOLUTE,
                        DEBUG_DRAW_MODE_PGI_LOW_QUALITY_SAMPLING,
                        DEBUG_DRAW_MODE_PGI_IRRADIANCE,
                        DEBUG_DRAW_MODE_PGI_RADIANCE,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO,
                        DEBUG_DRAW_MODE_AO,
                        DEBUG_DRAW_MODE_ALL_DIFFUSE,
                    }, debug_visualization_index_override);
                }
                ImGui::Checkbox("Enable", reinterpret_cast<bool *>(&render_data.pgi_settings.enabled));
                ImGui::Checkbox("Enable Probe Repositioning", reinterpret_cast<bool *>(&render_data.pgi_settings.probe_repositioning));
                ImGui::Checkbox("Enable Probe Repositioning Spring", reinterpret_cast<bool *>(&render_data.pgi_settings.probe_repositioning_spring_force));
                ImGui::InputInt("Cascade Count", &render_data.pgi_settings.cascade_count);
                ImGui::SliderFloat("Cascade Blend", &render_data.pgi_settings.cascade_blend, 0.0f, 1.0f);
                auto update_rates = std::array{
                    "FULL",    // PGI_UPDATE_RATE_FULL
                    "1_OF_2",  // PGI_UPDATE_RATE_1_OF_2
                    "1_OF_8",  // PGI_UPDATE_RATE_1_OF_8
                    "1_OF_16", // PGI_UPDATE_RATE_1_OF_16
                    "1_OF_32", // PGI_UPDATE_RATE_1_OF_32
                    "1_OF_64", // PGI_UPDATE_RATE_1_OF_64
                };
                ImGui::Combo("Update Rate", &render_data.pgi_settings.update_rate, update_rates.data(), s_cast<i32>(update_rates.size()));
                ImGui::InputInt("Probe Surface Resolution", &render_data.pgi_settings.probe_color_resolution);
                ImGui::InputInt("Probe Trace Resolution  ", &render_data.pgi_settings.probe_trace_resolution);
                ImGui::InputInt("Probe Visibility Resolution  ", &render_data.pgi_settings.probe_visibility_resolution);
                ImGui::InputFloat("Probe cos wrap around", &render_data.pgi_settings.cos_wrap_around);
                ImGui::InputFloat3("Probe range", &render_data.pgi_settings.probe_range.x);
                ImGui::InputInt3("Probe Count", &render_data.pgi_settings.probe_count.x);
                ImGui::SeparatorText("Debug");
                ImGui::Checkbox("Debug Draw Probe Influence", reinterpret_cast<bool *>(&render_data.pgi_settings.debug_probe_influence));
                ImGui::Checkbox("Debug Draw Probe Repositioning", reinterpret_cast<bool *>(&render_data.pgi_settings.debug_draw_repositioning));
                ImGui::Checkbox("Debug Draw Probe Repositioning Forces", reinterpret_cast<bool *>(&render_data.pgi_settings.debug_draw_repositioning_forces));
                ImGui::Checkbox("Debug Draw Probe Grid", reinterpret_cast<bool *>(&render_data.pgi_settings.debug_draw_grid));
                auto debug_daw_modes = std::array{
                    "OFF",                  // PGI_DEBUG_PROBE_DRAW_MODE_OFF
                    "IRRADIANCE",           // PGI_DEBUG_PROBE_DRAW_MODE_IRRADIANCE
                    "RADIANCE",             // PGI_DEBUG_PROBE_DRAW_MODE_RADIANCE
                    "RADIANCE_REFLECTION",  // PGI_DEBUG_PROBE_DRAW_MODE_RADIANCE_REFLECTION
                    "DISTANCE",             // PGI_DEBUG_PROBE_DRAW_MODE_DISTANCE
                    "UNCERTAINTY",          // PGI_DEBUG_PROBE_DRAW_MODE_UNCERTAINTY
                    "COLOR_TEXEL",          // PGI_DEBUG_PROBE_DRAW_MODE_COLOR_TEXEL
                    "DEPTH_TEXEL",          // PGI_DEBUG_PROBE_DRAW_MODE_DEPTH_TEXEL
                    "UV",                   // PGI_DEBUG_PROBE_DRAW_MODE_UV
                    "NORMAL",               // PGI_DEBUG_PROBE_DRAW_MODE_NORMAL
                    "HYSTERESIS",           // PGI_DEBUG_PROBE_DRAW_MODE_HYSTERESIS
                };
                tido::ui::filter_combo("Debug Probe Draw", &render_data.pgi_settings.debug_probe_draw_mode, debug_daw_modes.data(), s_cast<i32>(debug_daw_modes.size()));
                ImGui::InputInt("Debug Force Cascade", &render_data.pgi_settings.debug_force_cascade);
                ImGui::InputInt3("Debug Probe Index", &render_data.pgi_settings.debug_probe_index.x);
            }
            if (ImGui::CollapsingHeader("RTGI Settings"))
            {
                debug_mode_sub_selector("rtgi debug visualization", &rtgi_debug_visualization, {
                    DEBUG_DRAW_MODE_RTGI_DEBUG_PRIMARY_TRACE,
                    DEBUG_DRAW_MODE_RTGI_DIFFUSE_TRACE_CLOCKS,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_TRACE_CLOCKS,
                    DEBUG_DRAW_MODE_RTGI_TOTAL_RAYS_SHOT,
                    DEBUG_DRAW_MODE_RTGI_DIFFUSE_RAYS_SHOT,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_RAYS_SHOT,
                    DEBUG_DRAW_MODE_RTGI_TOTAL_RAYS_PER_TILE,
                    DEBUG_DRAW_MODE_RTGI_DIFFUSE_RAYS_PER_TILE,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_RAYS_PER_TILE,
                    DEBUG_DRAW_MODE_RTGI_RAY_SHARE,
                    DEBUG_DRAW_MODE_RTGI_RAY_MATERIAL,
                    DEBUG_DRAW_MODE_RTGI_GUIDE_DIRECTION,
                    DEBUG_DRAW_MODE_RTGI_GUIDE_LOW_RES_DIRECTION,
                    DEBUG_DRAW_MODE_RTGI_DIFFUSE_AO_GUIDE,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_HIT_DISTANCE,
                    DEBUG_DRAW_MODE_RTGI_DIFFUSE_PERCEPTUAL_MEAN,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_PERCEPTUAL_MEAN,
                    DEBUG_DRAW_MODE_RTGI_DIFFUSE_AO_GUIDE_TEMPORAL,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_HIT_DISTANCE_TEMPORAL,
                    DEBUG_DRAW_MODE_RTGI_DIFFUSE_PERCEPTUAL_MEAN_TEMPORAL,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_PERCEPTUAL_MEAN_TEMPORAL,
                    DEBUG_DRAW_MODE_RTGI_DIFFUSE_HISTORY_LENGTH,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_HISTORY_LENGTH,
                    DEBUG_DRAW_MODE_RTGI_DIFFUSE_TEMPORAL_REACTIVITY,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_TEMPORAL_REACTIVITY,
                    DEBUG_DRAW_MODE_RTGI_SPECULAR_VIRTUAL_AMOUNT,
                    DEBUG_DRAW_MODE_RTGI_HASH_GRID_CELLS,
                    DEBUG_DRAW_MODE_RTGI_HASH_GRID_RADIANCE,
                    DEBUG_DRAW_MODE_RTGI_HASH_GRID_SAMPLE_COUNT,
                    DEBUG_DRAW_MODE_RTGI_HASH_GRID_RADIANCE_JITTERED,
                    DEBUG_DRAW_MODE_INDIRECT_DIFFUSE,
                    DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO,
                    DEBUG_DRAW_MODE_AO,
                    DEBUG_DRAW_MODE_DIRECT_SPECULAR,
                    DEBUG_DRAW_MODE_INDIRECT_SPECULAR,
                    DEBUG_DRAW_MODE_ALL_DIFFUSE,
                    DEBUG_DRAW_MODE_ALL_SPECULAR,
                    DEBUG_DRAW_MODE_ALL_LIGHTING,
                    DEBUG_DRAW_MODE_ROUGHNESS,
                    DEBUG_DRAW_MODE_METALNESS,
                }, debug_visualization_index_override);
                ImGui::Checkbox("Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.enabled));
                // Accumulated GPU time (microseconds) for a set of render-time slots.
                auto rtgi_section_us = [&](std::initializer_list<u32> indices) -> f32 {
                    u64 sum = 0;
                    for (u32 i : indices) { sum += render_context.render_times.get_average(i); }
                    return static_cast<f32>(sum) * 0.001f;
                };
                // Section tree node that shows the section's accumulated GPU time on its header line.
                auto rtgi_section = [&](char const * name, std::initializer_list<u32> indices) -> bool {
                    bool const open = ImGui::TreeNodeEx(name, ImGuiTreeNodeFlags_None);
                    if (indices.size() > 0) // sections without timers (e.g. Specular) show no time
                    {
                        ImGui::SameLine();
                        ImGui::TextDisabled("%8.1f us", rtgi_section_us(indices));
                    }
                    return open;
                };
                // Per-category sub blocks: General / Diffuse / Specular. PushID keeps same-named controls apart.
                auto rtgi_sub_begin = [](char const * label) { ImGui::SeparatorText(label); ImGui::PushID(label); };
                auto rtgi_sub_end = []() { ImGui::PopID(); };
                if (rtgi_section("Specular", {}))
                {
                    // Specular signal + material overrides. Per-stage specular settings live in each stage's Specular block.
                    ImGui::Checkbox("Specular Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.specular_enabled));
                    ImGui::SliderFloat("Additive Upwards Gloss", &render_data.rtgi_settings.specular_upward_gloss, 0.0f, 1.0f);
                    ImGui::SetItemTooltip("Lowers roughness on surfaces facing up (wet / polished floors).");
                    ImGui::SliderFloat("Additive Total Gloss", &render_data.rtgi_settings.specular_total_gloss, 0.0f, 1.0f);
                    ImGui::SetItemTooltip("Lowers roughness on all surfaces (roughness *= 1 - gloss).");
                    ImGui::SliderFloat("Additive Metalness", &render_data.rtgi_settings.specular_additive_metalness, 0.0f, 1.0f);
                    ImGui::SetItemTooltip("Adds to material metalness on all surfaces (metalness = saturate(metalness + value)).");
                    ImGui::SliderFloat("Max Gloss", &render_data.rtgi_settings.specular_max_gloss, 0.0f, 1.0f);
                    ImGui::SetItemTooltip("Caps material gloss (1 - roughness) everywhere, applied after the additive gloss knobs.");
                    ImGui::TreePop();
                }
                if (rtgi_section("Trace", {
                        RenderTimes::index<"RTGI", "DISTRIBUTE_RAYS">(),
                        RenderTimes::index<"RTGI", "TRACE">(),
                        RenderTimes::index<"RTGI", "SHADE_HITS">(),
                        RenderTimes::index<"RTGI", "BLEND_RAYS">(),
                    }))
                {
                    rtgi_sub_begin("General");
                    ImGui::SliderFloat("Shading AO Range", &render_data.rtgi_settings.shading_ao_range, 0.0f, 4.0f);
                    ImGui::Checkbox("Use Repacked Ray Dispatch", reinterpret_cast<bool *>(&render_data.rtgi_settings.use_repacked_ray_dispatch));
                    ImGui::SetItemTooltip("Repacked: slim trace that only records hits, then specialized shade passes (sharp hits: full material + light shading, coarse hits: PGI radiance, misses: sky). Off: classic per-pixel trace that shades in the closest hit.");
                    ImGui::BeginDisabled(render_data.rtgi_settings.use_repacked_ray_dispatch == 0);
                    ImGui::Checkbox("Coarse Shade Wide Specular", reinterpret_cast<bool *>(&render_data.rtgi_settings.split_coarse_wide_specular));
                    ImGui::SetItemTooltip("Specular hits whose GGX cone at the hit is at least as wide as the PGI probe spacing there use coarse (PGI radiance) shading instead of full shading. Repacked ray dispatch only.");
                    ImGui::EndDisabled();
                    ImGui::Checkbox("Force Fine Hit Shading", reinterpret_cast<bool *>(&render_data.rtgi_settings.force_fine_hit_shading));
                    ImGui::SetItemTooltip("Every hit (diffuse rays too) gets full material + light shading with shadow rays instead of the PGI radiance shortcut.\n"
                                      "Overrides Coarse Shade Wide Specular. Both trace paths, not the pioneer trace.");
                    ImGui::SliderFloat("Ray Budget (rays/pixel)", &render_data.rtgi_settings.ray_percentage, 0.0f, 4.0f);
                    ImGui::SliderFloat("Min Ray Budget (guaranteed frac)", &render_data.rtgi_settings.min_ray_budget, 0.0f, 1.0f);
                    ImGui::SliderInt("Max Rays Per Pixel", &render_data.rtgi_settings.max_rays_per_pixel, 1, 32);
                    ImGui::SetItemTooltip("Max rays (diffuse + specular, base included) a pixel may request per frame. A fully fresh pixel asks for exactly this many. Too many rays in one frame are temporally similar and form stripes. Never below the base rays (2 with specular).");
                    {
                        ImGui::SliderFloat("Diffuse/Spec Share Slope", &render_data.rtgi_settings.ray_share_slope, 0.0f, 4.0f, "%.2f");
                        ImGui::SetItemTooltip("How strongly the visibly brighter signal (diffuse vs specular final radiance, in stops) takes the pixel's extra rays. 0 = even, 1 = linear ratio, 2 = squared ratio. Factors always sum to 2.");
                        ImGui::SliderFloat("Metal Diffuse Tolerance (stops)", &render_data.rtgi_settings.ray_diffuse_metal_tolerance_stops, 0.0f, 0.5f, "%.3f");
                        ImGui::SetItemTooltip("Diffuse extra rays fade out once leaving diffuse out would change the pixel by less than this many stops (material only). 0.1 -> full rays up to metalness ~0.93, none at 1. 0 = off.");
                        ImGui::SliderFloat("Rough Specular Cutoff Start", &render_data.rtgi_settings.ray_specular_roughness_cutoff_start, 0.0f, 1.0f, "%.2f");
                        ImGui::SliderFloat("Rough Specular Cutoff End", &render_data.rtgi_settings.ray_specular_roughness_cutoff_end, 0.0f, 1.0f, "%.2f");
                        ImGui::SetItemTooltip("Specular extra rays fade from 1 at Start roughness to 0 at End roughness (base ray stays). Start >= End = off.");
                    }
                    ImGui::Checkbox("Ray Redistribution", reinterpret_cast<bool *>(&render_data.rtgi_settings.use_ray_redistribution));
                    ImGui::Checkbox("Pioneer Guided Ray Bending", reinterpret_cast<bool *>(&render_data.rtgi_settings.pioneer_guiding_enabled));
                    ImGui::SliderFloat("Pioneer Trace Max Distance (m)", &render_data.rtgi_settings.guide_pioneer_trace_max_distance, 1.0f, 1024.0f);
                    rtgi_sub_end();
                    rtgi_sub_begin("Diffuse");
                    ImGui::Checkbox("Trace Use STBN", reinterpret_cast<bool *>(&render_data.rtgi_settings.trace_use_stbn));
                    ImGui::SliderFloat("Guide Concentration", &render_data.rtgi_settings.guide_concentration, 0.0f, 1.0f);
                    rtgi_sub_end();
                    rtgi_sub_begin("Specular");
                    ImGui::SliderFloat("Pioneer Guide Mix", &render_data.rtgi_settings.specular_guide_mix, 0.0f, 1.0f);
                    ImGui::SetItemTooltip("Probability (at roughness 1) of drawing a specular ray from the pioneer guide lobe (MIS weighted).");
                    ImGui::SliderFloat("Guide Concentration", &render_data.rtgi_settings.specular_guide_concentration, 0.0f, 1.0f);
                    rtgi_sub_end();
                    ImGui::TreePop();
                }
                if (rtgi_section("Prefilter", { RenderTimes::index<"RTGI", "PRE_FILTER">() }))
                {
                    rtgi_sub_begin("Diffuse");
                    ImGui::Checkbox("Firefly Filter Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.firefly_filter_enabled));
                    ImGui::RadioButton("Multichromatic", &render_data.rtgi_settings.firefly_clamp_mode, 0); ImGui::SameLine();
                    ImGui::RadioButton("Monochromatic", &render_data.rtgi_settings.firefly_clamp_mode, 1);
                    ImGui::SliderFloat("Firefly Perceptual Tolerance", &render_data.rtgi_settings.firefly_perceptual_tolerance, 0.25f, 16.0f);
                    ImGui::SliderFloat("Ambient Occlusion Guide Max Pixel Range", &render_data.rtgi_settings.max_visibility_pixel_range, 1.0f, 128.0f);
                    rtgi_sub_end();
                    rtgi_sub_begin("Specular");
                    ImGui::Checkbox("Firefly Filter Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.specular_firefly_filter_enabled));
                    ImGui::RadioButton("Multichromatic", &render_data.rtgi_settings.specular_firefly_clamp_mode, 0); ImGui::SameLine();
                    ImGui::RadioButton("Monochromatic", &render_data.rtgi_settings.specular_firefly_clamp_mode, 1);
                    ImGui::SliderFloat("Firefly Perceptual Tolerance", &render_data.rtgi_settings.specular_firefly_perceptual_tolerance, 0.25f, 16.0f);
                    rtgi_sub_end();
                    ImGui::TreePop();
                }
                if (rtgi_section("Pre Blur", {
                        RenderTimes::index<"RTGI", "PRE_BLUR0">(),
                        RenderTimes::index<"RTGI", "PRE_BLUR1">(),
                        RenderTimes::index<"RTGI", "PRE_BLUR2">(),
                        RenderTimes::index<"RTGI", "PRE_BLUR3">(),
                    }))
                {
                    rtgi_sub_begin("General");
                    ImGui::Checkbox("Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.pre_blur_enabled));
                    ImGui::SliderFloat("Base Width", &render_data.rtgi_settings.pre_blur_base_width, 1, 256);
                    ImGui::SliderInt("Sample Count", &render_data.rtgi_settings.pre_blur_sample_count, 1, 32);
                    ImGui::SliderInt("Iterations", &render_data.rtgi_settings.pre_blur_iterations, 1, 4);
                    ImGui::Checkbox("Ambient Occlusion Guide Radius Scaling", reinterpret_cast<bool *>(&render_data.rtgi_settings.pre_blur_ao_guiding));
                    ImGui::SetItemTooltip("Shrinks the blur radius in occluded areas (AO guide from the diffuse rays). Scales both the diffuse disc and the specular lobe radius.");
                    ImGui::SliderFloat("Radius Scaling Floor", &render_data.rtgi_settings.ao_guide_floor, 0.0f, 1.0f);
                    ImGui::Checkbox("Ray Count Sample Weighting", reinterpret_cast<bool *>(&render_data.rtgi_settings.pre_blur_ray_count_sample_weighting));
                    ImGui::SetItemTooltip("Taps weighted by the rays they shot this frame (diffuse taps by diffuse rays, specular taps by specular rays).");
                    ImGui::Checkbox("Firefly Energy Compensation Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.pre_blur_firefly_energy_compensation_enabled));
                    ImGui::SetItemTooltip("Iteration 0: each signal's firefly clamp energy factor boosts its tap weights, spreading the clamped energy instead of losing it.");
                    rtgi_sub_end();
                    rtgi_sub_begin("Diffuse");
                    ImGui::Checkbox("Perceptual Radiance Difference Guiding", reinterpret_cast<bool *>(&render_data.rtgi_settings.pre_blur_perceptual_difference_guiding));
                    ImGui::SliderFloat("Perceptual Radiance Tolerance", &render_data.rtgi_settings.pre_blur_perceptual_radiance_guide_tolerance, 0.0f, 2.0f);
                    rtgi_sub_end();
                    rtgi_sub_begin("Specular");
                    ImGui::SliderFloat("Blur Scale", &render_data.rtgi_settings.specular_pre_blur_scale, 0.0f, 4.0f);
                    ImGui::SetItemTooltip("Scales the specular lobe-footprint radius; taps outside it are dropped (0 = no specular pre blur).");
                    rtgi_sub_end();
                    ImGui::TreePop();
                }
                if (rtgi_section("Temporal", {
                        RenderTimes::index<"RTGI", "TEMPORAL_REPROJECT">(),
                        RenderTimes::index<"RTGI", "TEMPORAL_ACCUMULATION">(),
                        RenderTimes::index<"RTGI", "HASH_GRID_QUERY">(),
                        RenderTimes::index<"RTGI", "HASH_GRID_ALLOCATE">(),
                        RenderTimes::index<"RTGI", "HASH_GRID_SCATTER">(),
                        RenderTimes::index<"RTGI", "HASH_GRID_BLEND">(),
                        RenderTimes::index<"RTGI", "HASH_GRID_DEBUG">(),
                    }))
                {
                    rtgi_sub_begin("General");
                    ImGui::Checkbox("Temporal Noise Animation", reinterpret_cast<bool *>(&render_data.rtgi_settings.animate_noise));
                    rtgi_sub_end();
                    rtgi_sub_begin("Diffuse");
                    ImGui::Checkbox("Accumulation Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.temporal_accumulation_enabled));
                    ImGui::SliderInt("Max Temporal Samples", &render_data.rtgi_settings.max_temporal_samples, 1, 255);
                    ImGui::SliderFloat("Fast Convergence Samples", &render_data.rtgi_settings.fast_convergence_samples, 1.0f, 128.0f);
                    ImGui::Checkbox("Fast History Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.temporal_fast_history_enabled));
                    ImGui::SliderInt("Fast History Frames", &render_data.rtgi_settings.temporal_fast_history_frames, 1, 15);
                    ImGui::Checkbox("Temporal Firefly Filter Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.temporal_firefly_filter_enabled));
                    ImGui::SliderFloat("Temporal Firefly Std Dev Clamp", &render_data.rtgi_settings.temporal_firefly_std_dev_clamp, 0.0f, 8.0f);
                    ImGui::SliderFloat("Variance Fast History Blend", &render_data.rtgi_settings.temporal_variance_fast_history_blend, 0.0f, 8.0f);
                    ImGui::SliderFloat("Parallax Penalty Strength", &render_data.rtgi_settings.temporal_parallax_penalty_strength, 0.0f, 4.0f);
                    rtgi_sub_end();
                    rtgi_sub_begin("Specular");
                    ImGui::Checkbox("Accumulation Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.specular_temporal_accumulation_enabled));
                    ImGui::SliderFloat("Max Temporal Frames", &render_data.rtgi_settings.specular_max_temporal_frames, 1.0f, 128.0f);
                    ImGui::SliderFloat("Fast Convergence Samples", &render_data.rtgi_settings.specular_fast_convergence_samples, 1.0f, 128.0f);
                    ImGui::SetItemTooltip("Ray demand target for specular (capped by Max Temporal Frames).");
                    ImGui::Checkbox("Fast History Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.specular_fast_history_enabled));
                    ImGui::SliderInt("Fast History Frames", &render_data.rtgi_settings.specular_fast_history_frames, 1, 15);
                    ImGui::Checkbox("Temporal Firefly Filter Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.specular_temporal_firefly_filter_enabled));
                    ImGui::SliderFloat("Temporal Firefly Std Dev Clamp", &render_data.rtgi_settings.specular_temporal_firefly_std_dev_clamp, 0.0f, 8.0f);
                    ImGui::SliderFloat("Variance Fast History Blend", &render_data.rtgi_settings.specular_temporal_variance_fast_history_blend, 0.0f, 8.0f);
                    ImGui::SliderFloat("Parallax Penalty Strength", &render_data.rtgi_settings.specular_temporal_parallax_penalty_strength, 0.0f, 4.0f);
                    ImGui::Checkbox("Specular Virtual Motion Reprojection", reinterpret_cast<bool *>(&render_data.rtgi_settings.specular_virtual_reprojection));
                    ImGui::Checkbox("Catmull-Rom History", reinterpret_cast<bool *>(&render_data.rtgi_settings.specular_catrom_history));
                    ImGui::SetItemTooltip("Fetch specular history with a 12-tap Catmull-Rom (sharper under motion) where the full footprint is valid; bilinear otherwise.");
                    ImGui::SliderFloat("Reprojection Disagreement Strength", &render_data.rtgi_settings.specular_reprojection_disagreement_strength, 0.0f, 4.0f);
                    ImGui::SetItemTooltip("Shortens the specular history where surface and virtual motion reprojection disagree (in stops).");
                    rtgi_sub_end();
                    rtgi_sub_begin("Hash Grid");
                    ImGui::Checkbox("Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.hash_grid_enabled));
                    ImGui::SetItemTooltip("World-space cascaded hash grid of diffuse radiance. Fallback only: disocclusion fill and failed upscale taps.");
                    ImGui::BeginDisabled(render_data.rtgi_settings.hash_grid_enabled == 0);
                    ImGui::Checkbox("Hash Grid Revive", reinterpret_cast<bool *>(&render_data.rtgi_settings.hash_grid_disocclusion_fill));
                    ImGui::SetItemTooltip("Temporal accumulate: every pixel reads its hash grid cell and re-initializes its temporal history from it (color, the cell's average sample count, fast history) whenever the cell's history is longer than its own.");
                    ImGui::EndDisabled();
                    ImGui::SliderFloat("Cell Size (px)", &render_data.rtgi_settings.hash_grid_cell_size_px, 1.0f, 64.0f);
                    ImGui::SetItemTooltip("Target cell edge in half-res pixels. Rounded up to the next cascade level (power of two).");
                    ImGui::SliderFloat("Min Cell Size (m)", &render_data.rtgi_settings.hash_grid_min_cell_size, 0.005f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                    ImGui::SetItemTooltip("Cell size of cascade 0. Changing cell sizes orphans all cells (they age out).");
                    ImGui::SliderInt("Max Untouched Frames", &render_data.rtgi_settings.hash_grid_max_untouched_frames, 1, 256);
                    ImGui::SetItemTooltip("Cells no pixel queried for this many frames in a row are freed.");
                    ImGui::Checkbox("Draw Cell AABBs", reinterpret_cast<bool *>(&render_data.rtgi_settings.hash_grid_debug_draw_cells));
                    ImGui::SetItemTooltip("Draw the world-space box of every cell queried this frame, colored per cell. Cells of different normal buckets share a box. Capped by the debug AABB draw capacity.");
                    rtgi_sub_end();
                    ImGui::TreePop();
                }
                if (rtgi_section("Post Blur", {
                        RenderTimes::index<"RTGI", "POST_BLUR_VERTICAL">(),
                        RenderTimes::index<"RTGI", "POST_BLUR_HORIZONTAL">(),
                        RenderTimes::index<"RTGI", "POST_BLUR_ATROUS0">(),
                        RenderTimes::index<"RTGI", "POST_BLUR_ATROUS1">(),
                        RenderTimes::index<"RTGI", "POST_BLUR_ATROUS2">(),
                        RenderTimes::index<"RTGI", "POST_BLUR_ATROUS3">(),
                        RenderTimes::index<"RTGI", "POST_BLUR_ATROUS4">(),
                        RenderTimes::index<"RTGI", "POST_BLUR_ATROUS5">(),
                        RenderTimes::index<"RTGI", "POST_BLUR_ATROUS6">(),
                        RenderTimes::index<"RTGI", "POST_BLUR_ATROUS7">(),
                    }))
                {
                    rtgi_sub_begin("General");
                    ImGui::Checkbox("Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.post_blur_enabled));
                    ImGui::Checkbox("Disocclusion Blur", reinterpret_cast<bool *>(&render_data.rtgi_settings.post_blur_disocclusion_blur_enabled));
                    ImGui::RadioButton("Bilateral", &render_data.rtgi_settings.post_blur_mode, 0);
                    ImGui::SameLine();
                    ImGui::RadioButton("Atrous", &render_data.rtgi_settings.post_blur_mode, 1);
                    if (render_data.rtgi_settings.post_blur_mode == 1)
                    {
                        ImGui::SliderInt("Atrous Iterations", &render_data.rtgi_settings.post_blur_atrous_iterations, 1, 8);
                    }
                    else
                    {
                        ImGui::SliderInt("Stride", &render_data.rtgi_settings.post_blur_stride, 1, 8);
                        ImGui::SliderInt("Max Width", &render_data.rtgi_settings.post_blur_max_width, 1, 32);
                        // Groupshared (LDS) variant of the separable H/V blur. Same output, perf A/B.
                        // NOTE: the LDS variant caps the filter radius at 16 (its preloaded halo).
                        ImGui::Checkbox("Use Groupshared (LDS)", reinterpret_cast<bool *>(&render_data.rtgi_settings.post_blur_use_lds));
                    }
                    ImGui::Checkbox("Ambient Occlusion Guide Radius Scaling", reinterpret_cast<bool *>(&render_data.rtgi_settings.post_blur_ao_guiding));
                    ImGui::SetItemTooltip("Shrinks the blur width in occluded areas (temporal AO guide). Scales both the diffuse and the specular width.");
                    ImGui::SliderFloat("Radius Scaling Floor", &render_data.rtgi_settings.post_blur_ao_guide_floor, 0.0f, 1.0f);
                    rtgi_sub_end();
                    rtgi_sub_begin("Diffuse");
                    ImGui::Checkbox("Perceptual Radiance Difference Guiding", reinterpret_cast<bool *>(&render_data.rtgi_settings.post_blur_perceptual_difference_guiding));
                    ImGui::SliderFloat("Perceptual Radiance Difference Tolerance", &render_data.rtgi_settings.post_blur_perceptual_radiance_guide_tolerance, 0.0f, 2.0f);
                    rtgi_sub_end();
                    rtgi_sub_begin("Specular");
                    ImGui::SliderFloat("Blur Scale", &render_data.rtgi_settings.specular_post_blur_scale, 0.0f, 4.0f);
                    ImGui::SetItemTooltip("Scales the specular lobe-footprint width (capped at Max Width; 0 = no specular post blur).");
                    rtgi_sub_end();
                    ImGui::TreePop();
                }
                if (rtgi_section("Upscaling", { RenderTimes::index<"RTGI", "UPSCALE">() }))
                {
                    rtgi_sub_begin("General");
                    ImGui::Checkbox("Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.upscale_enabled));
                    ImGui::Checkbox("Wide 3x3 Footprint", reinterpret_cast<bool *>(&render_data.rtgi_settings.upscale_wide_footprint));
                    ImGui::SetItemTooltip("Weight all 9 half-res taps by a Gaussian of their distance to the full-res pixel. Off = classic tent, far-side taps get ~nothing (~2x2 footprint).");
                    rtgi_sub_end();
                    rtgi_sub_begin("Diffuse");
                    ImGui::Checkbox("SH Resolve Enabled", reinterpret_cast<bool *>(&render_data.rtgi_settings.sh_resolve_enabled));
                    rtgi_sub_end();
                    rtgi_sub_begin("Specular");
                    ImGui::Checkbox("Detail Normal + Gloss Weighting", reinterpret_cast<bool *>(&render_data.rtgi_settings.specular_upscale_detail_weighting));
                    ImGui::SetItemTooltip("Upscale weights specular taps by full-res detail normal vs half-res specular normal + gloss. Off = diffuse-style geometry weights only.");
                    rtgi_sub_end();
                    ImGui::TreePop();
                }
            }
            if (ImGui::CollapsingHeader("VSM Settings"))
            {
                {
                    debug_mode_sub_selector("vsm debug visualization", &vsm_debug_visualization, {
                        DEBUG_DRAW_MODE_VSM_OVERDRAW,
                        DEBUG_DRAW_MODE_VSM_CLIP_LEVEL,
                        DEBUG_DRAW_MODE_VSM_SPOT_LEVEL,
                        DEBUG_DRAW_MODE_VSM_POINT_LEVEL,
                        DEBUG_DRAW_MODE_DIRECT_DIFFUSE,
                        DEBUG_DRAW_MODE_SHADE_OPAQUE_CLOCKS,
                        DEBUG_DRAW_MODE_LIGHT_MASK_VOLUME,
                        DEBUG_DRAW_MODE_MESHLET_ID,
                        DEBUG_DRAW_MODE_ENTITY_ID,
                        DEBUG_DRAW_MODE_MESH_LOD,
                    }, debug_visualization_index_override);
                }
                bool enable = s_cast<bool>(render_context.render_data.vsm_settings.enable);
                bool shadow_everything = s_cast<bool>(render_context.render_data.vsm_settings.shadow_everything);
                bool force_clip_level = s_cast<bool>(render_context.render_data.vsm_settings.force_clip_level);
                bool enable_directional_caching = s_cast<bool>(render_context.render_data.vsm_settings.enable_directional_caching);
                bool enable_point_caching = s_cast<bool>(render_context.render_data.vsm_settings.enable_point_caching);
                ImGui::BeginChild("Checkboxes", ImVec2(0, ImGui::CalcTextSize("a").y * 6.0f));
                {
                    ImGui::Text("Draw cascade frustum");
                    ImGui::SetWindowFontScale(0.5);
                    for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
                    {
                        ImGui::Checkbox(fmt::format("##clips{}", clip).c_str(), &render_context.draw_clip_frustum.at(clip));
                        ImGui::SameLine();
                    }
                    ImGui::SetWindowFontScale(1.0);
                }
                ImGui::EndChild();

                ImGui::Checkbox("Enable VSM", &enable);

                ImGui::Checkbox("Shadow everything", &shadow_everything);
                ImGui::Checkbox("Force clip level", &force_clip_level);
                ImGui::SliderInt("Force mesh Lod", &render_context.render_data.vsm_settings.force_mesh_lod, -1, 15);
                ImGui::SliderFloat("Max acceptable mesh lod pixel error", &render_context.render_data.vsm_settings.mesh_lod_max_acceptable_pixel_error, 0.0f, 255.0f);
                ImGui::SliderInt("Vis point light idx", &render_context.render_data.vsm_settings.force_point_light_idx, -1, render_context.render_data.vsm_settings.point_light_count - 1);
                ImGui::SliderInt("Vis spot light idx", &render_context.render_data.vsm_settings.force_spot_light_idx, -1, render_context.render_data.vsm_settings.spot_light_count - 1);
                ImGui::Checkbox("Enable directional caching", &enable_directional_caching);
                ImGui::Checkbox("Enable point caching", &enable_point_caching);
                auto use_fixed_near_far = s_cast<bool>(render_context.render_data.vsm_settings.fixed_near_far);
                ImGui::Checkbox("Use fixed near far", &use_fixed_near_far);
                render_context.render_data.vsm_settings.fixed_near_far = use_fixed_near_far;
                ImGui::SliderFloat("Clip 0 scale", &render_context.render_data.vsm_settings.clip_0_frustum_scale, 0.1f, 20.0f);
                ImGui::SliderFloat("Clip selection bias", &render_context.render_data.vsm_settings.clip_selection_bias, -0.5f, 2.0f);
                ImGui::SliderFloat("Slope bias", &render_context.render_data.vsm_settings.slope_bias, 0.0, 10.0);
                ImGui::SliderFloat("Constant bias", &render_context.render_data.vsm_settings.constant_bias, 0.0, 20.0);
                ImGui::BeginDisabled(!force_clip_level);
                i32 forced_clip_level = render_context.render_data.vsm_settings.forced_clip_level;
                ImGui::SliderInt("Forced clip level", &forced_clip_level, 0, VSM_CLIP_LEVELS - 1);
                ImGui::EndDisabled();

                i32 forced_lower_point_mip_level = render_context.render_data.vsm_settings.forced_lower_point_mip_level;
                i32 forced_upper_point_mip_level = render_context.render_data.vsm_settings.forced_upper_point_mip_level;
                ImGui::SliderInt("Forced lower point mip level", &forced_lower_point_mip_level, 0, 6);
                ImGui::SliderInt("Forced upper point mip level", &forced_upper_point_mip_level, 0, 6);

                ImGui::InputInt("Debug cubemap face", &render_context.debug_frustum);
                ImGui::Checkbox("Visualize point frustum", &render_context.visualize_point_frustum);
                ImGui::Checkbox("Visualize spot frustum", &render_context.visualize_spot_frustum);
                render_context.debug_frustum = glm::clamp(render_context.debug_frustum, -1, 5);

                render_context.render_data.vsm_settings.enable = enable;
                render_context.render_data.vsm_settings.shadow_everything = shadow_everything;
                render_context.render_data.vsm_settings.force_clip_level = force_clip_level;
                render_context.render_data.vsm_settings.forced_clip_level = force_clip_level ? forced_clip_level : -1;
                render_context.render_data.vsm_settings.forced_lower_point_mip_level = std::min(forced_upper_point_mip_level, forced_lower_point_mip_level);
                render_context.render_data.vsm_settings.forced_upper_point_mip_level = std::max(forced_upper_point_mip_level, forced_lower_point_mip_level);
                render_context.render_data.vsm_settings.enable_directional_caching = enable_directional_caching;
                render_context.render_data.vsm_settings.enable_point_caching = enable_point_caching;

                ImGui::Checkbox("Show memory texture", &vsm_windows.view_meta_memory);
                ImGui::Checkbox("Show page texture", &vsm_windows.view_page_table);
                ImGui::Checkbox("Show reconstructed texture", &vsm_windows.view_reconstructed_shadow_map);

                u32 free_pages = render_context.general_readback.free_pages;
                u32 drawn_pages = render_context.general_readback.drawn_pages;
                u32 cached_pages = render_context.general_readback.cached_pages;
                u32 visible_pages = (VSM_META_MEMORY_TABLE_RESOLUTION * VSM_META_MEMORY_TABLE_RESOLUTION) - (cached_pages + free_pages);
                ImGui::Text("Drawn pages %i", drawn_pages);

                ImGui::Text("Free pages %i", free_pages);
                ImGui::Text("Cached not visible pages %i", cached_pages);
                ImGui::Text("Cached visible pages %i", visible_pages);
                ImGui::Text("Total %i", cached_pages + free_pages + visible_pages);

                u32 cached_point = render_context.general_readback.point_spot_cached_pages;
                u32 cached_point_visible = render_context.general_readback.point_spot_cached_visible_pages;
                u32 drawn_point = render_context.general_readback.drawn_point_spot_pages;
                u32 cached_directional = render_context.general_readback.directional_cached_pages;
                u32 cached_directional_visible = render_context.general_readback.directional_cached_visible_pages;
                u32 drawn_directional = render_context.general_readback.drawn_directional_pages;
                ImGui::Text("Cached point spot visible pages %i", cached_point_visible);
                ImGui::Text("Cached point spot pages %i", cached_point);
                ImGui::Text("Drawn point spot pages %i", drawn_point);
                ImGui::Text("Cached directional visible pages %i", cached_directional);
                ImGui::Text("Cached directional pages %i", cached_directional_visible);
                ImGui::Text("Drawn directional pages %i", drawn_directional);
                ImGui::Text("Drawn Total %i", drawn_directional + drawn_point);
                ImGui::Text("Cached Total %i", cached_point + cached_point_visible + cached_directional + cached_directional_visible);
                ImGui::Text("Total %i", cached_pages + free_pages + visible_pages);
            }
            if (ImGui::CollapsingHeader("Clouds Settings"))
            {
                bool enable = s_cast<bool>(render_context.render_data.volumetric_settings.enable);
                ImGui::Checkbox("Enable Volumetrics", &enable);

                ImGui::SeparatorText("Clouds");
                ImGui::Checkbox("Visualize bounds", &render_context.visualize_clouds_bounds);

                ImGui::Dummy({0.0, 3.0f});

                {
                    ImGui::Text("Max secondary ray march steps:");
                    ImGui::DragInt("##secondary_steps", &render_context.render_data.volumetric_settings.secondary_steps, 1, 0, 20);
                }

                {
                    f32 const indent_size = 10.0f;
                    ImGui::Indent(indent_size);
                    ImGui::SeparatorText("Phase function");

                    auto phase_functions = std::array{
                        "HENYEY_GREENSTEIN",
                        "HENYEY_GREENSTEIN_OCTAVES",
                        "DRAINE",
                    };
                    auto phase_function_mappings = std::array{
                        HENYEY_GREENSTEIN,
                        HENYEY_GREENSTEIN_OCTAVES,
                        DRAINE,
                    };

                    i32 & phase_model = render_context.render_data.volumetric_settings.phase_function_model;
                    ImGui::Combo("Model", &phase_model, phase_functions.data(), s_cast<int>(phase_functions.size()));

                    const i32 samples = 1000;
                    std::array<f32, samples> x_values = {};
                    std::array<f32, samples> y_values = {};

                    f32 & g = render_context.render_data.volumetric_settings.g;
                    f32 & diameter = render_context.render_data.volumetric_settings.diameter;

                    f32 & w_0 = render_context.render_data.volumetric_settings.w_0;
                    f32 & w_1 = render_context.render_data.volumetric_settings.w_1;
                    i32 & octaves = render_context.render_data.volumetric_settings.octaves;

                    static f32 extinction = 1.0f;

                    bool modulate_density = (render_context.render_data.volumetric_settings.use_density_modulated_g == 1u);
                    ImGui::Checkbox("Modulate g with density", &modulate_density);
                    render_context.render_data.volumetric_settings.use_density_modulated_g = modulate_density ? 1u : 0u;

                    switch(phase_model)
                    {
                        case HENYEY_GREENSTEIN:
                        {
                            ImGui::DragFloat("g", &g, 0.01f, 0.0f, 1.0f);
                            break;
                        }
                        case HENYEY_GREENSTEIN_OCTAVES:
                        {
                            ImGui::DragFloat("g", &g, 0.01f, 0.0f, 1.0f);
                            ImGui::DragFloat("w_0", &w_0, 0.01f, 0.0f, 10.0f);
                            ImGui::DragFloat("w_1", &w_1, 0.01f, 0.0f, 10.0f);
                            ImGui::InputInt("octaves", &octaves, 1, 1);
                            ImGui::DragFloat("extinction", &extinction, 0.1f, 0.0f, 100.0f);
                            break;
                        }
                        case DRAINE:
                        {
                            ImGui::DragFloat("g", &g, 0.01f, 0.0f, 1.0f);
                            ImGui::DragFloat("diameter", &diameter, 0.5f, 5.0f, 50.0f);
                        }
                    }

                    for (i32 i = 0; i < x_values.size(); i++)
                    {
                        f32 const angle_rad = (f32(i) / samples) * 2.0f * s_cast<f32>(PI);
                        f32 const cos_theta = std::cos(angle_rad);
                        f32 r = 1.0f;
                        switch(phase_model)
                        {
                            case HENYEY_GREENSTEIN: 
                            {
                                r = henyey_greenstein_phase(g, cos_theta);
                                break;
                            }
                            case HENYEY_GREENSTEIN_OCTAVES:
                            {
                                r = multi_octave_hg(g, cos_theta, extinction, w_0, w_1, octaves);
                                break;
                            }
                            case DRAINE:
                            {
                                r = draine_phase(diameter, g, cos_theta);
                                break;
                            }
                        }
                        x_values[i] = r * std::cos(angle_rad);
                        y_values[i] = r * std::sin(angle_rad);
                    }

                    ImPlot::PushStyleColor(ImPlotCol_FrameBg, bg_4);
                    f32 const size = ImGui::GetContentRegionAvail().x - 20.0f;
                    if (ImPlot::BeginPlot("##hg", {size, size}, ImPlotFlags_Equal))
                    {

                        ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::ColorConvertFloat4ToU32(select_blue_1));
                        ImPlot::SetupAxes("##x", "##y", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                        ImPlot::PlotLine("", x_values.data(), y_values.data(), samples);
                        ImPlot::PopStyleColor();

                        ImPlot::EndPlot();
                    }
                    ImPlot::PopStyleColor();
                    ImGui::Unindent(indent_size);
                }

                {
                    i32vec2 const screen_bounds = s_cast<i32vec2>(std::bit_cast<u32vec2>(render_context.render_data.settings.render_target_size));
                    f32 const text_size = s_cast<f32>(ImGui::GetContentRegionAvail().x * 0.4f);
                    f32 const drag_size = (s_cast<f32>(ImGui::GetContentRegionAvail().x) - text_size) * 0.5f;
                    ImGui::Text("Debug Pixel");
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(drag_size);
                    ImGui::DragInt("##debug_pixel_x", &render_context.render_data.volumetric_settings.debug_pixel.x, 10, -1, screen_bounds.x);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(drag_size);
                    ImGui::DragInt("##debug_pixel_y", &render_context.render_data.volumetric_settings.debug_pixel.y, 10, -1, screen_bounds.y);
                }

                const bool debug_pixel_selected = 
                    render_context.render_data.volumetric_settings.debug_pixel.x > 0 &&
                    render_context.render_data.volumetric_settings.debug_pixel.y > 0;

                if (!app_state.draw_observer && debug_pixel_selected) 
                {
                    f32vec2 aabb_size = {
                        render_context.render_data.settings.render_target_size_inv.x,
                        render_context.render_data.settings.render_target_size_inv.y 
                    };
                    daxa_f32vec3 aabb_pos = {
                        ((s_cast<f32>(render_context.render_data.volumetric_settings.debug_pixel.x) + 0.5f) * aabb_size.x) * 2.0f - 1.0f,
                        ((s_cast<f32>(render_context.render_data.volumetric_settings.debug_pixel.y) + 0.5f) * aabb_size.y) * 2.0f - 1.0f,
                        0.5f,
                    };
                    render_context.gpu_context->shader_debug_context.aabb_draws.draw({
                        ShaderDebugAABBDraw{
                            .position = aabb_pos,
                            .size = {0.01f, 0.01f},
                            .color = {1.0f, 0.0f, 0.0f},
                            .coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_NDC_MAIN_CAMERA,
                        }
                    });
                }
                
                render_context.render_data.volumetric_settings.enable = enable;
            }
        }
        ImGui::PopItemWidth();
    }
    ImGui::End();
}

// Programmed move sub option of an RT Lab test: camera + sun move in fixed steps (one step per test frame / move step).
static void rt_lab_test_move_ui(char const * id, RtLabState & lab, RtLabTestMove & move, ProgrammedMovePose const & camera, ProgrammedMovePose const & sun)
{
    ImGui::PushID(id);
    ImGui::Checkbox("Programmed Move", &move.enabled);
    ImGui::SetItemTooltip(
        "Moves the camera and / or the sun during the test in fixed steps, so runs are reproducible.\n"
        "The camera and sun return to their pre-test poses when the test ends.");
    if (move.enabled && ImGui::TreeNodeEx("Programmed Move Setup", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::InputFloat("Time Per Step (s)", &move.fixed_dt, 0.0f, 0.0f, "%.5f");
        move.fixed_dt = std::max(move.fixed_dt, 1e-4f);
        ImGui::SetItemTooltip("Move time per step: the moves advance by speed x this every step. Also the frame delta time while the test runs\n"
                              "(exposure adaption etc.). The denoisers only see frames.");
        ImGui::Text("Steps: %i", rt_lab_test_move_steps(move));
        ImGui::SameLine();
        if (lab.preview_move == &move)
        {
            if (ImGui::Button("Stop Preview"))
            {
                lab.preview_stop_requested = true;
            }
            ImGui::SameLine();
            ImGui::Text("step %i", lab.preview_step);
        }
        else
        {
            ImGui::BeginDisabled(lab.preview_move != nullptr);
            if (ImGui::Button("Preview Move"))
            {
                lab.preview_move = &move;
                lab.preview_step = 0;
            }
            ImGui::EndDisabled();
            ImGui::SetItemTooltip("Plays the move one step per frame (camera locked) without running the test, then returns the poses.");
        }
        ImGui::SeparatorText("Camera");
        programmed_move_ui("camera", move.camera, camera, false);
        ImGui::SeparatorText("Sun");
        programmed_move_ui("sun", move.sun, sun, false);
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void UIEngine::ui_rt_lab(RenderContext & render_context, ApplicationState & app_state)
{
    RenderGlobalData & render_data = render_context.render_data;
    if (ImGui::Begin("RT Lab", &widget_rt_lab, ImGuiWindowFlags_NoCollapse))
    {
        ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x * 0.35f);
        auto & lab = render_context.rt_lab;
        bool const estimating = lab.phase != RtLabConvergencePhase::IDLE;
        // Current poses ("Take Current Position" of the test moves).
        ProgrammedMovePose const current_camera_pose = {
            .position = app_state.camera_controller.position,
            .yaw = app_state.camera_controller.yaw,
            .pitch = app_state.camera_controller.pitch,
        };
        ProgrammedMovePose const current_sun_pose = programmed_move_sun_pose(std::bit_cast<f32vec3>(render_data.sky_settings.sun_direction));
        // Section tree node that shows the section's accumulated GPU time on its header line.
        auto lab_section = [&](char const * name, std::initializer_list<u32> indices) -> bool {
            bool const open = ImGui::TreeNodeEx(name, ImGuiTreeNodeFlags_None);
            u64 sum = 0;
            for (u32 i : indices)
            {
                sum += render_context.render_times.get_average(i);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("%8.1f us", static_cast<f32>(sum) * 0.001f);
            return open;
        };
        // All settings of one preset.
        auto lab_preset_settings = [&](RtLabSettings & settings) {
            ImGui::Checkbox("Half Resolution", reinterpret_cast<bool *>(&settings.half_res));
            ImGui::Checkbox("Animate Noise", reinterpret_cast<bool *>(&settings.animate_noise));
            if (lab_section("Trace", { RenderTimes::index<"RT_LAB", "RAY_DEMAND">(), RenderTimes::index<"RT_LAB", "RAY_DISTRIBUTE">(),
                                       RenderTimes::index<"RT_LAB", "TRACE">(), RenderTimes::index<"RT_LAB", "RAY_FILTER">(),
                                       RenderTimes::index<"RT_LAB", "RAY_GATHER">() }))
            {
                ImGui::TextDisabled("Cosine distributed diffuse rays.");
                ImGui::Checkbox("Ray Redistribution", reinterpret_cast<bool *>(&settings.redistribution_enabled));
                ImGui::SetItemTooltip("Off: 1 ray per pixel.\nOn: pixels with a short history (disocclusions) ask for extra rays (RTGI demand curve),\n"
                                      "the frame's rays are capped at Ray Budget per pixel: extras are drained first, down to Min Base Coverage.\n"
                                      "Debug view RTGI_DIFFUSE_RAYS_SHOT shows the rays per pixel.");
                ImGui::BeginDisabled(settings.redistribution_enabled == 0);
                ImGui::SliderFloat("Ray Budget (rays / px)", &settings.ray_budget, 0.125f, 4.0f, "%.3f");
                ImGui::SetItemTooltip("Hard budget: rays per lab pixel per frame. Rounding it up to the next integer resizes the ray list (graph rebuild).");
                ImGui::SliderFloat("Min Base Coverage", &settings.min_ray_budget, 0.0f, 1.0f);
                ImGui::SetItemTooltip("Fraction of the pixels that keep their base ray when the extras drain the budget.");
                ImGui::SliderInt("Max Rays Per Pixel", &settings.max_rays_per_pixel, 1, 32);
                ImGui::SliderFloat("Convergence Target (samples)", &settings.fast_convergence_samples, 1.0f, 256.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
                ImGui::SetItemTooltip("History length at which a pixel stops asking for extra rays (max extras at length 0, exp(-4 length / target) falloff).");
                ImGui::EndDisabled();
                ImGui::Checkbox("AO On Hit", reinterpret_cast<bool *>(&settings.hit_ao_enabled));
                ImGui::SetItemTooltip("Same short range AO as the RTGI coarse hit shading: hit PGI radiance x (min(t, range) / range)^2.\nPGI leaks at short hit distances.");
                if (settings.hit_ao_enabled)
                {
                    ImGui::SliderFloat("AO Range", &settings.hit_ao_range, 0.01f, 10.0f, "%.3f m", ImGuiSliderFlags_Logarithmic);
                }
                ImGui::SeparatorText("Ray Filter");
                ImGui::PushID("ray filter");
                char const * const ray_filter_modes[] = {"Off", "Variance Clamp", "Perceptual Clamp (RTGI style)"};
                ImGui::Combo("Mode", &settings.ray_filter_mode, ray_filter_modes, 3);
                ImGui::SameLine();
                ImGui::TextDisabled("%8.1f us", static_cast<f32>(render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "RAY_FILTER">())) * 0.001f);
                ImGui::SetItemTooltip("One thread per ray after the trace: statistics of all rays of the neighbor pixels on the same surface\n"
                                      "(the ray itself excluded), the ray clamped against them, hue preserving.\n"
                                      "Variance: luminance mean + sigmas x std (optionally also mean - sigmas x std).\n"
                                      "Perceptual: like the RTGI firefly clamp, per channel exp(mean of log rgb + tolerance), the worst channel decides.\n"
                                      "Pixel means, temporal SH and coarse clamp use the filtered rays, the temporal gradient the raw ones.");
                ImGui::BeginDisabled(settings.ray_filter_mode == RT_LAB_RAY_FILTER_OFF);
                ImGui::SliderInt("Radius", &settings.ray_filter_radius, 0, RT_LAB_RAY_FILTER_MAX_RADIUS);
                ImGui::SetItemTooltip("Neighbor pixels (2r + 1)^2 (2 = 5x5). 0 = only the pixel's own other rays.");
                if (settings.ray_filter_mode == RT_LAB_RAY_FILTER_PERCEPTUAL)
                {
                    ImGui::SliderFloat("Tolerance (log)", &settings.ray_filter_tolerance, 0.0f, 8.0f);
                    ImGui::SetItemTooltip("Ceiling = exp(neighbors' mean log + this) per channel (RTGI firefly_perceptual_tolerance, default 4).");
                }
                else
                {
                    ImGui::SliderFloat("Sigmas", &settings.ray_filter_sigmas, 0.25f, 8.0f);
                    ImGui::Checkbox("Clamp Low Too", reinterpret_cast<bool *>(&settings.ray_filter_clamp_low));
                    ImGui::SetItemTooltip("Also raise rays below mean - sigmas x std (adds energy to dark rays: biased brighter).");
                }
                ImGui::EndDisabled();
                ImGui::PopID();
                ImGui::TreePop();
            }
            if (lab_section("Temporal", { RenderTimes::index<"RT_LAB", "TEMPORAL">() }))
            {
                ImGui::Checkbox("Enabled", reinterpret_cast<bool *>(&settings.temporal_enabled));
                ImGui::Checkbox("Progressive Accumulation", reinterpret_cast<bool *>(&settings.temporal_progressive));
                ImGui::SetItemTooltip("Off = smart: fp16 history, short history (Max History Frames).\nOn = progressive: fp32 history, accumulates up to Progressive Max Samples.");
                if (settings.temporal_progressive)
                {
                    ImGui::SliderInt("Progressive Max Samples", &settings.progressive_max_samples, 1, 4096);
                }
                else
                {
                    ImGui::SliderInt("Max History Frames", &settings.max_history_frames, 1, 256);
                }
                ImGui::SliderFloat("Disocclusion Plane Tolerance (px)", &settings.disocclusion_plane_tolerance, 0.1f, 8.0f);
                ImGui::SliderFloat("Disocclusion Normal Threshold", &settings.disocclusion_normal_threshold, -1.0f, 1.0f);
                ImGui::Checkbox("Occluder Aware Soft Normalized Sample Count Reprojection", reinterpret_cast<bool *>(&settings.occluder_aware_soft_normalize));
                ImGui::SetItemTooltip("On: tido RTGI style. History length = normalized mean of the valid taps x (1 - occluder coverage)^0.34, only\n"
                                      "taps that hid the point last frame (in front of it) penalize; edge overhang does not.\n"
                                      "Off: old RTGI / ReLAX style soft normalize, sum(w * count) / W^0.66: every partial footprint is penalized.");
                ImGui::Checkbox("Bicubic Reprojection", reinterpret_cast<bool *>(&settings.bicubic_reprojection));
                ImGui::SetItemTooltip("Reproject the history with a 12-tap Catmull-Rom when all 12 taps lie on this surface (else bilinear).\nBilinear blurs the history a little every frame under motion.");
                ImGui::Checkbox("Split History (A/B Error Estimate)", reinterpret_cast<bool *>(&settings.split_history_enabled));
                ImGui::SetItemTooltip("Two extra histories, A takes even frames, B odd frames. |A - B| / 2 estimates the error of the accumulated mean.\nSee the RT_LAB_SPLIT_* debug views.");
                ImGui::SeparatorText("Temporal Gradient");
                ImGui::PushID("temporal gradient");
                ImGui::Checkbox("Enabled", reinterpret_cast<bool *>(&settings.temporal_gradient_enabled));
                ImGui::SameLine();
                ImGui::TextDisabled("%8.1f us", static_cast<f32>(render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "GRADIENT_TRACE">()) +
                                                                  render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "GRADIENT_BLUR_H">()) +
                                                                  render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "GRADIENT_BLUR_V">())) * 0.001f);
                ImGui::SetItemTooltip("A-SVGF style: one ray per 4x4 pixels is re-traced next frame with the exact same launch data.\n"
                                      "The luminance changes, blurred in screen space (edge aware, coarse plane + normal tests), give lambda = |sum delta| / sum max(old, new).\n"
                                      "Reprojected history lengths are scaled by 1 - lambda before the ray redistribution and in the temporal pass.\n"
                                      "Debug view RT_LAB_TEMPORAL_GRADIENT.");
                ImGui::BeginDisabled(settings.temporal_gradient_enabled == 0);
                ImGui::SliderInt("Blur Radius (cells)", &settings.temporal_gradient_blur_radius, 0, RT_LAB_GRADIENT_MAX_BLUR_RADIUS);
                ImGui::SetItemTooltip("Separable gaussian over the 4x4 pixel gradient cells (1/4 res), sigma = radius / 2.\nReach = 4 x radius pixels.");
                ImGui::SliderFloat("Scale", &settings.temporal_gradient_scale, 0.0f, 8.0f);
                ImGui::SetItemTooltip("lambda multiplier (saturated): history kept = 1 - saturate(lambda x scale).");
                ImGui::Checkbox("RIS Sample Selection", reinterpret_cast<bool *>(&settings.temporal_gradient_ris));
                ImGui::SetItemTooltip("On: each cell's re-traced ray is picked by RIS over ALL of the cell's rays, target = luminance + 0.1 x cell mean\n"
                                      "(bright rays carry most of a relative change; the uniform part keeps dark rays pickable).\n"
                                      "Delta and max are scaled by the RIS weight (sum w / M) / w_picked, so a cell still estimates its mean change.\n"
                                      "Off: uniform, first ray of a pixel with rays (pixel order rotated per frame).");
                ImGui::EndDisabled();
                ImGui::PopID();
                ImGui::SeparatorText("Coarse Temporal Anti-Lag");
                ImGui::PushID("coarse clamp");
                i32 const anti_lag_mode = settings.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MODE_MASK;
                i32 const anti_lag_flags = settings.coarse_anti_lag_enabled & ~RT_LAB_ANTI_LAG_MODE_MASK;
                bool anti_lag_enabled = anti_lag_mode != RT_LAB_ANTI_LAG_OFF;
                if (ImGui::Checkbox("Enabled", &anti_lag_enabled))
                {
                    settings.coarse_anti_lag_enabled = (anti_lag_enabled ? RT_LAB_ANTI_LAG_REDUCE_SAMPLES : RT_LAB_ANTI_LAG_OFF) | anti_lag_flags;
                }
                ImGui::SameLine();
                ImGui::TextDisabled("%8.1f us", static_cast<f32>(render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "ANTI_LAG_DOWNSAMPLE">()) +
                                                                  render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "ANTI_LAG_BLUR_H">()) +
                                                                  render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "ANTI_LAG_BLUR_V">())) * 0.001f);
                ImGui::SetItemTooltip("Adjusts the history COUNT, never the color. Ray luminance per 4x4 cell, blurred a lot (edge aware), feeds an RTGI\n"
                                      "style coarse fast history (fast mean + fast relative variance) next to a slow coarse mean. RTGI anti-lag: when the\n"
                                      "slow mean diverges from the fast mean, the history length is cut to N x (1 + sqrt(fast var) x Variance Blend)^2 x\n"
                                      "(1 / max(1, slow / fast))^2 (noise earns trust back); caps the coarse slow history and the pixel histories.\n"
                                      "Debug views: RT_LAB_COARSE_ANTI_LAG (history cut), _MEAN (slow coarse mean, per pixel), _FRAME_MEAN (this frame's\n"
                                      "blurred mean), _RELATIVE_STD (fast history relative std) / _FRAME_RELATIVE_STD (this frame's spatial sigma / mean),\n"
                                      "log heatmap 1 % .. 1000 %, the last three at cell res.");
                ImGui::BeginDisabled(anti_lag_mode == RT_LAB_ANTI_LAG_OFF);
                bool reduce_samples = anti_lag_mode != RT_LAB_ANTI_LAG_BLEND_ONLY;
                if (ImGui::Checkbox("Reduce Sample Count", &reduce_samples) && anti_lag_mode != RT_LAB_ANTI_LAG_OFF)
                {
                    settings.coarse_anti_lag_enabled = (reduce_samples ? RT_LAB_ANTI_LAG_REDUCE_SAMPLES : RT_LAB_ANTI_LAG_BLEND_ONLY) | anti_lag_flags;
                }
                ImGui::SetItemTooltip("On: the stored history length is cut, so the following frames keep converging fast on the new result.\n"
                                      "Off: RTGI style, only this frame's blend uses the cut length, the stored length keeps growing.");
                ImGui::SliderInt("Blur Radius (cells)", &settings.coarse_anti_lag_blur_radius, 0, RT_LAB_GRADIENT_MAX_BLUR_RADIUS);
                ImGui::SetItemTooltip("Separable edge aware gaussian over the 4x4 pixel cells, sigma = radius / 2. Reach = 4 x radius pixels.");
                ImGui::SliderInt("History Frames", &settings.coarse_anti_lag_history_frames, 1, 64);
                ImGui::SetItemTooltip("Fast history window in frames (RTGI temporal_fast_history_frames, default 4 = measured best).\n"
                                      "The slow side is the denoiser's own history length (Max History Frames / Progressive Max Samples).");
                bool mse_optimal = (settings.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MSE_OPTIMAL) != 0;
                if (ImGui::Checkbox("MSE Optimal Length", &mse_optimal))
                {
                    settings.coarse_anti_lag_enabled = mse_optimal ? (settings.coarse_anti_lag_enabled | RT_LAB_ANTI_LAG_MSE_OPTIMAL) : (settings.coarse_anti_lag_enabled & ~RT_LAB_ANTI_LAG_MSE_OPTIMAL);
                }
                ImGui::SetItemTooltip("Heuristic for the cut. Off: RTGI fast history anti-lag (variance scaling x (1 / slow-fast ratio)^2).\n"
                                      "On: MSE optimal length N* = k x fast rel variance / (slow-fast rel difference)^2 (difference debiased by the\n"
                                      "fast mean's own noise), never below History Frames. Continuous in the difference: no threshold, no abrupt cut.");
                ImGui::SliderFloat(mse_optimal ? "MSE k" : "Variance Blend", &settings.coarse_anti_lag_variance_blend, 0.0f, 16.0f);
                ImGui::SetItemTooltip(mse_optimal ? "MSE optimal length scale k: 1 = plain MSE optimum, larger = longer histories (less noise, more lag)."
                                                  : "How much fast relative variance (noise) earns trust back (RTGI temporal_variance_fast_history_blend, default 4).");
                ImGui::EndDisabled();
                ImGui::PopID();
                ImGui::TreePop();
            }
            if (lab_section("Spatial", {
                    RenderTimes::index<"RT_LAB", "BLUR_HORIZONTAL">(), RenderTimes::index<"RT_LAB", "BLUR_VERTICAL">(),
                    RenderTimes::index<"RT_LAB", "ATROUS0">(), RenderTimes::index<"RT_LAB", "ATROUS1">(),
                    RenderTimes::index<"RT_LAB", "ATROUS2">(), RenderTimes::index<"RT_LAB", "ATROUS3">(),
                    RenderTimes::index<"RT_LAB", "ATROUS4">(), RenderTimes::index<"RT_LAB", "ATROUS5">(),
                    RenderTimes::index<"RT_LAB", "ATROUS6">(), RenderTimes::index<"RT_LAB", "ATROUS7">(),
                    RenderTimes::index<"RT_LAB", "COARSE0">(), RenderTimes::index<"RT_LAB", "COARSE1">(),
                    RenderTimes::index<"RT_LAB", "COARSE2">(), RenderTimes::index<"RT_LAB", "COARSE3">(),
                    RenderTimes::index<"RT_LAB", "COARSE4">(), RenderTimes::index<"RT_LAB", "COARSE5">(),
                    RenderTimes::index<"RT_LAB", "COARSE6">(), RenderTimes::index<"RT_LAB", "COARSE7">() }))
            {
                ImGui::Checkbox("Enabled", reinterpret_cast<bool *>(&settings.spatial_enabled));
                char const * const spatial_modes[] = {"Separable Gaussian", "A-Trous 5x5"};
                ImGui::Combo("Mode", &settings.spatial_mode, spatial_modes, 2);
                ImGui::Checkbox("Variance Guide", reinterpret_cast<bool *>(&settings.variance_guide_enabled));
                ImGui::SetItemTooltip("Fine filter only. The temporal pass estimates the per sample rgb variance (5x5 raw samples on this surface,\n"
                                      "accumulated like the color). Center and tap are estimates of their means, N(mean, variance / effective samples);\n"
                                      "the tap weight is the likelihood both share a mean: exp(-z^2 / (2 k^2)), z^2 = mean over rgb of\n"
                                      "(mean_c - mean_t)^2 / (var_c / n_c + var_t / n_t). Few samples = loose test, many samples = tight test.\n"
                                      "Rejected taps also do not count into the effective sample count. Debug view RT_LAB_VARIANCE.");
                ImGui::BeginDisabled(settings.variance_guide_enabled == 0);
                ImGui::SliderFloat("Tolerance (sigmas)", &settings.variance_guide_sigmas, 0.25f, 16.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::EndDisabled();
                ImGui::SetItemTooltip("Separable gaussian: horizontal + vertical pass, Blur Radius taps per side.\nA-trous: passes of a 5x5 B3 spline kernel, tap spacing 1, 2, 4, 8, ... px.\nBoth use the same geometric edge stopping (plane tolerance, normal power).");
                if (settings.spatial_mode == 0)
                {
                    ImGui::SliderInt("Blur Radius (px)", &settings.blur_radius, 0, 32);
                }
                else
                {
                    ImGui::Indent();
                    ImGui::SliderInt("A-Trous Passes", &settings.atrous_passes, 1, RT_LAB_ATROUS_MAX_PASSES);
                    i32 const passes = std::clamp(settings.atrous_passes, 1, RT_LAB_ATROUS_MAX_PASSES);
                    // Reach: sum of 2 * step over the passes.
                    ImGui::TextDisabled("Tap spacing 1 .. %i px, reach +-%i px", 1 << (passes - 1), 2 * ((1 << passes) - 1));
                    ImGui::Unindent();
                }
                ImGui::SliderFloat("Plane Tolerance (px)", &settings.blur_plane_tolerance, 0.1f, 8.0f);
                ImGui::SliderFloat("Normal Power", &settings.blur_normal_power, 0.0f, 64.0f);
                ImGui::SliderFloat("Target Samples", &settings.fine_target_samples, 0.0f, 100000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
                ImGui::SetItemTooltip("Fine filter sample count budget (0 = off). Each pass accepts taps ring by ring from the center outwards;\n"
                                      "the ring that would exceed this effective sample count is weighted down so the pass reaches it exactly,\n"
                                      "further taps are dropped. Fine filter only. Also caps the coarse blend's Fully Fine Above.");
                ImGui::SeparatorText("Coarse Fallback");
                ImGui::PushID("coarse");
                ImGui::Checkbox("Enabled", reinterpret_cast<bool *>(&settings.coarse_enabled));
                ImGui::SetItemTooltip("Second, sloppy a-trous filter on the same input, blended in where the fine filter gathered few\neffective samples (edges, thin geometry, disocclusions). See RT_LAB_COARSE_BLEND.");
                ImGui::BeginDisabled(settings.coarse_enabled == 0);
                ImGui::SliderInt("Passes", &settings.coarse_atrous_passes, 1, RT_LAB_ATROUS_MAX_PASSES);
                ImGui::SliderFloat("Distance Tolerance", &settings.coarse_distance_tolerance, 0.1f, 32.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::SetItemTooltip("Smooth distance test, no plane test: weight = exp(-(max(d_rel - 1, 0) / tolerance)^2), d_rel = world distance\nrelative to what a camera facing surface at that screen offset would give. Higher = sloppier.");
                ImGui::SliderFloat("Normal Power##coarse", &settings.coarse_normal_power, 0.0f, 16.0f);
                ImGui::SetItemTooltip("Smooth normal test: weight = ((1 + dot) / 2)^power, 0 at 180 degrees.");
                ImGui::Checkbox("Variance Guide", reinterpret_cast<bool *>(&settings.coarse_variance_guide_enabled));
                ImGui::SetItemTooltip("Same likelihood test as the fine filter's variance guide, on the mean rgb of the SH-Y + CoCg\n"
                                      "(independent of the fine filter's toggle), with its own, usually wider, tolerance.");
                ImGui::BeginDisabled(settings.coarse_variance_guide_enabled == 0);
                ImGui::SliderFloat("Tolerance (sigmas)", &settings.coarse_variance_guide_sigmas, 0.25f, 16.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
                ImGui::EndDisabled();
                ImGui::SliderFloat("Top Up To (samples)", &settings.coarse_target_samples, 1.0f, 1000000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
                ImGui::SetItemTooltip("Blends in just enough of the coarse result to top the effective sample count up to this.\n"
                                      "Fine / coarse treated as estimates with variance 1 / N: N_eff = 1 / ((1 - t)^2 / N_fine + t^2 / N_coarse),\n"
                                      "t = smallest root of N_eff = target; unreachable target: minimum variance mix. Capped by Max Blend.");
                ImGui::SliderFloat("Max Blend", &settings.coarse_max_blend, 0.0f, 1.0f);
                ImGui::EndDisabled();
                ImGui::PopID();
                ImGui::TreePop();
            }
            if (lab_section("Resolve", { RenderTimes::index<"RT_LAB", "RESOLVE">() }))
            {
                ImGui::TextDisabled("Full res output, bilateral upsample when half res.");
                // Shared by both presets (not part of them): the same values show under each preset.
                ImGui::SeparatorText("Post Stabilization");
                ImGui::PushID("post stabilization");
                ImGui::Checkbox("Enabled", &lab.post_stabilization_enabled);
                ImGui::SetItemTooltip("TAA style temporal step, the LAST thing the resolve does, always at full res (both presets, not part of them):\n"
                                      "own history, reprojected, clamped to the current output's neighborhood box (mean +- gamma x sigma, YCoCg),\n"
                                      "blended with the current output by alpha. Replaces the output for the measurements / estimate too.\n"
                                      "Its cost shows in the RESOLVE timer (the halo of every 8x8 group is resolved twice).");
                ImGui::BeginDisabled(!lab.post_stabilization_enabled);
                ImGui::SliderFloat("Alpha", &lab.post_stabilization_alpha, 0.01f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::SetItemTooltip("Blend factor of the current output (1 = off).");
                ImGui::Checkbox("Variance Clamp", &lab.post_stabilization_clamp);
                ImGui::SetItemTooltip("Clamp the reprojected history into the neighborhood box. Off: plain exponential history (alpha only).");
                ImGui::BeginDisabled(!lab.post_stabilization_clamp);
                ImGui::SliderFloat("Gamma", &lab.post_stabilization_gamma, 0.25f, 4.0f);
                ImGui::SetItemTooltip("Clamp box half width in standard deviations of the neighborhood (smaller = less ghosting, more flicker).");
                ImGui::Checkbox("Temporal Variance", &lab.post_stabilization_temporal_variance);
                ImGui::SetItemTooltip("Short history (max 4 frames, reprojected) of the neighborhood variance loosens the box:\n"
                                      "half width = gamma x sqrt(max(neighborhood variance, its history)).\n"
                                      "A frame whose neighborhood spreads less than recently does not clamp harder.");
                ImGui::SliderInt("Variance Estimator Radius", &lab.post_stabilization_radius, 1, RT_LAB_STABILIZE_MAX_RADIUS);
                ImGui::SetItemTooltip("Neighborhood of the mean / sigma estimate: (2r + 1)^2 pixels (1 = 3x3). Larger = steadier box, the halo grows\n"
                                      "(more pixels resolved twice per 8x8 group).");
                ImGui::EndDisabled();
                ImGui::EndDisabled();
                ImGui::PopID();
                ImGui::TreePop();
            }
        };

        if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Use RT Lab Denoiser", reinterpret_cast<bool *>(&render_data.rt_lab_settings.enabled));
            ImGui::SetItemTooltip("Master switch: replaces the RTGI denoiser with the experimental RT Lab path (diffuse only). RTGI must be enabled.");
            if (render_data.rt_lab_settings.enabled && !render_data.rtgi_settings.enabled)
            {
                ImGui::TextDisabled("RTGI is disabled, RT Lab is not running.");
            }
            ImGui::Checkbox("Force Full PGI Update", &lab.force_full_pgi_update);
            ImGui::SetItemTooltip("While RT Lab runs (also during its tests), PGI updates every probe every frame (update rate FULL).\n"
                                  "Unchecked: the PGI settings' own update rate (restored when this turns off).");
            char const * const preset_names[RT_LAB_PRESET_COUNT] = {"Progressive", "Custom"};
            ImGui::BeginDisabled(estimating);
            ImGui::Combo("Active Preset", &lab.active_preset, preset_names, RT_LAB_PRESET_COUNT);
            ImGui::EndDisabled();
            ImGui::SetItemTooltip("Progressive: reference (full res, fp32 long history, 2 tight a-trous passes, no coarse fallback by default).\nCustom: the settings under test.");

            // Both presets, each editable with its own reset.
            for (i32 p = 0; p < RT_LAB_PRESET_COUNT; ++p)
            {
                ImGui::PushID(p);
                std::string const label = std::string(preset_names[p]) + " Preset" + (lab.active_preset == p ? " (active)" : "");
                if (ImGui::TreeNodeEx("##preset", ImGuiTreeNodeFlags_None, "%s", label.c_str()))
                {
                    if (ImGui::Button("Reset To Defaults"))
                    {
                        lab.presets[p] = rt_lab_default_settings(static_cast<RtLabPreset>(p));
                    }
                    lab_preset_settings(lab.presets[p]);
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
        }
        if (ImGui::CollapsingHeader("Debug Views", ImGuiTreeNodeFlags_DefaultOpen))
        {
            debug_mode_sub_selector("rt lab debug visualization", &rt_lab_debug_visualization, {
                DEBUG_DRAW_MODE_RT_LAB_HISTORY_LENGTH,
                DEBUG_DRAW_MODE_RT_LAB_TEMPORAL_GRADIENT,
                DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG,
                DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_MEAN,
                DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_FRAME_MEAN,
                DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_RELATIVE_STD,
                DEBUG_DRAW_MODE_RT_LAB_COARSE_ANTI_LAG_FRAME_RELATIVE_STD,
                DEBUG_DRAW_MODE_RT_LAB_VARIANCE,
                DEBUG_DRAW_MODE_RT_LAB_SPATIAL_SAMPLE_COUNT,
                DEBUG_DRAW_MODE_RT_LAB_COARSE_BLEND,
                DEBUG_DRAW_MODE_RT_LAB_COARSE_SAMPLE_COUNT,
                DEBUG_DRAW_MODE_RT_LAB_SPLIT_ERROR,
                DEBUG_DRAW_MODE_RT_LAB_SPLIT_RELATIVE_ERROR,
                DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_PROGRESSIVE,
                DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_CUSTOM,
                DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_DIFFERENCE,
                DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_DIFFERENCE_SQUARED,
                DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_RELATIVE_DIFFERENCE,
                DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_PREDICTED_ERROR,
                DEBUG_DRAW_MODE_RT_LAB_CONVERGENCE_PREDICTION_RATIO,
            }, debug_visualization_index_override);
            ImGui::SetItemTooltip(
                "History length: heatmap of length / max length.\n"
                "Temporal gradient: lambda (scaled), 0 = history kept (cold) .. 1 = history dropped (hot), magenta = off.\n"
                "Variance: relative std of the temporal mean, sqrt(per sample variance / history length) / mean, log heatmap 0.1%% .. 100%%.\n"
                "Spatial sample count: effective samples after the spatial passes, sum(weight / center weight * temporal count),\n"
                "heatmap up to Sample Count Max (after the fine / coarse blend).\n"
                "Coarse blend: t of the coarse spatial fallback top up, 0 = fine (cold) .. 1 = coarse (hot).\n"
                "Coarse sample count: the coarse fallback's own effective sample count, heatmap up to Sample Count Max.\n"
                "Split error: POST-SPATIAL |A - B| / 2 luminance (the difference goes through blur + upsample with the diffuse's\n"
                "weights), exposed irradiance, heatmap x debug visualization scale.\n"
                "Split relative error: |A - B| / min(A, B) (post-spatial), log heatmap 0.1%% (cold) .. 1%% .. 10%% .. 100%% (hot).\n"
                "Magenta = a half has no samples yet.\n"
                "Convergence difference: red = Custom brighter, blue = darker, x debug visualization scale.\n"
                "Convergence relative difference: |Custom - Progressive| / min(Custom, Progressive), same log heatmap as split relative.\n"
                "Convergence predicted error: Custom run split error |A - B| / 2 at its capture, heatmap x debug visualization scale.\n"
                "Convergence prediction ratio: log2(RMS predicted / RMS actual) over a 9x9 window (per pixel both are single\n"
                "random draws), red = overestimate, blue = underestimate, black = exact, +-3 stops.");
            ImGui::SeparatorText("Spatial Sample Count");
            ImGui::SliderFloat("Sample Count Max", &lab.sample_count_debug_max, 1.0f, 10000000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
            ImGui::SetItemTooltip("RT_LAB_SPATIAL_SAMPLE_COUNT heatmap saturates at this count.");
            ImGui::SliderFloat("Temporal Count Clamp", &lab.sample_count_input_clamp, 0.0f, 512.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
            ImGui::SetItemTooltip("The temporal history length is clamped to this before the spatial passes accumulate it.\n0 = off, 1 = count only the spatial samples. Metric only, the filtering is unchanged.");
        }
        if (ImGui::CollapsingHeader("Convergence Measurement", ImGuiTreeNodeFlags_DefaultOpen))
        {
            bool const measuring = lab.measure_phase != RtLabMeasurePhase::IDLE;
            ImGui::BeginDisabled(measuring || estimating);
            ImGui::SliderInt("Measure Frames", &lab.measure_frames, 16, 4096);
            ImGui::SliderInt("Instability Window (frames)", &lab.measure_window, 2, 128);
            ImGui::SetItemTooltip("Temporal instability: per pixel running mean / variance of the output luma, plain average for\nthis many frames, then an EMA over this many frames. Reprojected with the camera.");
            rt_lab_test_move_ui("measure_move", lab, lab.measure_move, current_camera_pose, current_sun_pose);
            if (lab.measure_move.enabled)
            {
                ImGui::TextDisabled("One move step per measured frame, then the poses hold still.");
            }
            ImGui::EndDisabled();
            if (!measuring)
            {
                ImGui::BeginDisabled(estimating);
                if (ImGui::Button("Measure Convergence"))
                {
                    lab.measure_start_requested = true;
                }
                ImGui::EndDisabled();
                ImGui::SetItemTooltip(
                    "Locks the camera, resets the history and runs the ACTIVE preset for Measure Frames frames (split history forced on).\n"
                    "Every frame logs 512 bin log histograms over 4x4 tiles of the post-spatial convergence error and the output\n"
                    "temporal instability. Reference free.");
            }
            else
            {
                if (ImGui::Button("Cancel##measure"))
                {
                    lab.measure_cancel_requested = true;
                }
                ImGui::SameLine();
                if (lab.measure_phase == RtLabMeasurePhase::RUNNING)
                {
                    ImGui::Text("Camera locked%s, measuring: %i / %i", lab.measurement.move.enabled ? " (programmed move)" : "", lab.measure_frame, lab.measurement.frames);
                    ImGui::ProgressBar(static_cast<f32>(lab.measure_frame) / static_cast<f32>(std::max(lab.measurement.frames, 1)));
                }
                else
                {
                    ImGui::Text("Collecting...");
                }
            }
            // == Results: list (latest runs unsaved, saved ones from disk), save, compare ==
            ImGui::SeparatorText("Results");
            if (!lab.results_scanned)
            {
                rt_lab_results_scan(lab);
            }
            if (ImGui::Button("Refresh"))
            {
                rt_lab_results_scan(lab);
            }
            ImGui::SetItemTooltip("Rescan " RT_LAB_RESULTS_DIRECTORY "/ for saved results.");
            ImGui::SameLine();
            bool const has_selection = lab.selected_result >= 0 && lab.selected_result < static_cast<i32>(lab.results.size());
            ImGui::BeginDisabled(!has_selection);
            if (ImGui::Button("Save Selected..."))
            {
                std::string name = lab.results[lab.selected_result].name;
                for (char & c : name)
                {
                    c = c == ' ' ? '_' : c;
                }
                std::snprintf(rt_lab_measure_ui.save_name.data(), rt_lab_measure_ui.save_name.size(), "%s", name.c_str());
                rt_lab_measure_ui.save_failed = false;
                ImGui::OpenPopup("Save Measurement Result");
            }
            ImGui::EndDisabled();
            ImGui::SetItemTooltip("Save the selected result to " RT_LAB_RESULTS_DIRECTORY "/<name>.rtlab (gitignored).");
            if (ImGui::BeginPopupModal("Save Measurement Result", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
            {
                if (ImGui::IsWindowAppearing())
                {
                    ImGui::SetKeyboardFocusHere();
                }
                bool const entered = ImGui::InputText("Name", rt_lab_measure_ui.save_name.data(), rt_lab_measure_ui.save_name.size(), ImGuiInputTextFlags_EnterReturnsTrue);
                // File name safe: letters, digits, '-', '_', '.', spaces become '_'.
                std::string name = rt_lab_measure_ui.save_name.data();
                for (char & c : name)
                {
                    bool const allowed = std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.';
                    c = allowed ? c : '_';
                }
                bool const exists = !name.empty() && std::filesystem::exists(std::filesystem::path(RT_LAB_RESULTS_DIRECTORY) / (name + ".rtlab"));
                ImGui::TextDisabled("File: " RT_LAB_RESULTS_DIRECTORY "/%s.rtlab%s", name.c_str(), exists ? "  (overwrites)" : "");
                if (rt_lab_measure_ui.save_failed)
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Saving failed.");
                }
                ImGui::BeginDisabled(name.empty() || !has_selection);
                if (ImGui::Button("Save") || (entered && !name.empty() && has_selection))
                {
                    rt_lab_measure_ui.save_failed = !rt_lab_result_save(lab, lab.selected_result, name);
                    if (!rt_lab_measure_ui.save_failed)
                    {
                        ImGui::CloseCurrentPopup();
                    }
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::Button("Cancel##save"))
                {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }

            char const * const preset_names[RT_LAB_PRESET_COUNT] = {"Progressive", "Custom"};
            auto result_label = [&](RtLabResult const & r) -> std::string
            {
                return fmt::format("{}{}  ({} frames)", r.path.empty() ? "* " : "", r.name, r.measurement.frames);
            };
            auto result_tooltip = [&](RtLabResult const & r)
            {
                auto const & m = r.measurement;
                auto const & s = m.settings;
                ImGui::SetItemTooltip(
                    "%s\n%s\nresolution %ux%u, camera (%.2f, %.2f, %.2f), instability window %i\n"
                    "half res %i, temporal %i (%s, max %i / %i), split %i, spatial %i (radius %i [-N = a-trous 5x5 x N passes], plane %.2f, normal power %.1f)\n"
                    "disocclusion plane %.2f, normal %.2f, bicubic reprojection %i, occluder aware soft normalize %i (-1 = plain)%s%s",
                    r.name.c_str(), m.timestamp.c_str(), m.resolution[0], m.resolution[1],
                    m.camera_position[0], m.camera_position[1], m.camera_position[2], m.window,
                    s.half_res, s.temporal_enabled, s.temporal_progressive ? "progressive" : "smart", s.max_history_frames, s.progressive_max_samples,
                    s.split_history_enabled, s.spatial_enabled, s.spatial_mode == 1 ? -s.atrous_passes : s.blur_radius, s.blur_plane_tolerance, s.blur_normal_power,
                    s.disocclusion_plane_tolerance, s.disocclusion_normal_threshold, s.bicubic_reprojection, s.occluder_aware_soft_normalize,
                    !m.move.enabled ? "" : fmt::format("\nprogrammed move: {} steps of {:.4f} s, camera velocity ({}, {}, {}) yaw {} pitch {} deg/s for {} s, sun {} / {} deg/s for {} s",
                        rt_lab_test_move_steps(m.move), m.move.fixed_dt, m.move.camera.velocity.x, m.move.camera.velocity.y, m.move.camera.velocity.z,
                        m.move.camera.yaw_speed, m.move.camera.pitch_speed, m.move.camera.duration, m.move.sun.yaw_speed, m.move.sun.pitch_speed, m.move.sun.duration).c_str(),
                    r.path.empty() ? "\n(unsaved)" : "");
            };
            // Right click on a result (in a dropdown list or on a closed dropdown) -> rename.
            auto result_context_menu = [&](i32 index)
            {
                if (ImGui::BeginPopupContextItem("##result_context"))
                {
                    if (ImGui::MenuItem("Rename..."))
                    {
                        rt_lab_measure_ui.rename_index = index;
                        rt_lab_measure_ui.rename_requested = true;
                    }
                    if (ImGui::MenuItem("Delete..."))
                    {
                        rt_lab_measure_ui.delete_index = index;
                        rt_lab_measure_ui.delete_requested = true;
                    }
                    ImGui::EndPopup();
                }
            };
            // Dropdown of all results; skip = index left out of the list (A in the B dropdown).
            auto result_dropdown = [&](char const * label, i32 & selection, bool allow_none, i32 skip) -> bool
            {
                bool const valid = selection >= 0 && selection < static_cast<i32>(lab.results.size()) && selection != skip;
                bool changed = false;
                ImGui::PushID(label);
                bool const open = ImGui::BeginCombo(label, valid ? result_label(lab.results[selection]).c_str() : "None");
                if (!open)
                {
                    if (valid)
                    {
                        result_tooltip(lab.results[selection]);
                        result_context_menu(selection);
                    }
                }
                else
                {
                    if (allow_none && ImGui::Selectable("None", !valid))
                    {
                        selection = -1;
                        changed = true;
                    }
                    for (i32 i = 0; i < static_cast<i32>(lab.results.size()); ++i)
                    {
                        if (i == skip)
                        {
                            continue;
                        }
                        ImGui::PushID(i);
                        if (ImGui::Selectable(result_label(lab.results[i]).c_str(), selection == i))
                        {
                            selection = i;
                            changed = true;
                        }
                        result_tooltip(lab.results[i]);
                        result_context_menu(i);
                        ImGui::PopID();
                    }
                    ImGui::EndCombo();
                }
                ImGui::PopID();
                return changed;
            };
            if (result_dropdown("Result (A)", lab.selected_result, false, -1) && lab.compare_result == lab.selected_result)
            {
                lab.compare_result = -1;
            }
            ImGui::SetItemTooltip("* = unsaved. Hover for the settings a result was recorded with, right click to rename / delete.");
            result_dropdown("Compare With (B)", lab.compare_result, true, lab.selected_result);
            ImGui::SetItemTooltip("Optional: B's overlay lines are drawn on A's heatmap (thin), and the table compares both.");

            // Rename popup (opened here, outside of the dropdowns: a modal opened from inside a combo popup would live in
            // the combo's ID scope).
            if (rt_lab_measure_ui.rename_requested)
            {
                rt_lab_measure_ui.rename_requested = false;
                i32 const index = rt_lab_measure_ui.rename_index;
                if (index >= 0 && index < static_cast<i32>(lab.results.size()))
                {
                    std::snprintf(rt_lab_measure_ui.rename_name.data(), rt_lab_measure_ui.rename_name.size(), "%s", lab.results[index].name.c_str());
                    rt_lab_measure_ui.rename_failed = false;
                    ImGui::OpenPopup("Rename Result");
                }
            }
            if (ImGui::BeginPopupModal("Rename Result", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
            {
                i32 const index = rt_lab_measure_ui.rename_index;
                bool const index_valid = index >= 0 && index < static_cast<i32>(lab.results.size());
                if (ImGui::IsWindowAppearing())
                {
                    ImGui::SetKeyboardFocusHere();
                }
                bool const entered = ImGui::InputText("Name", rt_lab_measure_ui.rename_name.data(), rt_lab_measure_ui.rename_name.size(), ImGuiInputTextFlags_EnterReturnsTrue);
                std::string name = rt_lab_measure_ui.rename_name.data();
                bool const saved = index_valid && !lab.results[index].path.empty();
                if (saved)
                {
                    // Saved results are files: file name safe characters only.
                    for (char & c : name)
                    {
                        bool const allowed = std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.';
                        c = allowed ? c : '_';
                    }
                    ImGui::TextDisabled("File: " RT_LAB_RESULTS_DIRECTORY "/%s.rtlab", name.c_str());
                }
                if (rt_lab_measure_ui.rename_failed)
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Renaming failed.");
                }
                ImGui::BeginDisabled(name.empty() || !index_valid);
                if (ImGui::Button("Rename") || (entered && !name.empty() && index_valid))
                {
                    rt_lab_measure_ui.rename_failed = !rt_lab_result_rename(lab, index, name);
                    if (!rt_lab_measure_ui.rename_failed)
                    {
                        ImGui::CloseCurrentPopup();
                    }
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::Button("Cancel##rename"))
                {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }

            // Delete popup (opened outside of the dropdowns, like the rename popup).
            if (rt_lab_measure_ui.delete_requested)
            {
                rt_lab_measure_ui.delete_requested = false;
                rt_lab_measure_ui.delete_failed = false;
                ImGui::OpenPopup("Delete Result");
            }
            if (ImGui::BeginPopupModal("Delete Result", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
            {
                i32 const index = rt_lab_measure_ui.delete_index;
                bool const index_valid = index >= 0 && index < static_cast<i32>(lab.results.size());
                if (index_valid)
                {
                    RtLabResult const & r = lab.results[index];
                    ImGui::Text("Delete \"%s\"?", r.name.c_str());
                    if (!r.path.empty())
                    {
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.4f, 1.0f), "This deletes the file %s.", r.path.string().c_str());
                    }
                }
                if (rt_lab_measure_ui.delete_failed)
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Deleting the file failed.");
                }
                ImGui::BeginDisabled(!index_valid);
                if (ImGui::Button("Delete"))
                {
                    rt_lab_measure_ui.delete_failed = !rt_lab_result_delete(lab, index);
                    if (!rt_lab_measure_ui.delete_failed)
                    {
                        ImGui::CloseCurrentPopup();
                    }
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::Button("Cancel##delete"))
                {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }

            // Re-check: a rename / delete above can have changed the list this frame.
            bool const show_selection = lab.selected_result >= 0 && lab.selected_result < static_cast<i32>(lab.results.size());
            bool const show_compare = lab.compare_result >= 0 && lab.compare_result < static_cast<i32>(lab.results.size()) && lab.compare_result != lab.selected_result;
            if (!measuring && show_selection)
            {
                RtLabResult & selected = lab.results[lab.selected_result];
                RtLabMeasurement const * compare = nullptr;
                if (show_compare && rt_lab_result_ensure_loaded(lab.results[lab.compare_result]))
                {
                    compare = &lab.results[lab.compare_result].measurement;
                }
                if (rt_lab_result_ensure_loaded(selected) && selected.measurement.frames > 0)
                {
                    rt_lab_measure_analytics(selected.measurement, compare, rt_lab_measure_ui);
                    // Temporal lag (step response): recomputed from the tile data at the UI tolerance when the result has it.
                    ImGui::SeparatorText("Temporal Lag");
                    ImGui::SliderFloat("Lag Tolerance", &lab.lag_tolerance, 0.002f, 0.2f, "%.3f", ImGuiSliderFlags_Logarithmic);
                    ImGui::SetItemTooltip("Relative: a tile counts as settled once it stays within this of its final value (mean of the last 8 frames).\n"
                                          "Only tiles that changed by more than 2 x this after the start frame count.");
                    RtLabLagSummary const lag = rt_lab_measure_lag(selected.measurement, lab.lag_tolerance);
                    if (!lag.valid)
                    {
                        ImGui::TextDisabled("No lag data (result recorded before the lag metric).");
                    }
                    else
                    {
                        ImGui::Text("Start frame %i (%s), tolerance %.1f %%%s", lag.start_frame,
                            selected.measurement.move.enabled ? "end of the programmed move" : "static run: convergence from reset", lag.tolerance * 100.0,
                            selected.measurement.lag_values.empty() ? " (saved summary)" : "");
                        ImGui::Text("Changed tiles: %u of %u", lag.changed_tiles, lag.tiles);
                        ImGui::Text("Settle frames: p50 %.0f   p90 %.0f   max %.0f   never settled: %.1f %%", lag.settle_p50, lag.settle_p90, lag.settle_max, lag.never_settled * 100.0);
                        if (!lag.settled_curve.empty() && ImPlot::BeginPlot("##rt_lab_lag_curve", ImVec2(-1, 160)))
                        {
                            ImPlot::SetupAxes("frames after start", "% of changed tiles settled", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_None);
                            ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 100.0, ImPlotCond_Always);
                            std::vector<f32> percent(lag.settled_curve.size());
                            for (size_t i = 0; i < percent.size(); ++i)
                            {
                                percent[i] = lag.settled_curve[i] * 100.0f;
                            }
                            ImPlot::PlotLine("settled", percent.data(), static_cast<i32>(percent.size()));
                            ImPlot::EndPlot();
                        }
                    }
                }
                else
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "Could not load %s.", selected.path.string().c_str());
                }
            }
        }
        if (ImGui::CollapsingHeader("Convergence Error vs Progressive"))
        {
            ImGui::BeginDisabled(estimating);
            rt_lab_test_move_ui("estimate_move", lab, lab.estimate_move, current_camera_pose, current_sun_pose);
            if (!lab.estimate_move.enabled)
            {
                ImGui::SliderInt("Frames Per Preset", &lab.test_frames, 1, 4096);
            }
            else
            {
                ImGui::SliderInt("Reference Frames Per Step", &lab.move_reference_frames, 2, 4096);
                ImGui::SetItemTooltip("Progressive frames accumulated from a reset at every step (rounded up to even: custom then alternates\n"
                                      "frame index parity, which the split history needs).");
                ImGui::SliderInt("Custom Warmup Frames", &lab.move_warmup_frames, 1, 4096);
                ImGui::SetItemTooltip("Custom accumulates this many frames at step 0 before the move starts.");
            }
            ImGui::EndDisabled();
            if (!estimating)
            {
                ImGui::BeginDisabled(lab.measure_phase != RtLabMeasurePhase::IDLE);
                if (ImGui::Button("Estimate Convergence Error"))
                {
                    lab.start_requested = true;
                }
                ImGui::EndDisabled();
                if (!lab.estimate_move.enabled)
                {
                    ImGui::SetItemTooltip("Locks the camera, accumulates the Progressive preset for Frames Per Preset frames and captures it,\nthen the same for the Custom preset, then compares Custom against Progressive.");
                }
                else
                {
                    ImGui::SetItemTooltip(
                        "Custom warms up at step 0, then for every step of the move: Custom reprojects its history to the step pose\n"
                        "and adds ONE frame (captured), then Progressive resets and accumulates Reference Frames Per Step frames at\n"
                        "the same pose (captured) and the two are compared. The custom history is kept across the progressive frames.");
                }
            }
            else
            {
                if (ImGui::Button("Cancel"))
                {
                    lab.cancel_requested = true;
                }
                ImGui::SameLine();
                switch (lab.phase)
                {
                    case RtLabConvergencePhase::PROGRESSIVE:
                    case RtLabConvergencePhase::CUSTOM:
                    {
                        ImGui::Text("Camera locked, accumulating %s: %i / %i",
                            lab.phase == RtLabConvergencePhase::PROGRESSIVE ? "Progressive" : "Custom", lab.phase_frame, lab.test_frames);
                        ImGui::ProgressBar(static_cast<f32>(lab.phase_frame) / static_cast<f32>(std::max(lab.test_frames, 1)));
                        break;
                    }
                    case RtLabConvergencePhase::MOVE_WARMUP:
                    {
                        ImGui::Text("Custom warmup: %i / %i", lab.phase_frame, lab.move_warmup_frames);
                        ImGui::ProgressBar(static_cast<f32>(lab.phase_frame) / static_cast<f32>(std::max(lab.move_warmup_frames, 1)));
                        break;
                    }
                    default:
                    {
                        ImGui::Text("Step %i / %i: %s", lab.move_step, lab.move_steps,
                            lab.phase == RtLabConvergencePhase::MOVE_CUSTOM ? "custom" : (lab.phase == RtLabConvergencePhase::MOVE_PROGRESSIVE ? "progressive" : "collecting"));
                        ImGui::ProgressBar(static_cast<f32>(lab.move_step) / static_cast<f32>(std::max(lab.move_steps, 1)));
                        break;
                    }
                }
            }
            // Per step results of the (last) programmed move estimate.
            if (!lab.move_results.empty())
            {
                ImGui::SeparatorText("Programmed Move: Error Per Step");
                ImGui::TextDisabled("Custom after reprojecting + 1 frame vs a fresh progressive reference at the same pose (exposed irradiance).");
                std::vector<f64> steps = {};
                std::vector<f64> rmse = {};
                std::vector<f64> bias = {};
                std::vector<f64> noise = {};
                std::vector<f64> systematic = {};
                for (RtLabConvergenceStep const & r : lab.move_results)
                {
                    steps.push_back(static_cast<f64>(r.step));
                    rmse.push_back(std::sqrt(r.mse));
                    bias.push_back(r.bias);
                    noise.push_back(r.rms_predicted);
                    systematic.push_back(std::sqrt(std::max(r.rms_actual * r.rms_actual - r.rms_predicted * r.rms_predicted, 0.0)));
                }
                if (ImPlot::BeginPlot("##rt_lab_move_error", ImVec2(-1, 220)))
                {
                    ImPlot::SetupAxes("step", "error", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    i32 const count = static_cast<i32>(steps.size());
                    ImPlot::PlotLine("RMSE", steps.data(), rmse.data(), count);
                    ImPlot::PlotLine("bias (mean signed)", steps.data(), bias.data(), count);
                    ImPlot::PlotLine("RMS noise (split prediction)", steps.data(), noise.data(), count);
                    ImPlot::PlotLine("RMS systematic (actual - noise)", steps.data(), systematic.data(), count);
                    ImPlot::EndPlot();
                }
                f64 mean_rmse = 0.0;
                f64 mean_bias = 0.0;
                for (size_t i = 0; i < rmse.size(); ++i)
                {
                    mean_rmse += rmse[i];
                    mean_bias += bias[i];
                }
                ImGui::Text("Steps: %i   mean RMSE: %.6g   mean bias: %.6g", static_cast<i32>(rmse.size()),
                    mean_rmse / static_cast<f64>(rmse.size()), mean_bias / static_cast<f64>(rmse.size()));
                if (ImGui::Button("Copy Per Step CSV"))
                {
                    std::string csv = "step,pixels,mse,rmse,bias,mean_abs_error,rms_predicted,rms_actual\n";
                    for (RtLabConvergenceStep const & r : lab.move_results)
                    {
                        csv += fmt::format("{},{},{},{},{},{},{},{}\n", r.step, r.pixels, r.mse, std::sqrt(r.mse), r.bias, r.mean_abs_error, r.rms_predicted, r.rms_actual);
                    }
                    ImGui::SetClipboardText(csv.c_str());
                }
                ImGui::TextDisabled("The single result below = the last compared step.");
            }
            ImGui::Checkbox("Histogram Auto Range", &lab.histogram_auto_range);
            ImGui::SetItemTooltip("The histogram range rescales itself over time to the smallest symmetric range holding Auto Coverage of the pixels.\nOff = the manual range.");
            if (lab.histogram_auto_range)
            {
                ImGui::SliderFloat("Histogram Auto Coverage", &lab.histogram_auto_coverage, 0.5f, 1.0f, "%.3f");
            }
            else
            {
                ImGui::SliderFloat("Histogram Range", &lab.histogram_range, 0.001f, 4.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
            }
            ImGui::SetItemTooltip("The histogram covers [-range, range] of signed luminance difference. Pixels outside are counted separately.");
            if (lab.result.pixels > 0)
            {
                auto const & result = lab.result;
                ImGui::TextDisabled("Exposed irradiance (as the INDIRECT_DIFFUSE debug view), Custom - Progressive");
                ImGui::Text("Pixels: %u", result.pixels);
                ImGui::Text("MSE:  %.6g", result.mse);
                ImGui::Text("RMSE: %.6g", std::sqrt(result.mse));
                ImGui::Text("Mean Abs Luminance Error: %.6g", result.mean_abs_error);
                ImGui::Text("Bias (mean signed luminance error): %.6g", result.bias);
                if (result.predicted_pixels > 0)
                {
                    ImGui::SeparatorText("Split History Error Prediction");
                    ImGui::TextDisabled("Predicted sqrt(custom^2 + progressive^2) of both runs' post-spatial |A - B| / 2 (the actual\n"
                                        "error contains both) vs actual |Custom - Progressive| luminance.");
                    ImGui::Text("Pixels with a prediction: %u", result.predicted_pixels);
                    ImGui::Text("Mean predicted: %.6g   Mean actual: %.6g", result.mean_predicted, result.mean_actual);
                    ImGui::Text("RMS predicted:  %.6g   RMS actual:  %.6g   (ratio %.3f)", result.rms_predicted, result.rms_actual,
                        result.rms_actual > 0.0 ? result.rms_predicted / result.rms_actual : 0.0);
                    ImGui::Text("Correlation (predicted vs actual): %.3f", result.correlation);
                    // MSE = variance + bias^2. The split halves share every systematic error (blur bias, lag), so the
                    // prediction is the noise part only; what the actual error has on top is bias.
                    f64 const bias_squared = std::max(result.rms_actual * result.rms_actual - result.rms_predicted * result.rms_predicted, 0.0);
                    ImGui::Text("RMS noise (predicted): %.6g   RMS bias (actual - noise): %.6g", result.rms_predicted, std::sqrt(bias_squared));
                    ImGui::SetItemTooltip("MSE = variance + bias^2. Both split halves go through the same blur and see the same lag,\n"
                                          "so the split only measures noise. sqrt(rms_actual^2 - rms_predicted^2) estimates the RMS\n"
                                          "systematic error (blur bias, temporal lag, ...).");
                }
                else if (lab.presets[RT_LAB_PRESET_CUSTOM].split_history_enabled == 0)
                {
                    ImGui::TextDisabled("Enable Split History in the Custom preset for the error prediction check.");
                }
                f64 const total = static_cast<f64>(result.pixels);
                ImGui::Text("Histogram range: +-%.4g, outside: %.2f%% below, %.2f%% above",
                    result.histogram_range, 100.0 * result.below_range / total, 100.0 * result.above_range / total);
                ImGui::Checkbox("Log Scale", &rt_lab_histogram_log_scale);
                std::array<f64, RT_LAB_CONVERGENCE_HISTOGRAM_BINS> xs = {};
                std::array<f64, RT_LAB_CONVERGENCE_HISTOGRAM_BINS> ys = {};
                f64 const bin_width = 2.0 * static_cast<f64>(result.histogram_range) / static_cast<f64>(RT_LAB_CONVERGENCE_HISTOGRAM_BINS);
                for (u32 b = 0; b < RT_LAB_CONVERGENCE_HISTOGRAM_BINS; ++b)
                {
                    xs[b] = -static_cast<f64>(result.histogram_range) + (static_cast<f64>(b) + 0.5) * bin_width;
                    ys[b] = 100.0 * static_cast<f64>(result.histogram[b]) / static_cast<f64>(result.pixels);
                }
                if (ImPlot::BeginPlot("##rt_lab_convergence_histogram", ImVec2(-1, 220)))
                {
                    ImPlot::SetupAxes("luminance difference", "% of pixels", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
                    if (rt_lab_histogram_log_scale)
                    {
                        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
                    }
                    ImPlot::PlotBars("pixels", xs.data(), ys.data(), RT_LAB_CONVERGENCE_HISTOGRAM_BINS, bin_width);
                    f64 const zero = 0.0;
                    ImPlot::PlotInfLines("##zero", &zero, 1);
                    ImPlot::EndPlot();
                }
            }
            else if (lab.has_result)
            {
                ImGui::TextDisabled("Waiting for the analysis readback (RT Lab must be enabled).");
            }
        }
        ImGui::PopItemWidth();
    }
    ImGui::End();
}


void UIEngine::ui_vsm_textures(RenderContext & render_context)
{
    if(vsm_windows.view_page_table)
    {
        ImGui::Begin("VSM Page Texture", nullptr, 0);
        {
            ImGui::Image(
                imgui_renderer.create_texture_id({
                    .image_view = render_context.gpu_context->shader_debug_context.vsm_debug_page_table.id().default_view(),
                    .sampler = std::bit_cast<daxa::SamplerId>(render_context.render_data.samplers.nearest_clamp),
                }),
                ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x));
        }
        ImGui::End();
    }
    if(vsm_windows.view_meta_memory) 
    {
        ImGui::Begin("VSM Memory Texture", nullptr, 0);
        ImGui::Image(
            imgui_renderer.create_texture_id({
                .image_view = render_context.gpu_context->shader_debug_context.vsm_debug_meta_memory_table.id().default_view(),
                .sampler = std::bit_cast<daxa::SamplerId>(render_context.render_data.samplers.nearest_clamp),
            }),
            ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x));
        ImGui::End();
    }
    if(vsm_windows.view_reconstructed_shadow_map)
    {
        ImGui::Begin("VSM Reconstructed Directional Texture", nullptr, 0);
        ImGui::Image(
            imgui_renderer.create_texture_id({
                .image_view = render_context.gpu_context->shader_debug_context.vsm_recreated_shadowmap_memory_table.id().default_view(),
                .sampler = std::bit_cast<daxa::SamplerId>(render_context.render_data.samplers.nearest_clamp),
            }),
            ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().x));
        ImGui::End();
    }
}

void UIEngine::ui_convert_vdb_windw(ApplicationState & app_state, ThreadPool * _threadpool)
{
    if(ImGui::Begin("Load and Convert VDB", nullptr, 0))
    {
#if TIDO_BUILT_WITH_UTILS_VDB_LOADER
        auto & vdb_state = app_state.vdb_management_state;
        bool load_metadata = false;
        // Input path
        {
            std::array<char, 512> in_path_buffer = {};
            std::memcpy(in_path_buffer.data(), vdb_state.desired_load_path.data(), vdb_state.desired_load_path.size());
            ImGui::Text("VDB file path:");
            ImGui::SameLine(200);
            ImGui::InputText("##In VDB Path", in_path_buffer.data(), in_path_buffer.size());
            vdb_state.desired_load_path = std::string(in_path_buffer.data());
            ImGui::SameLine();
            if(ImGui::Button("..."))
            {
                vdb_state.desired_load_path = open_file_dialog("VDB\0*.vdb\0");
                vdb_state.desired_save_path = std::filesystem::path(vdb_state.desired_load_path).replace_extension(".cloudbin").string();
                load_metadata = !vdb_state.desired_load_path.empty();
            }
        }

        {
            std::array<char, 512> out_path_buffer = {};
            std::memcpy(out_path_buffer.data(), vdb_state.desired_save_path.data(), vdb_state.desired_save_path.size());
            ImGui::Text("Result File Path:");
            ImGui::SameLine(200);
            ImGui::InputText("##Out VDB Path", out_path_buffer.data(), out_path_buffer.size());
            vdb_state.desired_save_path = std::string(out_path_buffer.data());
        }

        static bool attempted_vdb_header_load = false;
        if(ImGui::Button("Load VDB metadata") || load_metadata)
        {
            fmt::print("Requested VDB metadata load: {}\n", vdb_state.desired_load_path);
            vdb_state.vdb_grids_metadata = read_vdb_header(vdb_state.desired_load_path);
            attempted_vdb_header_load = true;
            if(! vdb_state.vdb_grids_metadata.empty())
            {
                vdb_state.grid_included_in_conversion.resize(vdb_state.vdb_grids_metadata.size());
                vdb_state.grids_to_load.resize(vdb_state.vdb_grids_metadata.size());
                std::fill(vdb_state.grid_included_in_conversion.begin(), vdb_state.grid_included_in_conversion.end(), true);
            }
        };

        if(vdb_state.vdb_grids_metadata.empty() && attempted_vdb_header_load)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertFloat4ToU32(ImVec4(1.00f, 0.37f, 0.37f, 1.00f)));
            ImGui::Text("File header could not be read");
            ImGui::PopStyleColor();
        }

        std::array<const char *, 2> conversion_formats = {
            to_string(TidoVolumetricCloudFileFormat::RAW).data(),
            to_string(TidoVolumetricCloudFileFormat::CLOUD_SDF_BC1_DATA_BC6).data(),
        };

        static const std::array<std::vector<ImVec4>, s_cast<TidoVolumetricCloudFileFormat>(TidoVolumetricCloudFileFormat::COUNT)> format_colors = {
            std::vector<ImVec4>{
                ImVec4(1.00f, 0.37f, 0.37f, 1.00f), // RAW
                ImVec4(1.00f, 0.37f, 0.37f, 1.00f), // RAW
                ImVec4(1.00f, 0.37f, 0.37f, 1.00f), // RAW
                ImVec4(1.00f, 0.37f, 0.37f, 1.00f), // RAW
            },
            std::vector<ImVec4>{
                ImVec4(0.078f, 0.106f, 0.961f, 1.0f), // BC6
                ImVec4(0.078f, 0.106f, 0.961f, 1.0f), // BC6
                ImVec4(0.078f, 0.106f, 0.961f, 1.0f), // BC6
                ImVec4(0.40f, 1.00f, 0.52f, 1.00f), // SDF BC1 
            },
        };

        i32 conversion_format = s_cast<i32>(vdb_state.conversion_format);
        ImGui::Combo("Conversion format", &conversion_format, conversion_formats.data(), s_cast<i32>(conversion_formats.size()));
        vdb_state.conversion_format = s_cast<TidoVolumetricCloudFileFormat>(conversion_format);
        auto const & format_coloring = format_colors[s_cast<u32>(vdb_state.conversion_format)];

        if(! vdb_state.vdb_grids_metadata.empty())
        {
            if (ImGui::BeginTable("Metadata", 5, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
            {
                ImGui::TableSetupColumn("Name", {});
                ImGui::TableSetupColumn("Min extents", {});
                ImGui::TableSetupColumn("Max extents", {});
                ImGui::TableSetupColumn("Include", {});
                ImGui::TableSetupColumn("Clamp Values", {});
                ImGui::TableHeadersRow();

                u32 active_grid_counter = 0;
                for(u32 grid_index = 0; grid_index < vdb_state.vdb_grids_metadata.size(); grid_index++)
                {
                    bool const grid_included_in_conversion = vdb_state.grid_included_in_conversion[grid_index];
                    auto & grid_metadata = vdb_state.vdb_grids_metadata[grid_index];

                    ImVec4 row_color = ImVec4(0.1f, 0.1f, 0.1f, 1.0f);
                    if(grid_included_in_conversion && active_grid_counter < format_coloring.size())
                    {
                        row_color = format_coloring[active_grid_counter];
                        active_grid_counter++;
                    }
                    ImGui::PushStyleColor(ImGuiCol_TableRowBg, row_color);
                    ImGui::PushStyleColor(ImGuiCol_TableRowBgAlt, row_color);

                    ImGui::TableNextRow();

                    // --- Column 0 (drag handle) ---
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Selectable(grid_metadata.name.c_str(), false);

                    // Begin drag source
                    if (ImGui::BeginDragDropSource())
                    {
                        ImGui::SetDragDropPayload("ROW_INDEX", &grid_index, sizeof(u32));
                        ImGui::Text("Move row %d", grid_index);
                        ImGui::EndDragDropSource();
                    }

                    // Begin drop target
                    if (ImGui::BeginDragDropTarget())
                    {
                        if (const ImGuiPayload* payload =
                            ImGui::AcceptDragDropPayload("ROW_INDEX"))
                        {
                            u32 src_index = *(const u32*)payload->Data;

                            if (src_index != grid_index)
                            {
                                std::swap(vdb_state.vdb_grids_metadata[src_index], vdb_state.vdb_grids_metadata[grid_index]);
                            }
                        }
                        ImGui::EndDragDropTarget();
                    }

                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%i, %i, %i", grid_metadata.min_extents.x, grid_metadata.min_extents.y, grid_metadata.min_extents.z);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%i, %i, %i", grid_metadata.max_extents.x, grid_metadata.max_extents.y, grid_metadata.max_extents.z);
                    ImGui::TableSetColumnIndex(3);
                    bool check_value = vdb_state.grid_included_in_conversion[grid_index] != 0;
                    ImGui::Checkbox(fmt::format("##include{}", grid_metadata.name).c_str(), &check_value);
                    vdb_state.grid_included_in_conversion[grid_index] = check_value ? 1u : 0u;

                    ImGui::TableSetColumnIndex(4);
                    ImGui::InputFloat2(fmt::format("##clamp_values{}", grid_metadata.name).c_str(), &vdb_state.grids_to_load[grid_index].value_range.x);
                    ImGui::TableEndRow(ImGui::GetCurrentTable());

                    ImGui::PopStyleColor(2);
                }
                ImGui::EndTable();
            }
        }

        ImGui::BeginDisabled(vdb_state.convert_vdb_task != nullptr);
        if(ImGui::Button("Load and Convert"))
        {
            AssetProcessor::ConvertVDBTaskInfo task_info{
                .vdb_path = vdb_state.desired_load_path,
                .converted_result_path = vdb_state.desired_save_path,
                .output_format = vdb_state.conversion_format,
                .threadpool = _threadpool,
            };
            for(u32 grid_index = 0; grid_index < vdb_state.vdb_grids_metadata.size(); grid_index++)
            {
                if(vdb_state.grid_included_in_conversion[grid_index])
                {
                    bool const is_sdf_compressed = vdb_state.conversion_format == TidoVolumetricCloudFileFormat::CLOUD_SDF_BC1_DATA_BC6;
                    bool const is_sdf_channel = task_info.grids_to_convert.size() == 3;

                    // We don't convert to fp16 only if we are using a custom compression of the sdf values.
                    // This is because only BC1 sdf compression expects values to be in fp32.
                    bool const convert_to_fp16 = !(is_sdf_compressed && is_sdf_channel);
                    task_info.grids_to_convert.push_back(VDBGridInfo{
                        .name = vdb_state.vdb_grids_metadata[grid_index].name,
                        .value_range = vdb_state.grids_to_load[grid_index].value_range,
                        .convert_to_fp16 = convert_to_fp16,
                    });
                }
            }
            vdb_state.convert_vdb_task = std::make_shared<AssetProcessor::ConvertVDBTask>(task_info);
            _threadpool->async_dispatch(vdb_state.convert_vdb_task);
        }
        ImGui::EndDisabled();
        if(vdb_state.convert_vdb_task != nullptr)
        {
            // Position at bottom-right corner with padding
            ImVec2 work_pos = ImGui::GetMainViewport()->WorkPos;
            ImVec2 work_size = ImGui::GetMainViewport()->WorkSize;
            ImVec2 window_pos = ImVec2(work_pos.x + work_size.x - 10.0f, work_pos.y + work_size.y - 10.0f);
            ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, ImVec2(1.0f, 1.0f)); // pivot bottom-right
    
            ImGui::SetNextWindowBgAlpha(0.9f); // Slightly transparent
    
            static int ticks_since_finished = 0;
            if (ImGui::Begin("VDB Conversion", nullptr, 
                ImGuiWindowFlags_NoDecoration | 
                ImGuiWindowFlags_AlwaysAutoResize | 
                ImGuiWindowFlags_NoSavedSettings | 
                ImGuiWindowFlags_NoFocusOnAppearing | 
                ImGuiWindowFlags_NoNav))
            {
                if(vdb_state.convert_vdb_task->not_finished == 0)
                {
                    ticks_since_finished += 1;
                    if(vdb_state.convert_vdb_task->result)
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertFloat4ToU32(ImVec4(0.40f, 1.00f, 0.52f, 1.00f)));
                        ImGui::Text("VDB conversion finished successfully!");
                        ImGui::PopStyleColor();
                    }
                    else
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertFloat4ToU32(ImVec4(1.00f, 0.37f, 0.37f, 1.00f)));
                        ImGui::Text("VDB conversion failed!");
                        ImGui::Text(vdb_state.convert_vdb_task->error_message.c_str());
                        ImGui::PopStyleColor();
                    }
                }
                else
                {
                    ImGui::Text("VDB conversion in process!");
                    ImGui::Text(vdb_state.convert_vdb_task->progress_message.c_str());
                    Task * current_subtask = vdb_state.convert_vdb_task->current_subtask;
                    if(current_subtask)
                    {
                        ImGui::ProgressBar(
                            s_cast<f32>(current_subtask->chunk_count - current_subtask->not_finished) /
                            s_cast<f32>(current_subtask->chunk_count),
                        ImVec2(-1.0f, 15.0f));
                    }
                }
            }
            ImGui::End();
            if(ticks_since_finished > 1000)
            {
                ticks_since_finished = 0;
                vdb_state.convert_vdb_task = nullptr;
            }
        }
#else // TIDO_BUILT_WITH_UTILS_VDB_LOADER
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertFloat4ToU32(ImVec4(1.00f, 0.37f, 0.37f, 1.00f)));
        ImGui::Text("VDB loading and converting is not available in this build.");
        ImGui::PopStyleColor();
#endif // TIDO_BUILT_WITH_UTILS_VDB_LOADER
    }
    ImGui::End();
}

void UIEngine::ui_visbuffer_pipeline_statistics(RenderContext & render_context)
{
    // Calculate Statistics:
    u32 mesh_instance_count = render_context.mesh_instance_counts.mesh_instance_count;
    u32 first_pass_meshes_post_cull = render_context.general_readback.first_pass_mesh_count_post_cull[0] + render_context.general_readback.first_pass_mesh_count_post_cull[1];
    u32 first_pass_meshlets_pre_cull = render_context.general_readback.first_pass_meshlet_count_pre_cull[0] + render_context.general_readback.first_pass_meshlet_count_pre_cull[1];
    u32 first_pass_meshlets_post_cull = render_context.general_readback.first_pass_meshlet_count_post_cull;
    u32 second_pass_meshes_post_cull = render_context.general_readback.second_pass_mesh_count_post_cull[0] + render_context.general_readback.second_pass_mesh_count_post_cull[1];
    u32 second_pass_meshlets_pre_cull = render_context.general_readback.second_pass_meshlet_count_pre_cull[0] + render_context.general_readback.second_pass_meshlet_count_pre_cull[1];
    u32 second_pass_meshlets_post_cull = render_context.general_readback.second_pass_meshlet_count_post_cull;
    u32 total_meshlets_drawn = first_pass_meshlets_post_cull + second_pass_meshlets_post_cull;

    u32 meshlet_bitfield_used_size = ((FIRST_PASS_MESHLET_BITFIELD_U32_OFFSETS_SIZE + render_context.general_readback.first_pass_meshlet_bitfield_requested_dynamic_size) * 4u) / 1000u;
    u32 meshlet_bitfield_total_size = ((FIRST_PASS_MESHLET_BITFIELD_U32_SIZE) * 4u) / 1000u;
    struct VisbufferPipelineStat
    {
        char const * name = {};
        char const * unit = {};
        u32 value = {};
        u32 max_value = {};
    };
    std::array visbuffer_pipeline_stats = {
        VisbufferPipelineStat{"Mesh Instances Pre Cull", "", mesh_instance_count, MAX_MESH_INSTANCES},
        VisbufferPipelineStat{"First Pass Meshes Post Cull", "", first_pass_meshes_post_cull, MAX_MESH_INSTANCES},
        VisbufferPipelineStat{"Second Pass Meshes Post Cull", "", second_pass_meshes_post_cull, MAX_MESH_INSTANCES},
        VisbufferPipelineStat{"First Pass Meshlets Pre Cull", "", first_pass_meshlets_pre_cull, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS},
        VisbufferPipelineStat{"First Pass Meshlets Post Cull", "", first_pass_meshlets_post_cull, MAX_MESHLET_INSTANCES},
        VisbufferPipelineStat{"Second Pass Meshlets Pre Cull", "", second_pass_meshlets_pre_cull, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS},
        VisbufferPipelineStat{"Second Pass Meshlets Post Cull", "", second_pass_meshlets_post_cull, MAX_MESHLET_INSTANCES},
        VisbufferPipelineStat{"Total Meshlet Instances Post Cull", "", total_meshlets_drawn, MAX_MESHLET_INSTANCES},
        VisbufferPipelineStat{"First Pass Bitfield Use", "kb", meshlet_bitfield_used_size, meshlet_bitfield_total_size},
    };
    Settings const & settings = render_context.render_data.settings;
    f32 const soft_limit_percentage = settings.visbuffer_usage_soft_limit * 100.0f;
    ImVec4 const LOD_BIAS_RED = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
    ImVec4 const LOD_BIAS_ORANGE = ImVec4(1.0f, 0.6f, 0.15f, 1.0f);
    ImVec4 const LOD_BIAS_YELLOW = ImVec4(1.0f, 0.9f, 0.25f, 1.0f);
    ImVec4 const LOD_BIAS_GREEN = ImVec4(0.4f, 0.85f, 0.45f, 1.0f);

    // Short history for the lod bias / usage plots.
    static constexpr i32 LOD_BIAS_HISTORY_SIZE = 256;
    static std::array<f32, LOD_BIAS_HISTORY_SIZE> lod_bias_history = {};
    static std::array<f32, LOD_BIAS_HISTORY_SIZE> usage_history = {};
    static i32 lod_bias_history_offset = 0;
    lod_bias_history[lod_bias_history_offset] = settings.lod_bias;
    usage_history[lod_bias_history_offset] = render_context.lod_bias_max_usage * 100.0f;
    lod_bias_history_offset = (lod_bias_history_offset + 1) % LOD_BIAS_HISTORY_SIZE;

    if (ImGui::CollapsingHeader("Visbuffer Pipeline"))
    {
        if (ImGui::BeginTable("Visbuffer GPU Buffer Metrics", 4, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
        {
            ImGui::TableSetupColumn("Value Name", {});
            ImGui::TableSetupColumn("Value", {});
            ImGui::TableSetupColumn("Value Max", {});
            ImGui::TableSetupColumn("Value %", {});
            ImGui::TableHeadersRow();
            for (auto const & stat : visbuffer_pipeline_stats)
            {
                f32 const percentage = static_cast<f32>(stat.value) / static_cast<f32>(stat.max_value) * 100.0f;
                if (percentage > 100.0f)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, 1.0f)));
                }
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text(stat.name);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%i%s", stat.value, stat.unit);
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%i%s", stat.max_value, stat.unit);
                ImGui::TableSetColumnIndex(3);
                bool const over_soft_limit = percentage <= 100.0f && percentage > soft_limit_percentage;
                if (over_soft_limit)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetColorU32(LOD_BIAS_ORANGE));
                }
                ImGui::Text("%f%%", percentage);
                if (over_soft_limit)
                {
                    ImGui::PopStyleColor();
                }
                if (percentage > 100.0f)
                {
                    ImGui::PopStyleColor();
                }
            }
            ImGui::EndTable();
        }

        ImGui::SeparatorText("Automatic Lod Bias");
        {
            using LodBiasState = RenderContext::LodBiasState;
            f32 const usage = render_context.lod_bias_max_usage;

            ImVec4 state_color = LOD_BIAS_GREEN;
            char const * state_name = "SETTLED";
            char const * state_tooltip = "Usage below the soft limit. The bias slowly descends.";
            if (!settings.enable_auto_lod_bias)
            {
                state_color = ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled);
                state_name = "OFF";
                state_tooltip = "Auto lod bias is disabled (Settings > Geometry).";
            }
            else
            {
                switch (render_context.lod_bias_state)
                {
                    case LodBiasState::SPIKE:
                        state_color = LOD_BIAS_RED;
                        state_name = "SPIKE";
                        state_tooltip = "Usage exceeded 100%. The bias was raised sharply.";
                        break;
                    case LodBiasState::RISING:
                        state_color = LOD_BIAS_ORANGE;
                        state_name = "RISING";
                        state_tooltip = "Usage above the soft limit. The bias slowly rises.";
                        break;
                    case LodBiasState::DECREASING:
                        state_color = LOD_BIAS_YELLOW;
                        state_name = "HOLDING";
                        state_tooltip = "Usage above the soft limit but already decreasing. The bias is held.";
                        break;
                    default: break;
                }
            }

            ImVec4 usage_color = LOD_BIAS_GREEN;
            if (usage > 1.0f)
            {
                usage_color = LOD_BIAS_RED;
            }
            else if (usage > settings.visbuffer_usage_soft_limit)
            {
                usage_color = LOD_BIAS_ORANGE;
            }

            if (ImGui::BeginTable("Lod Bias Status", 2, ImGuiTableFlags_SizingStretchProp))
            {
                ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed);
                ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("State");
                ImGui::TableSetColumnIndex(1);
                {
                    // Colored status chip.
                    ImVec4 const chip_bg = ImVec4(state_color.x, state_color.y, state_color.z, 0.25f);
                    ImGui::PushStyleColor(ImGuiCol_Button, chip_bg);
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, chip_bg);
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive, chip_bg);
                    ImGui::PushStyleColor(ImGuiCol_Text, state_color);
                    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);
                    ImGui::Button(state_name, ImVec2(ImGui::CalcTextSize("HOLDING").x + ImGui::GetStyle().FramePadding.x * 4.0f, 0.0f));
                    ImGui::PopStyleVar();
                    ImGui::PopStyleColor(4);
                    ImGui::SetItemTooltip("%s", state_tooltip);
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Lod Bias");
                ImGui::TableSetColumnIndex(1);
                {
                    std::string const overlay = fmt::format("{:.3f}  (x{:.2f} pixel error)", settings.lod_bias, std::exp2(settings.lod_bias));
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, state_color);
                    f32 const reset_width = ImGui::CalcTextSize("Reset").x + ImGui::GetStyle().FramePadding.x * 2.0f;
                    ImGui::ProgressBar(settings.lod_bias / RenderContext::AUTO_LOD_BIAS_MAX, ImVec2(-(reset_width + ImGui::GetStyle().ItemSpacing.x), 0.0f), overlay.c_str());
                    ImGui::PopStyleColor();
                    ImGui::SameLine();
                    if (ImGui::Button("Reset##lod_bias_stats"))
                    {
                        render_context.render_data.settings.lod_bias = 0.0f;
                    }
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Max Buffer Usage");
                ImGui::TableSetColumnIndex(1);
                {
                    std::string const overlay = fmt::format("{:.1f}%  (soft limit {:.0f}%)", usage * 100.0f, soft_limit_percentage);
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, usage_color);
                    ImGui::ProgressBar(std::min(usage, 1.0f), ImVec2(-FLT_MIN, 0.0f), overlay.c_str());
                    ImGui::PopStyleColor();
                    // Soft limit marker.
                    ImVec2 const bar_min = ImGui::GetItemRectMin();
                    ImVec2 const bar_max = ImGui::GetItemRectMax();
                    f32 const marker_x = bar_min.x + (bar_max.x - bar_min.x) * settings.visbuffer_usage_soft_limit;
                    ImGui::GetWindowDrawList()->AddLine(ImVec2(marker_x, bar_min.y), ImVec2(marker_x, bar_max.y), ImGui::GetColorU32(LOD_BIAS_ORANGE), 2.0f);
                }

                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextUnformatted("History");
                ImGui::TableSetColumnIndex(1);
                {
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, state_color);
                    ImGui::PlotLines("##lod_bias_history", lod_bias_history.data(), LOD_BIAS_HISTORY_SIZE, lod_bias_history_offset, "lod bias", 0.0f, RenderContext::AUTO_LOD_BIAS_MAX, ImVec2(-FLT_MIN, 40.0f));
                    ImGui::PopStyleColor();
                    ImGui::PushStyleColor(ImGuiCol_PlotLines, usage_color);
                    ImGui::PlotLines("##usage_history", usage_history.data(), LOD_BIAS_HISTORY_SIZE, lod_bias_history_offset, "buffer usage %", 0.0f, 100.0f, ImVec2(-FLT_MIN, 40.0f));
                    ImGui::PopStyleColor();
                }
                ImGui::EndTable();
            }
        }
    }
}

void UIEngine::ui_pgi_statistics(RenderContext & render_context)
{
    if (ImGui::CollapsingHeader("Probe Global Illumination"))
    {
        u32 heavy_ray_count = 512000;
        u32 super_heavy_ray_count = 1024000;
        auto & render_data = render_context.render_data;
        u32 total_probes = render_data.pgi_settings.probe_count.x * render_data.pgi_settings.probe_count.y * render_data.pgi_settings.probe_count.z * render_data.pgi_settings.cascade_count;
        static u32 updated_probes = 0;
        static u32 ray_count = 0;
        if ((render_data.frame_index % 11) == 0)
        {
            updated_probes = render_context.general_readback.requested_probes;
            ray_count = render_data.pgi_settings.probe_trace_resolution * render_data.pgi_settings.probe_trace_resolution * render_context.general_readback.requested_probes;
        }
        ImGui::Text(fmt::format("Total Probe Count {}", total_probes).c_str());
        ImGui::Text(fmt::format("PGI Updated Probes {:>7} / {:>7} = {}%%",
            updated_probes,
            total_probes,
            float(updated_probes) / float(total_probes) * 100.0f)
                .c_str());
        if (ray_count > super_heavy_ray_count)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetColorU32(ImVec4(1.0f, 0.0f, 0.0f, 1.0f)));
        }
        else if (ray_count > heavy_ray_count)
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetColorU32(ImVec4(1.0f, 1.0f, 0.0f, 1.0f)));
        }
        ImGui::Text(fmt::format("Rays Shoot this frame {}", ray_count).c_str());
        if (ray_count > heavy_ray_count)
        {
            ImGui::PopStyleColor(1);
        }
    }
}

void ui_rtgi_statistics(RenderContext & render_context)
{
    if (ImGui::CollapsingHeader("Ray Traced Global Illumination"))
    {
        auto & render_data = render_context.render_data;
        {
            auto const & rb = render_context.general_readback;
            u32 const base_rays = rb.rtgi_requested_base_rays;
            u32 const extra_rays = rb.rtgi_requested_extra_rays;
            u32 const budget = rb.rtgi_ray_budget;
            f32 const half_res_pixels = std::max(1.0f, static_cast<f32>((render_data.settings.render_target_size.x / 2) * (render_data.settings.render_target_size.y / 2)));
            u32 const requested = base_rays + extra_rays;
            u32 const req_d = rb.rtgi_requested_diffuse_rays;
            u32 const req_s = rb.rtgi_requested_specular_rays;
            u32 const shot_d = rb.rtgi_shot_diffuse_rays;
            u32 const shot_s = rb.rtgi_shot_specular_rays;
            u32 const shot = shot_d + shot_s;
            auto pct = [](u32 part, u32 whole) -> f32 { return whole > 0 ? 100.0f * static_cast<f32>(part) / static_cast<f32>(whole) : 0.0f; };
            ImGui::SeparatorText("Rays this frame");
            if (ImGui::BeginTable("##rtgi_ray_stats", 4, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV))
            {
                ImGui::TableSetupColumn("");
                ImGui::TableSetupColumn("Diffuse");
                ImGui::TableSetupColumn("Specular");
                ImGui::TableSetupColumn("Total");
                ImGui::TableHeadersRow();
                auto row = [](char const * label, char const * fmt, auto d, auto s, auto t)
                {
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    ImGui::TextUnformatted(label);
                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text(fmt, d);
                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text(fmt, s);
                    ImGui::TableSetColumnIndex(3);
                    ImGui::Text(fmt, t);
                };
                row("Requested", "%u", req_d, req_s, req_d + req_s);
                row("Shot", "%u", shot_d, shot_s, shot);
                row("Share of shot", "%.0f%%", pct(shot_d, shot), pct(shot_s, shot), shot > 0 ? 100.0f : 0.0f);
                row("Fulfilled", "%.0f%%", pct(shot_d, req_d), pct(shot_s, req_s), pct(shot, req_d + req_s));
                row("Shot / half-res px", "%.2f", static_cast<f32>(shot_d) / half_res_pixels, static_cast<f32>(shot_s) / half_res_pixels, static_cast<f32>(shot) / half_res_pixels);
                ImGui::EndTable();
            }
            ImGui::Text("Requested base %u + extra %u, budget %u", base_rays, extra_rays, budget);
            if (render_data.rtgi_settings.use_repacked_ray_dispatch != 0)
            {
                u32 const shade_sharp = rb.rtgi_shade_sharp_rays;
                u32 const shade_coarse = rb.rtgi_shade_coarse_rays;
                u32 const shade_miss = rb.rtgi_shade_miss_rays;
                u32 const shade_total = shade_sharp + shade_coarse + shade_miss;
                ImGui::Text("Hit shading: sharp %u (%.0f%%), coarse %u (%.0f%%), miss %u (%.0f%%)", shade_sharp, pct(shade_sharp, shade_total), shade_coarse, pct(shade_coarse, shade_total), shade_miss, pct(shade_miss, shade_total));
                ImGui::SetItemTooltip("Rays shaded with full material + light shading (sharp) vs PGI radiance (coarse) vs sky (miss) this frame.");
            }
            if (shot > 0)
            {
                ImGui::Text("%s gets more rays (%.0f%% of shot)", shot_d >= shot_s ? "Diffuse" : "Specular", std::max(pct(shot_d, shot), pct(shot_s, shot)));
            }
            {
                // Sum over geometry pixels of min(history / max history, 1), averaged.
                f32 const conv_pixels = static_cast<f32>(rb.rtgi_convergence_pixels) * static_cast<f32>(RTGI_CONVERGENCE_SCALE);
                f32 const conv_d = conv_pixels > 0.0f ? static_cast<f32>(rb.rtgi_convergence_diffuse_sum) / conv_pixels : 0.0f;
                f32 const conv_s = conv_pixels > 0.0f ? static_cast<f32>(rb.rtgi_convergence_specular_sum) / conv_pixels : 0.0f;
                bool const spec_on = render_data.rtgi_settings.specular_enabled != 0;
                f32 const conv_t = spec_on ? 0.5f * (conv_d + conv_s) : conv_d;
                ImGui::SeparatorText("Ray distribution & history length");
                ImGui::Text("Mean convergence: diffuse %.1f%%, specular %.1f%%, total %.1f%%", 100.0f * conv_d, 100.0f * conv_s, 100.0f * conv_t);
                // Percent of geometry pixels per bucket of history / fast convergence target: buckets 0..14 split
                // [0, target) evenly, bucket 15 = at or above the target (no extra rays requested).
                constexpr int bucket_count = RTGI_CONVERGENCE_BUCKETS;
                f32 const pixel_count = std::max(1.0f, static_cast<f32>(rb.rtgi_convergence_pixels));
                std::array<f32, 2 * bucket_count> hist = {};
                for (int b = 0; b < bucket_count; ++b)
                {
                    hist[b] = 100.0f * static_cast<f32>(rb.rtgi_convergence_histogram_diffuse[b]) / pixel_count;
                    hist[bucket_count + b] = 100.0f * static_cast<f32>(rb.rtgi_convergence_histogram_specular[b]) / pixel_count;
                }
                f32 const diffuse_target = std::max(render_data.rtgi_settings.fast_convergence_samples, 1.0f);
                f32 const specular_target = std::max(std::min(render_data.rtgi_settings.specular_fast_convergence_samples, render_data.rtgi_settings.specular_max_temporal_frames), 1.0f);
                constexpr int line_points = bucket_count - 1; // last bucket (>= target) is shown as text, not on the line
                std::array<f32, line_points> line_x = {};
                for (int b = 0; b < line_points; ++b)
                {
                    line_x[b] = 100.0f * (static_cast<f32>(b) + 0.5f) / static_cast<f32>(line_points); // bucket center, % of target
                }
                ImGui::Text("At target (no extra rays): diffuse %.1f%%, specular %.1f%%", hist[bucket_count - 1], spec_on ? hist[2 * bucket_count - 1] : 0.0f);

                // Left: requested vs shot rays per half-res pixel (blocks), budget as a line. Right: history distribution.
                if (ImGui::BeginTable("##rtgi_ray_plots", 2, ImGuiTableFlags_SizingStretchSame))
                {
                    constexpr f32 plot_height = 200.0f;
                    ImGui::TableNextColumn();
                    if (ImPlot::BeginPlot("Samples / half-res px##rtgi_ray_blocks", ImVec2(-1.0f, plot_height), ImPlotFlags_NoMouseText))
                    {
                        // Series = signal (diffuse blue, specular orange, same colors as the history lines),
                        // groups = requested / shot / revived. Series-major: [diffuse req, shot, revived][specular ...].
                        // Revived = samples the hash grid revive wrote this frame (diffuse only; the grid holds no specular).
                        f32 const revived_samples = static_cast<f32>(rb.rtgi_revived_samples) / static_cast<f32>(RTGI_REVIVE_SAMPLE_SCALE);
                        std::array<f32, 6> blocks = {
                            static_cast<f32>(req_d) / half_res_pixels,
                            static_cast<f32>(shot_d) / half_res_pixels,
                            revived_samples / half_res_pixels,
                            static_cast<f32>(req_s) / half_res_pixels,
                            static_cast<f32>(shot_s) / half_res_pixels,
                            0.0f,
                        };
                        static char const * const block_series[] = {"diffuse", "specular"};
                        static constexpr double group_ticks[] = {0.0, 1.0, 2.0};
                        static char const * const group_labels[] = {"requested", "shot", "revived"};
                        double const budget_per_px = static_cast<double>(budget) / static_cast<double>(half_res_pixels);
                        // Left axis: samples per half-res pixel. Right axis: the same range as total samples (left
                        // value x half-res pixel count). Both get the same explicit limits so their ticks line up.
                        double y_max = budget_per_px;
                        for (f32 const v : blocks) { y_max = std::max(y_max, static_cast<double>(v)); }
                        y_max = y_max > 0.0 ? y_max * 1.15 : 1.0;
                        static double total_samples_scale = 1.0;
                        total_samples_scale = static_cast<double>(half_res_pixels);
                        ImPlot::SetupAxes(nullptr, nullptr, ImPlotAxisFlags_None, ImPlotAxisFlags_None);
                        ImPlot::SetupAxisLimits(ImAxis_X1, -0.6, 2.6, ImGuiCond_Always);
                        ImPlot::SetupAxisTicks(ImAxis_X1, group_ticks, 3, group_labels);
                        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, y_max, ImGuiCond_Always);
                        ImPlot::SetupAxis(ImAxis_Y2, nullptr, ImPlotAxisFlags_AuxDefault | ImPlotAxisFlags_NoGridLines);
                        ImPlot::SetupAxisLimits(ImAxis_Y2, 0.0, y_max, ImGuiCond_Always);
                        ImPlot::SetupAxisFormat(ImAxis_Y2, [](double value, char * buff, int size, void * user_data) -> int
                        {
                            double const total = value * *static_cast<double const *>(user_data);
                            if (std::abs(total) >= 1e6) { return std::snprintf(buff, size, "%.2fM", total / 1e6); }
                            if (std::abs(total) >= 1e3) { return std::snprintf(buff, size, "%.0fK", total / 1e3); }
                            return std::snprintf(buff, size, "%.0f", total);
                        }, &total_samples_scale);
                        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
                        ImPlot::PlotBarGroups(block_series, blocks.data(), spec_on ? 2 : 1, 3, 0.7);
                        ImPlot::PlotInfLines("budget", &budget_per_px, 1, ImPlotInfLinesFlags_Horizontal);
                        ImPlot::EndPlot();
                    }
                    ImGui::SetItemTooltip("Samples per half-res pixel this frame (blue diffuse, orange specular; a traced ray is one sample): requested, shot, and revived (history samples the hash grid revive wrote, %u pixels). Horizontal line = hard sample budget (diffuse + specular). Left axis: samples per half-res pixel, right axis: total samples.", rb.rtgi_revived_pixels);
                    ImGui::TableNextColumn();
                    if (ImPlot::BeginPlot("History distribution##rtgi_history_lines", ImVec2(-1.0f, plot_height), ImPlotFlags_NoMouseText))
                    {
                        char diffuse_label[48];
                        char specular_label[48];
                        std::snprintf(diffuse_label, sizeof(diffuse_label), "diffuse (target %.0f)", diffuse_target);
                        std::snprintf(specular_label, sizeof(specular_label), "specular (target %.0f)", specular_target);
                        // Fixed scale on both axes (locked, no auto fit / pan / zoom) so frames and A/B runs compare directly.
                        ImPlot::SetupAxes("history, % of target", "% of pixels", ImPlotAxisFlags_Lock, ImPlotAxisFlags_Lock);
                        ImPlot::SetupAxisLimits(ImAxis_X1, 0.0, 100.0, ImGuiCond_Always);
                        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 25.0, ImGuiCond_Always);
                        ImPlot::SetupLegend(ImPlotLocation_NorthEast);
                        ImPlot::PlotLine(diffuse_label, line_x.data(), hist.data(), line_points);
                        if (spec_on)
                        {
                            ImPlot::PlotLine(specular_label, line_x.data(), hist.data() + bucket_count, line_points);
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::SetItemTooltip("Geometry pixels per history bucket below the fast convergence target (the ray demand target): %d buckets of %.2f diffuse / %.2f specular samples. Pixels at or above the target are listed above. %u pixels.",
                        line_points, diffuse_target / line_points, specular_target / line_points, rb.rtgi_convergence_pixels);
                    ImGui::EndTable();
                }
            }
            if (render_data.rtgi_settings.hash_grid_enabled)
            {
                ImGui::SeparatorText("Hash grid");
                f32 const occupancy = static_cast<f32>(rb.rtgi_hash_grid_live_cells) / static_cast<f32>(RTGI_HASH_GRID_CAPACITY);
                char grid_overlay[64];
                std::snprintf(grid_overlay, sizeof(grid_overlay), "%u / %u cells live", rb.rtgi_hash_grid_live_cells, RTGI_HASH_GRID_CAPACITY);
                ImGui::ProgressBar(occupancy, ImVec2(-FLT_MIN, 0.0f), grid_overlay);
                ImGui::Text("Touched this frame: %u", rb.rtgi_hash_grid_touched_cells);
                ImGui::Text("Failed queries:     %u px", rb.rtgi_hash_grid_failed_queries);
                ImGui::SetItemTooltip("Geometry pixels whose probe window had no free slot.");
            }
            f32 const coverage = static_cast<f32>(budget) / std::max(1.0f, static_cast<f32>(requested));
            char overlay[32];
            std::snprintf(overlay, sizeof(overlay), "budget covers %.0f%%", 100.0f * coverage);
            ImGui::ProgressBar(std::min(coverage, 1.0f), ImVec2(-FLT_MIN, 0.0f), overlay);
        }
    }
}

void ui_light_statistics(RenderContext & render_context)
{
    if (ImGui::CollapsingHeader("Lights"))
    {
        ImGui::Text("Max Point Lights: %i", MAX_POINT_LIGHTS);
        ImGui::Text("Max Spot Lights: %i", MAX_SPOT_LIGHTS);
        ImGui::Text("Max Light Instances: %i", MAX_LIGHT_INSTANCES_PER_FRAME);
        ImGui::Text("Point Lights: %i", render_context.render_data.light_settings.point_light_count);
        ImGui::Text("Spot Lights: %i", render_context.render_data.light_settings.spot_light_count);
    }
}

void UIEngine::ui_render_statistics(RenderContext & render_context, ApplicationState & app_state)
{
    static float t = 0;
    t += gather_perm_measurements ? render_context.render_data.delta_time : 0.0f;
    bool auto_reset_timings = false;
    if (ImGui::Begin("Render statistics", nullptr, ImGuiWindowFlags_NoCollapse))
    {
        static float history = 20.0f;
        // Copies the shown part of a scrolling buffer (x >= the interval start) into plain arrays, oldest point first,
        // reduced to at most MAX_PLOT_POINTS buckets. Each bucket keeps its largest sample so spikes stay visible.
        // `band` is a second buffer written in lockstep with `buffer` (same size, same frames); it is read at the
        // indices picked from `buffer`. Copying everything every frame was expensive in Debug with 10000 point buffers.
        static constexpr i32 MAX_PLOT_POINTS = 1024;
        f32 const interval_start = show_entire_interval ? -std::numeric_limits<f32>::infinity() : t - history;
        auto unroll = [&](auto const & buffer, auto const & band, std::vector<f32> & xs, std::vector<f32> & ys, std::vector<f32> & band_ys)
        {
            i32 const size = static_cast<i32>(buffer.data.size());
            i32 const count = buffer.wrapped ? size : buffer.offset;
            i32 const start = buffer.wrapped ? buffer.offset : 0;
            auto at = [&](i32 i) { return (start + i) % size; };
            i32 first = count;
            while (first > 0 && buffer.data[at(first - 1)].x >= interval_start) { --first; }
            i32 const shown = count - first;
            i32 const buckets = std::min(shown, MAX_PLOT_POINTS);
            xs.resize(buckets);
            ys.resize(buckets);
            band_ys.resize(buckets);
            for (i32 b = 0; b < buckets; ++b)
            {
                i32 const begin = first + static_cast<i32>(static_cast<i64>(b) * shown / buckets);
                i32 const end = first + static_cast<i32>(static_cast<i64>(b + 1) * shown / buckets);
                i32 best = at(begin);
                for (i32 i = begin + 1; i < end; ++i)
                {
                    if (buffer.data[at(i)].y > buffer.data[best].y) { best = at(i); }
                }
                xs[b] = buffer.data[best].x;
                ys[b] = buffer.data[best].y;
                band_ys[b] = band.data[best].y;
            }
        };
        // Line + shaded +-1 std dev band in the same color.
        auto plot_with_band = [&](char const * label, i32 color_index, auto const & line_buffer, auto const & std_dev_buffer)
        {
            static std::vector<f32> xs, ys, std_ys, low, high;
            unroll(line_buffer, std_dev_buffer, xs, ys, std_ys);
            i32 const count = static_cast<i32>(xs.size());
            if (count == 0) { return; }
            low.resize(count);
            high.resize(count);
            for (i32 i = 0; i < count; ++i)
            {
                low[i] = std::max(0.0f, ys[i] - std_ys[i]);
                high[i] = ys[i] + std_ys[i];
            }
            ImVec4 const color = ImPlot::GetColormapColor(color_index);
            ImPlot::SetNextFillStyle(color, 0.25f);
            ImPlot::PlotShaded(label, xs.data(), low.data(), high.data(), count);
            ImPlot::SetNextLineStyle(color);
            ImPlot::PlotLine(label, xs.data(), ys.data(), count);
        };
        auto setup_time_axes = [&](f32 first_t)
        {
            ImPlot::SetupAxes("Timeline", "Execution time", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
            if (!show_entire_interval)
            {
                ImPlot::SetupAxisLimits(ImAxis_X1, t - history, t, ImGuiCond_Always);
            }
            else
            {
                ImPlot::SetupAxisLimits(ImAxis_X1, first_t, t, ImGuiCond_Always);
            }
            ImPlot::SetupAxisFormat(ImAxis_Y1, "%.0f us");
            ImPlot::SetupAxisFormat(ImAxis_X1, "%.0fs");
        };

        ImGui::SeparatorText("CPU Group Timings");
        {
            {
                // Total frame rate: frames counted over a FPS_WINDOW_S window, refreshed once per window (no long smoothing lag).
                static constexpr f64 FPS_WINDOW_S = 0.25;
                static f64 window_time = 0.0;
                static f64 window_time_sq = 0.0;
                static u32 window_frames = 0;
                static f64 fps = 0.0;
                static f64 window_frame_ms = 0.0;
                static f64 window_frame_std_dev_ms = 0.0;
                f64 const delta = static_cast<f64>(app_state.delta_time);
                window_time += delta;
                window_time_sq += delta * delta;
                window_frames += 1;
                if (window_time >= FPS_WINDOW_S)
                {
                    f64 const mean = window_time / static_cast<f64>(window_frames);
                    fps = static_cast<f64>(window_frames) / window_time;
                    window_frame_ms = mean * 1000.0;
                    window_frame_std_dev_ms = std::sqrt(std::max(0.0, window_time_sq / static_cast<f64>(window_frames) - mean * mean)) * 1000.0;
                    window_time = 0.0;
                    window_time_sq = 0.0;
                    window_frames = 0;
                }
                ImGui::Text("Total FPS:                    %.1f  (frame time %.2f ms, std dev %.2f ms)", fps, window_frame_ms, window_frame_std_dev_ms);
                ImGui::Text("Last frame:                   %.2f ms (%.1f fps)", app_state.delta_time * 1000.0f, 1.0f / app_state.delta_time);
            }
            ImGui::Text("CPU Windowing:                %fms", app_state.time_taken_cpu_windowing * 1000.0f);
            ImGui::Text("CPU Application:              %fms", app_state.time_taken_cpu_application * 1000.0f);
            ImGui::Text("CPU Wait For GPU:             %fms", app_state.time_taken_cpu_wait_for_gpu * 1000.0f);
            ImGui::Text("CPU Renderer Prepare:         %fms", app_state.time_taken_cpu_renderer_prepare * 1000.0f);
            ImGui::Text("CPU Renderer Record + Submit: %fms", app_state.time_taken_cpu_renderer_record * 1000.0f);
        }
        ImGui::SeparatorText("GPU Group Timings");
        {
            if (gather_perm_measurements)
            {
                if (ImGui::Button("Stop gathering")) { gather_perm_measurements = false; }
            }
            else
            {
                if (ImGui::Button("Start gathering"))
                {
                    gather_perm_measurements = true;
                    auto_reset_timings = true;
                }
            }

            ImGui::SameLine();
            if (ImGui::Button("Reset timings") || auto_reset_timings)
            {
                t = 0;
                for (i32 i = 0; i < render_times_history.scrolling_ewa.size(); i++)
                {
                    render_times_history.scrolling_ewa.at(i).erase();
                    render_times_history.scrolling_mean.at(i).erase();
                    render_times_history.scrolling_raw.at(i).erase();
                    render_times_history.scrolling_std_dev.at(i).erase();
                }
                for (u32 i = 0; i < RenderTimes::FLAT_TIMINGS_COUNT; i++)
                {
                    render_times_history.pass_average.at(i).erase();
                    render_times_history.pass_std_dev.at(i).erase();
                }
            }
            ImGui::SameLine();
            ImGui::Checkbox("Show entire measured interval", &show_entire_interval);

            // ==== update scrolling times for render time groups =====

            f32 const rolling_mean_weight = s_cast<f32>(render_times_history.mean_sample_count) / s_cast<f32>(render_times_history.mean_sample_count + 1);
            f32 const ewa_weight = 0.95f;
            for (u32 group_i = 0; group_i < RenderTimes::GROUP_COUNT; ++group_i)
            {
                u32 group_first_idx = RenderTimes::group_first_flat_index(group_i);
                u32 group_size = RenderTimes::group_size(group_i);

                f32 raw = 0.0f;
                for (u32 time = group_first_idx; time < (group_first_idx + group_size); ++time)
                {
                    raw += static_cast<u32>(render_context.render_times.get(time)) * 0.001f;
                }
                render_times_history.scrolling_raw.at(group_i).add_point(ImVec2(t, raw));

                f32 const new_rolling_average = render_times_history.scrolling_mean.at(group_i).back().y * rolling_mean_weight + raw * (1.0f - rolling_mean_weight);
                render_times_history.scrolling_mean.at(group_i).add_point(ImVec2(t, new_rolling_average));

                f32 const new_ewa = render_times_history.scrolling_ewa.at(group_i).back().y * ewa_weight + (1.0f - ewa_weight) * raw;
                render_times_history.scrolling_ewa.at(group_i).add_point(ImVec2(t, new_ewa));

                // Variance is in ns^2: std dev in us = sqrt(variance) * 0.001.
                f32 const group_std_dev = static_cast<f32>(std::sqrt(static_cast<f64>(render_context.render_times.get_group_variance(group_i)))) * 0.001f;
                render_times_history.scrolling_std_dev.at(group_i).add_point(ImVec2(t, group_std_dev));
                for (u32 time = group_first_idx; time < (group_first_idx + group_size); ++time)
                {
                    f32 const average = static_cast<f32>(render_context.render_times.get_average(time)) * 0.001f;
                    f32 const std_dev = static_cast<f32>(std::sqrt(static_cast<f64>(render_context.render_times.get_variance(time)))) * 0.001f;
                    render_times_history.pass_average.at(time).add_point(ImVec2(t, average));
                    render_times_history.pass_std_dev.at(time).add_point(ImVec2(t, std_dev));
                }
            }
            render_times_history.mean_sample_count += 1;

            // ========================================================

            static i32 selected_item = 0;
            std::array<char const *, 3> items = {"exp. weight average", "rolling average", "raw values"};
            if (ImGui::BeginCombo("##combo", items.at(selected_item)))
            {
                for (int i = 0; i < 3; i++)
                {
                    bool is_selected = (selected_item == i);
                    if (ImGui::Selectable(items[i], is_selected))
                    {
                        selected_item = i;
                    }
                    if (is_selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            static bool calculate_percentiles = false;
            if (selected_item == 2)
            {
                ImGui::SameLine();
                ImGui::Checkbox("calculate percentiles", &calculate_percentiles);
            }
            else
            {
                calculate_percentiles = false;
            }

            // Groups: selected series with a +-1 std dev band (smoothed variance of the group sum).
            if (ImPlot::BeginPlot("##Scrolling"))
            {
                decltype(render_times_history.scrolling_ewa) * measurements_scrolling_selected = {};
                if (selected_item == 0) { measurements_scrolling_selected = &render_times_history.scrolling_ewa; }
                else if (selected_item == 1) { measurements_scrolling_selected = &render_times_history.scrolling_mean; }
                else { measurements_scrolling_selected = &render_times_history.scrolling_raw; }
                setup_time_axes(measurements_scrolling_selected->at(0).front().x);
                for (i32 i = 0; i < s_cast<i32>(RenderTimes::GROUP_COUNT); i++)
                {
                    plot_with_band(std::string(RenderTimes::group_name(i)).c_str(), i,
                        measurements_scrolling_selected->at(i), render_times_history.scrolling_std_dev.at(i));
                }
                ImPlot::EndPlot();
            }

            for (i32 group_i = 0; group_i < RenderTimes::GROUP_COUNT; group_i++)
            {
                bool open = ImGui::CollapsingHeader(fmt::format("{:<25}", RenderTimes::group_name(group_i)).c_str());
                ImGui::SameLine();
                ImGui::Text("%8.1fus", render_context.render_times.get_group_average(group_i) * 0.001f);
                if (open)
                {
                    if (ImGui::BeginTable("Detail Timings", 4, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Timing", {});
                        ImGui::TableSetupColumn("Average", {});
                        ImGui::TableSetupColumn("Std Dev", {});
                        ImGui::TableSetupColumn("Raw", {});
                        ImGui::TableHeadersRow();
                        for (u32 in_group_i = 0; in_group_i < RenderTimes::group_size(group_i); ++in_group_i)
                        {
                            u32 timing_index = RenderTimes::group_first_flat_index(group_i) + in_group_i;
                            std::string_view name = RenderTimes::group_names(group_i)[in_group_i];

                            bool skip = render_context.render_times.get(timing_index) == 0;
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name.data());
                            ImGui::TableSetColumnIndex(1);
                            if (skip)
                                ImGui::Text("-");
                            else
                                ImGui::Text("%8.1fus", static_cast<f32>(render_context.render_times.get_average(timing_index)) * 0.001f);
                            ImGui::TableSetColumnIndex(2);
                            if (skip)
                                ImGui::Text("-");
                            else
                                ImGui::Text("%8.1fus", std::sqrt(static_cast<f32>(render_context.render_times.get_variance(timing_index))) * 0.001f);
                            ImGui::TableSetColumnIndex(3);
                            if (skip)
                                ImGui::Text("-");
                            else
                                ImGui::Text("%8.1fus", static_cast<f32>(render_context.render_times.get(timing_index)) * 0.001f);
                        }
                        ImGui::EndTable();
                    }
                    // Passes of this group: smoothed average with a +-1 std dev band. Passes that did not run are skipped.
                    if (ImPlot::BeginPlot(fmt::format("##passes{}", group_i).c_str(), ImVec2(-1, 220)))
                    {
                        u32 const first = RenderTimes::group_first_flat_index(group_i);
                        setup_time_axes(render_times_history.pass_average.at(first).front().x);
                        for (u32 in_group_i = 0; in_group_i < RenderTimes::group_size(group_i); ++in_group_i)
                        {
                            u32 const timing_index = first + in_group_i;
                            if (render_context.render_times.get_average(timing_index) == 0) { continue; }
                            std::string const name{RenderTimes::group_names(group_i)[in_group_i]};
                            plot_with_band(name.c_str(), s_cast<i32>(in_group_i),
                                render_times_history.pass_average.at(timing_index), render_times_history.pass_std_dev.at(timing_index));
                        }
                        ImPlot::EndPlot();
                    }
                }
                if (calculate_percentiles)
                {
                    auto sorted_values = render_times_history.scrolling_raw.at(group_i).data;
                    std::sort(sorted_values.begin(), sorted_values.end(),
                        [](auto const & a, auto const & b) -> bool
                        { return a.y < b.y; });
                    ImGui::TextUnformatted(fmt::format("{:<30} {:>10.2f} us", "\t 95th percentile: ", sorted_values.at(s_cast<u64>(sorted_values.size() * 0.95f)).y).c_str());
                    ImGui::TextUnformatted(fmt::format("{:<30} {:>10.2f} us", "\t 99th percentile: ", sorted_values.at(s_cast<u64>(sorted_values.size() * 0.99f)).y).c_str());
                }
            }
        }
        ImGui::SeparatorText("Render System Statistics");
        {
            ui_visbuffer_pipeline_statistics(render_context);
            ui_pgi_statistics(render_context);
            ui_rtgi_statistics(render_context);
            ui_light_statistics(render_context);
        }
        ImGui::SeparatorText("Device Memory Use");
        {
            auto mem_report = render_context.gpu_context->device.device_memory_report_convenient();

            {
                bool open = (ImGui::CollapsingHeader("Total Buffer VRAM Use:        "));
                ImGui::SameLine();
                ImGui::Text(fmt::format("{:>10.2f} mb", s_cast<f32>(mem_report.total_buffer_device_memory_use) / 1024.0f  / 1024.0f).c_str());
                if (open){
                    std::stable_sort(mem_report.buffer_list.begin(), mem_report.buffer_list.end(), [](auto& a, auto& b){ return a.size > b.size; });
                    usize const list_len_max = 30;
                    usize const list_len = std::min(list_len_max, mem_report.buffer_list.size());
                    if (ImGui::BeginTable("Buffer Mem Use", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Name", {});
                        ImGui::TableSetupColumn("Size", {});
                        ImGui::TableSetupColumn("Block Allocated?", {});
                        ImGui::TableHeadersRow();
                        for (auto i = 0; i < list_len; ++i)
                        {
                            auto buf = mem_report.buffer_list[i];
                            char const * name = render_context.gpu_context->device.buffer_info(buf.buffer).value().name.data();
    
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name);
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%fmb", s_cast<f32>(buf.size) / 1024.0f / 1024.0f);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text(buf.block_allocated ? "Yes" : "No");
                        }
                    }
                    ImGui::EndTable();
                }
            }
            {
                bool open = (ImGui::CollapsingHeader("Total Image VRAM Use:         "));
                ImGui::SameLine();
                ImGui::Text(fmt::format("{:>10.2f} mb", s_cast<f32>(mem_report.total_image_device_memory_use) / 1024.0f  / 1024.0f).c_str());
                if (open){
                    std::stable_sort(mem_report.image_list.begin(), mem_report.image_list.end(), [](auto& a, auto& b){ return a.size > b.size; });
                    usize const list_len_max = 30;
                    usize const list_len = std::min(list_len_max, mem_report.image_list.size());
                    if (ImGui::BeginTable("Image Mem Use", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Name", {});
                        ImGui::TableSetupColumn("Size", {});
                        ImGui::TableSetupColumn("Block Allocated?", {});
                        ImGui::TableHeadersRow();
                        for (auto i = 0; i < list_len; ++i)
                        {
                            auto img = mem_report.image_list[i];
                            char const * name = render_context.gpu_context->device.image_info(img.image).value().name.data();
    
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name);
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%fmb", s_cast<f32>(img.size) / 1024.0f / 1024.0f);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text(img.block_allocated ? "Yes" : "No");
                        }
                    }
                    ImGui::EndTable();
                }
            }
            {
                bool open = (ImGui::CollapsingHeader("Total Aliased Tlas VRAM Use:   "));
                ImGui::SameLine();
                ImGui::Text(fmt::format("{:>10.2f} mb", s_cast<f32>(mem_report.total_aliased_tlas_device_memory_use) / 1024.0f  / 1024.0f).c_str());
                if (open){
                    std::stable_sort(mem_report.tlas_list.begin(), mem_report.tlas_list.end(), [](auto& a, auto& b){ return a.size > b.size; });
                    usize const list_len_max = 30;
                    usize const list_len = std::min(list_len_max, mem_report.tlas_list.size());
                    if (ImGui::BeginTable("Tlas Aliased Mem Use", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Name", {});
                        ImGui::TableSetupColumn("Size", {});
                        ImGui::TableSetupColumn("Buffer Allocated?", {});
                        ImGui::TableHeadersRow();
                        for (auto i = 0; i < list_len; ++i)
                        {
                            auto tlas = mem_report.tlas_list[i];
                            char const * name = render_context.gpu_context->device.tlas_info(tlas.tlas).value().name.data();
    
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name);
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%fmb", s_cast<f32>(tlas.size) / 1024.0f / 1024.0f);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("Yes");
                        }
                    }
                    ImGui::EndTable();
                }
            }
            {
                bool open = (ImGui::CollapsingHeader("Total Aliased Blas VRAM Use:   "));
                ImGui::SameLine();
                ImGui::Text(fmt::format("{:>10.2f} mb", s_cast<f32>(mem_report.total_aliased_blas_device_memory_use) / 1024.0f  / 1024.0f).c_str());
                if (open){
                    std::stable_sort(mem_report.blas_list.begin(), mem_report.blas_list.end(), [](auto& a, auto& b){ return a.size > b.size; });
                    usize const list_len_max = 30;
                    usize const list_len = std::min(list_len_max, mem_report.blas_list.size());
                    if (ImGui::BeginTable("Blas Aliased Mem Use", 3, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
                    {
                        ImGui::TableSetupColumn("Name", {});
                        ImGui::TableSetupColumn("Size", {});
                        ImGui::TableSetupColumn("Buffer Allocated?", {});
                        ImGui::TableHeadersRow();
                        for (auto i = 0; i < list_len; ++i)
                        {
                            auto blas = mem_report.blas_list[i];
                            char const * name = render_context.gpu_context->device.blas_info(blas.blas).value().name.data();
    
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex(0);
                            ImGui::Text("%s", name);
                            ImGui::TableSetColumnIndex(1);
                            ImGui::Text("%fmb", s_cast<f32>(blas.size) / 1024.0f / 1024.0f);
                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("Yes");
                        }
                    }
                    ImGui::EndTable();
                }
            }
            ImGui::CollapsingHeader(fmt::format("Total  Block VRAM Use:        {:>10.2f} mb", s_cast<f32>(mem_report.total_memory_block_device_memory_use) / 1024.0f / 1024.0f ).c_str());
            ImGui::CollapsingHeader(fmt::format("Total  VRAM Use:              {:>10.2f} mb", s_cast<f32>(mem_report.total_device_memory_use) / 1024.0f / 1024.0f ).c_str());
        }
    }
    ImGui::End();
}

UIEngine::~UIEngine()
{
    for (u32 icon_idx = 0; icon_idx < s_cast<u32>(ICONS::SIZE); icon_idx++)
    {
        if (!icons.at(icon_idx).is_empty())
        {
            gpu_context->device.destroy_image(icons.at(icon_idx));
        }
    }
    ImPlot::DestroyContext();
}