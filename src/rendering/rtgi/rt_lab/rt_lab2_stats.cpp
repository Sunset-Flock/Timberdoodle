#include "rt_lab2_stats.hpp"

#include "rt_lab2.hpp"
#include "rt_lab2.inl"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>
#include <implot.h>

// == Readback ===========================================================================================================

static void stats_ingest(RenderContext & render_context, RtLab2StatsState & s)
{
    u32 const * words = render_context.gpu_context->device.buffer_host_address_as<u32>(s.buffer.id()).value();
    u32 const capacity = rt_lab2_cell_capacity(render_context.rt_lab.lab2.settings);
    std::vector<std::pair<u32, u32>> ready = {}; // (issued frame, slot)
    for (u32 slot = 0; slot < RT_LAB2_STATS_SLOTS; ++slot)
    {
        if (s.pending[slot] && s.frame_counter >= s.issued_frame[slot] + MAX_GPU_FRAMES_IN_FLIGHT + 2)
        {
            ready.emplace_back(s.issued_frame[slot], slot);
        }
    }
    std::sort(ready.begin(), ready.end());
    for (auto const & [frame, slot] : ready)
    {
        s.pending[slot] = false;
        if (s.pause)
        {
            continue;
        }
        u32 const * w = words + slot * RT_LAB2_STATS_SLOT_WORDS;
        RtLab2Stats stats = {};
        stats.valid = true;
        stats.live = w[0];
        stats.touched = w[1];
        stats.with_data = w[2];
        stats.stale = w[3];
        stats.pixels = w[4];
        stats.rays = w[5];
        stats.cells_with_rays = w[6];
        f64 fill_sum = 0.0;
        f64 fill_cells = 0.0;
        for (u32 b = 0; b < RT_LAB2_STATS_UI_BUCKETS; ++b)
        {
            stats.level_live[b] = w[RT_LAB2_STATS_LEVEL_LIVE + b];
            stats.level_touched[b] = w[RT_LAB2_STATS_LEVEL_TOUCHED + b];
            stats.fill[b] = w[RT_LAB2_STATS_FILL + b];
            stats.coverage[b] = w[RT_LAB2_STATS_COVERAGE + b];
            stats.rays_hist[b] = w[RT_LAB2_STATS_RAYS + b];
            stats.untouched[b] = w[RT_LAB2_STATS_UNTOUCHED + b];
            fill_sum += static_cast<f64>(stats.fill[b]) * (static_cast<f64>(b) + 0.5) / static_cast<f64>(RT_LAB2_STATS_UI_BUCKETS);
            fill_cells += static_cast<f64>(stats.fill[b]);
        }
        stats.mean_fill = fill_cells > 0.0 ? static_cast<f32>(fill_sum / fill_cells) : 0.0f;
        s.latest = stats;
        RtLab2StatsPoint point = {};
        point.live = static_cast<f32>(stats.live);
        point.touched = static_cast<f32>(stats.touched);
        point.with_data = static_cast<f32>(stats.with_data);
        point.pixels = static_cast<f32>(stats.pixels);
        point.rays = static_cast<f32>(stats.rays);
        point.mean_fill = stats.mean_fill;
        point.rays_per_pixel = stats.pixels > 0 ? static_cast<f32>(stats.rays) / static_cast<f32>(stats.pixels) : 0.0f;
        point.load = capacity > 0 ? static_cast<f32>(stats.live) / static_cast<f32>(capacity) : 0.0f;
        s.history.push_back(point);
        while (s.history.size() > RT_LAB2_STATS_HISTORY_FRAMES)
        {
            s.history.pop_front();
        }
    }
}

auto rt_lab2_stats_frame_update(RenderContext & render_context) -> bool
{
    auto & s = render_context.rt_lab.lab2.stats;
    auto & device = render_context.gpu_context->device;
    ++s.frame_counter;
    s.dispatch_pending = false;
    // Test hook (no UI needed): TIDO_LAB2_STATS=1 collects the statistics and prints them every 50 frames.
    static bool const hook = std::getenv("TIDO_LAB2_STATS") != nullptr;
    if (hook)
    {
        s.enabled = true;
        if (s.latest.valid && s.frame_counter % 50 == 0)
        {
            RtLab2Stats const & st = s.latest;
            fmt::print("[Lab2Stats] capacity {} live {} queried {} with data {} stale {} | pixels {} rays {} cells with rays {} | mean fill {:.3f} | levels live {} {} {} {} {} {} {}\n",
                rt_lab2_cell_capacity(render_context.rt_lab.lab2.settings), st.live, st.touched, st.with_data, st.stale, st.pixels, st.rays, st.cells_with_rays, st.mean_fill,
                st.level_live[0], st.level_live[1], st.level_live[2], st.level_live[3], st.level_live[4], st.level_live[5], st.level_live[6]);
        }
    }
    bool rebuild = s.enabled != s.built;
    s.built = s.enabled;
    if (!s.enabled)
    {
        return rebuild;
    }
    if (s.buffer.id().is_empty())
    {
        u64 const size = sizeof(u32) * RT_LAB2_STATS_SLOT_WORDS * RT_LAB2_STATS_SLOTS;
        s.buffer.set_buffer(device.create_buffer({
            .size = size,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "rt_lab2_stats",
        }));
        std::memset(device.buffer_host_address(s.buffer.id()).value(), 0, size);
        s.pending.fill(false);
        s.issued_frame.fill(0);
        rebuild = true;
    }
    stats_ingest(render_context, s);
    // Issue this frame's slot: zeroed here (the GPU finished with it long ago: the ring is longer than the frames in flight).
    u32 const slot = s.next_slot++ % RT_LAB2_STATS_SLOTS;
    u32 * words = device.buffer_host_address_as<u32>(s.buffer.id()).value();
    std::memset(words + slot * RT_LAB2_STATS_SLOT_WORDS, 0, sizeof(u32) * RT_LAB2_STATS_SLOT_WORDS);
    s.issued_frame[slot] = s.frame_counter;
    s.pending[slot] = true;
    s.dispatch_slot = slot;
    s.dispatch_pending = true;
    return rebuild;
}

void rt_lab2_stats_destroy(RenderContext & render_context)
{
    auto & s = render_context.rt_lab.lab2.stats;
    if (!s.buffer.id().is_empty())
    {
        render_context.gpu_context->device.destroy_buffer(s.buffer.id());
        s.buffer.set_buffer(daxa::BufferId{});
    }
}

void rt_lab2_stats_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto & s = render_context->rt_lab.lab2.stats;
    if (!s.dispatch_pending)
    {
        return;
    }
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RT_LAB", "LAB2_STATS">());
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab2_stats_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLab2Settings settings = render_context->rt_lab.lab2.settings;
    auto settings_alloc = ti.allocator->allocate(sizeof(RtLab2Settings)).value();
    std::memcpy(settings_alloc.host_address, &settings, sizeof(RtLab2Settings));
    RtLab2StatsPush push = {};
    push.attach = alloc.device_address;
    push.settings = settings_alloc.device_address;
    push.slot = s.dispatch_slot;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(rt_lab2_cell_capacity(settings), static_cast<u32>(RT_LAB2_CELL_PASS_X)), 1, 1});
}

// == UI =================================================================================================================

static auto stats_to_floats(std::array<u32, RT_LAB2_STATS_UI_BUCKETS> const & values) -> std::array<f32, RT_LAB2_STATS_UI_BUCKETS>
{
    std::array<f32, RT_LAB2_STATS_UI_BUCKETS> result = {};
    for (u32 i = 0; i < RT_LAB2_STATS_UI_BUCKETS; ++i)
    {
        result[i] = static_cast<f32>(values[i]);
    }
    return result;
}

// A bar plot with 16 buckets. `labels` = the x tick labels (16), `b` = optional second series (grouped bars).
static void stats_histogram(char const * title, char const * x_label, std::array<std::string, RT_LAB2_STATS_UI_BUCKETS> const & labels, char const * a_name,
    std::array<u32, RT_LAB2_STATS_UI_BUCKETS> const & a, char const * b_name = nullptr, std::array<u32, RT_LAB2_STATS_UI_BUCKETS> const * b = nullptr, f32 height = 150.0f)
{
    if (!ImPlot::BeginPlot(title, ImVec2(-1.0f, height)))
    {
        return;
    }
    std::array<f64, RT_LAB2_STATS_UI_BUCKETS> positions = {};
    std::array<char const *, RT_LAB2_STATS_UI_BUCKETS> label_pointers = {};
    for (u32 i = 0; i < RT_LAB2_STATS_UI_BUCKETS; ++i)
    {
        positions[i] = static_cast<f64>(i);
        label_pointers[i] = labels[i].c_str();
    }
    ImPlot::SetupAxes(x_label, "cells", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
    ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), static_cast<i32>(RT_LAB2_STATS_UI_BUCKETS), label_pointers.data());
    ImPlot::SetupAxisLimits(ImAxis_X1, -0.7, RT_LAB2_STATS_UI_BUCKETS - 0.3, ImPlotCond_Always);
    auto const a_values = stats_to_floats(a);
    if (b != nullptr)
    {
        auto const b_values = stats_to_floats(*b);
        ImPlot::PlotBars(a_name, a_values.data(), static_cast<i32>(RT_LAB2_STATS_UI_BUCKETS), 0.38, -0.2);
        ImPlot::PlotBars(b_name, b_values.data(), static_cast<i32>(RT_LAB2_STATS_UI_BUCKETS), 0.38, 0.2);
    }
    else
    {
        ImPlot::PlotBars(a_name, a_values.data(), static_cast<i32>(RT_LAB2_STATS_UI_BUCKETS), 0.7);
    }
    ImPlot::EndPlot();
}

static auto stats_pow2_labels(char const * prefix) -> std::array<std::string, RT_LAB2_STATS_UI_BUCKETS>
{
    std::array<std::string, RT_LAB2_STATS_UI_BUCKETS> labels = {};
    for (u32 i = 0; i < RT_LAB2_STATS_UI_BUCKETS; ++i)
    {
        labels[i] = fmt::format("{}{}", prefix, i);
    }
    return labels;
}

void rt_lab2_stats_ui(RenderContext & render_context)
{
    auto & lab2 = render_context.rt_lab.lab2;
    auto & s = lab2.stats;
    ImGui::Checkbox("Collect Statistics", &s.enabled);
    ImGui::SetItemTooltip("Adds the statistics pass to the RT Lab 2 graph (a rebuild): it scans every cell slot after the cell integrate pass.\n"
                          "The numbers and graphs below describe the frame that just ended.");
    if (!s.enabled)
    {
        return;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Pause##stats", &s.pause);
    if (!lab2.active)
    {
        ImGui::TextDisabled("RT Lab 2 is not the active denoiser: the statistics show nothing new.");
    }
    RtLab2Stats const & st = s.latest;
    if (!st.valid)
    {
        ImGui::TextDisabled("Waiting for the first sample...");
        return;
    }
    u32 const capacity = rt_lab2_cell_capacity(lab2.settings);
    f64 const cell_bytes = static_cast<f64>(sizeof(RtLab2Cell));
    f32 const load = static_cast<f32>(st.live) / static_cast<f32>(capacity);

    ImGui::SeparatorText("Grid");
    ImGui::Text("Capacity %u cells (2^%d), %.1f MB (%.0f B per cell)", capacity, static_cast<i32>(std::round(std::log2(static_cast<f64>(capacity)))), static_cast<f64>(capacity) * cell_bytes / (1024.0 * 1024.0), cell_bytes);
    ImGui::Text("Live cells %u   load %.2f %%   (%.1f MB in use)", st.live, load * 100.0f, static_cast<f64>(st.live) * cell_bytes / (1024.0 * 1024.0));
    if (load > 0.5f)
    {
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.9f, 0.3f, 0.2f, 1.0f));
    }
    ImGui::ProgressBar(std::clamp(load, 0.0f, 1.0f), ImVec2(-1.0f, 0.0f));
    if (load > 0.5f)
    {
        ImGui::PopStyleColor();
        ImGui::TextColored(ImVec4(0.95f, 0.5f, 0.3f, 1.0f), "The grid is more than half full: probe windows overflow and queries fail. Raise Cell Capacity.");
    }
    ImGui::Text("Queried last frame %u (%.1f %%)   not queried %u   with data %u", st.touched, st.live > 0 ? 100.0f * static_cast<f32>(st.touched) / static_cast<f32>(st.live) : 0.0f, st.stale, st.with_data);

    ImGui::SeparatorText("Coverage + Rays (last frame)");
    f32 const rays_per_pixel = st.pixels > 0 ? static_cast<f32>(st.rays) / static_cast<f32>(st.pixels) : 0.0f;
    ImGui::Text("Pixel coverage (sum over cells) %llu   rays %llu   rays per covered pixel %.3f", static_cast<unsigned long long>(st.pixels), static_cast<unsigned long long>(st.rays), rays_per_pixel);
    ImGui::Text("Mean coverage per queried cell %.1f pixels   mean rays per queried cell %.1f   cells that got rays %u", st.touched > 0 ? static_cast<f32>(st.pixels) / static_cast<f32>(st.touched) : 0.0f,
        st.touched > 0 ? static_cast<f32>(st.rays) / static_cast<f32>(st.touched) : 0.0f, st.cells_with_rays);
    ImGui::SetItemTooltip("Pixels inside a cascade blend region feed two cells, so the summed coverage can exceed the screen's pixel count.");
    ImGui::Text("Mean history fill %.1f %% of the window", st.mean_fill * 100.0f);

    ImGui::SeparatorText("Pass Timings");
    {
        struct Row
        {
            char const * name;
            u64 nanoseconds;
        };
        std::vector<Row> rows = {
            {"Ray Demand (old)", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "RAY_DEMAND">())},
            {"Cell Query", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB2_QUERY">())},
            {"Cell Prepare", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB2_CELL_PREPARE">())},
            {"Cell Demand / Rays Debug", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB2_DEMAND">())},
            {"Ray Distribute (old)", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "RAY_DISTRIBUTE">())},
            {"Trace", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "TRACE">())},
            {"Ray Filter", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "RAY_FILTER">())},
            {"Ray Gather (old)", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "RAY_GATHER">())},
            {"Cell Accumulate", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB2_ACCUMULATE">())},
            {"Cell Integrate", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB2_CELL_INTEGRATE">())},
            {"Lookup", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB2_LOOKUP">())},
            {"Integrator", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB2_INTEGRATOR">())},
            {"Resolve (old)", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "RESOLVE">())},
            {"Statistics", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB2_STATS">())},
            {"Cell Reset", render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB2_RESET">())},
        };
        u64 total = 0;
        u64 maximum = 1;
        for (Row const & row : rows)
        {
            total += row.nanoseconds;
            maximum = std::max(maximum, row.nanoseconds);
        }
        if (ImGui::BeginTable("##lab2_timings", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("Pass", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("us", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 3.0f);
            for (Row const & row : rows)
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(row.name);
                ImGui::TableNextColumn();
                ImGui::Text("%.1f", static_cast<f32>(row.nanoseconds) * 0.001f);
                ImGui::TableNextColumn();
                ImGui::ProgressBar(static_cast<f32>(row.nanoseconds) / static_cast<f32>(maximum), ImVec2(-1.0f, 0.0f), "");
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted("Total");
            ImGui::TableNextColumn();
            ImGui::Text("%.1f", static_cast<f32>(total) * 0.001f);
            ImGui::EndTable();
        }
    }

    // Time series.
    if (s.history.size() > 1)
    {
        u32 const n = static_cast<u32>(s.history.size());
        std::vector<f32> x(n), live(n), touched(n), with_data(n), pixels(n), rays(n), fill(n), rays_per_pixel_curve(n), load_curve(n);
        for (u32 i = 0; i < n; ++i)
        {
            RtLab2StatsPoint const & p = s.history[i];
            x[i] = static_cast<f32>(i) - static_cast<f32>(n - 1);
            live[i] = p.live;
            touched[i] = p.touched;
            with_data[i] = p.with_data;
            pixels[i] = p.pixels;
            rays[i] = p.rays;
            fill[i] = p.mean_fill * 100.0f;
            rays_per_pixel_curve[i] = p.rays_per_pixel;
            load_curve[i] = p.load * 100.0f;
        }
        ImGui::SeparatorText("Over Time");
        if (ImPlot::BeginPlot("Cells##lab2_stats_cells", ImVec2(-1.0f, 150.0f)))
        {
            ImPlot::SetupAxes("frames ago", "cells", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("live", x.data(), live.data(), static_cast<i32>(n));
            ImPlot::PlotLine("queried", x.data(), touched.data(), static_cast<i32>(n));
            ImPlot::PlotLine("with data", x.data(), with_data.data(), static_cast<i32>(n));
            ImPlot::EndPlot();
        }
        if (ImPlot::BeginPlot("Coverage + Rays##lab2_stats_rays", ImVec2(-1.0f, 150.0f)))
        {
            ImPlot::SetupAxes("frames ago", "pixels / rays", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("pixel coverage", x.data(), pixels.data(), static_cast<i32>(n));
            ImPlot::PlotLine("rays", x.data(), rays.data(), static_cast<i32>(n));
            ImPlot::EndPlot();
        }
        if (ImPlot::BeginPlot("History Fill / Load / Rays Per Pixel##lab2_stats_fill", ImVec2(-1.0f, 150.0f)))
        {
            ImPlot::SetupAxes("frames ago", "% / rays per pixel x 100", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
            for (u32 i = 0; i < n; ++i)
            {
                rays_per_pixel_curve[i] *= 100.0f;
            }
            ImPlot::PlotLine("mean history fill %", x.data(), fill.data(), static_cast<i32>(n));
            ImPlot::PlotLine("grid load %", x.data(), load_curve.data(), static_cast<i32>(n));
            ImPlot::PlotLine("rays per pixel x 100", x.data(), rays_per_pixel_curve.data(), static_cast<i32>(n));
            ImPlot::EndPlot();
        }
    }

    // Distributions.
    ImGui::SeparatorText("Distributions");
    {
        std::array<std::string, RT_LAB2_STATS_UI_BUCKETS> level_labels = stats_pow2_labels("");
        stats_histogram("Cells Per Cascade Level##lab2_stats_levels", "level", level_labels, "live", st.level_live, "queried", &st.level_touched);
        std::array<std::string, RT_LAB2_STATS_UI_BUCKETS> fill_labels = {};
        for (u32 i = 0; i < RT_LAB2_STATS_UI_BUCKETS; ++i)
        {
            fill_labels[i] = fmt::format("{}", (i + 1) * 100 / RT_LAB2_STATS_UI_BUCKETS);
        }
        stats_histogram("History Fill (queried cells)##lab2_stats_fill_hist", "samples / window, % (upper edge)", fill_labels, "cells", st.fill);
        ImGui::SetItemTooltip("How full the history window of every queried cell with data is. Cells piling up at the left are new or were reset (disocclusion, importance\n"
                              "scaling), at the right they are converged to their window.");
        stats_histogram("Pixel Coverage (queried cells)##lab2_stats_coverage", "log2(1 + pixels)", stats_pow2_labels(""), "cells", st.coverage);
        stats_histogram("Rays Last Frame (queried cells)##lab2_stats_rays_hist", "log2(1 + rays)", stats_pow2_labels(""), "cells", st.rays_hist);
        stats_histogram("Frames Not Queried (live cells)##lab2_stats_untouched", "log2(1 + frames)", stats_pow2_labels(""), "cells", st.untouched);
        ImGui::SetItemTooltip("Cells of geometry that left the view age until Free After frames, then their slot is freed.");
    }
}
