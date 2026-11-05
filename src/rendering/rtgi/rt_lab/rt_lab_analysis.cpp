#include "rt_lab_analysis.hpp"

#include "rt_lab.hpp"
#include "rt_lab_analysis.inl"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numbers>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>
#include <implot.h>

static constexpr f64 ANALYSIS_LOG_FLOOR = -12.0; // the range never goes below / above this (log10)
static constexpr f64 ANALYSIS_LOG_CEILING = 8.0;
static constexpr f64 ANALYSIS_MARGIN = 0.3;      // decades added around the percentile targets
static constexpr f64 ANALYSIS_TAIL_FRACTION = 0.001;
static constexpr u32 PICKER_HISTOGRAM_BINS = 64;

static constexpr ImVec4 PICKER_COLORS[RT_LAB_PICKER_UI_COUNT] = {ImVec4(0.30f, 0.65f, 1.00f, 1.0f), ImVec4(1.00f, 0.55f, 0.20f, 1.0f)};
static constexpr char const * PICKER_NAMES[RT_LAB_PICKER_UI_COUNT] = {"A", "B"};

u64 RtLabStatsHistogram::total() const
{
    u64 sum = under + over;
    for (u32 count : bins)
    {
        sum += count;
    }
    return sum;
}

// == Range smoothing ====================================================================================================

// Edges move quickly when the data leaves the range (nothing gets clipped for long), and slowly when the data shrinks, and only when it is
// well inside (hysteresis): the range does not jitter with the frame to frame noise of the percentiles.
static void smooth_range_edge(f64 & edge, f64 target, bool is_low)
{
    bool const widening = is_low ? target < edge : target > edge;
    if (widening)
    {
        edge += 0.7 * (target - edge);
    }
    else if (std::abs(target - edge) > 0.15)
    {
        edge += 0.25 * (target - edge);
    }
}

static void smooth_range(RtLabSmoothRange & range, f64 target_low, f64 target_high)
{
    smooth_range_edge(range.low, target_low, true);
    smooth_range_edge(range.high, target_high, false);
    range.low = std::clamp(range.low, ANALYSIS_LOG_FLOOR, ANALYSIS_LOG_CEILING - 1.0);
    range.high = std::clamp(range.high, ANALYSIS_LOG_FLOOR + 1.0, ANALYSIS_LOG_CEILING);
    if (range.high - range.low < 1.0)
    {
        f64 const center = 0.5 * (range.low + range.high);
        range.low = center - 0.5;
        range.high = center + 0.5;
    }
}

// Percentile targets of a GPU histogram: a tail in the under / over bucket widens by a decade per update.
static auto histogram_targets(RtLabStatsHistogram const & h, f64 & low, f64 & high) -> bool
{
    u64 const total = h.total();
    if (total == 0)
    {
        return false;
    }
    f64 const need = ANALYSIS_TAIL_FRACTION * static_cast<f64>(total);
    f64 const width = (h.high - h.low) / static_cast<f64>(h.bins.size());
    low = h.low;
    high = h.high;
    u64 cumulative = h.under;
    if (static_cast<f64>(cumulative) > need)
    {
        low = h.low > ANALYSIS_LOG_FLOOR + 0.5 ? h.low - 1.0 : h.low;
    }
    else
    {
        for (u32 i = 0; i < h.bins.size(); ++i)
        {
            cumulative += h.bins[i];
            if (static_cast<f64>(cumulative) > need)
            {
                low = h.low + static_cast<f64>(i) * width;
                break;
            }
        }
    }
    cumulative = h.over;
    if (static_cast<f64>(cumulative) > need)
    {
        high = h.high < ANALYSIS_LOG_CEILING - 0.5 ? h.high + 1.0 : h.high;
    }
    else
    {
        for (u32 i = static_cast<u32>(h.bins.size()); i-- > 0;)
        {
            cumulative += h.bins[i];
            if (static_cast<f64>(cumulative) > need)
            {
                high = h.low + static_cast<f64>(i + 1) * width;
                break;
            }
        }
    }
    low -= ANALYSIS_MARGIN;
    high += ANALYSIS_MARGIN;
    return true;
}

static void smooth_peak(f64 & peak, f64 value)
{
    peak += (value > peak ? 0.7 : 0.25) * (value - peak);
}

// log10 value below which the fraction p of the counted values lies (NaN = empty).
static auto histogram_percentile(RtLabStatsHistogram const & h, f64 p) -> f64
{
    u64 const total = h.total();
    if (total == 0)
    {
        return std::numeric_limits<f64>::quiet_NaN();
    }
    f64 const need = p * static_cast<f64>(total);
    f64 const width = (h.high - h.low) / static_cast<f64>(h.bins.size());
    u64 cumulative = h.under;
    if (static_cast<f64>(cumulative) >= need)
    {
        return h.low;
    }
    for (u32 i = 0; i < h.bins.size(); ++i)
    {
        cumulative += h.bins[i];
        if (static_cast<f64>(cumulative) >= need)
        {
            return h.low + (static_cast<f64>(i) + 0.5) * width;
        }
    }
    return h.high;
}

// == Frame update / readback ============================================================================================

static void stats_ingest(RenderContext & render_context, RtLabAnalysisState & s)
{
    u32 * words = render_context.gpu_context->device.buffer_host_address_as<u32>(s.stats_buffer.id()).value();
    std::vector<std::pair<u32, u32>> ready = {}; // (stamp, slot)
    for (u32 slot = 0; slot < RT_LAB_STATS_SLOTS; ++slot)
    {
        u32 const stamp = words[slot * RT_LAB_STATS_SLOT_WORDS];
        bool const finished = s.frame_counter >= s.stats_issued_frame[slot] + MAX_GPU_FRAMES_IN_FLIGHT + 2;
        if (stamp != 0 && stamp == s.stats_issued_stamp[slot] && finished && stamp > s.stats_last_ingested)
        {
            ready.emplace_back(stamp, slot);
        }
    }
    std::sort(ready.begin(), ready.end());
    for (auto const & [stamp, slot] : ready)
    {
        s.stats_last_ingested = stamp;
        u32 const * slot_words = words + slot * RT_LAB_STATS_SLOT_WORDS + RT_LAB_STATS_HEADER_WORDS;
        auto read = [&](u32 const * hist_words, RtLabStatsHistogram & hist, f64 low, f64 high)
        {
            hist.valid = true;
            hist.low = low;
            hist.high = high;
            for (u32 i = 0; i < RT_LAB_STATS_BINS; ++i)
            {
                hist.bins[i] = hist_words[i];
            }
            hist.under = hist_words[RT_LAB_STATS_BINS];
            hist.over = hist_words[RT_LAB_STATS_BINS + 1];
        };
        auto const & range = s.stats_issued_range[slot];
        read(slot_words, s.luma_histogram, range[0], range[1]);
        read(slot_words + RT_LAB_STATS_HIST_WORDS, s.variance_histogram, range[2], range[3]);

        // Exact moments from the per group partials.
        {
            u32 const * partials = slot_words + 2 * RT_LAB_STATS_HIST_WORDS;
            u32 const groups = std::min(s.stats_issued_groups[slot][0] * s.stats_issued_groups[slot][1], static_cast<u32>(RT_LAB_STATS_MAX_GROUPS));
            auto summarize = [&](u32 metric, RtLabStatsSummary & out)
            {
                f64 count = 0.0, sum = 0.0, sum_squares = 0.0, positive = 0.0, log_sum = 0.0, log_squares = 0.0;
                f64 min = std::numeric_limits<f64>::max(), max = 0.0;
                for (u32 g = 0; g < groups; ++g)
                {
                    u32 const * w = partials + g * RT_LAB_STATS_PARTIAL_WORDS + metric * RT_LAB_STATS_METRIC_WORDS;
                    auto f = [&](u32 i) -> f64
                    {
                        f32 v = 0.0f;
                        std::memcpy(&v, &w[i], sizeof(f32));
                        return static_cast<f64>(v);
                    };
                    if (f(0) <= 0.0)
                    {
                        continue;
                    }
                    count += f(0);
                    sum += f(1);
                    sum_squares += f(2);
                    min = std::min(min, f(3));
                    max = std::max(max, f(4));
                    positive += f(5);
                    log_sum += f(6);
                    log_squares += f(7);
                }
                out = {};
                if (count < 1.0)
                {
                    return;
                }
                out.valid = true;
                out.count = static_cast<u64>(count);
                out.mean = sum / count;
                out.std_dev = std::sqrt(std::max(sum_squares / count - out.mean * out.mean, 0.0));
                out.min = min;
                out.max = max;
                if (positive >= 1.0)
                {
                    f64 const log_mean = log_sum / positive;
                    out.positive_count = static_cast<u64>(positive);
                    out.geometric_mean = std::pow(10.0, log_mean);
                    out.log10_std = std::sqrt(std::max(log_squares / positive - log_mean * log_mean, 0.0));
                }
            };
            summarize(0, s.luma_summary);
            summarize(1, s.variance_summary);
        }

        f64 low = 0.0, high = 0.0;
        if (histogram_targets(s.luma_histogram, low, high))
        {
            smooth_range(s.luma_range, low, high);
        }
        if (histogram_targets(s.variance_histogram, low, high))
        {
            smooth_range(s.variance_range, low, high);
        }
        auto peak_share = [](RtLabStatsHistogram const & hist) -> f64
        {
            u64 const total = hist.total();
            u32 peak = 0;
            for (u32 count : hist.bins)
            {
                peak = std::max(peak, count);
            }
            return total > 0 ? static_cast<f64>(peak) / static_cast<f64>(total) : 0.0;
        };
        smooth_peak(s.luma_peak, peak_share(s.luma_histogram));
        smooth_peak(s.variance_peak, peak_share(s.variance_histogram));
    }
}

static void picker_ingest(RenderContext & render_context, RtLabAnalysisState & s)
{
    u32 const * words = render_context.gpu_context->device.buffer_host_address_as<u32>(s.picker_buffer.id()).value();
    std::vector<std::pair<u32, u32>> ready = {};
    for (u32 slot = 0; slot < RT_LAB_PICKER_SLOTS; ++slot)
    {
        u32 const stamp = words[slot * RT_LAB_PICKER_SLOT_WORDS];
        bool const finished = s.frame_counter >= s.picker_issued_frame[slot] + MAX_GPU_FRAMES_IN_FLIGHT + 2;
        if (stamp != 0 && stamp == s.picker_issued_stamp[slot] && finished && stamp > s.picker_last_ingested)
        {
            ready.emplace_back(stamp, slot);
        }
    }
    std::sort(ready.begin(), ready.end());
    for (auto const & [stamp, slot] : ready)
    {
        s.picker_last_ingested = stamp;
        u32 const * slot_words = words + slot * RT_LAB_PICKER_SLOT_WORDS;
        for (u32 index = 0; index < RT_LAB_PICKER_UI_COUNT; ++index)
        {
            RtLabPicker & picker = s.pickers[index];
            if (!picker.active || picker.pause || ((slot_words[1] >> index) & 1u) == 0)
            {
                continue;
            }
            u32 const * w = slot_words + RT_LAB_PICKER_HEADER_WORDS + index * RT_LAB_PICKER_WORDS;
            if (static_cast<i32>(w[1]) != picker.pixel[0] || static_cast<i32>(w[2]) != picker.pixel[1])
            {
                continue; // another pixel (re-picked while this slot was in flight)
            }
            auto f = [&](u32 i) -> f32
            {
                f32 v = 0.0f;
                std::memcpy(&v, &w[i], sizeof(f32));
                return v;
            };
            RtLabPickerSample sample = {};
            sample.geometry = (w[0] & 1u) != 0;
            for (u32 c = 0; c < 3; ++c)
            {
                sample.rgb[c] = f(3 + c);
            }
            sample.window_mean = f(6);
            sample.window_variance = f(7);
            sample.window_count = f(8);
            sample.exposure = f(9);
            sample.window_min = f(10);
            sample.window_max = f(11);
            sample.luma = sample.geometry ? 0.2126f * sample.rgb[0] + 0.7152f * sample.rgb[1] + 0.0722f * sample.rgb[2] : 0.0f;
            picker.samples.push_back(sample);
            ++picker.frames;
            while (picker.samples.size() > RT_LAB_PICKER_HISTORY_FRAMES)
            {
                picker.samples.pop_front();
            }
        }
    }
}

auto rt_lab_analysis_frame_update(RenderContext & render_context) -> bool
{
    auto & s = render_context.rt_lab.analysis;
    auto & device = render_context.gpu_context->device;
    ++s.frame_counter;
    s.stats_dispatch_pending = false;
    s.picker_dispatch_pending = false;

    bool const any_picker = std::any_of(s.pickers.begin(), s.pickers.end(), [](RtLabPicker const & p) { return p.active; });
    bool rebuild = s.stats_enabled != s.stats_built || any_picker != s.picker_built;

    if (s.stats_enabled)
    {
        if (s.stats_buffer.id().is_empty())
        {
            u64 const size = sizeof(u32) * RT_LAB_STATS_SLOT_WORDS * RT_LAB_STATS_SLOTS;
            s.stats_buffer.set_buffer(device.create_buffer({.size = size, .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, .name = "rt_lab_stats"}));
            std::memset(device.buffer_host_address(s.stats_buffer.id()).value(), 0, size);
            s.stats_issued_stamp.fill(0);
            s.stats_issued_frame.fill(0);
            rebuild = true;
        }
        stats_ingest(render_context, s);
        u32 const stamp = s.next_stamp++;
        u32 const slot = stamp % RT_LAB_STATS_SLOTS;
        // The slot holds a finished frame's counts (ingested long ago): clear it before the GPU adds to it.
        u32 * words = device.buffer_host_address_as<u32>(s.stats_buffer.id()).value() + slot * RT_LAB_STATS_SLOT_WORDS;
        std::memset(words, 0, sizeof(u32) * RT_LAB_STATS_PARTIALS_OFFSET); // header + histograms (every group rewrites its partial)
        s.stats_issued_stamp[slot] = stamp;
        s.stats_issued_frame[slot] = s.frame_counter;
        s.stats_issued_range[slot] = {static_cast<f32>(s.luma_range.low), static_cast<f32>(s.luma_range.high), static_cast<f32>(s.variance_range.low),
            static_cast<f32>(s.variance_range.high)};
        auto const target = render_context.render_data.settings.render_target_size;
        s.stats_issued_groups[slot] = {round_up_div(target.x, static_cast<u32>(RT_LAB_STATS_GROUP)), round_up_div(target.y, static_cast<u32>(RT_LAB_STATS_GROUP))};
        s.stats_dispatch_groups[0] = s.stats_issued_groups[slot][0];
        s.stats_dispatch_groups[1] = s.stats_issued_groups[slot][1];
        s.stats_dispatch_stamp = stamp;
        std::copy(s.stats_issued_range[slot].begin(), s.stats_issued_range[slot].end(), s.stats_dispatch_range);
        s.stats_dispatch_pending = true;
    }

    if (any_picker)
    {
        if (s.picker_buffer.id().is_empty())
        {
            u64 const size = sizeof(u32) * RT_LAB_PICKER_SLOT_WORDS * RT_LAB_PICKER_SLOTS;
            s.picker_buffer.set_buffer(device.create_buffer({.size = size, .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM, .name = "rt_lab_picker"}));
            std::memset(device.buffer_host_address(s.picker_buffer.id()).value(), 0, size);
            s.picker_issued_stamp.fill(0);
            s.picker_issued_frame.fill(0);
            rebuild = true;
        }
        picker_ingest(render_context, s);
        u32 mask = 0;
        for (u32 index = 0; index < RT_LAB_PICKER_UI_COUNT; ++index)
        {
            RtLabPicker const & picker = s.pickers[index];
            mask |= picker.active ? (1u << index) : 0u;
            s.picker_dispatch_pixel[index][0] = static_cast<u32>(std::max(picker.pixel[0], 0));
            s.picker_dispatch_pixel[index][1] = static_cast<u32>(std::max(picker.pixel[1], 0));
        }
        u32 const stamp = s.next_stamp++;
        u32 const slot = stamp % RT_LAB_PICKER_SLOTS;
        s.picker_issued_stamp[slot] = stamp;
        s.picker_issued_frame[slot] = s.frame_counter;
        s.picker_dispatch_stamp = stamp;
        s.picker_dispatch_mask = mask;
        s.picker_dispatch_pending = true;
    }
    return rebuild;
}

void rt_lab_analysis_destroy(RenderContext & render_context)
{
    auto & s = render_context.rt_lab.analysis;
    for (auto * buffer : {&s.stats_buffer, &s.picker_buffer})
    {
        if (!buffer->id().is_empty())
        {
            render_context.gpu_context->device.destroy_buffer(buffer->id());
            buffer->set_buffer(daxa::BufferId{});
        }
    }
}

// == Task graph =========================================================================================================

void rt_lab_add_analysis(TasksRtgiInfo const & info, daxa::TaskImageView resolved)
{
    auto & render_context = info.render_context;
    auto & s = render_context.rt_lab.analysis;
    if (s.stats_enabled && !s.stats_buffer.id().is_empty())
    {
        info.tg.register_buffer(s.stats_buffer);
        info.tg.add_task(daxa::HeadTask<RtLabStatsH::Info>()
            .head_views(RtLabStatsH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .view_cam_depth = info.view_cam_depth,
                .resolved = resolved,
                .stats = s.stats_buffer.view(),
            })
            .executes(rt_lab_stats_callback, &render_context));
    }
    if (!s.picker_buffer.id().is_empty() && std::any_of(s.pickers.begin(), s.pickers.end(), [](RtLabPicker const & p) { return p.active; }))
    {
        info.tg.register_buffer(s.picker_buffer);
        info.tg.add_task(daxa::HeadTask<RtLabPickerH::Info>()
            .head_views(RtLabPickerH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .view_cam_depth = info.view_cam_depth,
                .resolved = resolved,
                .picker = s.picker_buffer.view(),
            })
            .executes(rt_lab_picker_callback, &render_context));
    }
}

void rt_lab_stats_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto & s = render_context->rt_lab.analysis;
    if (!s.stats_dispatch_pending)
    {
        return;
    }
    RtLabStatsPush push = {};
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    push.attach = alloc.device_address;
    push.luma_log_min = s.stats_dispatch_range[0];
    push.luma_log_max = s.stats_dispatch_range[1];
    push.variance_log_min = s.stats_dispatch_range[2];
    push.variance_log_max = s.stats_dispatch_range[3];
    push.slot = s.stats_dispatch_stamp % RT_LAB_STATS_SLOTS;
    push.stamp = s.stats_dispatch_stamp;
    push.groups_x = s.stats_dispatch_groups[0];
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_stats_compile_info().name));
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({s.stats_dispatch_groups[0], s.stats_dispatch_groups[1], 1});
}

void rt_lab_picker_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto & s = render_context->rt_lab.analysis;
    if (!s.picker_dispatch_pending)
    {
        return;
    }
    auto const size = render_context->render_data.settings.render_target_size;
    RtLabPickerPush push = {};
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    push.attach = alloc.device_address;
    for (u32 index = 0; index < RT_LAB_PICKER_UI_COUNT; ++index)
    {
        push.screen_pixel[index] = {std::min(s.picker_dispatch_pixel[index][0], size.x - 1), std::min(s.picker_dispatch_pixel[index][1], size.y - 1)};
    }
    push.slot = s.picker_dispatch_stamp % RT_LAB_PICKER_SLOTS;
    push.stamp = s.picker_dispatch_stamp;
    push.active_mask = s.picker_dispatch_mask;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_picker_compile_info().name));
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({1, 1, 1});
}

// == Input ==============================================================================================================

auto rt_lab_picker_input(RenderContext & render_context, bool mouse_down, i32 cursor_x, i32 cursor_y, i32 window_width, i32 window_height) -> bool
{
    auto & s = render_context.rt_lab.analysis;
    if (window_width <= 0 || window_height <= 0)
    {
        s.picking = -1;
        return false;
    }
    auto const size = render_context.render_data.settings.render_target_size;
    // Markers of the followed pixels on the viewport.
    ImVec2 const display = ImGui::GetIO().DisplaySize;
    for (u32 index = 0; index < RT_LAB_PICKER_UI_COUNT; ++index)
    {
        RtLabPicker const & picker = s.pickers[index];
        if (picker.active)
        {
            ImVec2 const center = ImVec2((static_cast<f32>(picker.pixel[0]) + 0.5f) * display.x / static_cast<f32>(size.x),
                (static_cast<f32>(picker.pixel[1]) + 0.5f) * display.y / static_cast<f32>(size.y));
            ImU32 const color = ImGui::ColorConvertFloat4ToU32(PICKER_COLORS[index]);
            ImDrawList * draw = ImGui::GetForegroundDrawList();
            draw->AddCircle(center, 7.0f, color, 0, 2.0f);
            draw->AddText(ImVec2(center.x + 9.0f, center.y - 18.0f), color, PICKER_NAMES[index]);
        }
    }
    if (s.picking < 0 || !mouse_down)
    {
        return false;
    }
    RtLabPicker & picker = s.pickers[s.picking];
    picker.pixel[0] = std::clamp(static_cast<i32>(static_cast<i64>(cursor_x) * size.x / window_width), 0, static_cast<i32>(size.x) - 1);
    picker.pixel[1] = std::clamp(static_cast<i32>(static_cast<i64>(cursor_y) * size.y / window_height), 0, static_cast<i32>(size.y) - 1);
    picker.active = true;
    picker.samples.clear();
    picker.frames = 0;
    s.picking = -1;
    return true;
}

// == UI =================================================================================================================

// Histogram as filled steps over a log10 x axis in REAL units (the axis ticks are the luminance / variance values, not their logarithms). Bin i covers
// 10^(log_low + i w) .. 10^(log_low + (i + 1) w).
static void plot_step_histogram(char const * label, f64 log_low, f64 log_width, std::vector<f64> const & ys, ImVec4 color, f32 fill_alpha)
{
    std::vector<f64> xs(ys.size() * 2), steps(ys.size() * 2);
    for (u32 i = 0; i < ys.size(); ++i)
    {
        xs[2 * i] = std::pow(10.0, log_low + static_cast<f64>(i) * log_width);
        xs[2 * i + 1] = std::pow(10.0, log_low + static_cast<f64>(i + 1) * log_width);
        steps[2 * i] = ys[i];
        steps[2 * i + 1] = ys[i];
    }
    ImPlot::SetNextFillStyle(color, fill_alpha);
    ImPlot::PlotShaded(label, xs.data(), steps.data(), static_cast<i32>(xs.size()), 0.0);
    ImPlot::SetNextLineStyle(color, 1.5f);
    ImPlot::PlotLine(fmt::format("{}##outline", label).c_str(), xs.data(), steps.data(), static_cast<i32>(xs.size()));
}

// Axes of a histogram plot: x = value on a log10 scale (the range and the y peak are smoothed by the caller, so they move without jumps).
static auto begin_histogram_plot(char const * id, char const * x_label, char const * y_label, f64 log_low, f64 log_high, f64 y_peak) -> bool
{
    if (!ImPlot::BeginPlot(id, ImVec2(-1.0f, 190.0f)))
    {
        return false;
    }
    ImPlot::SetupAxes(x_label, y_label, ImPlotAxisFlags_None, ImPlotAxisFlags_LockMin);
    ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
    ImPlot::SetupAxisLimits(ImAxis_X1, std::pow(10.0, log_low), std::pow(10.0, log_high), ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, std::max(y_peak * 1.15, 1e-4), ImPlotCond_Always);
    return true;
}

static void statistics_summary_ui(char const * id, RtLabStatsSummary const & m, RtLabStatsHistogram const & h)
{
    if (!m.valid)
    {
        return;
    }
    auto percentile = [&](f64 p) { return std::pow(10.0, histogram_percentile(h, p)); };
    struct Row
    {
        char const * name;
        f64 value;
    };
    Row const rows[] = {
        {"Count", static_cast<f64>(m.count)},
        {"Mean", m.mean},
        {"Std Dev", m.std_dev},
        {"Coeff. of Variation", m.mean > 0.0 ? m.std_dev / m.mean : 0.0},
        {"Min", m.min},
        {"Max", m.max},
        {"Geometric Mean", m.geometric_mean},
        {"Spread (x/ factor)", std::pow(10.0, m.log10_std)},
        {"p1", percentile(0.01)},
        {"p5", percentile(0.05)},
        {"p25", percentile(0.25)},
        {"p50 (median)", percentile(0.50)},
        {"p75", percentile(0.75)},
        {"p95", percentile(0.95)},
        {"p99", percentile(0.99)},
    };
    if (ImGui::BeginTable(id, 4, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV))
    {
        constexpr u32 row_count = sizeof(rows) / sizeof(rows[0]);
        for (u32 i = 0; i < (row_count + 1) / 2; ++i)
        {
            ImGui::TableNextRow();
            for (u32 column = 0; column < 2; ++column)
            {
                u32 const index = i + column * ((row_count + 1) / 2);
                ImGui::TableSetColumnIndex(column * 2);
                if (index < row_count)
                {
                    ImGui::TextUnformatted(rows[index].name);
                    ImGui::TableSetColumnIndex(column * 2 + 1);
                    if (index == 0)
                    {
                        ImGui::Text("%llu", static_cast<unsigned long long>(m.count));
                    }
                    else
                    {
                        ImGui::Text("%.4g", rows[index].value);
                    }
                }
            }
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Count, mean, std, min, max, geometric mean: exact. Percentiles: from the histogram bins (%.2f%% below / %.2f%% above its range).",
        h.total() > 0 ? 100.0 * h.under / static_cast<f64>(h.total()) : 0.0, h.total() > 0 ? 100.0 * h.over / static_cast<f64>(h.total()) : 0.0);
}

static void statistics_histogram_ui(char const * id, char const * x_label, RtLabStatsHistogram const & h, RtLabStatsSummary const & summary, RtLabSmoothRange const & range,
    f64 peak, ImVec4 color)
{
    if (!h.valid)
    {
        ImGui::TextDisabled("Waiting for the readback.");
        return;
    }
    u64 const total = h.total();
    f64 const width = (h.high - h.low) / static_cast<f64>(h.bins.size());
    std::vector<f64> ys(h.bins.size());
    for (u32 i = 0; i < h.bins.size(); ++i)
    {
        ys[i] = total > 0 ? static_cast<f64>(h.bins[i]) / static_cast<f64>(total) : 0.0;
    }
    if (begin_histogram_plot(id, x_label, "share of pixels", range.low, range.high, peak))
    {
        plot_step_histogram("pixels", h.low, width, ys, color, 0.55f);
        if (summary.valid && summary.mean > 0.0)
        {
            ImPlot::SetNextLineStyle(ImVec4(1.0f, 1.0f, 1.0f, 0.8f), 1.0f);
            ImPlot::PlotInfLines("mean", &summary.mean, 1);
        }
        ImPlot::EndPlot();
    }
    statistics_summary_ui(fmt::format("{}summary", id).c_str(), summary, h);
}

void rt_lab_statistics_ui(RenderContext & render_context)
{
    auto & s = render_context.rt_lab.analysis;
    ImGui::Checkbox("Enabled##statistics", &s.stats_enabled);
    ImGui::SetItemTooltip("Spatial statistics of the resolved output over all geometry pixels, from the current frame only (no history).\n"
                          "The bucket range follows the data smoothly.");
    if (!s.stats_enabled)
    {
        return;
    }
    ImGui::SeparatorText("Luminance");
    statistics_histogram_ui("##luma histogram", "luminance", s.luma_histogram, s.luma_summary, s.luma_range, s.luma_peak, ImVec4(0.30f, 0.65f, 1.0f, 1.0f));
    ImGui::SeparatorText("5x5 Spatial Variance");
    statistics_histogram_ui("##variance histogram", "variance of luminance", s.variance_histogram, s.variance_summary, s.variance_range, s.variance_peak, ImVec4(1.0f, 0.55f, 0.2f, 1.0f));
    ImGui::SetItemTooltip("Per pixel: unbiased variance of the luminance over the 5x5 window (geometry pixels only), that pixel's own frame.");
}

// The last `window` samples of a picker.
struct PickerSampleWindow
{
    std::deque<RtLabPickerSample>::const_iterator first;
    std::deque<RtLabPickerSample>::const_iterator last;
    auto begin() const { return first; }
    auto end() const { return last; }
    auto size() const { return static_cast<u32>(last - first); }
};

static auto picker_window(RtLabPicker const & picker, i32 window) -> PickerSampleWindow
{
    auto const count = static_cast<i64>(picker.samples.size());
    return {picker.samples.end() - std::min<i64>(count, std::max(window, 2)), picker.samples.end()};
}

void rt_lab_picker_ui(RenderContext & render_context)
{
    auto & s = render_context.rt_lab.analysis;
    ImGui::Checkbox("Exposed Units", &s.exposed_units);
    ImGui::SetItemTooltip("Luminance x exposure x pi, the units of the INDIRECT_DIFFUSE debug view. Off: scene radiance.");
    ImGui::SliderInt("Window (frames)", &s.temporal_window, 2, RT_LAB_PICKER_HISTORY_FRAMES, "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("The temporal statistics, the trend plot and the histogram over time only look at the last this many frames.");
    for (u32 index = 0; index < RT_LAB_PICKER_UI_COUNT; ++index)
    {
        RtLabPicker & picker = s.pickers[index];
        ImGui::PushID(static_cast<i32>(index));
        ImGui::PushStyleColor(ImGuiCol_Text, PICKER_COLORS[index]);
        ImGui::Text("Pixel %s", PICKER_NAMES[index]);
        ImGui::PopStyleColor();
        ImGui::SameLine();
        if (ImGui::Button(s.picking == static_cast<i32>(index) ? "Click the viewport..." : "Pick"))
        {
            s.picking = s.picking == static_cast<i32>(index) ? -1 : static_cast<i32>(index);
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!picker.active);
        if (ImGui::Button("Clear"))
        {
            picker.active = false;
            picker.samples.clear();
            picker.frames = 0;
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset History"))
        {
            picker.samples.clear();
            picker.frames = 0;
        }
        ImGui::SameLine();
        ImGui::Checkbox("Pause", &picker.pause);
        ImGui::EndDisabled();
        if (picker.active && !picker.samples.empty())
        {
            RtLabPickerSample const & p = picker.samples.back();
            f32 const scale = s.exposed_units ? p.exposure * std::numbers::pi_v<f32> : 1.0f;
            ImGui::Text("(%d, %d)  %s   luminance %.4g   rgb %.4g %.4g %.4g", picker.pixel[0], picker.pixel[1], p.geometry ? "" : "sky",
                p.luma * scale, p.rgb[0] * scale, p.rgb[1] * scale, p.rgb[2] * scale);
            f32 const window_std = std::sqrt(std::max(p.window_variance, 0.0f));
            ImGui::Text("5x5 window: mean %.4g   std %.4g   min %.4g   max %.4g   (%.0f px)", p.window_mean * scale, window_std * scale, p.window_min * scale,
                p.window_max * scale, p.window_count);
            // Color now + the last three frames, as displayed: x exposure x pi (the INDIRECT_DIFFUSE units), clamped, sRGB encoded.
            auto swatch = [&](RtLabPickerSample const & sample, char const * id, ImVec2 size)
            {
                auto encode = [](f32 linear) -> f32
                {
                    f32 const v = std::clamp(linear, 0.0f, 1.0f);
                    return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
                };
                f32 const exposed_scale = sample.exposure * std::numbers::pi_v<f32>;
                ImVec4 const color = sample.geometry ? ImVec4(encode(sample.rgb[0] * exposed_scale), encode(sample.rgb[1] * exposed_scale), encode(sample.rgb[2] * exposed_scale), 1.0f)
                                                       : ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
                ImGui::ColorButton(id, color, ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip, size);
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("rgb %.4g %.4g %.4g (scene radiance), luminance %.4g", sample.rgb[0], sample.rgb[1], sample.rgb[2], sample.luma);
                }
            };
            ImGui::TextUnformatted("Now");
            ImGui::SameLine();
            swatch(p, "##now", ImVec2(64.0f, 40.0f));
            for (u32 ago = 1; ago <= 3; ++ago)
            {
                ImGui::SameLine();
                if (picker.samples.size() > ago)
                {
                    ImGui::Text("-%u", ago);
                    ImGui::SameLine();
                    swatch(picker.samples[picker.samples.size() - 1 - ago], fmt::format("##ago{}", ago).c_str(), ImVec2(40.0f, 40.0f));
                }
            }
            // Temporal statistics of the pixel.
            f64 sum = 0.0, sum_squares = 0.0, value_min = std::numeric_limits<f64>::max(), value_max = 0.0;
            u64 count = 0;
            for (RtLabPickerSample const & sample : picker_window(picker, s.temporal_window))
            {
                if (sample.geometry)
                {
                    f64 const v = sample.luma * (s.exposed_units ? sample.exposure * std::numbers::pi : 1.0);
                    value_min = std::min(value_min, v);
                    value_max = std::max(value_max, v);
                    sum += v;
                    sum_squares += v * v;
                    ++count;
                }
            }
            if (count > 1)
            {
                f64 const mean = sum / static_cast<f64>(count);
                f64 const variance = std::max(sum_squares - static_cast<f64>(count) * mean * mean, 0.0) / static_cast<f64>(count - 1);
                ImGui::Text("last %llu frames: mean %.4g   std %.4g   relative std %.2f%%   min %.4g   max %.4g", static_cast<unsigned long long>(count), mean, std::sqrt(variance),
                    mean > 0.0 ? 100.0 * std::sqrt(variance) / mean : 0.0, value_min, value_max);
            }
        }
        else if (picker.active)
        {
            ImGui::TextDisabled("(%d, %d) waiting for the readback.", picker.pixel[0], picker.pixel[1]);
        }
        ImGui::PopID();
    }

    // Shared plots of the active pickers.
    bool const any_samples = std::any_of(s.pickers.begin(), s.pickers.end(), [](RtLabPicker const & p) { return p.active && !p.samples.empty(); });
    if (!any_samples)
    {
        return;
    }
    auto exposed = [&](RtLabPickerSample const & sample) -> f64
    {
        return sample.luma * (s.exposed_units ? sample.exposure * std::numbers::pi : 1.0);
    };
    ImGui::SeparatorText("Temporal Trend");
    if (ImPlot::BeginPlot("##picker trend", ImVec2(-1.0f, 200.0f)))
    {
        ImPlot::SetupAxes("frames ago", s.exposed_units ? "exposed luminance" : "luminance", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
        for (u32 index = 0; index < RT_LAB_PICKER_UI_COUNT; ++index)
        {
            RtLabPicker const & picker = s.pickers[index];
            if (!picker.active || picker.samples.empty())
            {
                continue;
            }
            std::vector<f64> xs, value, window_mean;
            PickerSampleWindow const samples = picker_window(picker, s.temporal_window);
            f64 const n = static_cast<f64>(samples.size());
            u32 i = 0;
            for (RtLabPickerSample const & sample : samples)
            {
                f64 const scale = s.exposed_units ? sample.exposure * std::numbers::pi : 1.0;
                xs.push_back(static_cast<f64>(i++) - n + 1.0);
                value.push_back(std::max(exposed(sample), 1e-9));
                window_mean.push_back(std::max(sample.window_mean * scale, 1e-9));
            }
            ImVec4 faded = PICKER_COLORS[index];
            faded.w = 0.45f;
            ImPlot::SetNextLineStyle(PICKER_COLORS[index], 1.5f);
            ImPlot::PlotLine(fmt::format("{} pixel", PICKER_NAMES[index]).c_str(), xs.data(), value.data(), static_cast<i32>(xs.size()));
            ImPlot::SetNextLineStyle(faded, 1.0f);
            ImPlot::PlotLine(fmt::format("{} 5x5 mean", PICKER_NAMES[index]).c_str(), xs.data(), window_mean.data(), static_cast<i32>(xs.size()));
        }
        ImPlot::EndPlot();
    }

    ImGui::SeparatorText("Luminance Histogram Over Time");
    // One shared log10 range (smoothed per picker, the plot uses the union).
    f64 shared_low = std::numeric_limits<f64>::max(), shared_high = std::numeric_limits<f64>::lowest(), shared_peak = 0.0;
    struct Series
    {
        std::vector<f64> ys;
        u32 index;
    };
    std::vector<Series> series;
    for (u32 index = 0; index < RT_LAB_PICKER_UI_COUNT; ++index)
    {
        RtLabPicker & picker = s.pickers[index];
        std::vector<f64> logs;
        for (RtLabPickerSample const & sample : picker_window(picker, s.temporal_window))
        {
            f64 const v = exposed(sample);
            if (sample.geometry && v > 1e-12)
            {
                logs.push_back(std::log10(v));
            }
        }
        if (!picker.active || logs.size() < 2)
        {
            continue;
        }
        std::sort(logs.begin(), logs.end());
        f64 const target_low = logs[static_cast<size_t>(0.01 * static_cast<f64>(logs.size() - 1))] - ANALYSIS_MARGIN;
        f64 const target_high = logs[static_cast<size_t>(0.99 * static_cast<f64>(logs.size() - 1))] + ANALYSIS_MARGIN;
        smooth_range(picker.histogram_range, target_low, target_high);
        shared_low = std::min(shared_low, picker.histogram_range.low);
        shared_high = std::max(shared_high, picker.histogram_range.high);
        Series entry = {};
        entry.index = index;
        entry.ys.assign(PICKER_HISTOGRAM_BINS, 0.0);
        series.push_back(std::move(entry));
    }
    if (!series.empty())
    {
        f64 const width = (shared_high - shared_low) / static_cast<f64>(PICKER_HISTOGRAM_BINS);
        for (Series & entry : series)
        {
            RtLabPicker & picker = s.pickers[entry.index];
            f64 total = 0.0;
            for (RtLabPickerSample const & sample : picker_window(picker, s.temporal_window))
            {
                f64 const v = exposed(sample);
                if (sample.geometry && v > 1e-12)
                {
                    f64 const t = (std::log10(v) - shared_low) / (shared_high - shared_low);
                    if (t >= 0.0 && t < 1.0)
                    {
                        entry.ys[static_cast<size_t>(t * PICKER_HISTOGRAM_BINS)] += 1.0;
                    }
                    total += 1.0;
                }
            }
            f64 peak = 0.0;
            for (f64 & y : entry.ys)
            {
                y = total > 0.0 ? y / total : 0.0;
                peak = std::max(peak, y);
            }
            smooth_peak(picker.histogram_peak, peak);
            shared_peak = std::max(shared_peak, picker.histogram_peak);
        }
        if (begin_histogram_plot("##picker histogram", s.exposed_units ? "exposed luminance" : "luminance", "share of frames", shared_low, shared_high, shared_peak))
        {
            for (Series const & entry : series)
            {
                plot_step_histogram(fmt::format("Pixel {}", PICKER_NAMES[entry.index]).c_str(), shared_low, width, entry.ys, PICKER_COLORS[entry.index], 0.4f);
            }
            ImPlot::EndPlot();
        }
    }
}
