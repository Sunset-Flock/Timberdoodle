#include "rt_lab_probe.hpp"

#include "rt_lab.hpp"
#include "rt_lab_probe.inl"
#include "rt_lab_telemetry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numbers>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>
#include <implot.h>

static constexpr f64 PROBE_RADIANCE_SCALE_LN = 9.210340371976184; // ln(RTGI_RADIANCE_SCALE = 1e4): y = ln(1 + 1e4 L) = this + ln(L + 1e-4)
static constexpr f64 PROBE_EPSILON = 1e-4;
static constexpr u32 PROBE_PLOT_POINTS = 240;

static constexpr ImVec4 PROBE_COLORS[RT_LAB_PROBE_UI_COUNT] = {ImVec4(0.30f, 0.65f, 1.00f, 1.0f), ImVec4(1.00f, 0.55f, 0.20f, 1.0f)};
static constexpr char const * PROBE_NAMES[RT_LAB_PROBE_UI_COUNT] = {"A", "B"};

static auto probe_luma(f32 const (&rgb)[3]) -> f64
{
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

static auto probe_normal_cdf(f64 z) -> f64
{
    return 0.5 * std::erfc(-z * std::numbers::sqrt2 * 0.5);
}

static void probe_clear(RtLabProbe & probe)
{
    probe.valid = false;
    probe.frames = 0;
    probe.latest = {};
    probe.trend.clear();
}

// == Readback ===========================================================================================================

static void probe_ingest(RenderContext & render_context, RtLabProbeState & s)
{
    u32 const * words = render_context.gpu_context->device.buffer_host_address_as<u32>(s.buffer.id()).value();
    std::vector<std::pair<u32, u32>> ready = {}; // (stamp, slot)
    for (u32 slot = 0; slot < RT_LAB_PROBE_SLOTS; ++slot)
    {
        u32 const stamp = words[slot * RT_LAB_PROBE_SLOT_WORDS];
        bool const finished = s.frame_counter >= s.issued_frame[slot] + MAX_GPU_FRAMES_IN_FLIGHT + 2;
        if (stamp != 0 && stamp == s.issued_stamp[slot] && finished && stamp > s.last_ingested)
        {
            ready.emplace_back(stamp, slot);
        }
    }
    std::sort(ready.begin(), ready.end());
    for (auto const & [stamp, slot] : ready)
    {
        s.last_ingested = stamp;
        u32 const * slot_words = words + slot * RT_LAB_PROBE_SLOT_WORDS;
        for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
        {
            RtLabProbe & probe = s.probes[index];
            if (!probe.active || probe.pause || ((slot_words[1] >> index) & 1u) == 0)
            {
                continue;
            }
            u32 const * w = slot_words + RT_LAB_PROBE_HEADER_WORDS + index * RT_LAB_PROBE_WORDS;
            if (static_cast<i32>(w[1]) != probe.pixel[0] || static_cast<i32>(w[2]) != probe.pixel[1])
            {
                continue; // another pixel (re-picked while this slot was in flight)
            }
            auto f = [&](u32 i) -> f32
            {
                f32 v = 0.0f;
                std::memcpy(&v, &w[i], sizeof(f32));
                return v;
            };
            RtLabProbeSample sample = {};
            sample.geometry = (w[0] & RT_LAB_PROBE_FLAG_GEOMETRY) != 0;
            sample.has_variance = (w[0] & RT_LAB_PROBE_FLAG_VARIANCE) != 0;
            sample.has_dist = (w[0] & RT_LAB_PROBE_FLAG_DIST_AB) != 0;
            sample.has_dist_low = (w[0] & RT_LAB_PROBE_FLAG_DIST_C) != 0;
            sample.lab_pixel[0] = w[3];
            sample.lab_pixel[1] = w[4];
            sample.linear_depth = f(5);
            for (u32 c = 0; c < 3; ++c)
            {
                sample.temporal[c] = f(6 + c);
                sample.filtered[c] = f(10 + c);
                sample.raw[c] = f(14 + c);
                sample.resolved[c] = f(18 + c);
                sample.variance[c] = f(22 + c);
            }
            sample.history_length = f(9);
            sample.effective_count = f(13);
            sample.rays = f(17);
            sample.exposure = f(25);
            for (u32 c = 0; c < 4; ++c)
            {
                sample.dist_a[c] = f(26 + c);
                sample.dist_b[c] = f(30 + c);
                sample.dist_c[c] = f(34 + c);
            }
            sample.pair_geometry = f(38);
            sample.pair_variance = f(39);
            sample.pair_distribution = f(40);
            sample.pair_total = f(41);
            sample.pair_valid = (w[42] & 1u) != 0;
            sample.pair_variance_on = (w[42] & 2u) != 0;
            sample.pair_distribution_on = (w[42] & 4u) != 0;
            probe.latest = sample;
            probe.valid = true;
            ++probe.frames;
            RtLabProbeTrend point = {};
            point.raw = sample.rays > 0.0f ? static_cast<f32>(probe_luma(sample.raw)) : 0.0f;
            point.temporal = static_cast<f32>(probe_luma(sample.temporal));
            point.filtered = static_cast<f32>(probe_luma(sample.filtered));
            point.resolved = static_cast<f32>(probe_luma(sample.resolved));
            point.length = sample.history_length;
            point.tail_fraction = sample.has_dist ? sample.dist_b[0] : 0.0f;
            probe.trend.push_back(point);
            while (probe.trend.size() > RT_LAB_PROBE_HISTORY_FRAMES)
            {
                probe.trend.pop_front();
            }
        }
    }
}

auto rt_lab_probe_frame_update(RenderContext & render_context) -> bool
{
    auto & s = render_context.rt_lab.probe;
    auto & device = render_context.gpu_context->device;
    ++s.frame_counter;
    s.dispatch_pending = false;
    // Test hook (no mouse needed): TIDO_PROBE_PIXELS="ax ay bx by" follows those screen pixels and prints their readback every 50 frames.
    static i32 hook[4] = {-1, -1, -1, -1};
    static bool hook_parsed = false;
    if (!hook_parsed)
    {
        hook_parsed = true;
        if (char const * text = std::getenv("TIDO_PROBE_PIXELS"))
        {
            std::sscanf(text, "%d %d %d %d", &hook[0], &hook[1], &hook[2], &hook[3]);
        }
    }
    if (hook[0] >= 0)
    {
        s.enabled = true;
        for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
        {
            if (!s.probes[index].active)
            {
                s.probes[index].active = true;
                s.probes[index].pixel[0] = hook[index * 2];
                s.probes[index].pixel[1] = hook[index * 2 + 1];
            }
            RtLabProbe const & probe = s.probes[index];
            // TIDO_PROBE_CSV=1: one machine readable line per frame instead (frame, state A, state B, this frame's mean ray luminance + ray count).
            static bool const csv = std::getenv("TIDO_PROBE_CSV") != nullptr;
            if (csv && probe.valid)
            {
                RtLabProbeSample const & p = probe.latest;
                fmt::print("[PC] {} {} {:.6g} {:.6g} {:.6g} {:.3f} {:.6g} {:.6g} {:.6g} {:.3f} {:.8g} {:.1f} {:.8g}\n", PROBE_NAMES[index], probe.frames, p.dist_a[0], p.dist_a[1], p.dist_a[2], p.dist_a[3],
                    p.dist_b[0], p.dist_b[1], p.dist_b[2], p.dist_b[3], probe_luma(p.raw), p.rays, probe_luma(p.temporal));
            }
            else if (probe.valid && s.frame_counter % 50 == 0)
            {
                RtLabProbeSample const & p = probe.latest;
                fmt::print("[Probe {}] frames {} geometry {} temporal L {:.5g} len {:.1f} filtered L {:.5g} n {:.4g} resolved L {:.5g} rays {} dist {} body {:.3f}/{:.3f}/{:.1f} tail {:.3f}/{:.3f}/{:.1f} | pair valid {} geo {:.3f} var {:.3f} dist {:.3f} total {:.3f}\n",
                    PROBE_NAMES[index], probe.frames, p.geometry, probe_luma(p.temporal), p.history_length, probe_luma(p.filtered), p.effective_count, probe_luma(p.resolved),
                    p.rays, p.has_dist, p.dist_a[0], p.dist_a[1], p.dist_a[3], p.dist_b[0], p.dist_b[1], p.dist_b[3], p.pair_valid, p.pair_geometry, p.pair_variance, p.pair_distribution, p.pair_total);
            }
        }
    }
    bool rebuild = s.enabled != s.built;
    if (!s.enabled)
    {
        return rebuild;
    }
    if (s.buffer.id().is_empty())
    {
        u64 const size = sizeof(u32) * RT_LAB_PROBE_SLOT_WORDS * RT_LAB_PROBE_SLOTS;
        s.buffer.set_buffer(device.create_buffer({
            .size = size,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "rt_lab_probe",
        }));
        std::memset(device.buffer_host_address(s.buffer.id()).value(), 0, size);
        s.issued_stamp.fill(0);
        s.issued_frame.fill(0);
        rebuild = true;
    }
    probe_ingest(render_context, s);
    u32 mask = 0;
    for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
    {
        RtLabProbe const & probe = s.probes[index];
        mask |= probe.active ? (1u << index) : 0u;
        s.dispatch_pixel[index][0] = static_cast<u32>(std::max(probe.pixel[0], 0));
        s.dispatch_pixel[index][1] = static_cast<u32>(std::max(probe.pixel[1], 0));
    }
    if (mask != 0)
    {
        u32 const stamp = s.next_stamp++;
        u32 const slot = stamp % RT_LAB_PROBE_SLOTS;
        s.issued_stamp[slot] = stamp;
        s.issued_frame[slot] = s.frame_counter;
        s.dispatch_stamp = stamp;
        s.dispatch_active_mask = mask;
        s.dispatch_pending = true;
    }
    return rebuild;
}

void rt_lab_probe_destroy(RenderContext & render_context)
{
    auto & s = render_context.rt_lab.probe;
    if (!s.buffer.id().is_empty())
    {
        render_context.gpu_context->device.destroy_buffer(s.buffer.id());
        s.buffer.set_buffer(daxa::BufferId{});
    }
}

void rt_lab_probe_callback(daxa::TaskInterface ti, RenderContext * render_context, bool spatial_ran, bool variance_bound, bool dist_ab_bound, bool dist_c_bound)
{
    auto & s = render_context->rt_lab.probe;
    if (!s.dispatch_pending)
    {
        return;
    }
    auto const & AT = RtLabProbeH::Info::AT;
    auto const lab_size = ti.info(AT.lab_depth).value().size;
    auto const full_size = render_context->render_data.settings.render_target_size;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_probe_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLabProbePush push = {};
    push.attach = alloc.device_address;
    for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
    {
        u32 const x = std::min(s.dispatch_pixel[index][0], full_size.x - 1);
        u32 const y = std::min(s.dispatch_pixel[index][1], full_size.y - 1);
        push.screen_pixel[index] = {x, y};
        push.lab_pixel[index] = {
            std::min(static_cast<u32>(static_cast<u64>(x) * lab_size.x / full_size.x), lab_size.x - 1),
            std::min(static_cast<u32>(static_cast<u64>(y) * lab_size.y / full_size.y), lab_size.y - 1),
        };
    }
    push.slot = s.dispatch_stamp % RT_LAB_PROBE_SLOTS;
    push.stamp = s.dispatch_stamp;
    push.active_mask = s.dispatch_active_mask;
    push.bound = (spatial_ran ? RT_LAB_PROBE_FLAG_SPATIAL_RAN : 0u) | (variance_bound ? RT_LAB_PROBE_FLAG_VARIANCE : 0u) |
                 (dist_ab_bound ? RT_LAB_PROBE_FLAG_DIST_AB : 0u) | (dist_c_bound ? RT_LAB_PROBE_FLAG_DIST_C : 0u);
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({1, 1, 1});
}

// == Input ==============================================================================================================

auto rt_lab_probe_input(RenderContext & render_context, bool mouse_down, i32 cursor_x, i32 cursor_y, i32 window_width, i32 window_height) -> bool
{
    auto & s = render_context.rt_lab.probe;
    if (!s.enabled || window_width <= 0 || window_height <= 0)
    {
        s.picking = -1;
        return false;
    }
    auto const size = render_context.render_data.settings.render_target_size;
    // Markers of the followed pixels on the viewport.
    ImVec2 const display = ImGui::GetIO().DisplaySize;
    for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
    {
        RtLabProbe const & probe = s.probes[index];
        if (probe.active)
        {
            ImVec2 const center = ImVec2((static_cast<f32>(probe.pixel[0]) + 0.5f) * display.x / static_cast<f32>(size.x),
                (static_cast<f32>(probe.pixel[1]) + 0.5f) * display.y / static_cast<f32>(size.y));
            ImU32 const color = ImGui::ColorConvertFloat4ToU32(PROBE_COLORS[index]);
            ImDrawList * draw = ImGui::GetForegroundDrawList();
            draw->AddCircle(center, 7.0f, color, 0, 2.0f);
            draw->AddText(ImVec2(center.x + 9.0f, center.y - 18.0f), color, PROBE_NAMES[index]);
        }
    }
    if (s.picking < 0 || !mouse_down)
    {
        return false;
    }
    RtLabProbe & probe = s.probes[s.picking];
    probe.pixel[0] = std::clamp(static_cast<i32>(static_cast<i64>(cursor_x) * size.x / window_width), 0, static_cast<i32>(size.x) - 1);
    probe.pixel[1] = std::clamp(static_cast<i32>(static_cast<i64>(cursor_y) * size.y / window_height), 0, static_cast<i32>(size.y) - 1);
    probe.active = true;
    probe_clear(probe);
    s.picking = -1;
    return true;
}

// == UI =================================================================================================================

// The tracked state as a mixture3 (rt_lab_telemetry.hpp) over luminance: u = ln(x + 1e-4), y = u + ln(1e4).
static auto probe_mixture(RtLabProbeSample const & p) -> RtLabTelemetryMixture3
{
    RtLabTelemetryMixture3 m = {};
    m.eps = PROBE_EPSILON;
    m.body_mean = static_cast<f64>(p.dist_a[0]) - PROBE_RADIANCE_SCALE_LN;
    m.body_variance = p.dist_a[1];
    m.perceptual_mean = static_cast<f64>(p.dist_a[2]) - PROBE_RADIANCE_SCALE_LN;
    m.body_count = static_cast<u64>(std::max(p.dist_a[3], 0.0f));
    bool const has_tail = p.dist_b[3] >= 1.0f;
    m.high_count = has_tail ? static_cast<u64>(p.dist_b[3]) : 0;
    m.high_fraction = has_tail ? p.dist_b[0] : 0.0;
    m.high_mean = static_cast<f64>(p.dist_b[1]) - PROBE_RADIANCE_SCALE_LN;
    m.high_variance = p.dist_b[2];
    bool const has_low = p.has_dist_low && p.dist_c[3] >= 1.0f;
    m.low_count = has_low ? static_cast<u64>(p.dist_c[3]) : 0;
    m.low_fraction = has_low ? p.dist_c[0] : 0.0;
    m.low_mean = static_cast<f64>(p.dist_c[1]) - PROBE_RADIANCE_SCALE_LN;
    m.low_variance = p.dist_c[2];
    m.valid = m.body_count > 0;
    return m;
}

// Mean of x under the mixture: lognormal components E[exp(u)] = exp(mu + var / 2).
static auto probe_mixture_mean(RtLabTelemetryMixture3 const & m) -> f64
{
    f64 const body_fraction = std::max(1.0 - m.low_fraction - m.high_fraction, 0.0);
    f64 const body = std::exp(m.body_mean + 0.5 * std::max(m.body_variance, 0.0));
    f64 const high = m.high_count > 0 ? std::exp(m.high_mean + 0.5 * m.high_variance) : 0.0;
    f64 const low = m.low_count > 0 ? std::exp(m.low_mean + 0.5 * m.low_variance) : 0.0;
    f64 const total_u = body_fraction * body + m.high_fraction * high + m.low_fraction * low;
    return std::max(total_u - PROBE_EPSILON, 0.0);
}

// One stage row: swatch of the exposed color (what the screen would show before tonemapping, x exposure x pi), luminance in scene and exposed units, rgb
// in the selected units.
static void probe_color_row(char const * label, f32 const (&rgb)[3], f32 unit, f32 exposed_unit)
{
    auto encode = [&](f32 v) -> f32 { return std::pow(std::clamp(v * exposed_unit, 0.0f, 1.0f), 1.0f / 2.2f); };
    ImGui::ColorButton(label, ImVec4(encode(rgb[0]), encode(rgb[1]), encode(rgb[2]), 1.0f), ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoAlpha, ImVec2(22.0f, 22.0f));
    ImGui::SameLine();
    ImGui::Text("%-9s L %.5g  exposed L %.4g   exposed rgb %.4g %.4g %.4g", label, probe_luma(rgb) * unit, probe_luma(rgb) * exposed_unit,
        rgb[0] * exposed_unit, rgb[1] * exposed_unit, rgb[2] * exposed_unit);
}

static void probe_column(RenderContext & render_context, RtLabProbeState & s, u32 index)
{
    RtLabProbe & probe = s.probes[index];
    ImGui::PushID(static_cast<i32>(index));
    ImGui::TextColored(PROBE_COLORS[index], "Probe %s", PROBE_NAMES[index]);
    ImGui::SameLine();
    bool const picking = s.picking == static_cast<i32>(index);
    if (ImGui::Button(picking ? "Click on the viewport..." : "Pick Pixel"))
    {
        s.picking = picking ? -1 : static_cast<i32>(index);
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!probe.active);
    if (ImGui::Button("Clear"))
    {
        probe.active = false;
        probe_clear(probe);
    }
    ImGui::SameLine();
    ImGui::Checkbox("Pause", &probe.pause);
    ImGui::EndDisabled();
    if (!probe.active)
    {
        ImGui::TextDisabled("No pixel. Pick one, then click the viewport.");
        ImGui::PopID();
        return;
    }
    if (!probe.valid)
    {
        ImGui::TextDisabled("pixel (%d, %d): waiting for data...", probe.pixel[0], probe.pixel[1]);
        ImGui::PopID();
        return;
    }
    RtLabProbeSample const & p = probe.latest;
    ImGui::Text("pixel (%d, %d)  lab pixel (%u, %u)  depth %.3g m  %s", probe.pixel[0], probe.pixel[1], p.lab_pixel[0], p.lab_pixel[1], p.linear_depth, p.geometry ? "" : "(sky)");
    if (!p.geometry)
    {
        ImGui::PopID();
        return;
    }
    f32 const exposed_unit = p.exposure * static_cast<f32>(std::numbers::pi);
    f32 const unit = s.exposed_units ? exposed_unit : 1.0f;
    ImGui::SeparatorText("Average Color");
    {
        f64 const accumulated_luma = probe_luma(p.temporal);
        ImGui::TextColored(PROBE_COLORS[index], "Temporal luminance  %.6g   (exposed %.5g)", accumulated_luma, accumulated_luma * p.exposure * std::numbers::pi);
        ImGui::SetItemTooltip("Luminance of the temporally accumulated color (history mean before the spatial passes), scene radiance units.\nExposed = x exposure x pi, the INDIRECT_DIFFUSE debug view's units.");
    }
    probe_color_row("rays", p.raw, unit, exposed_unit);
    ImGui::SameLine();
    ImGui::TextDisabled("(%g rays this frame)", p.rays);
    probe_color_row("temporal", p.temporal, unit, exposed_unit);
    probe_color_row("filtered", p.filtered, unit, exposed_unit);
    probe_color_row("resolved", p.resolved, unit, exposed_unit);
    f64 const temporal_luma = probe_luma(p.temporal);
    ImGui::Text("history length %.1f   filtered effective samples %.4g", p.history_length, p.effective_count);
    if (p.has_variance && p.history_length > 0.0f && temporal_luma > 1e-9)
    {
        f64 const variance = std::max(probe_luma(p.variance), 0.0);
        ImGui::Text("per sample rel std %.3g   rel std of the temporal mean %.3g %%", std::sqrt(variance) / temporal_luma,
            100.0 * std::sqrt(variance / p.history_length) / temporal_luma);
    }
    ImGui::SeparatorText("Distribution Estimate");
    if (!p.has_dist)
    {
        ImGui::TextDisabled("Distribution guide off (Settings -> Spatial -> Radiance Guides).");
    }
    else if (p.dist_a[3] < 1.0f)
    {
        ImGui::TextDisabled("No body yet.");
    }
    else
    {
        RtLabTelemetryMixture3 const m = probe_mixture(p);
        ImGui::Text("body   median %.4g  log std %.3g  samples %.1f", std::exp(m.body_mean) * unit, std::sqrt(std::max(m.body_variance, 0.0)), p.dist_a[3]);
        ImGui::SetItemTooltip("Median luminance (exp of the log mean) and std of the log.");
        ImGui::Text("perceptual mean L %.4g  (tail thresholds: x e^2 above, / e^2 below)", std::exp(m.perceptual_mean) * unit);
        ImGui::SetItemTooltip("exp of the tracked mean of y over ALL samples: the tail thresholds are this +- 2 in the log domain.");
        if (m.high_count > 0)
        {
            ImGui::Text("tail   p %.2f %%  median %.4g  log std %.3g  samples %.1f", 100.0 * m.high_fraction, std::exp(m.high_mean) * unit, std::sqrt(std::max(m.high_variance, 0.0)), p.dist_b[3]);
        }
        else
        {
            ImGui::TextDisabled("tail   none");
        }
        if (p.has_dist_low)
        {
            if (m.low_count > 0)
            {
                ImGui::Text("dark   p %.2f %%  median %.4g  log std %.3g  samples %.1f", 100.0 * m.low_fraction, std::exp(m.low_mean) * unit, std::sqrt(std::max(m.low_variance, 0.0)), p.dist_c[3]);
            }
            else
            {
                ImGui::TextDisabled("dark   none");
            }
        }
        ImGui::Text("model mean L %.5g  (temporal mean L %.5g)", probe_mixture_mean(m) * unit, temporal_luma * unit);
    }
    ImGui::PopID();
}

void rt_lab_probe_ui(RenderContext & render_context)
{
    auto & s = render_context.rt_lab.probe;
    ImGui::Checkbox("Enable Live Pixel Probes", &s.enabled);
    ImGui::SetItemTooltip("Follow up to two pixels of the viewport and show the denoiser state accumulated at them, every frame: the average color at each\n"
                          "stage and the live body / tail distribution estimate of the distribution guide. Reads back the existing temporal images, no\n"
                          "rays are collected. Enabling rebuilds the task graph.");
    if (!s.enabled)
    {
        return;
    }
    if (render_context.render_data.rt_lab_settings.enabled == 0)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "RT Lab is off.");
    }
    ImGui::Checkbox("Exposed Units", &s.exposed_units);
    ImGui::SetItemTooltip("Luminance x exposure x pi: the INDIRECT_DIFFUSE debug view's units. Off: scene radiance.");
    ImGui::SameLine();
    ImGui::Checkbox("Log Axis", &s.log_axis);
    ImGui::SetItemTooltip("Log10 luminance axis of the distribution plot (the density is then per log10 unit).");

    if (ImGui::BeginTable("##probes", 2, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame))
    {
        ImGui::TableNextRow();
        for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
        {
            ImGui::TableSetColumnIndex(static_cast<i32>(index));
            probe_column(render_context, s, index);
        }
        ImGui::EndTable();
    }

    // Weights the first spatial pass gives each pixel as a tap of the other one.
    {
        RtLabProbe const & a = s.probes[0];
        RtLabProbe const & b = s.probes[1];
        ImGui::SeparatorText("Spatial Filter Weights (first pass)");
        if (!(a.active && b.active && a.valid && b.valid && a.latest.pair_valid && b.latest.pair_valid))
        {
            ImGui::TextDisabled("Needs both probes on surfaces (and the spatial filter on).");
        }
        else if (ImGui::BeginTable("##pairweights", 6, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("center -> tap");
            ImGui::TableSetupColumn("geometry");
            ImGui::TableSetupColumn("variance guide");
            ImGui::TableSetupColumn("distribution guide");
            ImGui::TableSetupColumn("product");
            ImGui::TableSetupColumn("");
            ImGui::TableHeadersRow();
            for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
            {
                RtLabProbeSample const & p = s.probes[index].latest;
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextColored(PROBE_COLORS[index], "%s -> %s", PROBE_NAMES[index], PROBE_NAMES[1 - index]);
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%.4f", p.pair_geometry);
                ImGui::TableSetColumnIndex(2);
                if (p.pair_variance_on)
                {
                    ImGui::Text("%.4f", p.pair_variance);
                }
                else
                {
                    ImGui::TextDisabled("off");
                }
                ImGui::TableSetColumnIndex(3);
                if (p.pair_distribution_on)
                {
                    ImGui::Text("%.4f", p.pair_distribution);
                }
                else
                {
                    ImGui::TextDisabled("off");
                }
                ImGui::TableSetColumnIndex(4);
                ImGui::Text("%.4f", p.pair_total);
            }
            ImGui::EndTable();
            ImGui::TextDisabled("Weight the center pixel gives the other one as a tap of the first a-trous pass (input = the temporal output). Without the B3 kernel,\nthe ray length / SH guides and the sample budget. Not symmetric: the plane test uses the center's normal and pixel width.");
        }
    }

    // Distribution plot: both probes' mixtures over the same axis.
    std::array<RtLabTelemetryMixture3, RT_LAB_PROBE_UI_COUNT> mixtures = {};
    std::array<bool, RT_LAB_PROBE_UI_COUNT> shown = {};
    std::array<f32, RT_LAB_PROBE_UI_COUNT> units = {};
    f64 u_lo = 1e30, u_hi = -1e30;
    for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
    {
        RtLabProbe const & probe = s.probes[index];
        if (probe.active && probe.valid && probe.latest.geometry && probe.latest.has_dist && probe.latest.dist_a[3] >= 1.0f)
        {
            RtLabProbeSample const & p = probe.latest;
            shown[index] = true;
            mixtures[index] = probe_mixture(p);
            units[index] = s.exposed_units ? p.exposure * static_cast<f32>(std::numbers::pi) : 1.0f;
            RtLabTelemetryMixture3 const & m = mixtures[index];
            auto widen = [&](f64 mean, f64 variance)
            {
                f64 const sigma = std::sqrt(std::max(variance, 0.02));
                u_lo = std::min(u_lo, mean - 3.5 * sigma);
                u_hi = std::max(u_hi, mean + 3.5 * sigma);
            };
            widen(m.body_mean, std::max(m.body_variance, 1e-4));
            if (m.high_count > 0)
            {
                widen(m.high_mean, m.high_variance);
            }
            if (m.low_count > 0)
            {
                widen(m.low_mean, m.low_variance);
            }
        }
    }
    if (u_hi > u_lo)
    {
        ImGui::SeparatorText("Distribution Of The Incoming Luminance");
        if (ImPlot::BeginPlot("##distribution", ImVec2(-1.0f, 230.0f)))
        {
            ImPlot::SetupAxes(s.exposed_units ? "exposed luminance" : "luminance", s.log_axis ? "density per log10 L" : "density", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            if (s.log_axis)
            {
                ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
            }
            f64 const x_lo = std::max(std::exp(u_lo) - PROBE_EPSILON, 1e-7);
            f64 const x_hi = std::max(std::exp(u_hi) - PROBE_EPSILON, x_lo * 2.0);
            for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
            {
                if (!shown[index])
                {
                    continue;
                }
                std::array<f32, PROBE_PLOT_POINTS> xs, ys;
                for (u32 i = 0; i < PROBE_PLOT_POINTS; ++i)
                {
                    f64 const t = static_cast<f64>(i) / (PROBE_PLOT_POINTS - 1.0);
                    f64 const x = s.log_axis ? x_lo * std::pow(x_hi / x_lo, t) : x_lo + (x_hi - x_lo) * t;
                    f64 const pdf = rt_lab_telemetry_mixture3_pdf(mixtures[index], x);
                    xs[i] = static_cast<f32>(x * units[index]);
                    ys[i] = static_cast<f32>(s.log_axis ? pdf * x * std::numbers::ln10 : pdf / units[index]);
                }
                ImPlot::SetNextLineStyle(PROBE_COLORS[index], 2.0f);
                ImPlot::PlotLine(fmt::format("Probe {}", PROBE_NAMES[index]).c_str(), xs.data(), ys.data(), static_cast<i32>(PROBE_PLOT_POINTS));
                f64 const mean_x = probe_luma(s.probes[index].latest.temporal) * units[index];
                ImPlot::SetNextLineStyle(PROBE_COLORS[index], 1.0f);
                ImPlot::PlotInfLines(fmt::format("mean {}", PROBE_NAMES[index]).c_str(), &mean_x, 1);
                // The tracked perceptual mean (state A.z, the mean of y over all samples) converted back to luminance: the geometric center of the samples.
                f64 const perceptual_x = std::max(std::exp(mixtures[index].perceptual_mean) - PROBE_EPSILON, 1e-7) * units[index];
                ImVec4 faint = PROBE_COLORS[index];
                faint.w = 0.55f;
                ImPlot::SetNextLineStyle(faint, 1.0f);
                ImPlot::PlotInfLines(fmt::format("perceptual mean {}", PROBE_NAMES[index]).c_str(), &perceptual_x, 1);
            }
            ImPlot::EndPlot();
        }
        ImGui::TextDisabled("Curves: tracked mixture (dark tail + body + bright tail). Vertical lines: the temporal mean luminance (solid) and the tracked perceptual mean converted back to luminance (faint).");
    }

    // Trend of the luminance over the last frames.
    ImGui::SeparatorText("Luminance Over The Last Frames");
    if (ImPlot::BeginPlot("##trend", ImVec2(-1.0f, 200.0f)))
    {
        ImPlot::SetupAxes("frame", s.exposed_units ? "exposed luminance" : "luminance", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
        for (u32 index = 0; index < RT_LAB_PROBE_UI_COUNT; ++index)
        {
            RtLabProbe const & probe = s.probes[index];
            if (!probe.active || probe.trend.size() < 2)
            {
                continue;
            }
            f32 const unit = s.exposed_units ? probe.latest.exposure * static_cast<f32>(std::numbers::pi) : 1.0f;
            std::vector<f32> xs(probe.trend.size()), temporal(probe.trend.size()), resolved(probe.trend.size());
            for (size_t i = 0; i < probe.trend.size(); ++i)
            {
                xs[i] = static_cast<f32>(i) - static_cast<f32>(probe.trend.size());
                temporal[i] = probe.trend[i].temporal * unit;
                resolved[i] = probe.trend[i].resolved * unit;
            }
            ImPlot::SetNextLineStyle(PROBE_COLORS[index], 1.5f);
            ImPlot::PlotLine(fmt::format("{} temporal", PROBE_NAMES[index]).c_str(), xs.data(), temporal.data(), static_cast<i32>(xs.size()));
            ImVec4 faded = PROBE_COLORS[index];
            faded.w = 0.55f;
            ImPlot::SetNextLineStyle(faded, 1.0f);
            ImPlot::PlotLine(fmt::format("{} resolved", PROBE_NAMES[index]).c_str(), xs.data(), resolved.data(), static_cast<i32>(xs.size()));
        }
        ImPlot::EndPlot();
    }
}
