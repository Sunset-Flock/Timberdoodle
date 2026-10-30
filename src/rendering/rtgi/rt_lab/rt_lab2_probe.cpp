#include "rt_lab2_probe.hpp"

#include "rt_lab2.hpp"
#include "rt_lab2.inl"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <fmt/format.h>
#include <imgui.h>
#include <implot.h>

static_assert(RT_LAB2_PROBE_UI_COUNT == RT_LAB2_PROBE_COUNT, "the UI follows as many probes as the shader writes");

static constexpr ImVec4 PROBE_COLORS[RT_LAB2_PROBE_UI_COUNT] = {ImVec4(0.30f, 0.65f, 1.00f, 1.0f), ImVec4(1.00f, 0.55f, 0.20f, 1.0f)};
static constexpr char const * PROBE_NAMES[RT_LAB2_PROBE_UI_COUNT] = {"A", "B"};
static constexpr char const * NORMAL_BUCKET_NAMES[6] = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};

static auto probe_luma(f32 const (&rgb)[3]) -> f64
{
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
}

// Same rule as rt_lab2_history_cap (rt_lab2.hlsl).
static auto probe_history_cap(RtLab2Settings const & s, f32 exposure) -> f32
{
    f32 const upper = std::max(s.max_history_samples, 1.0f);
    if (s.history_samples_per_pixel <= 0.0f)
    {
        return upper;
    }
    return std::clamp(s.history_samples_per_pixel * std::max(exposure, 1.0f), std::min(std::max(s.min_history_samples, 1.0f), upper), upper);
}

static void probe_clear(RtLab2Probe & probe)
{
    probe.valid = false;
    probe.frames = 0;
    probe.latest = {};
    probe.trend.clear();
}

// == Readback ===========================================================================================================

static void probe_ingest(RenderContext & render_context, RtLab2ProbeState & s)
{
    auto const & settings = render_context.rt_lab.lab2.settings;
    u32 const * words = render_context.gpu_context->device.buffer_host_address_as<u32>(s.buffer.id()).value();
    std::vector<std::pair<u32, u32>> ready = {}; // (stamp, slot)
    for (u32 slot = 0; slot < RT_LAB2_PROBE_SLOTS; ++slot)
    {
        u32 const stamp = words[slot * RT_LAB2_PROBE_SLOT_WORDS];
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
        u32 const * slot_words = words + slot * RT_LAB2_PROBE_SLOT_WORDS;
        for (u32 index = 0; index < RT_LAB2_PROBE_UI_COUNT; ++index)
        {
            RtLab2Probe & probe = s.probes[index];
            if (!probe.active || probe.pause || ((slot_words[1] >> index) & 1u) == 0)
            {
                continue;
            }
            u32 const * w = slot_words + RT_LAB2_PROBE_HEADER_WORDS + index * RT_LAB2_PROBE_WORDS;
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
            RtLab2ProbeSample sample = {};
            sample.geometry = (w[0] & RT_LAB2_PROBE_FLAG_GEOMETRY) != 0;
            sample.cell = (w[0] & RT_LAB2_PROBE_FLAG_CELL) != 0;
            sample.lookup = (w[0] & RT_LAB2_PROBE_FLAG_LOOKUP) != 0;
            sample.pixel[0] = w[1];
            sample.pixel[1] = w[2];
            sample.cell_index = w[3];
            sample.level = w[4];
            sample.normal_bucket = w[5];
            sample.frames_untouched = w[6];
            sample.key = w[7];
            for (u32 c = 0; c < 3; ++c)
            {
                sample.coord[c] = static_cast<i32>(w[8 + c]);
                sample.position[c] = f(12 + c);
                sample.normal[c] = f(16 + c);
                sample.history[c] = f(20 + c);
                sample.frame_mean[c] = f(28 + c);
                sample.lookup_value[c] = f(32 + c);
                sample.output[c] = f(36 + c);
            }
            sample.cell_size = f(11);
            sample.linear_depth = f(15);
            sample.samples = f(19);
            sample.camera_exposure = f(23);
            sample.exposure = f(24);
            sample.last_pixels = w[25];
            sample.last_rays = w[26];
            sample.output_frames = f(39);
            sample.gk_x = f(72);
            sample.gk_p = f(73);
            sample.gk_applied = f(74);
            sample.gk_loss = f(75);
            sample.meas_z = f(76);
            sample.meas_r = f(77);
            sample.rec_valid = w[78];
            sample.gk_pending = f(79);
            for (u32 t = 0; t < 8; ++t)
            {
                sample.taps[t] = RtLab2ProbeTap{
                    .found = w[40 + t * 4] != 0,
                    .weight = f(41 + t * 4),
                    .samples = f(42 + t * 4),
                    .luminance = f(43 + t * 4),
                };
            }
            probe.latest = sample;
            probe.valid = true;
            ++probe.frames;
            RtLab2ProbeTrend point = {};
            point.frame_mean = sample.last_rays > 0 ? static_cast<f32>(probe_luma(sample.frame_mean)) : 0.0f;
            point.history = static_cast<f32>(probe_luma(sample.history));
            point.lookup = sample.lookup ? static_cast<f32>(probe_luma(sample.lookup_value)) : 0.0f;
            point.output = static_cast<f32>(probe_luma(sample.output));
            point.samples = sample.samples;
            point.cap = probe_history_cap(settings, sample.exposure);
            point.exposure = sample.exposure;
            point.rays = static_cast<f32>(sample.last_rays);
            probe.trend.push_back(point);
            while (probe.trend.size() > RT_LAB2_PROBE_HISTORY_FRAMES)
            {
                probe.trend.pop_front();
            }
        }
    }
}

auto rt_lab2_probe_frame_update(RenderContext & render_context) -> bool
{
    auto & s = render_context.rt_lab.lab2.probe;
    auto & device = render_context.gpu_context->device;
    ++s.frame_counter;
    s.dispatch_pending = false;
    // Test hook (no mouse needed): TIDO_CELL_PROBE_PIXELS="ax ay bx by" follows those screen pixels and prints their cells every 50 frames.
    static i32 hook[4] = {-1, -1, -1, -1};
    static bool hook_parsed = false;
    if (!hook_parsed)
    {
        hook_parsed = true;
        if (char const * text = std::getenv("TIDO_CELL_PROBE_PIXELS"))
        {
            std::sscanf(text, "%d %d %d %d", &hook[0], &hook[1], &hook[2], &hook[3]);
        }
    }
    if (hook[0] >= 0)
    {
        s.enabled = true;
        for (u32 index = 0; index < RT_LAB2_PROBE_UI_COUNT; ++index)
        {
            if (!s.probes[index].active)
            {
                s.probes[index].active = true;
                s.probes[index].pixel[0] = hook[index * 2];
                s.probes[index].pixel[1] = hook[index * 2 + 1];
            }
            RtLab2Probe const & probe = s.probes[index];
            static u32 const print_every = std::getenv("TIDO_CELL_PROBE_EVERY") ? static_cast<u32>(std::max(std::atoi(std::getenv("TIDO_CELL_PROBE_EVERY")), 1)) : 50u;
            if (probe.valid && s.frame_counter % print_every == 0)
            {
                RtLab2ProbeSample const & p = probe.latest;
                fmt::print("[CellProbe {}] camera exposure {:.4g} | frames {} cell {} level {} size {:.3f} m | exposure {:.1f} pixels {} rays {} | samples {:.1f} cap {:.1f} | history L {:.5g} frame L {:.5g} lookup L {:.5g} output L {:.5g} frames {:.0f} | grad rec {} meas z {:+.4f} r {:.4f} x {:+.4f} applied {:+.4f} loss {:.5f}\n",
                    PROBE_NAMES[index], p.camera_exposure, probe.frames, p.cell, p.level, p.cell_size, p.exposure, p.last_pixels, p.last_rays, p.samples,
                    probe_history_cap(render_context.rt_lab.lab2.settings, p.exposure), probe_luma(p.history), probe_luma(p.frame_mean), probe_luma(p.lookup_value),
                    probe_luma(p.output), p.output_frames, p.rec_valid, p.meas_z, p.meas_r, p.gk_x, p.gk_applied, p.gk_loss);
            }
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
        u64 const size = sizeof(u32) * RT_LAB2_PROBE_SLOT_WORDS * RT_LAB2_PROBE_SLOTS;
        s.buffer.set_buffer(device.create_buffer({
            .size = size,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "rt_lab2_probe",
        }));
        std::memset(device.buffer_host_address(s.buffer.id()).value(), 0, size);
        s.issued_stamp.fill(0);
        s.issued_frame.fill(0);
        rebuild = true;
    }
    probe_ingest(render_context, s);
    u32 mask = 0;
    for (u32 index = 0; index < RT_LAB2_PROBE_UI_COUNT; ++index)
    {
        RtLab2Probe const & probe = s.probes[index];
        mask |= probe.active ? (1u << index) : 0u;
        s.dispatch_pixel[index][0] = static_cast<u32>(std::max(probe.pixel[0], 0));
        s.dispatch_pixel[index][1] = static_cast<u32>(std::max(probe.pixel[1], 0));
    }
    if (mask != 0)
    {
        u32 const stamp = s.next_stamp++;
        u32 const slot = stamp % RT_LAB2_PROBE_SLOTS;
        s.issued_stamp[slot] = stamp;
        s.issued_frame[slot] = s.frame_counter;
        s.dispatch_stamp = stamp;
        s.dispatch_active_mask = mask;
        s.dispatch_pending = true;
    }
    return rebuild;
}

void rt_lab2_probe_destroy(RenderContext & render_context)
{
    auto & s = render_context.rt_lab.lab2.probe;
    if (!s.buffer.id().is_empty())
    {
        render_context.gpu_context->device.destroy_buffer(s.buffer.id());
        s.buffer.set_buffer(daxa::BufferId{});
    }
}

void rt_lab2_probe_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto & s = render_context->rt_lab.lab2.probe;
    if (!s.dispatch_pending)
    {
        return;
    }
    auto const & AT = RtLab2ProbeH::Info::AT;
    auto const size = ti.info(AT.lab_depth).value().size;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab2_probe_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLab2Settings settings = render_context->rt_lab.lab2.settings;
    auto settings_alloc = ti.allocator->allocate(sizeof(RtLab2Settings)).value();
    std::memcpy(settings_alloc.host_address, &settings, sizeof(RtLab2Settings));
    RtLab2ProbePush push = {};
    push.attach = alloc.device_address;
    push.settings = settings_alloc.device_address;
    push.size = {size.x, size.y};
    for (u32 index = 0; index < RT_LAB2_PROBE_UI_COUNT; ++index)
    {
        push.screen_pixel[index] = {std::min(s.dispatch_pixel[index][0], size.x - 1), std::min(s.dispatch_pixel[index][1], size.y - 1)};
    }
    push.slot = s.dispatch_stamp % RT_LAB2_PROBE_SLOTS;
    push.stamp = s.dispatch_stamp;
    push.active_mask = s.dispatch_active_mask;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({1, 1, 1});
}

// == Input ==============================================================================================================

auto rt_lab2_probe_input(RenderContext & render_context, bool mouse_down, i32 cursor_x, i32 cursor_y, i32 window_width, i32 window_height) -> bool
{
    auto & s = render_context.rt_lab.lab2.probe;
    if (!s.enabled || window_width <= 0 || window_height <= 0)
    {
        s.picking = -1;
        return false;
    }
    auto const size = render_context.render_data.settings.render_target_size;
    // Markers of the followed pixels on the viewport (squares: the pixel probes draw circles).
    ImVec2 const display = ImGui::GetIO().DisplaySize;
    for (u32 index = 0; index < RT_LAB2_PROBE_UI_COUNT; ++index)
    {
        RtLab2Probe const & probe = s.probes[index];
        if (probe.active)
        {
            ImVec2 const center = ImVec2((static_cast<f32>(probe.pixel[0]) + 0.5f) * display.x / static_cast<f32>(size.x),
                (static_cast<f32>(probe.pixel[1]) + 0.5f) * display.y / static_cast<f32>(size.y));
            ImU32 const color = ImGui::ColorConvertFloat4ToU32(PROBE_COLORS[index]);
            ImDrawList * draw = ImGui::GetForegroundDrawList();
            draw->AddRect(ImVec2(center.x - 7.0f, center.y - 7.0f), ImVec2(center.x + 7.0f, center.y + 7.0f), color, 0.0f, 0, 2.0f);
            draw->AddText(ImVec2(center.x + 9.0f, center.y - 18.0f), color, fmt::format("cell {}", PROBE_NAMES[index]).c_str());
        }
    }
    if (s.picking < 0 || !mouse_down)
    {
        return false;
    }
    RtLab2Probe & probe = s.probes[s.picking];
    probe.pixel[0] = std::clamp(static_cast<i32>(static_cast<i64>(cursor_x) * size.x / window_width), 0, static_cast<i32>(size.x) - 1);
    probe.pixel[1] = std::clamp(static_cast<i32>(static_cast<i64>(cursor_y) * size.y / window_height), 0, static_cast<i32>(size.y) - 1);
    probe.active = true;
    probe_clear(probe);
    s.picking = -1;
    return true;
}

// == UI =================================================================================================================

// A color swatch + the luminance in scene units and in the display units of the INDIRECT_DIFFUSE debug view.
static void probe_color_row(char const * label, f32 const (&rgb)[3], f32 exposed_unit)
{
    auto encode = [&](f32 v) -> f32 { return std::pow(std::clamp(v * exposed_unit, 0.0f, 1.0f), 1.0f / 2.2f); };
    ImGui::ColorButton(label, ImVec4(encode(rgb[0]), encode(rgb[1]), encode(rgb[2]), 1.0f), ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoAlpha, ImVec2(22.0f, 22.0f));
    ImGui::SameLine();
    ImGui::Text("%-10s L %.5g   exposed %.4g   rgb %.4g %.4g %.4g", label, probe_luma(rgb), probe_luma(rgb) * exposed_unit, rgb[0] * exposed_unit, rgb[1] * exposed_unit, rgb[2] * exposed_unit);
}

static void probe_column(RenderContext & render_context, RtLab2ProbeState & s, u32 index)
{
    auto const & settings = render_context.rt_lab.lab2.settings;
    RtLab2Probe & probe = s.probes[index];
    ImGui::PushID(static_cast<i32>(index));
    ImGui::TextColored(PROBE_COLORS[index], "Cell Probe %s", PROBE_NAMES[index]);
    ImGui::SameLine();
    if (ImGui::Button(s.picking == static_cast<i32>(index) ? "Click In Viewport..." : "Pick Pixel"))
    {
        s.picking = s.picking == static_cast<i32>(index) ? -1 : static_cast<i32>(index);
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear"))
    {
        probe.active = false;
        probe_clear(probe);
    }
    ImGui::SameLine();
    ImGui::Checkbox("Pause", &probe.pause);
    if (!probe.active)
    {
        ImGui::TextDisabled("Pick a pixel to follow its hash grid cell.");
        ImGui::PopID();
        return;
    }
    ImGui::Text("Pixel %d, %d   frames %llu", probe.pixel[0], probe.pixel[1], static_cast<unsigned long long>(probe.frames));
    if (!probe.valid)
    {
        ImGui::TextDisabled("Waiting for the first sample...");
        ImGui::PopID();
        return;
    }
    RtLab2ProbeSample const & p = probe.latest;
    if (!p.geometry)
    {
        ImGui::TextDisabled("No geometry (sky).");
        ImGui::PopID();
        return;
    }
    f32 const exposed_unit = 3.14159265f * p.camera_exposure; // the display units of the INDIRECT_DIFFUSE debug view
    f32 const cap = probe_history_cap(settings, p.exposure);
    if (!p.cell)
    {
        ImGui::TextDisabled("No cell yet (the pixel's cell was not queried: no geometry last frame or the grid is full).");
    }
    else
    {
        ImGui::SeparatorText("Cell");
        ImGui::Text("Index %u   key %08x   normal bucket %s", p.cell_index, p.key, NORMAL_BUCKET_NAMES[std::min(p.normal_bucket, 5u)]);
        ImGui::Text("Level %u   size %.4f m (about %.0f px wide)   coord %d %d %d", p.level, p.cell_size, std::sqrt(std::max(p.exposure, 1.0f)), p.coord[0], p.coord[1], p.coord[2]);
        ImGui::Text("Frames untouched %u   position %.2f %.2f %.2f   depth %.2f m", p.frames_untouched, p.position[0], p.position[1], p.position[2], p.linear_depth);
        ImGui::SeparatorText("Temporal Gradient");
        ImGui::Text("Re-traced rays %u   measurement %s ln(new / old) %+.4f  var %.4f", p.rec_valid, p.meas_r > 0.0f ? "yes" : "none", p.meas_z, p.meas_r);
        ImGui::Text("Kalman change %+.4f  var %.5f   pending %+.4f", p.gk_x, p.gk_p, p.gk_pending);
        ImGui::Text("Applied correction %+.4f (x%.3f)   uncertainty left %.5f", p.gk_applied, std::exp(p.gk_applied), p.gk_loss);
        ImGui::SeparatorText("Coverage + Rays");
        ImGui::Text("Exposure (smoothed visible pixels) %.1f", p.exposure);
        ImGui::Text("Last frame: %u pixels query it, %u rays hit it", p.last_pixels, p.last_rays);
        f32 const rays_per_pixel = p.last_pixels > 0 ? static_cast<f32>(p.last_rays) / static_cast<f32>(p.last_pixels) : 0.0f;
        ImGui::Text("Rays per covered pixel %.3f", rays_per_pixel);
        ImGui::SeparatorText("History");
        ImGui::Text("Samples %.1f / window %.1f", p.samples, cap);
        ImGui::ProgressBar(std::clamp(p.samples / std::max(cap, 1.0f), 0.0f, 1.0f), ImVec2(-1.0f, 0.0f));
        if (p.last_rays > 0)
        {
            f32 const blend = static_cast<f32>(p.last_rays) / (std::min(p.samples, cap) + static_cast<f32>(p.last_rays));
            ImGui::Text("Newest frame weight %.1f %%   window %.1f frames at this ray rate", blend * 100.0f, cap / static_cast<f32>(p.last_rays));
        }
        else
        {
            ImGui::TextDisabled("No rays last frame: the history is held.");
        }
        ImGui::SetItemTooltip("Adaption speed = window / rays per frame. The window is history samples per pixel x the cell's smoothed pixel coverage,\n"
                              "so a cell with fewer covering pixels gets proportionally fewer rays AND a proportionally shorter window.");
    }
    ImGui::SeparatorText("Radiance");
    if (p.cell)
    {
        probe_color_row("history", p.history, exposed_unit);
        if (p.last_rays > 0)
        {
            probe_color_row("frame mean", p.frame_mean, exposed_unit);
        }
    }
    if (p.lookup)
    {
        probe_color_row("lookup", p.lookup_value, exposed_unit);
    }
    else
    {
        ImGui::TextDisabled("lookup: no value (no tap found a cell)");
    }
    probe_color_row("output", p.output, exposed_unit);
    ImGui::Text("Integrator history %.0f frames   camera exposure %.3g", p.output_frames, p.camera_exposure);

    if (ImGui::TreeNodeEx("Neighbor Cells", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::TextDisabled("The 8 neighbor cells around the pixel at its base cascade (the lookup itself draws random offsets from its kernel).");
        if (ImGui::BeginTable("##taps", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
        {
            ImGui::TableSetupColumn("Tap");
            ImGui::TableSetupColumn("Offset");
            ImGui::TableSetupColumn("Weight");
            ImGui::TableSetupColumn("Samples");
            ImGui::TableSetupColumn("Exposed L");
            ImGui::TableHeadersRow();
            for (u32 t = 0; t < 8; ++t)
            {
                RtLab2ProbeTap const & tap = p.taps[t];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%u", t);
                ImGui::TableNextColumn();
                ImGui::Text("%c%c%c", (t & 1u) ? '+' : '-', (t & 2u) ? '+' : '-', (t & 4u) ? '+' : '-');
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", tap.weight);
                ImGui::TableNextColumn();
                if (tap.found)
                {
                    ImGui::Text("%.1f", tap.samples);
                    ImGui::TableNextColumn();
                    ImGui::Text("%.4g", tap.luminance * 3.14159265f * p.camera_exposure);
                }
                else
                {
                    ImGui::TextDisabled("no cell");
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("-");
                }
            }
            ImGui::EndTable();
        }
        ImGui::TreePop();
    }

    // Trends.
    if (probe.trend.size() > 1)
    {
        u32 const n = static_cast<u32>(probe.trend.size());
        std::vector<f32> x(n), frame_mean(n), history(n), lookup(n), output(n), samples(n), cap_curve(n), exposure(n), rays(n);
        f32 const unit_scale = s.exposed_units ? 3.14159265f * p.camera_exposure : 1.0f;
        for (u32 i = 0; i < n; ++i)
        {
            RtLab2ProbeTrend const & t = probe.trend[i];
            x[i] = static_cast<f32>(i) - static_cast<f32>(n - 1);
            frame_mean[i] = t.frame_mean * unit_scale;
            history[i] = t.history * unit_scale;
            lookup[i] = t.lookup * unit_scale;
            output[i] = t.output * unit_scale;
            samples[i] = t.samples;
            cap_curve[i] = t.cap;
            exposure[i] = t.exposure;
            rays[i] = t.rays;
        }
        if (ImPlot::BeginPlot("Luminance##cell_probe_luminance", ImVec2(-1.0f, 150.0f)))
        {
            ImPlot::SetupAxes("frames ago", s.exposed_units ? "exposed L" : "L", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("history", x.data(), history.data(), static_cast<i32>(n));
            ImPlot::PlotLine("lookup", x.data(), lookup.data(), static_cast<i32>(n));
            ImPlot::PlotLine("output", x.data(), output.data(), static_cast<i32>(n));
            ImPlot::PlotScatter("frame mean", x.data(), frame_mean.data(), static_cast<i32>(n));
            ImPlot::EndPlot();
        }
        if (ImPlot::BeginPlot("Samples / Coverage##cell_probe_samples", ImVec2(-1.0f, 130.0f)))
        {
            ImPlot::SetupAxes("frames ago", "samples / pixels / rays", ImPlotAxisFlags_None, ImPlotAxisFlags_AutoFit);
            ImPlot::PlotLine("history samples", x.data(), samples.data(), static_cast<i32>(n));
            ImPlot::PlotLine("window", x.data(), cap_curve.data(), static_cast<i32>(n));
            ImPlot::PlotLine("exposure", x.data(), exposure.data(), static_cast<i32>(n));
            ImPlot::PlotLine("rays", x.data(), rays.data(), static_cast<i32>(n));
            ImPlot::EndPlot();
        }
    }
    ImGui::PopID();
}

void rt_lab2_probe_ui(RenderContext & render_context)
{
    auto & lab2 = render_context.rt_lab.lab2;
    auto & s = lab2.probe;
    ImGui::Checkbox("Enable Cell Probes", &s.enabled);
    ImGui::SetItemTooltip("Adds the probe pass to the RT Lab 2 graph (a rebuild). Pick up to two pixels with the buttons, then click in the viewport.\n"
                          "Every frame the hash grid cell under the pixel is read back and shown with its coverage, rays, history window and radiance.");
    if (!s.enabled)
    {
        return;
    }
    if (!lab2.active)
    {
        ImGui::TextDisabled("RT Lab 2 is not the active denoiser: the probes show nothing.");
    }
    ImGui::SameLine();
    ImGui::Checkbox("Exposed Units In Plots", &s.exposed_units);
    ImGui::SetItemTooltip("Plot luminance x camera exposure x pi (the INDIRECT_DIFFUSE debug view's units) instead of scene units.");
    if (s.picking >= 0)
    {
        ImGui::TextColored(PROBE_COLORS[s.picking], "Click a pixel in the viewport for probe %s.", PROBE_NAMES[s.picking]);
    }
    RtLab2Probe const & a = s.probes[0];
    RtLab2Probe const & b = s.probes[1];
    if (a.valid && b.valid && a.latest.cell && b.latest.cell && a.latest.cell_index == b.latest.cell_index)
    {
        ImGui::TextDisabled("A and B read the same cell.");
    }
    if (ImGui::BeginTable("##cell_probes", 2, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame))
    {
        ImGui::TableNextRow();
        for (u32 index = 0; index < RT_LAB2_PROBE_UI_COUNT; ++index)
        {
            ImGui::TableSetColumnIndex(static_cast<i32>(index));
            probe_column(render_context, s, index);
        }
        ImGui::EndTable();
    }
}
