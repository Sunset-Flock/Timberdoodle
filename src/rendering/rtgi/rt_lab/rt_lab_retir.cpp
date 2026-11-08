#include "rt_lab_retir.hpp"
#include "rt_lab_internal.hpp"

#include "rt_lab3.inl"

#include <cstring>

#include <imgui.h>
#include <nlohmann/json.hpp>

auto rt_lab_retir_to_lab3_settings(RtLabRetirSettings const & retir) -> RtLab3Settings
{
    RtLab3Settings settings = rt_lab3_default_settings();
    settings.temporal_enabled = retir.temporal_enabled;
    settings.temporal_max_m = retir.temporal_max_m;
    settings.temporal_plane_tolerance = retir.temporal_plane_tolerance;
    settings.temporal_normal_threshold = retir.temporal_normal_threshold;
    settings.temporal_visibility = retir.temporal_visibility;
    settings.temporal_max_age = retir.temporal_max_age;
    settings.temporal_max_age_jitter = retir.temporal_max_age_jitter;
    settings.temporal_partial_scale = retir.temporal_partial_scale;
    settings.jacobian_max = retir.jacobian_max;
    settings.min_reconnection_distance = retir.min_reconnection_distance;
    settings.spatial_passes = 0;
    return settings;
}

void rt_lab_retir_temporal_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & AT = RtLab3TemporalH::Info::AT;
    RtLab3Settings settings = rt_lab_retir_to_lab3_settings(render_context->rt_lab.retir.settings);
    settings.reset = render_context->rt_lab.lab3.frame_reset ? 1 : 0;
    auto alloc = ti.allocator->allocate(sizeof(RtLab3Settings)).value();
    std::memcpy(alloc.host_address, &settings, sizeof(RtLab3Settings));
    RtLab3TemporalPush push = {};
    push.settings = alloc.device_address;
    rt_lab_dispatch(ti, render_context, push, AT.out_a, RT_LAB3_PASS_X, RenderTimes::index<"RT_LAB", "LAB3_TEMPORAL">(), rt_lab_retir_temporal_compile_info().name);
}

void rt_lab_retir_ui(RenderContext & render_context)
{
    auto & retir = render_context.rt_lab.retir;
    auto & s = retir.settings;
    ImGui::PushID("rt lab retir");
    if (ImGui::Button("Reset ReTIR Settings"))
    {
        static_cast<RtLabRetirConfig &>(retir) = RtLabRetirConfig{};
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%8.1f us", static_cast<f32>(render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "LAB3_TEMPORAL">())) * 0.001f);
    ImGui::TextDisabled("Temporal only. The reservoir is processed like the ReSTIR temporal pass, but the rays are added one at a time and the estimate");
    ImGui::TextDisabled("after every ray is kept: the denoiser gets all of them (mean, fresh ray count as weight).");
    bool temporal = s.temporal_enabled != 0;
    if (ImGui::Checkbox("Temporal", &temporal))
    {
        s.temporal_enabled = temporal ? 1 : 0;
    }
    ImGui::SliderFloat("Max M", &s.temporal_max_m, 1.0f, 128.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Cap of the confidence of the reprojected reservoir. Lower = the history forgets faster (less lag, more noise).");
    ImGui::SliderInt("Sample Age Cap (frames)", &s.temporal_max_age, 0, 256, s.temporal_max_age == 0 ? "off" : "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("A reservoir whose sample would get older than the cap is not reused. 0 = off.");
    ImGui::BeginDisabled(s.temporal_max_age == 0);
    ImGui::SliderFloat("Age Cap Jitter", &s.temporal_max_age_jitter, 0.0f, 1.0f, "%.2f");
    ImGui::EndDisabled();
    ImGui::SliderFloat("Plane Tolerance", &s.temporal_plane_tolerance, 0.1f, 8.0f, "%.2f px");
    ImGui::SliderFloat("Normal Threshold", &s.temporal_normal_threshold, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Partial Match M Scale", &s.temporal_partial_scale, 0.0f, 1.0f, "%.2f");
    bool visibility = s.temporal_visibility != 0;
    if (ImGui::Checkbox("Visibility Ray", &visibility))
    {
        s.temporal_visibility = visibility ? 1 : 0;
    }
    ImGui::SliderFloat("Jacobian Max", &s.jacobian_max, 1.0f, 100.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Min Reconnection Distance", &s.min_reconnection_distance, 0.0f, 1.0f, "%.3f m");
    ImGui::PopID();
}

void rt_lab_retir_apply_json(RtLabRetirState & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys)
{
    auto number = [](nlohmann::json const & value) -> f64
    {
        return value.is_boolean() ? (value.get<bool>() ? 1.0 : 0.0) : value.get<f64>();
    };
    for (auto const & [key, value] : json.items())
    {
        if (false)
        {
        }
#define RT_LAB_RETIR_JSON_FIELD(FIELD)                                                         \
    else if (key == #FIELD)                                                                    \
    {                                                                                          \
        state.settings.FIELD = static_cast<decltype(state.settings.FIELD)>(number(value));     \
    }
        RT_LAB_RETIR_JSON_FIELD(temporal_enabled)
        RT_LAB_RETIR_JSON_FIELD(temporal_max_m)
        RT_LAB_RETIR_JSON_FIELD(temporal_plane_tolerance)
        RT_LAB_RETIR_JSON_FIELD(temporal_normal_threshold)
        RT_LAB_RETIR_JSON_FIELD(temporal_visibility)
        RT_LAB_RETIR_JSON_FIELD(temporal_max_age)
        RT_LAB_RETIR_JSON_FIELD(temporal_max_age_jitter)
        RT_LAB_RETIR_JSON_FIELD(temporal_partial_scale)
        RT_LAB_RETIR_JSON_FIELD(jacobian_max)
        RT_LAB_RETIR_JSON_FIELD(min_reconnection_distance)
#undef RT_LAB_RETIR_JSON_FIELD
        else
        {
            unknown_keys.push_back(key);
        }
    }
}
