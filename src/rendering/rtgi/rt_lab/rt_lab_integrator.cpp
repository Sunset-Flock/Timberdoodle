#include "rt_lab_integrator.hpp"
#include "rt_lab_internal.hpp"

#include <algorithm>
#include <cstring>

#include <imgui.h>
#include <nlohmann/json.hpp>

static void rt_lab_integrator_callback(daxa::TaskInterface ti, RenderContext * render_context, RtLabIntegratorState * state)
{
    auto const & AT = RtLabIntegratorH::Info::AT;
    RtLabIntegratorSettings settings = state->settings;
    settings.reset = (render_context->render_data.rt_lab_settings.reset_history != 0 || state->clear_requested) ? 1 : 0;
    state->clear_requested = false;
    settings.window_radius = std::clamp(settings.window_radius, 0, RT_LAB_INTEGRATOR_MAX_WINDOW_RADIUS);
    auto alloc = ti.allocator->allocate(sizeof(RtLabIntegratorSettings)).value();
    std::memcpy(alloc.host_address, &settings, sizeof(RtLabIntegratorSettings));
    RtLabIntegratorPush push = {};
    push.settings = alloc.device_address;
    rt_lab_dispatch(ti, render_context, push, AT.integrated, RT_LAB_INTEGRATOR_X, RenderTimes::index<"RT_LAB", "INTEGRATOR">(), rt_lab_integrator_compile_info().name);
}

auto rt_lab_add_integrator(
    TasksRtgiInfo const & info,
    RtLabIntegratorState & state,
    daxa::TaskImageView lab_depth,
    daxa::TaskImageView lab_face_normal,
    daxa::TaskImageView input,
    daxa::Extent3D size,
    char const * history_name) -> daxa::TaskImageView
{
    auto & render_context = info.render_context;
    state.clear_requested = true; // the graph's persistent history is undefined
    auto integrated = info.tg.create_task_image({
        .format = daxa::Format::R32G32B32A32_SFLOAT, // .rgb radiance (RTGI_RADIANCE_SCALE units overflow f16), .a frames
        .size = size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = history_name,
    });
    info.tg.add_task(daxa::HeadTask<RtLabIntegratorH::Info>()
        .head_views(RtLabIntegratorH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .lab_depth_history = lab_depth.previous(),
            .lab_face_normal_history = lab_face_normal.previous(),
            .input = input,
            .history = integrated.previous(),
            .integrated = integrated.current(),
        })
        .executes(rt_lab_integrator_callback, &render_context, &state));
    return integrated.current();
}

auto rt_lab_integrator_frame_update(RtLabIntegratorState & state) -> bool
{
    bool const rebuild = state.enabled != state.built_enabled;
    state.built_enabled = state.enabled;
    return rebuild;
}

void rt_lab_integrator_ui(RtLabIntegratorState & state, RenderContext & render_context, char const * id)
{
    auto & s = state.settings;
    ImGui::PushID(id);
    ImGui::Checkbox("Temporal Resolve", &state.enabled);
    ImGui::SetItemTooltip("TAA / temporal analyzer hybrid on the final per pixel radiance, right before the resolve:\n"
                          "the window variance of this frame decides how much history is kept (quiet = trust the new value, noisy = keep the\n"
                          "history and move the new value toward the window mean); the history is clamped into mean +- gamma sigma of the window.\n"
                          "Graph rebuild when toggled.");
    ImGui::SameLine();
    ImGui::TextDisabled("%8.1f us", static_cast<f32>(render_context.render_times.get_average(RenderTimes::index<"RT_LAB", "INTEGRATOR">())) * 0.001f);
    if (ImGui::Button("Reset Temporal Resolve Settings"))
    {
        s = rt_lab_integrator_default_settings();
    }
    ImGui::BeginDisabled(!state.enabled);
    ImGui::SliderInt("Window Radius", &s.window_radius, 0, RT_LAB_INTEGRATOR_MAX_WINDOW_RADIUS);
    ImGui::SliderFloat("Variance Gamma", &s.variance_gamma, 0.0f, 8.0f, "%.2f");
    ImGui::SetItemTooltip("Half width of the history clamp box in window standard deviations. 0 = no clamp.");
    ImGui::SliderFloat("Alpha Min", &s.alpha_min, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Blend factor of the new value at the noisiest windows (keeps most history).");
    ImGui::SliderFloat("Alpha Max", &s.alpha_max, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Blend factor of the new value at quiet windows (trusts the new value).");
    ImGui::SliderFloat("Variance Reference", &s.variance_reference, 0.01f, 2.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Window relative standard deviation at which the alpha reaches Alpha Min.");
    ImGui::SliderFloat("Spatial Strength", &s.spatial_strength, 0.0f, 1.0f, "%.2f");
    ImGui::SetItemTooltip("How far a noisy window moves the new value toward the window mean.");
    ImGui::SliderInt("Max Frames", &s.max_frames, 1, 512);
    ImGui::SetItemTooltip("Cap of the history length (the blend factor never goes below 1 / length).");
    ImGui::SliderFloat("Plane Tolerance", &s.plane_tolerance, 0.1f, 8.0f, "%.2f px");
    ImGui::SliderFloat("Normal Threshold", &s.normal_threshold, 0.0f, 1.0f, "%.3f");
    ImGui::EndDisabled();
    ImGui::PopID();
}

void rt_lab_integrator_apply_json(RtLabIntegratorState & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys)
{
    auto number = [](nlohmann::json const & value) -> f64
    {
        return value.is_boolean() ? (value.get<bool>() ? 1.0 : 0.0) : value.get<f64>();
    };
    for (auto const & [key, value] : json.items())
    {
        if (key == "enabled")
        {
            state.enabled = number(value) != 0.0;
        }
#define RT_LAB_INTEGRATOR_JSON_FIELD(FIELD)                                                    \
    else if (key == #FIELD)                                                                    \
    {                                                                                          \
        state.settings.FIELD = static_cast<decltype(state.settings.FIELD)>(number(value));     \
    }
        RT_LAB_INTEGRATOR_JSON_FIELD(window_radius)
        RT_LAB_INTEGRATOR_JSON_FIELD(variance_gamma)
        RT_LAB_INTEGRATOR_JSON_FIELD(alpha_min)
        RT_LAB_INTEGRATOR_JSON_FIELD(alpha_max)
        RT_LAB_INTEGRATOR_JSON_FIELD(variance_reference)
        RT_LAB_INTEGRATOR_JSON_FIELD(spatial_strength)
        RT_LAB_INTEGRATOR_JSON_FIELD(max_frames)
        RT_LAB_INTEGRATOR_JSON_FIELD(plane_tolerance)
        RT_LAB_INTEGRATOR_JSON_FIELD(normal_threshold)
#undef RT_LAB_INTEGRATOR_JSON_FIELD
        else
        {
            unknown_keys.push_back(key);
        }
    }
}
