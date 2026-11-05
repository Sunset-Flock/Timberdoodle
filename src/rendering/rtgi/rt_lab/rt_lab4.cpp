#include "rt_lab4.hpp"
#include "rt_lab_internal.hpp"

#include "rt_lab2.hpp"
#include "rt_lab3.hpp"
#include "rt_lab_resolve.inl"

#include <algorithm>

#include <imgui.h>
#include <nlohmann/json.hpp>

auto rt_lab4_frame_update(RenderContext & render_context, bool reference_phase) -> bool
{
    auto & lab4 = render_context.rt_lab.lab4;
    lab4.cell_weight_mode = std::clamp(lab4.cell_weight_mode, 0, 2);
    lab4.cell_source = std::clamp(lab4.cell_source, 0, 1);
    lab4.active = lab4.enabled && !reference_phase && render_context.render_data.rt_lab_settings.enabled != 0;
    bool const rebuild = lab4.active != lab4.built_active || lab4.cell_weight_mode != lab4.built_cell_weight_mode || lab4.cell_source != lab4.built_cell_source;
    lab4.built_active = lab4.active;
    lab4.built_cell_weight_mode = lab4.cell_weight_mode;
    lab4.built_cell_source = lab4.cell_source;
    return rebuild;
}

auto tasks_rt_lab4_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult
{
    auto & render_context = info.render_context;
    auto & lab4 = render_context.rt_lab.lab4;

    // ReSTIR stage: ray redistribution, trace, reservoir reuse, ray filter. cell_input = the filtered estimate with the per pixel cell weight in .a.
    auto const stage = rt_lab3_add_restir(info, std::clamp(lab4.cell_weight_mode, 0, 2), std::clamp(lab4.cell_source, 0, 1));
    // Cell grid + the temporal resolve at the end.
    auto const integrated = rt_lab2_add_cell_stage(info, stage.lab_depth, stage.lab_face_normal, stage.cell_input, stage.size);

    // The old lab's resolve: RTGI_RADIANCE_SCALE divided out, at half res the bilateral upsample to the full res output.
    auto resolved = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = stage.full_size,
        .name = "rt_lab_resolved_diffuse",
    });
    info.tg.add_task(daxa::HeadTask<RtLabResolveH::Info>()
        .head_views(RtLabResolveH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .view_cam_depth = info.view_cam_depth,
            .view_cam_face_normals = info.view_cam_face_normals,
            .lab_depth = stage.lab_depth.current(),
            .lab_face_normal = stage.lab_face_normal.current(),
            .lab_diffuse = integrated,
            .resolved = resolved,
            .lab_split_difference = daxa::NullTaskImage,
            .split_difference_resolved = daxa::NullTaskImage,
        })
        .executes(rt_lab_resolve_callback, &render_context, false, true));
    rt_lab_add_analysis(info, resolved);

    rt_lab_add_test_passes(info, resolved, stage.full_size);

    return TasksRtgiMainResult{
        .opaque_diffuse = resolved,
        .opaque_specular = daxa::NullTaskImage,
    };
}

// The ReSTIR -> denoiser glue of a preset's settings (live copy of the preset's config, see rt_lab_preset_config_ui).
void rt_lab4_ui(RenderContext & render_context)
{
    auto & lab4 = render_context.rt_lab.lab4;
    ImGui::PushID("rt lab 4");
    char const * sources[] = {"ReSTIR estimate (after ray filter)", "Fresh rays only (independent samples)"};
    ImGui::Combo("Denoiser Input", &lab4.cell_source, sources, 2);
    ImGui::SetItemTooltip("What the denoiser behind ReSTIR (hash grid cells or the classic temporal pass) integrates.\n"
                          "ReSTIR estimate: the filtered per pixel result of the reservoir reuse. Neighboring pixels share samples (spatial reuse) and a pixel keeps one\n"
                          "sample for many frames (temporal reuse), so the denoiser sees a handful of samples copied into many pixels: blobs of about the reuse radius.\n"
                          "Fresh rays only: the mean of the rays the pixel shot THIS frame (each clamped against the ray filter ceiling): independent samples, the denoiser does the\n"
                          "temporal accumulation. ReSTIR then only drives the ray redistribution. Pixels without a fresh ray feed nothing.\n"
                          "Graph rebuild when changed.");
    ImGui::BeginDisabled(lab4.cell_source != RT_LAB3_CELL_SOURCE_RESTIR);
    char const * modes[] = {"One per pixel with a reservoir", "Fresh rays shot this frame", "Accumulated rays shot (ray history)"};
    ImGui::Combo("Input Sample Weight", &lab4.cell_weight_mode, modes, 3);
    ImGui::SetItemTooltip("Only for the ReSTIR estimate input: what a pixel's estimate counts as in the denoiser (the weight of its mean):\n"
                          "one sample per pixel that has a reservoir, the fresh rays the pixel shot this frame (pixels without a fresh ray feed nothing),\n"
                          "or the rays shot at it over time (ray history; the reuse correlates the samples, so this still overstates the information).\n"
                          "Graph rebuild when changed.");
    ImGui::EndDisabled();
    ImGui::TextDisabled("Hash grid blend: Max Frame Samples (Hash Grid section) caps how hard one frame pulls a cell.");
    ImGui::PopID();
}

void rt_lab4_apply_json(RtLab4State & state, nlohmann::json const & json, std::vector<std::string> & unknown_keys)
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
        else if (key == "cell_source")
        {
            state.cell_source = static_cast<i32>(number(value));
        }
        else if (key == "cell_weight_mode")
        {
            state.cell_weight_mode = static_cast<i32>(number(value));
        }
        else
        {
            unknown_keys.push_back(key);
        }
    }
}
