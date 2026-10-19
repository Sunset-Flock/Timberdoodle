#include "rt_lab.hpp"

#include "rt_lab_trace.inl"
#include "rt_lab_temporal.inl"
#include "rt_lab_anti_lag.inl"
#include "rt_lab_coarse.inl"
#include "rt_lab_blur.inl"
#include "rt_lab_resolve.inl"
#include "rt_lab_convergence.inl"
#include "rt_lab_measure.inl"

#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

auto rt_lab_default_settings(RtLabPreset preset) -> RtLabSettings
{
    if (preset == RT_LAB_PRESET_PROGRESSIVE)
    {
        // Reference: full res, fp32 history accumulating up to 512 samples. Light spatial: 2 a-trous passes with a tight
        // variance guide (1 sigma) and no coarse fallback, so the reference converges faster with little blur bias.
        return RtLabSettings{
            .enabled                       = 0,
            .half_res                      = 0,
            .animate_noise                 = 1,
            .hit_ao_enabled                = 1,
            .hit_ao_range                  = 1.0f,
            .redistribution_enabled        = 0,
            .ray_budget                    = 1.0f,
            .variance_laziness_enabled     = 1,
            .variance_laziness_target_error = 0.2f,
            .variance_laziness_min_samples = 4.0f,
            .variance_laziness_perceptual_floor = 0.05f,
            .min_ray_budget                = 0.125f,
            .max_rays_per_pixel            = 12,
            .fast_convergence_samples      = 32.0f,
            .temporal_gradient_enabled     = 0,
            .temporal_gradient_blur_radius = 10,
            .temporal_gradient_all_base_rays = 0,
            .temporal_gradient_dilation_radius = 16,
            .temporal_gradient_scale       = 0.75f,
            .temporal_gradient_cap_mode = 0,
            .temporal_gradient_cap_k = 0.05f,
            .temporal_gradient_cap_min = 2.0f,
            .temporal_gradient_settle_frames = 8.0f,
            .temporal_gradient_settle_tolerance = 0.02f,
            .temporal_gradient_rescale = 0,
            .temporal_gradient_rescale_strength = 1.0f,
            .temporal_gradient_pixel_relative = 0,
            .temporal_gradient_history_alpha = 1.0f,
            .temporal_gradient_history_shock = 0.3f,
            .temporal_gradient_deadzone = 0.0f,
            .temporal_gradient_kalman_q = 1e-4f,
            .temporal_gradient_kalman_gate = 3.0f,
            .temporal_gradient_kalman_k = 20.0f,
            .temporal_enabled              = 1,
            .temporal_progressive          = 1,
            .max_history_frames            = 64,
            .progressive_max_samples       = 512,
            .disocclusion_plane_tolerance  = 2.0f,
            .disocclusion_normal_threshold = 0.8f,
            .bicubic_reprojection          = 1,
            .occluder_aware_soft_normalize = 1,
            .variance_guide_enabled        = 1,
            .variance_guide_sigmas         = 1.0f,
            .variance_guide_heavy_tail     = 0,
            .variance_guide_tail_nu        = 3.0f,
            .variance_guide_count_cap_enabled = 0,
            .variance_guide_count_cap      = 64.0f,
            .variance_guide_gradient       = 0,
            .split_history_enabled         = 1,
            .spatial_enabled               = 1,
            .spatial_mode                  = 1,
            .atrous_passes                 = 2,
            .blur_radius                   = 8,
            .blur_plane_tolerance          = 1.5f,
            .blur_normal_power             = 8.0f,
            .fine_target_samples           = 32000.0f,
            .coarse_enabled                = 0,
            .coarse_atrous_passes          = 5,
            .coarse_distance_tolerance     = 2.0f,
            .coarse_normal_power           = 1.0f,
            .coarse_variance_guide_enabled = 1,
            .coarse_variance_guide_sigmas  = 3.0f,
            .coarse_target_samples         = 32000.0f,
            .coarse_max_blend              = 1.0f,
            .coarse_anti_lag_enabled          = 0,
            .coarse_anti_lag_blur_radius      = 8,
            .coarse_anti_lag_history_frames   = 4,
            .coarse_anti_lag_variance_blend   = 64.0f,
            .ray_filter_mode               = RT_LAB_RAY_FILTER_OFF,
            .ray_filter_radius             = 2,
            .ray_filter_sigmas             = 3.0f,
            .ray_filter_clamp_low          = 0,
            .ray_filter_tolerance          = 7.0f,
            .firefly_weight_enabled        = 0,
            .ray_length_guide_mode                 = 0,
            .ray_length_guide_radius_px            = 128.0f,
            .ray_length_guide_distance_falloff     = 0.5f,
            .ray_length_guide_similarity_tolerance = 0.25f,
            .reset_history                 = 0,
        };
    }
    // Custom: full res, short fp16 history + spatial.
    return RtLabSettings{
        .enabled                       = 0,
        .half_res                      = 0,
        .animate_noise                 = 1,
        .hit_ao_enabled                = 1,
        .hit_ao_range                  = 1.0f,
        .redistribution_enabled        = 1,
        .ray_budget                    = 1.0f,
        .variance_laziness_enabled     = 1,
        .variance_laziness_target_error = 0.2f,
        .variance_laziness_min_samples = 4.0f,
        .variance_laziness_perceptual_floor = 0.05f,
        .min_ray_budget                = 0.125f,
        .max_rays_per_pixel            = 12,
        .fast_convergence_samples      = 32.0f,
        .temporal_gradient_enabled     = 1,
        .temporal_gradient_blur_radius = 10,
        .temporal_gradient_all_base_rays = 0,
        .temporal_gradient_dilation_radius = 16,
        .temporal_gradient_scale       = 0.75f,
        .temporal_gradient_cap_mode = RT_LAB_GRADIENT_CAP_KALMAN, // 2026-10: was SETTLE (+ deadzone / smoothing)
        .temporal_gradient_cap_k = 0.05f,
        .temporal_gradient_cap_min = 2.0f,
        .temporal_gradient_settle_frames = 8.0f,
        .temporal_gradient_settle_tolerance = 0.02f,
        .temporal_gradient_rescale = 1,
        .temporal_gradient_rescale_strength = 1.0f, // Kalman: the correction is already shrunk (SETTLE mode: 0.2, more = ugly artifacts)
        .temporal_gradient_pixel_relative = 0,
        .temporal_gradient_history_alpha = 0.33f, // 2026-10: smoothing + deadzone vs whole screen jitter, sharp cap for shocks
        .temporal_gradient_history_shock = 0.3f,
        .temporal_gradient_deadzone = 0.3f,
        .temporal_gradient_kalman_q = 1e-4f,
        .temporal_gradient_kalman_gate = 3.0f,
        .temporal_gradient_kalman_k = 20.0f,
        .temporal_enabled              = 1,
        .temporal_progressive          = 0,
        .max_history_frames            = 64,
        .progressive_max_samples       = 512,
        .disocclusion_plane_tolerance  = 2.0f,
        .disocclusion_normal_threshold = 0.8f,
        .bicubic_reprojection          = 1,
        .occluder_aware_soft_normalize = 1,
        .variance_guide_enabled        = 1,
        .variance_guide_sigmas         = 2.0f,
        .variance_guide_heavy_tail     = 0,
        .variance_guide_tail_nu        = 3.0f,
        .variance_guide_count_cap_enabled = 0,
        .variance_guide_count_cap      = 64.0f,
        .variance_guide_gradient       = 0,
        .split_history_enabled         = 1,
        .spatial_enabled               = 1,
        .spatial_mode                  = 1,
        .atrous_passes                 = 5,
        .blur_radius                   = 8,
        .blur_plane_tolerance          = 1.5f,
        .blur_normal_power             = 8.0f,
        .fine_target_samples           = 32000.0f,
        .coarse_enabled                = 1,
        .coarse_atrous_passes          = 5,
        .coarse_distance_tolerance     = 2.0f,
        .coarse_normal_power           = 1.0f,
        .coarse_variance_guide_enabled = 1,
        .coarse_variance_guide_sigmas  = 3.0f,
        .coarse_target_samples        = 32000.0f,
        .coarse_max_blend              = 1.0f,
        .coarse_anti_lag_enabled          = RT_LAB_ANTI_LAG_REDUCE_SAMPLES | RT_LAB_ANTI_LAG_KALMAN, // 2026-10: was BLEND_ONLY | MSE_OPTIMAL
        .coarse_anti_lag_blur_radius      = 8,
        .coarse_anti_lag_history_frames   = 4,
        .coarse_anti_lag_variance_blend   = 64.0f, // MSE k (measured best of 1 / 4 / 16 / 64 on the sun sweep)
        .ray_filter_mode               = RT_LAB_RAY_FILTER_PERCEPTUAL,
        .ray_filter_radius             = 2,
        .ray_filter_sigmas             = 3.0f,
        .ray_filter_clamp_low          = 0,
        .ray_filter_tolerance          = 7.0f,
        .firefly_weight_enabled        = 0,
        .ray_length_guide_mode                 = RT_LAB_RAY_LENGTH_GUIDE_SIMILARITY, // radius weighting (user 2026-10)
        .ray_length_guide_radius_px            = 128.0f,
        .ray_length_guide_distance_falloff     = 0.5f,
        .ray_length_guide_similarity_tolerance = 0.25f,
        .reset_history                 = 0,
    };
}

void rt_lab_init(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    for (i32 p = 0; p < RT_LAB_PRESET_COUNT; ++p)
    {
        lab.presets[p] = rt_lab_default_settings(static_cast<RtLabPreset>(p));
    }
    render_context.render_data.rt_lab_settings = lab.presets[lab.active_preset];
}

static void rt_lab_destroy_capture_images(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    for (daxa::ExternalTaskImage * image : {&lab.progressive_image, &lab.custom_image, &lab.predicted_error_image})
    {
        if (!image->id().is_empty())
        {
            render_context.gpu_context->device.destroy_image(image->id());
            image->set_image(daxa::ImageId{});
        }
    }
    lab.live_analysis = false;
}

static void rt_lab_destroy_measure_buffers(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    for (daxa::ExternalTaskBuffer * buffer : {&lab.measure_frame_histogram, &lab.measure_run_histograms, &lab.measure_lag_run})
    {
        if (!buffer->id().is_empty())
        {
            render_context.gpu_context->device.destroy_buffer(buffer->id());
            buffer->set_buffer(daxa::BufferId{});
        }
    }
    lab.measure_run_capacity = 0;
    lab.measure_lag_capacity = 0;
}

static void rt_lab_destroy_history_backups(RenderContext & render_context);

void rt_lab_destroy(RenderContext & render_context)
{
    rt_lab_destroy_capture_images(render_context);
    rt_lab_destroy_measure_buffers(render_context);
    rt_lab_destroy_history_backups(render_context);
}

// == Convergence measurement analytics ==

static auto rt_lab_measure_bin_width(RtLabMeasurement const & m) -> f64
{
    return (static_cast<f64>(m.log_max) - static_cast<f64>(m.log_min)) / static_cast<f64>(RT_LAB_MEASURE_BINS);
}

auto rt_lab_measure_bin_edge(RtLabMeasurement const & m, u32 k) -> f64
{
    return std::pow(10.0, static_cast<f64>(m.log_min) + static_cast<f64>(k) * rt_lab_measure_bin_width(m));
}

auto rt_lab_measure_fraction_below(RtLabMeasurement const & m, i32 metric, i32 frame, f64 value) -> f64
{
    if (frame < 0 || frame >= m.frames || m.denominator[metric] == 0 || !(value > 0.0))
    {
        return 0.0;
    }
    f64 const t = (std::log10(value) - static_cast<f64>(m.log_min)) / rt_lab_measure_bin_width(m);
    u32 const k = static_cast<u32>(std::clamp(std::floor(t), 0.0, static_cast<f64>(RT_LAB_MEASURE_BINS)));
    return static_cast<f64>(m.cumulative[metric][static_cast<size_t>(frame) * (RT_LAB_MEASURE_BINS + 1) + k]) / static_cast<f64>(m.denominator[metric]);
}

auto rt_lab_measure_percentile(RtLabMeasurement const & m, i32 metric, i32 frame, f64 fraction) -> f64
{
    if (frame < 0 || frame >= m.frames || m.denominator[metric] == 0)
    {
        return std::numeric_limits<f64>::infinity();
    }
    f64 const target = fraction * static_cast<f64>(m.denominator[metric]);
    u32 const * cumulative = m.cumulative[metric].data() + static_cast<size_t>(frame) * (RT_LAB_MEASURE_BINS + 1);
    for (u32 k = 0; k <= RT_LAB_MEASURE_BINS; ++k)
    {
        if (static_cast<f64>(cumulative[k]) >= target)
        {
            return rt_lab_measure_bin_edge(m, k);
        }
    }
    return std::numeric_limits<f64>::infinity(); // that many tiles never got below the histogram range
}

auto rt_lab_measure_frames_to_reach(RtLabMeasurement const & m, i32 metric, f64 value, f64 fraction) -> i32
{
    // Walk backwards: the first frame from which on the fraction stays reached until the end of the run.
    i32 first = -1;
    for (i32 frame = m.frames - 1; frame >= 0; --frame)
    {
        if (rt_lab_measure_fraction_below(m, metric, frame, value) < fraction)
        {
            break;
        }
        first = frame;
    }
    return first < 0 ? -1 : first + 1; // frames of accumulation
}

auto rt_lab_measure_lag(RtLabMeasurement const & m, f64 tolerance) -> RtLabLagSummary
{
    u32 const tiles = static_cast<u32>(std::max(m.lag_tiles_x, 0) * std::max(m.lag_tiles_y, 0));
    if (tiles == 0 || m.lag_values.size() < static_cast<size_t>(m.frames) * tiles || m.frames <= RT_LAB_MEASURE_LAG_FINAL_FRAMES)
    {
        return m.lag;
    }
    RtLabLagSummary summary = {};
    summary.valid = true;
    summary.tolerance = tolerance;
    summary.start_frame = std::clamp(m.lag_start_frame, 0, m.frames - 1);
    summary.mean_curve.assign(static_cast<size_t>(m.frames), 0.0f);
    for (i32 f = 0; f < m.frames; ++f)
    {
        f64 sum = 0.0;
        u32 count = 0;
        for (u32 t = 0; t < tiles; ++t)
        {
            f32 const v = m.lag_values[static_cast<size_t>(f) * tiles + t];
            if (v >= 0.0f)
            {
                sum += v;
                count += 1;
            }
        }
        summary.mean_curve[f] = count > 0 ? static_cast<f32>(sum / count) : 0.0f;
    }
    i32 const final_begin = m.frames - RT_LAB_MEASURE_LAG_FINAL_FRAMES;
    i32 const window = std::max(final_begin - summary.start_frame, 1); // frames the settle search covers
    std::vector<f64> settle_frames = {};
    std::vector<u32> settled_count(static_cast<size_t>(window), 0u);
    u32 never = 0;
    for (u32 tile = 0; tile < tiles; ++tile)
    {
        auto const value = [&](i32 frame) -> f64 { return static_cast<f64>(m.lag_values[static_cast<size_t>(frame) * tiles + tile]); };
        f64 final_sum = 0.0;
        bool geometry = true;
        for (i32 f = final_begin; f < m.frames; ++f)
        {
            geometry = geometry && value(f) >= 0.0;
            final_sum += value(f);
        }
        if (!geometry)
        {
            continue;
        }
        summary.tiles += 1;
        f64 const final_value = final_sum / static_cast<f64>(RT_LAB_MEASURE_LAG_FINAL_FRAMES);
        f64 const scale = std::max(final_value, 1e-6);
        // Changed: away from the final value by more than 2 x tolerance at some frame after the start.
        bool changed = false;
        for (i32 f = summary.start_frame; f < final_begin && !changed; ++f)
        {
            changed = value(f) >= 0.0 && std::abs(value(f) - final_value) / scale > 2.0 * tolerance;
        }
        if (!changed)
        {
            continue;
        }
        summary.changed_tiles += 1;
        // Settle frame: the first frame from which on it stays within the tolerance (walk backwards).
        i32 settle = final_begin;
        for (i32 f = final_begin - 1; f >= summary.start_frame; --f)
        {
            if (value(f) < 0.0 || std::abs(value(f) - final_value) / scale > tolerance)
            {
                break;
            }
            settle = f;
        }
        if (settle >= final_begin)
        {
            never += 1;
            continue;
        }
        settle_frames.push_back(static_cast<f64>(settle - summary.start_frame));
        for (i32 f = settle - summary.start_frame; f < window; ++f)
        {
            settled_count[static_cast<size_t>(f)] += 1;
        }
    }
    if (summary.changed_tiles > 0)
    {
        summary.never_settled = static_cast<f64>(never) / static_cast<f64>(summary.changed_tiles);
        summary.settled_curve.resize(static_cast<size_t>(window));
        for (i32 f = 0; f < window; ++f)
        {
            summary.settled_curve[static_cast<size_t>(f)] = static_cast<f32>(static_cast<f64>(settled_count[static_cast<size_t>(f)]) / static_cast<f64>(summary.changed_tiles));
        }
    }
    if (!settle_frames.empty())
    {
        std::sort(settle_frames.begin(), settle_frames.end());
        auto const percentile = [&](f64 q) { return settle_frames[std::min(static_cast<size_t>(q * static_cast<f64>(settle_frames.size())), settle_frames.size() - 1)]; };
        summary.settle_p50 = percentile(0.5);
        summary.settle_p90 = percentile(0.9);
        summary.settle_max = settle_frames.back();
    }
    return summary;
}

// Builds the cumulative histograms + denominators from m.rows and bumps the generation.
static void rt_lab_measure_finalize(RtLabMeasurement & m)
{
    for (u32 metric = 0; metric < RT_LAB_MEASURE_METRICS; ++metric)
    {
        m.denominator[metric] = 0;
        m.cumulative[metric].assign(static_cast<size_t>(m.frames) * (RT_LAB_MEASURE_BINS + 1), 0u);
        for (i32 frame = 0; frame < m.frames; ++frame)
        {
            u32 const * row = m.rows.data() + static_cast<size_t>(frame) * RT_LAB_MEASURE_ROW_SIZE + metric * RT_LAB_MEASURE_METRIC_STRIDE;
            m.denominator[metric] = std::max(m.denominator[metric], row[RT_LAB_MEASURE_TILES]);
            u32 * cumulative = m.cumulative[metric].data() + static_cast<size_t>(frame) * (RT_LAB_MEASURE_BINS + 1);
            cumulative[0] = row[RT_LAB_MEASURE_UNDER];
            for (u32 k = 0; k < RT_LAB_MEASURE_BINS; ++k)
            {
                cumulative[k + 1] = cumulative[k] + row[k];
            }
        }
    }
    static u32 generation = 0;
    m.generation = ++generation;
}

// == Measurement results: files ==
// <name>.rtlab: "RTLABRES" magic, u32 version, u32 header size, JSON metadata header, u32 row count, raw rows
// (u32 x RT_LAB_MEASURE_ROW_SIZE each).
static constexpr char RT_LAB_RESULT_MAGIC[8] = {'R', 'T', 'L', 'A', 'B', 'R', 'E', 'S'};
static constexpr u32 RT_LAB_RESULT_VERSION = 1;

static auto rt_lab_settings_to_json(RtLabSettings const & s) -> nlohmann::json
{
    return {
        {"half_res", s.half_res},
        {"animate_noise", s.animate_noise},
        {"hit_ao_enabled", s.hit_ao_enabled},
        {"hit_ao_range", s.hit_ao_range},
        {"redistribution_enabled", s.redistribution_enabled},
        {"ray_budget", s.ray_budget},
        {"min_ray_budget", s.min_ray_budget},
        {"max_rays_per_pixel", s.max_rays_per_pixel},
        {"fast_convergence_samples", s.fast_convergence_samples},
        {"variance_laziness_enabled", s.variance_laziness_enabled},
        {"variance_laziness_target_error", s.variance_laziness_target_error},
        {"variance_laziness_min_samples", s.variance_laziness_min_samples},
        {"variance_laziness_perceptual_floor", s.variance_laziness_perceptual_floor},
        {"temporal_gradient_enabled", s.temporal_gradient_enabled},
        {"temporal_gradient_blur_radius", s.temporal_gradient_blur_radius},
        {"temporal_gradient_dilation_radius", s.temporal_gradient_dilation_radius},
        {"temporal_gradient_all_base_rays", s.temporal_gradient_all_base_rays},
        {"temporal_gradient_scale", s.temporal_gradient_scale},
        {"temporal_gradient_cap_mode", s.temporal_gradient_cap_mode},
        {"temporal_gradient_cap_k", s.temporal_gradient_cap_k},
        {"temporal_gradient_cap_min", s.temporal_gradient_cap_min},
        {"temporal_gradient_settle_frames", s.temporal_gradient_settle_frames},
        {"temporal_gradient_settle_tolerance", s.temporal_gradient_settle_tolerance},
        {"temporal_gradient_rescale", s.temporal_gradient_rescale},
        {"temporal_gradient_rescale_strength", s.temporal_gradient_rescale_strength},
        {"temporal_gradient_pixel_relative", s.temporal_gradient_pixel_relative},
        {"temporal_gradient_history_alpha", s.temporal_gradient_history_alpha},
        {"temporal_gradient_history_shock", s.temporal_gradient_history_shock},
        {"temporal_gradient_deadzone", s.temporal_gradient_deadzone},
        {"temporal_gradient_kalman_q", s.temporal_gradient_kalman_q},
        {"temporal_gradient_kalman_gate", s.temporal_gradient_kalman_gate},
        {"temporal_gradient_kalman_k", s.temporal_gradient_kalman_k},
        {"temporal_enabled", s.temporal_enabled},
        {"temporal_progressive", s.temporal_progressive},
        {"max_history_frames", s.max_history_frames},
        {"progressive_max_samples", s.progressive_max_samples},
        {"disocclusion_plane_tolerance", s.disocclusion_plane_tolerance},
        {"disocclusion_normal_threshold", s.disocclusion_normal_threshold},
        {"bicubic_reprojection", s.bicubic_reprojection},
        {"occluder_aware_soft_normalize", s.occluder_aware_soft_normalize},
        {"variance_guide_enabled", s.variance_guide_enabled},
        {"variance_guide_sigmas", s.variance_guide_sigmas},
        {"variance_guide_heavy_tail", s.variance_guide_heavy_tail},
        {"variance_guide_tail_nu", s.variance_guide_tail_nu},
        {"variance_guide_count_cap_enabled", s.variance_guide_count_cap_enabled},
        {"variance_guide_count_cap", s.variance_guide_count_cap},
        {"variance_guide_gradient", s.variance_guide_gradient},
        {"split_history_enabled", s.split_history_enabled},
        {"spatial_enabled", s.spatial_enabled},
        {"spatial_mode", s.spatial_mode},
        {"atrous_passes", s.atrous_passes},
        {"blur_radius", s.blur_radius},
        {"blur_plane_tolerance", s.blur_plane_tolerance},
        {"blur_normal_power", s.blur_normal_power},
        {"fine_target_samples", s.fine_target_samples},
        {"coarse_enabled", s.coarse_enabled},
        {"coarse_atrous_passes", s.coarse_atrous_passes},
        {"coarse_distance_tolerance", s.coarse_distance_tolerance},
        {"coarse_normal_power", s.coarse_normal_power},
        {"coarse_variance_guide_enabled", s.coarse_variance_guide_enabled},
        {"coarse_variance_guide_sigmas", s.coarse_variance_guide_sigmas},
        {"coarse_target_samples", s.coarse_target_samples},
        {"coarse_max_blend", s.coarse_max_blend},
        {"coarse_anti_lag_enabled", s.coarse_anti_lag_enabled},
        {"coarse_anti_lag_blur_radius", s.coarse_anti_lag_blur_radius},
        {"coarse_anti_lag_history_frames", s.coarse_anti_lag_history_frames},
        {"coarse_anti_lag_variance_blend", s.coarse_anti_lag_variance_blend},
        {"ray_filter_mode", s.ray_filter_mode},
        {"ray_filter_radius", s.ray_filter_radius},
        {"ray_filter_sigmas", s.ray_filter_sigmas},
        {"ray_filter_clamp_low", s.ray_filter_clamp_low},
        {"ray_filter_tolerance", s.ray_filter_tolerance},
        {"firefly_weight_enabled", s.firefly_weight_enabled},
        {"ray_length_guide_mode", s.ray_length_guide_mode},
        {"ray_length_guide_radius_px", s.ray_length_guide_radius_px},
        {"ray_length_guide_distance_falloff", s.ray_length_guide_distance_falloff},
        {"ray_length_guide_similarity_tolerance", s.ray_length_guide_similarity_tolerance},
    };
}

static auto rt_lab_settings_from_json(nlohmann::json const & j) -> RtLabSettings
{
    RtLabSettings s = {};
    s.half_res = j.value("half_res", 0);
    s.animate_noise = j.value("animate_noise", 1);
    // Results saved before the setting: AO on hit was always applied with rtgi_settings.shading_ao_range (default 1).
    s.hit_ao_enabled = j.value("hit_ao_enabled", 1);
    s.hit_ao_range = j.value("hit_ao_range", 1.0f);
    s.redistribution_enabled = j.value("redistribution_enabled", 0); // results saved before ray redistribution: 1 ray per pixel
    s.ray_budget = j.value("ray_budget", 1.0f);
    s.min_ray_budget = j.value("min_ray_budget", 0.125f);
    s.max_rays_per_pixel = j.value("max_rays_per_pixel", 12);
    s.fast_convergence_samples = j.value("fast_convergence_samples", 32.0f);
    s.variance_laziness_enabled = j.value("variance_laziness_enabled", 0);
    s.variance_laziness_target_error = j.value("variance_laziness_target_error", 0.2f);
    s.variance_laziness_min_samples = j.value("variance_laziness_min_samples", 4.0f);
    s.variance_laziness_perceptual_floor = j.value("variance_laziness_perceptual_floor", 0.05f);
    s.temporal_gradient_enabled = j.value("temporal_gradient_enabled", 0); // results saved before the temporal gradient: off
    s.temporal_gradient_blur_radius = j.value("temporal_gradient_blur_radius", 3);
    s.temporal_gradient_dilation_radius = j.value("temporal_gradient_dilation_radius", 0); // results saved before the dilation: off
    s.temporal_gradient_all_base_rays = j.value("temporal_gradient_all_base_rays", 0); // results saved before: one ray per cell
    s.temporal_gradient_scale = j.value("temporal_gradient_scale", 1.0f);
    s.temporal_gradient_cap_mode = j.value("temporal_gradient_cap_mode", 0);
    s.temporal_gradient_cap_k = j.value("temporal_gradient_cap_k", 0.05f);
    s.temporal_gradient_cap_min = j.value("temporal_gradient_cap_min", 2.0f);
    s.temporal_gradient_settle_frames = j.value("temporal_gradient_settle_frames", 8.0f);
    s.temporal_gradient_settle_tolerance = j.value("temporal_gradient_settle_tolerance", 0.02f);
    s.temporal_gradient_rescale = j.value("temporal_gradient_rescale", 0);
    s.temporal_gradient_rescale_strength = j.value("temporal_gradient_rescale_strength", 1.0f);
    s.temporal_gradient_pixel_relative = j.value("temporal_gradient_pixel_relative", 0);
    s.temporal_gradient_history_alpha = j.value("temporal_gradient_history_alpha", 1.0f);
    s.temporal_gradient_history_shock = j.value("temporal_gradient_history_shock", 0.3f);
    s.temporal_gradient_deadzone = j.value("temporal_gradient_deadzone", 0.0f);
    s.temporal_gradient_kalman_q = j.value("temporal_gradient_kalman_q", 1e-4f);
    s.temporal_gradient_kalman_gate = j.value("temporal_gradient_kalman_gate", 3.0f);
    s.temporal_gradient_kalman_k = j.value("temporal_gradient_kalman_k", 20.0f);
    s.temporal_enabled = j.value("temporal_enabled", 1);
    s.temporal_progressive = j.value("temporal_progressive", 0);
    s.max_history_frames = j.value("max_history_frames", 32);
    s.progressive_max_samples = j.value("progressive_max_samples", 512);
    s.disocclusion_plane_tolerance = j.value("disocclusion_plane_tolerance", 2.0f);
    s.disocclusion_normal_threshold = j.value("disocclusion_normal_threshold", 0.8f);
    s.bicubic_reprojection = j.value("bicubic_reprojection", 0); // results saved before the setting: bilinear
    // -1: saved before the setting (plain normalized mean). "occluder_normalized_reprojection" = the setting's first name.
    s.occluder_aware_soft_normalize = j.value("occluder_aware_soft_normalize", j.value("occluder_normalized_reprojection", -1));
    s.variance_guide_enabled = j.value("variance_guide_enabled", 0); // results saved before the variance guide: off
    s.variance_guide_sigmas = j.value("variance_guide_sigmas", 2.0f);
    s.variance_guide_heavy_tail = j.value("variance_guide_heavy_tail", 0);
    s.variance_guide_tail_nu = j.value("variance_guide_tail_nu", 3.0f);
    s.variance_guide_count_cap_enabled = j.value("variance_guide_count_cap_enabled", 0);
    s.variance_guide_count_cap = j.value("variance_guide_count_cap", 64.0f);
    s.variance_guide_gradient = j.value("variance_guide_gradient", 0);
    s.split_history_enabled = j.value("split_history_enabled", 1);
    s.spatial_enabled = j.value("spatial_enabled", 1);
    s.spatial_mode = j.value("spatial_mode", 0);
    s.atrous_passes = j.value("atrous_passes", 4); // results saved before the setting: the fixed 4 passes
    s.blur_radius = j.value("blur_radius", 8);
    s.blur_plane_tolerance = j.value("blur_plane_tolerance", 1.5f);
    s.blur_normal_power = j.value("blur_normal_power", 8.0f);
    s.fine_target_samples = j.value("fine_target_samples", 0.0f); // results saved before the budget: none
    s.coarse_enabled = j.value("coarse_enabled", 0); // results saved before the coarse fallback: off
    s.coarse_atrous_passes = j.value("coarse_atrous_passes", 5);
    s.coarse_distance_tolerance = j.value("coarse_distance_tolerance", 2.0f);
    s.coarse_normal_power = j.value("coarse_normal_power", 1.0f);
    s.coarse_variance_guide_enabled = j.value("coarse_variance_guide_enabled", 0); // results saved before the coarse variance guide: off
    s.coarse_variance_guide_sigmas = j.value("coarse_variance_guide_sigmas", 3.0f);
    s.coarse_target_samples = j.value("coarse_target_samples", 0.0f); // 0: saved with the old smoothstep blend (coarse_count_low / _high)
    s.coarse_max_blend = j.value("coarse_max_blend", 1.0f);
    s.coarse_anti_lag_enabled = j.value("coarse_anti_lag_enabled", j.value("coarse_clamp_enabled", 0)); // results saved before the coarse temporal anti-lag: off
    s.coarse_anti_lag_blur_radius = j.value("coarse_anti_lag_blur_radius", j.value("coarse_clamp_blur_radius", 16));
    s.coarse_anti_lag_history_frames = j.value("coarse_anti_lag_history_frames", j.value("coarse_clamp_history_frames", 8));
    s.coarse_anti_lag_variance_blend = j.value("coarse_anti_lag_variance_blend", j.value("coarse_clamp_variance_blend", 4.0f));
    s.ray_filter_mode = j.value("ray_filter_mode", RT_LAB_RAY_FILTER_OFF); // results saved before the ray filter: off
    s.ray_filter_radius = j.value("ray_filter_radius", 2);
    s.ray_filter_sigmas = j.value("ray_filter_sigmas", 3.0f);
    s.ray_filter_clamp_low = j.value("ray_filter_clamp_low", 0);
    s.ray_filter_tolerance = j.value("ray_filter_tolerance", 4.0f);
    s.firefly_weight_enabled = j.value("firefly_weight_enabled", 0); // results saved before the firefly weighting: off
    s.ray_length_guide_mode = j.value("ray_length_guide_mode", 0); // results saved before the ray length guide: off
    s.ray_length_guide_radius_px = j.value("ray_length_guide_radius_px", 20.0f);
    s.ray_length_guide_distance_falloff = j.value("ray_length_guide_distance_falloff", 1.0f);
    s.ray_length_guide_similarity_tolerance = j.value("ray_length_guide_similarity_tolerance", 1.0f);
    return s;
}

auto rt_lab_settings_apply_json(RtLabSettings const & base, nlohmann::json const & overrides, std::vector<std::string> & unknown_keys) -> RtLabSettings
{
    nlohmann::json merged = rt_lab_settings_to_json(base);
    if (overrides.is_object())
    {
        for (auto const & [key, value] : overrides.items())
        {
            if (!merged.contains(key))
            {
                unknown_keys.push_back(key);
                continue;
            }
            merged[key] = value;
        }
    }
    // Every preset key is present in `merged`: from_json's "old result" fallbacks never apply.
    RtLabSettings settings = rt_lab_settings_from_json(merged);
    settings.enabled = base.enabled;
    settings.reset_history = base.reset_history;
    settings.sample_count_input_clamp = base.sample_count_input_clamp;
    settings.sample_count_debug_max = base.sample_count_debug_max;
    return settings;
}

static auto rt_lab_pose_to_json(ProgrammedMovePose const & p) -> nlohmann::json
{
    return {p.position.x, p.position.y, p.position.z, p.yaw, p.pitch};
}

static auto rt_lab_pose_from_json(nlohmann::json const & j) -> ProgrammedMovePose
{
    ProgrammedMovePose p = {};
    if (j.is_array() && j.size() == 5)
    {
        p.position = {j[0].get<f32>(), j[1].get<f32>(), j[2].get<f32>()};
        p.yaw = j[3].get<f32>();
        p.pitch = j[4].get<f32>();
    }
    return p;
}

static auto rt_lab_test_move_to_json(RtLabTestMove const & move) -> nlohmann::json
{
    return {
        {"enabled", move.enabled},
        {"fixed_dt", move.fixed_dt},
        {"camera", programmed_move_to_json(move.camera)},
        {"sun", programmed_move_to_json(move.sun)},
    };
}

static auto rt_lab_test_move_from_json(nlohmann::json const & j) -> RtLabTestMove
{
    RtLabTestMove move = {};
    move.enabled = j.value("enabled", false);
    move.fixed_dt = j.value("fixed_dt", move.fixed_dt);
    if (j.contains("camera"))
    {
        move.camera = programmed_move_from_json(j["camera"], move.camera);
    }
    if (j.contains("sun"))
    {
        move.sun = programmed_move_from_json(j["sun"], move.sun);
    }
    return move;
}

auto rt_lab_convergence_result_to_json(RtLabConvergenceResult const & r) -> nlohmann::json
{
    return {
        {"pixels", r.pixels},
        {"mse", r.mse},
        {"rmse", std::sqrt(r.mse)},
        {"mean_abs_error", r.mean_abs_error},
        {"bias", r.bias},
        {"histogram_range", r.histogram_range},
        {"histogram", r.histogram},
        {"below_range", r.below_range},
        {"above_range", r.above_range},
        {"predicted_pixels", r.predicted_pixels},
        {"mean_predicted", r.mean_predicted},
        {"mean_actual", r.mean_actual},
        {"rms_predicted", r.rms_predicted},
        {"rms_actual", r.rms_actual},
        {"correlation", r.correlation},
        {"rms_bias", std::sqrt(std::max(r.rms_actual * r.rms_actual - r.rms_predicted * r.rms_predicted, 0.0))},
    };
}

static auto rt_lab_convergence_result_from_json(nlohmann::json const & j) -> RtLabConvergenceResult
{
    RtLabConvergenceResult r = {};
    r.pixels = j.value("pixels", 0u);
    r.mse = j.value("mse", 0.0);
    r.mean_abs_error = j.value("mean_abs_error", 0.0);
    r.bias = j.value("bias", 0.0);
    r.histogram_range = j.value("histogram_range", 0.0f);
    if (j.contains("histogram") && j["histogram"].size() == RT_LAB_CONVERGENCE_HISTOGRAM_BINS)
    {
        r.histogram = j["histogram"].get<std::array<u32, RT_LAB_CONVERGENCE_HISTOGRAM_BINS>>();
    }
    r.below_range = j.value("below_range", 0u);
    r.above_range = j.value("above_range", 0u);
    r.predicted_pixels = j.value("predicted_pixels", 0u);
    r.mean_predicted = j.value("mean_predicted", 0.0);
    r.mean_actual = j.value("mean_actual", 0.0);
    r.rms_predicted = j.value("rms_predicted", 0.0);
    r.rms_actual = j.value("rms_actual", 0.0);
    r.correlation = j.value("correlation", 0.0);
    return r;
}

auto rt_lab_convergence_steps_to_json(std::vector<RtLabConvergenceStep> const & steps) -> nlohmann::json
{
    nlohmann::json json = nlohmann::json::array();
    for (RtLabConvergenceStep const & r : steps)
    {
        json.push_back({
            {"step", r.step},
            {"pixels", r.pixels},
            {"mse", r.mse},
            {"rmse", std::sqrt(r.mse)},
            {"bias", r.bias},
            {"mean_abs_error", r.mean_abs_error},
            {"rms_predicted", r.rms_predicted},
            {"rms_actual", r.rms_actual},
        });
    }
    return json;
}

static auto rt_lab_convergence_steps_from_json(nlohmann::json const & json) -> std::vector<RtLabConvergenceStep>
{
    std::vector<RtLabConvergenceStep> steps = {};
    for (nlohmann::json const & j : json)
    {
        steps.push_back(RtLabConvergenceStep{
            .step = j.value("step", 0),
            .pixels = j.value("pixels", 0u),
            .mse = j.value("mse", 0.0),
            .bias = j.value("bias", 0.0),
            .mean_abs_error = j.value("mean_abs_error", 0.0),
            .rms_predicted = j.value("rms_predicted", 0.0),
            .rms_actual = j.value("rms_actual", 0.0),
        });
    }
    return steps;
}

static auto rt_lab_measurement_header(RtLabMeasurement const & m, std::string const & name) -> nlohmann::json
{
    nlohmann::json header = {
        {"name", name},
        {"timestamp", m.timestamp},
        {"frames", m.frames},
        {"window", m.window},
        {"preset", m.preset},
        {"metrics", m.metrics},
        {"settings", rt_lab_settings_to_json(m.settings)},
        {"camera_position", {m.camera_position[0], m.camera_position[1], m.camera_position[2]}},
        {"move", rt_lab_test_move_to_json(m.move)},
        {"move_camera_origin", rt_lab_pose_to_json(m.move_camera_origin)},
        {"move_sun_origin", rt_lab_pose_to_json(m.move_sun_origin)},
        {"resolution", {m.resolution[0], m.resolution[1]}},
        {"row_size", RT_LAB_MEASURE_ROW_SIZE},
        {"bins", RT_LAB_MEASURE_BINS},
        {"log_min", m.log_min},
        {"log_max", m.log_max},
        {"tile", RT_LAB_MEASURE_TILE},
        {"lag", {
            {"valid", m.lag.valid},
            {"tolerance", m.lag.tolerance},
            {"start_frame", m.lag.start_frame},
            {"tiles", m.lag.tiles},
            {"changed_tiles", m.lag.changed_tiles},
            {"settle_p50", m.lag.settle_p50},
            {"settle_p90", m.lag.settle_p90},
            {"settle_max", m.lag.settle_max},
            {"never_settled", m.lag.never_settled},
            {"settled_curve", m.lag.settled_curve},
        }},
    };
    if ((m.metrics & RT_LAB_TEST_METRIC_REFERENCE) != 0)
    {
        header["reference"] = rt_lab_convergence_result_to_json(m.reference);
        header["reference_frames"] = m.reference_frames;
        header["reference_warmup_frames"] = m.reference_warmup_frames;
        header["reference_steps"] = rt_lab_convergence_steps_to_json(m.reference_steps);
    }
    return header;
}

// Reads a result file's header (and the rows if read_rows). false = not a (compatible) result file.
static auto rt_lab_result_read(std::filesystem::path const & path, bool read_rows, RtLabResult & result) -> bool
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        return false;
    }
    char magic[8] = {};
    u32 version = 0;
    u32 header_size = 0;
    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
    if (!file || std::memcmp(magic, RT_LAB_RESULT_MAGIC, sizeof(magic)) != 0 || version != RT_LAB_RESULT_VERSION)
    {
        return false;
    }
    std::string header_text(header_size, '\0');
    file.read(header_text.data(), header_size);
    nlohmann::json const header = nlohmann::json::parse(header_text, nullptr, false);
    // Rows saved before the jitter metric hold 2 metrics: padded with an empty jitter metric (not in m.metrics).
    u32 const file_row_size = header.is_discarded() ? 0u : header.value("row_size", 0u);
    if (!file || header.is_discarded() || (file_row_size != RT_LAB_MEASURE_ROW_SIZE && file_row_size != RT_LAB_MEASURE_METRIC_STRIDE * 2))
    {
        return false;
    }
    auto & m = result.measurement;
    result.name = header.value("name", path.stem().string());
    result.path = path;
    m.timestamp = header.value("timestamp", std::string{});
    m.frames = header.value("frames", 0);
    m.window = header.value("window", 0);
    m.preset = header.value("preset", 0);
    // Results saved before the test setup: the per frame metrics of then (no jitter).
    m.metrics = header.value("metrics", RT_LAB_TEST_METRIC_CONVERGENCE | RT_LAB_TEST_METRIC_INSTABILITY | RT_LAB_TEST_METRIC_LAG);
    if (header.contains("reference"))
    {
        m.reference = rt_lab_convergence_result_from_json(header["reference"]);
        m.reference_frames = header.value("reference_frames", 0);
        m.reference_warmup_frames = header.value("reference_warmup_frames", 0);
        m.reference_steps = rt_lab_convergence_steps_from_json(header.value("reference_steps", nlohmann::json::array()));
    }
    m.log_min = header.value("log_min", -4.0f); // every file so far wrote its range; -4 / 1 was the first one
    m.log_max = header.value("log_max", 1.0f);
    if (header.contains("settings"))
    {
        m.settings = rt_lab_settings_from_json(header["settings"]);
    }
    if (header.contains("camera_position") && header["camera_position"].size() == 3)
    {
        for (u32 i = 0; i < 3; ++i)
        {
            m.camera_position[i] = header["camera_position"][i].get<f32>();
        }
    }
    if (header.contains("lag")) // results saved before the lag metric: none
    {
        nlohmann::json const & lag = header["lag"];
        m.lag.valid = lag.value("valid", false);
        m.lag.tolerance = lag.value("tolerance", 0.02);
        m.lag.start_frame = lag.value("start_frame", 0);
        m.lag.tiles = lag.value("tiles", 0u);
        m.lag.changed_tiles = lag.value("changed_tiles", 0u);
        m.lag.settle_p50 = lag.value("settle_p50", 0.0);
        m.lag.settle_p90 = lag.value("settle_p90", 0.0);
        m.lag.settle_max = lag.value("settle_max", 0.0);
        m.lag.never_settled = lag.value("never_settled", 0.0);
        m.lag.settled_curve = lag.value("settled_curve", std::vector<f32>{});
    }
    if (header.contains("move")) // results saved before programmed moves: static camera
    {
        m.move = rt_lab_test_move_from_json(header["move"]);
        m.move_camera_origin = rt_lab_pose_from_json(header.value("move_camera_origin", nlohmann::json{}));
        m.move_sun_origin = rt_lab_pose_from_json(header.value("move_sun_origin", nlohmann::json{}));
    }
    if (header.contains("resolution") && header["resolution"].size() == 2)
    {
        m.resolution[0] = header["resolution"][0].get<u32>();
        m.resolution[1] = header["resolution"][1].get<u32>();
    }
    if (!read_rows)
    {
        return true;
    }
    u32 row_count = 0;
    file.read(reinterpret_cast<char *>(&row_count), sizeof(row_count));
    if (!file || static_cast<i32>(row_count) != m.frames)
    {
        return false;
    }
    std::vector<u32> file_rows(static_cast<size_t>(row_count) * file_row_size);
    file.read(reinterpret_cast<char *>(file_rows.data()), static_cast<std::streamsize>(file_rows.size() * sizeof(u32)));
    if (!file)
    {
        return false;
    }
    m.rows.assign(static_cast<size_t>(row_count) * RT_LAB_MEASURE_ROW_SIZE, 0u);
    for (u32 row = 0; row < row_count; ++row)
    {
        std::memcpy(m.rows.data() + static_cast<size_t>(row) * RT_LAB_MEASURE_ROW_SIZE, file_rows.data() + static_cast<size_t>(row) * file_row_size, file_row_size * sizeof(u32));
    }
    rt_lab_measure_finalize(m);
    result.loaded = true;
    return true;
}

// Keeps selected_result / compare_result pointing at the same entries after results[removed] was erased.
static void rt_lab_results_erased(RtLabState & lab, i32 removed)
{
    for (i32 * selection : {&lab.selected_result, &lab.compare_result})
    {
        if (*selection == removed)
        {
            *selection = -1;
        }
        else if (*selection > removed)
        {
            *selection -= 1;
        }
    }
}

void rt_lab_results_scan(RtLabState & lab)
{
    lab.results_scanned = true;
    std::error_code error = {};
    if (!std::filesystem::is_directory(RT_LAB_RESULTS_DIRECTORY, error))
    {
        return;
    }
    std::vector<std::filesystem::path> paths = {};
    for (auto const & entry : std::filesystem::directory_iterator(RT_LAB_RESULTS_DIRECTORY, error))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".rtlab")
        {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    for (auto const & path : paths)
    {
        bool const known = std::any_of(lab.results.begin(), lab.results.end(), [&](RtLabResult const & r)
            { return !r.path.empty() && std::filesystem::equivalent(r.path, path, error); });
        if (known)
        {
            continue;
        }
        RtLabResult result = {};
        if (rt_lab_result_read(path, false, result))
        {
            lab.results.push_back(std::move(result));
        }
    }
}

auto rt_lab_result_save(RtLabState & lab, i32 index, std::string const & name) -> bool
{
    if (index < 0 || index >= static_cast<i32>(lab.results.size()) || name.empty())
    {
        return false;
    }
    auto & result = lab.results[index];
    if (!rt_lab_result_ensure_loaded(result))
    {
        return false;
    }
    std::error_code error = {};
    std::filesystem::create_directories(RT_LAB_RESULTS_DIRECTORY, error);
    std::filesystem::path const path = std::filesystem::path(RT_LAB_RESULTS_DIRECTORY) / (name + ".rtlab");
    std::string const header = rt_lab_measurement_header(result.measurement, name).dump(2);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        return false;
    }
    u32 const version = RT_LAB_RESULT_VERSION;
    u32 const header_size = static_cast<u32>(header.size());
    u32 const row_count = static_cast<u32>(result.measurement.frames);
    file.write(RT_LAB_RESULT_MAGIC, sizeof(RT_LAB_RESULT_MAGIC));
    file.write(reinterpret_cast<char const *>(&version), sizeof(version));
    file.write(reinterpret_cast<char const *>(&header_size), sizeof(header_size));
    file.write(header.data(), header_size);
    file.write(reinterpret_cast<char const *>(&row_count), sizeof(row_count));
    file.write(reinterpret_cast<char const *>(result.measurement.rows.data()), static_cast<std::streamsize>(result.measurement.rows.size() * sizeof(u32)));
    if (!file)
    {
        return false;
    }
    // Saving under the name of another listed saved result replaces that entry.
    for (i32 i = 0; i < static_cast<i32>(lab.results.size()); ++i)
    {
        if (i != index && !lab.results[i].path.empty() && std::filesystem::equivalent(lab.results[i].path, path, error))
        {
            lab.results.erase(lab.results.begin() + i);
            rt_lab_results_erased(lab, i);
            index -= index > i ? 1 : 0;
            break;
        }
    }
    lab.results[index].name = name;
    lab.results[index].path = path;
    return true;
}

auto rt_lab_result_delete(RtLabState & lab, i32 index) -> bool
{
    if (index < 0 || index >= static_cast<i32>(lab.results.size()))
    {
        return false;
    }
    if (!lab.results[index].path.empty())
    {
        std::error_code error = {};
        std::filesystem::remove(lab.results[index].path, error);
        if (error)
        {
            return false;
        }
    }
    lab.results.erase(lab.results.begin() + index);
    rt_lab_results_erased(lab, index);
    if (lab.selected_result < 0 && !lab.results.empty())
    {
        lab.selected_result = static_cast<i32>(lab.results.size()) - 1;
    }
    return true;
}

auto rt_lab_result_rename(RtLabState & lab, i32 index, std::string const & name) -> bool
{
    if (index < 0 || index >= static_cast<i32>(lab.results.size()) || name.empty())
    {
        return false;
    }
    if (lab.results[index].path.empty())
    {
        lab.results[index].name = name;
        return true;
    }
    std::filesystem::path const old_path = lab.results[index].path;
    std::filesystem::path const new_path = std::filesystem::path(RT_LAB_RESULTS_DIRECTORY) / (name + ".rtlab");
    std::error_code error = {};
    bool const same_file = std::filesystem::equivalent(old_path, new_path, error);
    if (!rt_lab_result_save(lab, index, name))
    {
        return false;
    }
    if (!same_file)
    {
        std::filesystem::remove(old_path, error);
    }
    return true;
}

auto rt_lab_result_ensure_loaded(RtLabResult & result) -> bool
{
    if (result.loaded)
    {
        return true;
    }
    if (result.path.empty())
    {
        return false;
    }
    return rt_lab_result_read(result.path, true, result);
}

// == Test state machine (RtLabTestSetup, rt_lab_state.hpp) ==

// History backups of the move reference. The graph registered them: callers rebuild it (rebuild_requested).
static void rt_lab_destroy_history_backups(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    for (auto & [name, backup] : lab.history_backups)
    {
        if (!backup.image.id().is_empty())
        {
            render_context.gpu_context->device.destroy_image(backup.image.id());
        }
    }
    lab.history_backups.clear();
    lab.restore_custom_history = false;
    lab.save_custom_history = false;
}

auto rt_lab_test_metrics_from_names(std::string_view list, u32 & metrics, std::string & unknown) -> bool
{
    metrics = 0;
    while (!list.empty())
    {
        size_t const comma = list.find(',');
        std::string_view name = list.substr(0, comma);
        list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
        while (!name.empty() && name.front() == ' ')
        {
            name.remove_prefix(1);
        }
        while (!name.empty() && name.back() == ' ')
        {
            name.remove_suffix(1);
        }
        if (name.empty() || name == "none")
        {
            continue;
        }
        u32 const metric = name == "convergence" ? RT_LAB_TEST_METRIC_CONVERGENCE
                         : name == "instability" ? RT_LAB_TEST_METRIC_INSTABILITY
                         : name == "jitter"      ? RT_LAB_TEST_METRIC_JITTER
                         : name == "lag"         ? RT_LAB_TEST_METRIC_LAG
                         : name == "reference"   ? RT_LAB_TEST_METRIC_REFERENCE
                         : name == "all"         ? (RT_LAB_TEST_METRIC_PER_FRAME | RT_LAB_TEST_METRIC_REFERENCE)
                                                 : 0u;
        if (metric == 0)
        {
            unknown = std::string(name);
            return false;
        }
        metrics |= metric;
    }
    return true;
}

auto rt_lab_test_move_steps(RtLabTestMove const & move) -> i32
{
    return std::max(programmed_move_step_count(move.camera, move.fixed_dt), programmed_move_step_count(move.sun, move.fixed_dt));
}

void rt_lab_set_test_move_origins(RtLabState & lab, ProgrammedMovePose const & camera, ProgrammedMovePose const & sun)
{
    lab.test_camera_origin = camera;
    lab.test_sun_origin = sun;
    if (lab.phase == RtLabTestPhase::RUN)
    {
        lab.measurement.move_camera_origin = camera;
        lab.measurement.move_sun_origin = sun;
    }
}

static auto rt_lab_test_has_reference(RtLabTestSetup const & setup) -> bool
{
    return (setup.metrics & RT_LAB_TEST_METRIC_REFERENCE) != 0;
}

static auto rt_lab_test_move_phase(RtLabTestPhase phase) -> bool
{
    return phase == RtLabTestPhase::MOVE_WARMUP || phase == RtLabTestPhase::MOVE_TESTED || phase == RtLabTestPhase::MOVE_REFERENCE;
}

// Moves the auto histogram range towards the smallest symmetric range holding histogram_auto_coverage of the pixels.
static void rt_lab_update_histogram_auto_range(RtLabState & lab)
{
    auto const & result = lab.live_result;
    if (result.pixels == 0 || result.histogram_range <= 0.0f)
    {
        return;
    }
    if (lab.histogram_auto_range_value <= 0.0f)
    {
        lab.histogram_auto_range_value = std::max(static_cast<f32>(std::sqrt(result.mse)) * 4.0f, 1e-5f);
        return;
    }
    // The readback lags a few frames: only react to results binned with the current range.
    if (std::abs(result.histogram_range - lab.histogram_auto_range_value) > lab.histogram_auto_range_value * 1e-4f)
    {
        return;
    }
    f64 const total = static_cast<f64>(result.pixels);
    f64 const allowed_outside = total * (1.0 - static_cast<f64>(std::clamp(lab.histogram_auto_coverage, 0.5f, 1.0f)));
    f32 target = lab.histogram_auto_range_value;
    if (static_cast<f64>(result.below_range + result.above_range) > allowed_outside)
    {
        target *= 1.25f; // too much outside: grow
    }
    else
    {
        // Shrink: drop bin pairs from both ends while the pixels outside stay within the allowance.
        u32 const half_bins = RT_LAB_CONVERGENCE_HISTOGRAM_BINS / 2;
        f64 outside = static_cast<f64>(result.below_range + result.above_range);
        u32 keep = half_bins;
        while (keep > 1)
        {
            u32 const lo = half_bins - keep;
            u32 const hi = half_bins + keep - 1;
            f64 const next_outside = outside + static_cast<f64>(result.histogram[lo]) + static_cast<f64>(result.histogram[hi]);
            if (next_outside > allowed_outside)
            {
                break;
            }
            outside = next_outside;
            keep -= 1;
        }
        target = result.histogram_range * static_cast<f32>(keep) / static_cast<f32>(half_bins);
    }
    lab.histogram_auto_range_value = std::max(lab.histogram_auto_range_value + (target - lab.histogram_auto_range_value) * 0.25f, 1e-5f);
}

// Creates the reference capture images if missing.
static void rt_lab_create_capture_images(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    if (!lab.progressive_image.id().is_empty())
    {
        return;
    }
    auto const render_target_size = render_context.render_data.settings.render_target_size;
    auto const info = daxa::ImageInfo{
        .format = daxa::Format::R16G16B16A16_SFLOAT, // same as the resolved diffuse: lossless capture
        .size = {render_target_size.x, render_target_size.y, 1},
        .usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST,
    };
    auto progressive_info = info;
    progressive_info.name = "rt_lab_convergence_progressive";
    auto custom_info = info;
    custom_info.name = "rt_lab_convergence_custom";
    auto predicted_error_info = info;
    predicted_error_info.format = daxa::Format::R32G32_SFLOAT;
    predicted_error_info.name = "rt_lab_convergence_predicted_error";
    lab.progressive_image.set_image(render_context.gpu_context->device.create_image(progressive_info));
    lab.custom_image.set_image(render_context.gpu_context->device.create_image(custom_info));
    lab.predicted_error_image.set_image(render_context.gpu_context->device.create_image(predicted_error_info));
}

// Grows the measurement buffers to `frames` rows / `lag_floats` lag values (host visible: the store / lag passes write
// one small row per frame, the CPU reads the whole run at the end) and clears the rows.
static void rt_lab_prepare_measure_buffers(RenderContext & render_context, i32 frames, u64 lag_floats)
{
    auto & lab = render_context.rt_lab;
    auto & device = render_context.gpu_context->device;
    if (lab.measure_frame_histogram.id().is_empty())
    {
        lab.measure_frame_histogram.set_buffer(device.create_buffer({
            .size = sizeof(u32) * RT_LAB_MEASURE_ROW_SIZE,
            .name = "rt_lab_measure_frame_histogram",
        }));
    }
    if (lab.measure_run_capacity < frames)
    {
        if (!lab.measure_run_histograms.id().is_empty())
        {
            device.destroy_buffer(lab.measure_run_histograms.id());
        }
        lab.measure_run_histograms.set_buffer(device.create_buffer({
            .size = sizeof(u32) * RT_LAB_MEASURE_ROW_SIZE * static_cast<u32>(frames),
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "rt_lab_measure_run_histograms",
        }));
        lab.measure_run_capacity = frames;
    }
    std::memset(device.buffer_host_address(lab.measure_run_histograms.id()).value(), 0, sizeof(u32) * RT_LAB_MEASURE_ROW_SIZE * static_cast<size_t>(frames));
    if (lab.measure_lag_capacity < std::max(lag_floats, u64{1}))
    {
        if (!lab.measure_lag_run.id().is_empty())
        {
            device.destroy_buffer(lab.measure_lag_run.id());
        }
        lab.measure_lag_capacity = std::max(lag_floats, u64{1});
        lab.measure_lag_run.set_buffer(device.create_buffer({
            .size = sizeof(f32) * lab.measure_lag_capacity,
            .memory_flags = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
            .name = "rt_lab_measure_lag_run",
        }));
    }
}

static void rt_lab_test_begin_phase(RtLabState & lab, RtLabTestPhase phase)
{
    // Every phase switch rebuilds the graph (preset switches, history backup tasks), except the wait after a static run:
    // the tested preset keeps running (the live analysis compares the captures).
    lab.rebuild_requested = lab.rebuild_requested || !(phase == RtLabTestPhase::COLLECTING && lab.phase == RtLabTestPhase::RUN);
    lab.phase = phase;
    lab.phase_frame = 0;
    if (phase == RtLabTestPhase::COLLECTING)
    {
        lab.collect_countdown = MAX_GPU_FRAMES_IN_FLIGHT + 2;
    }
}

// Ends the running (or cancelled) test: poses / camera / enables return, history backups go.
static void rt_lab_test_end(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    if (!lab.history_backups.empty())
    {
        rt_lab_destroy_history_backups(render_context);
        lab.rebuild_requested = true;
    }
    lab.phase = RtLabTestPhase::IDLE;
    lab.phase_frame = 0;
    lab.camera_locked = false;
    render_context.render_data.rt_lab_settings.enabled = lab.saved_lab_enabled;
    render_context.render_data.rtgi_settings.enabled = lab.saved_rtgi_enabled;
}

static void rt_lab_test_start(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    auto & render_data = render_context.render_data;
    auto const render_target_size = render_data.settings.render_target_size;
    RtLabTestSetup & setup = lab.running;
    setup = lab.test;
    setup.frames = std::clamp(setup.frames, 1, 4096);
    setup.window = std::max(setup.window, 1);
    setup.reference_frames = std::max(setup.reference_frames, 1);
    setup.move_warmup_frames = std::max(setup.move_warmup_frames, 1);
    // The tested preset runs one frame every move_reference_frames + 1 frames: an EVEN reference count keeps that period
    // odd, so its frames alternate frame_index parity (split history A takes even frames, B odd ones).
    setup.move_reference_frames = std::max(setup.move_reference_frames + (setup.move_reference_frames & 1), 2);
    if (!setup.move.enabled)
    {
        setup.move = {};
    }
    bool const reference = rt_lab_test_has_reference(setup);

    lab.saved_lab_enabled = render_data.rt_lab_settings.enabled;
    lab.saved_rtgi_enabled = render_data.rtgi_settings.enabled;
    lab.camera_locked = true;
    lab.live_analysis = false;
    lab.live_result = {};
    lab.histogram_auto_range_value = 0.0f;
    if (reference)
    {
        rt_lab_create_capture_images(render_context);
    }
    i32 const lag_tiles_x = (setup.metrics & RT_LAB_TEST_METRIC_LAG) != 0 ? static_cast<i32>(round_up_div(render_target_size.x, static_cast<u32>(RT_LAB_MEASURE_LAG_TILE))) : 0;
    i32 const lag_tiles_y = (setup.metrics & RT_LAB_TEST_METRIC_LAG) != 0 ? static_cast<i32>(round_up_div(render_target_size.y, static_cast<u32>(RT_LAB_MEASURE_LAG_TILE))) : 0;
    rt_lab_prepare_measure_buffers(render_context, setup.frames, static_cast<u64>(setup.frames) * static_cast<u64>(lag_tiles_x * lag_tiles_y));

    auto & m = lab.measurement;
    m = {};
    m.frames = setup.frames;
    m.window = setup.window;
    m.preset = lab.active_preset;
    m.metrics = setup.metrics;
    m.settings = lab.presets[lab.active_preset];
    m.settings.split_history_enabled = 1;
    m.camera_position[0] = render_data.view_camera.position.x;
    m.camera_position[1] = render_data.view_camera.position.y;
    m.camera_position[2] = render_data.view_camera.position.z;
    m.move = setup.move;
    m.lag_tiles_x = lag_tiles_x;
    m.lag_tiles_y = lag_tiles_y;
    m.lag_start_frame = setup.shock_frames > 0 ? std::min(setup.shock_start + setup.shock_frames, setup.frames - 1)
                        : setup.move.enabled   ? std::min(rt_lab_test_move_steps(setup.move), setup.frames - 1)
                                               : 0;
    m.reference_frames = !reference ? 0 : (setup.move.enabled ? setup.move_reference_frames : setup.reference_frames);
    m.reference_warmup_frames = reference && setup.move.enabled ? setup.move_warmup_frames : 0;
    m.resolution[0] = render_target_size.x;
    m.resolution[1] = render_target_size.y;
    {
        std::time_t const now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm local = {};
        localtime_s(&local, &now);
        char buffer[32] = {};
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local);
        m.timestamp = buffer;
    }
    // The static reference runs first (no move: nothing to restore in between).
    rt_lab_test_begin_phase(lab, reference && !setup.move.enabled ? RtLabTestPhase::REFERENCE : RtLabTestPhase::RUN);
    lab.test_move_begin = lab.phase == RtLabTestPhase::RUN && setup.move.enabled;
}

// Reads the run's rows + lag tiles and adds the result (selected) as "run <n> <preset>", n continuing after the highest
// "run <n>" listed (saved ones too).
static void rt_lab_test_collect(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    auto & m = lab.measurement;
    u32 const * rows = render_context.gpu_context->device.buffer_host_address_as<u32>(lab.measure_run_histograms.id()).value();
    m.rows.assign(rows, rows + static_cast<size_t>(m.frames) * RT_LAB_MEASURE_ROW_SIZE);
    rt_lab_measure_finalize(m);
    if (m.lag_tiles_x > 0)
    {
        size_t const count = static_cast<size_t>(m.frames) * static_cast<size_t>(m.lag_tiles_x * m.lag_tiles_y);
        f32 const * lag = render_context.gpu_context->device.buffer_host_address_as<f32>(lab.measure_lag_run.id()).value();
        m.lag_values.assign(lag, lag + count);
        m.lag = rt_lab_measure_lag(m, lab.lag_tolerance);
    }
    i32 run = 0;
    for (RtLabResult const & r : lab.results)
    {
        i32 number = 0;
        if (std::sscanf(r.name.c_str(), "run%*[ _]%d", &number) == 1)
        {
            run = std::max(run, number);
        }
    }
    char const * const preset_names[RT_LAB_PRESET_COUNT] = {"progressive", "custom"};
    lab.results.push_back(RtLabResult{
        .name = fmt::format("run {} {}", run + 1, preset_names[std::clamp(m.preset, 0, RT_LAB_PRESET_COUNT - 1)]),
        .loaded = true,
        .measurement = m,
    });
    lab.selected_result = static_cast<i32>(lab.results.size()) - 1;
}

void rt_lab_tests_begin_frame(RenderContext & render_context)
{
    auto & lab = render_context.rt_lab;
    RtLabTestSetup const & setup = lab.running;
    bool const reference = rt_lab_test_has_reference(setup);
    lab.test_move_begin = false;

    // Phase transitions (phase_frame = frames rendered in the phase so far, counted by rt_lab_frame_update).
    switch (lab.phase)
    {
        case RtLabTestPhase::IDLE: break;
        case RtLabTestPhase::REFERENCE:
        {
            if (lab.phase_frame >= setup.reference_frames)
            {
                rt_lab_test_begin_phase(lab, RtLabTestPhase::RUN);
            }
            break;
        }
        case RtLabTestPhase::RUN:
        {
            if (lab.phase_frame >= setup.frames)
            {
                if (reference && setup.move.enabled)
                {
                    rt_lab_test_begin_phase(lab, RtLabTestPhase::MOVE_WARMUP);
                    lab.move_step = 0;
                    lab.move_steps = rt_lab_test_move_steps(setup.move);
                    lab.test_move_begin = true;
                }
                else
                {
                    // Static reference: both captures exist now, the analysis runs every frame from here on.
                    lab.live_analysis = reference;
                    rt_lab_test_begin_phase(lab, RtLabTestPhase::COLLECTING);
                }
            }
            break;
        }
        case RtLabTestPhase::MOVE_WARMUP:
        {
            if (lab.phase_frame >= setup.move_warmup_frames)
            {
                rt_lab_test_begin_phase(lab, RtLabTestPhase::MOVE_REFERENCE);
            }
            break;
        }
        case RtLabTestPhase::MOVE_REFERENCE:
        {
            if (lab.phase_frame >= setup.move_reference_frames)
            {
                if (lab.move_step >= lab.move_steps)
                {
                    rt_lab_test_begin_phase(lab, RtLabTestPhase::COLLECTING); // waits for the last step's analysis readback
                }
                else
                {
                    lab.move_step += 1;
                    rt_lab_test_begin_phase(lab, RtLabTestPhase::MOVE_TESTED);
                }
            }
            break;
        }
        case RtLabTestPhase::MOVE_TESTED:
        {
            if (lab.phase_frame >= 1)
            {
                rt_lab_test_begin_phase(lab, RtLabTestPhase::MOVE_REFERENCE);
            }
            break;
        }
        case RtLabTestPhase::COLLECTING:
        {
            lab.collect_countdown -= 1;
            // The static reference's analysis readback arrives a few frames after the captures (give up after a while).
            bool const waiting_for_reference = reference && !setup.move.enabled && lab.live_result.pixels == 0 && lab.phase_frame < 64;
            if (lab.collect_countdown <= 0 && !waiting_for_reference)
            {
                if (reference && !setup.move.enabled)
                {
                    lab.measurement.reference = lab.live_result;
                }
                rt_lab_test_collect(render_context);
                rt_lab_test_end(render_context);
            }
            break;
        }
    }

    if (lab.test_cancel_requested && rt_lab_test_running(lab))
    {
        rt_lab_test_end(render_context);
        lab.measurement = {};
        lab.live_analysis = false;
    }
    if (lab.test_start_requested && !rt_lab_test_running(lab))
    {
        rt_lab_test_start(render_context);
    }
    lab.test_start_requested = false;
    lab.test_cancel_requested = false;

    // == Programmed move pose of this frame ==
    // Run: one move step per recorded frame, holds the end pose after the move. Move reference: the current step (the
    // warmup sits at step 0).
    lab.test_move = nullptr;
    lab.test_move_step = 0;
    if (lab.phase == RtLabTestPhase::RUN && lab.running.move.enabled)
    {
        lab.test_move = &lab.running.move;
        lab.test_move_step = lab.phase_frame;
    }
    else if (rt_lab_test_move_phase(lab.phase))
    {
        lab.test_move = &lab.running.move;
        lab.test_move_step = lab.move_step;
    }

    // Move preview: a starting test cancels it, it ends one frame after its last step (the end pose stays visible).
    bool const test_running = rt_lab_test_running(lab);
    if (lab.preview_move != nullptr && (test_running || lab.preview_stop_requested || lab.preview_step > rt_lab_test_move_steps(*lab.preview_move)))
    {
        lab.preview_move = nullptr;
        if (!test_running)
        {
            lab.camera_locked = false;
        }
    }
    lab.preview_stop_requested = false;
    if (lab.preview_move != nullptr)
    {
        lab.test_move = lab.preview_move;
        lab.test_move_step = lab.preview_step;
        lab.test_move_begin = lab.preview_step == 0;
        lab.camera_locked = true;
        lab.preview_step += 1;
    }
}

static auto rt_lab_convergence_result_from_readback(ReadbackValues const & readback) -> RtLabConvergenceResult
{
    RtLabConvergenceResult result = {};
    f64 const pixels = static_cast<f64>(readback.rt_lab_convergence_pixels);
    f64 const scale = static_cast<f64>(RT_LAB_CONVERGENCE_FIXED_SCALE);
    result.pixels = readback.rt_lab_convergence_pixels;
    result.mse = static_cast<f64>(readback.rt_lab_convergence_squared_error_sum) / scale / pixels;
    result.bias = (static_cast<f64>(readback.rt_lab_convergence_positive_error_sum) - static_cast<f64>(readback.rt_lab_convergence_negative_error_sum)) / scale / pixels;
    result.mean_abs_error = (static_cast<f64>(readback.rt_lab_convergence_positive_error_sum) + static_cast<f64>(readback.rt_lab_convergence_negative_error_sum)) / scale / pixels;
    result.histogram_range = readback.rt_lab_convergence_histogram_range;
    result.below_range = readback.rt_lab_convergence_below_range;
    result.above_range = readback.rt_lab_convergence_above_range;
    for (u32 b = 0; b < RT_LAB_CONVERGENCE_HISTOGRAM_BINS; ++b)
    {
        result.histogram[b] = readback.rt_lab_convergence_histogram[b];
    }
    result.predicted_pixels = readback.rt_lab_convergence_predicted_pixels;
    if (readback.rt_lab_convergence_predicted_pixels > 0)
    {
        f64 const n = static_cast<f64>(readback.rt_lab_convergence_predicted_pixels);
        f64 const sp = static_cast<f64>(readback.rt_lab_convergence_predicted_sum) / scale;
        f64 const sa = static_cast<f64>(readback.rt_lab_convergence_actual_sum) / scale;
        f64 const spp = static_cast<f64>(readback.rt_lab_convergence_predicted_squared_sum) / scale;
        f64 const saa = static_cast<f64>(readback.rt_lab_convergence_actual_squared_sum) / scale;
        f64 const spa = static_cast<f64>(readback.rt_lab_convergence_predicted_actual_sum) / scale;
        result.mean_predicted = sp / n;
        result.mean_actual = sa / n;
        result.rms_predicted = std::sqrt(spp / n);
        result.rms_actual = std::sqrt(saa / n);
        f64 const covariance = n * spa - sp * sa;
        f64 const variance_product = (n * spp - sp * sp) * (n * saa - sa * sa);
        result.correlation = variance_product > 0.0 ? covariance / std::sqrt(variance_product) : 0.0;
    }
    return result;
}

auto rt_lab_frame_update(RenderContext & render_context) -> bool
{
    auto & lab = render_context.rt_lab;
    auto & render_data = render_context.render_data;
    auto const render_target_size = render_data.settings.render_target_size;
    lab.capture_target = RT_LAB_CAPTURE_NONE;
    lab.analyze_step = ~0u;
    lab.restore_custom_history = false;
    lab.save_custom_history = false;
    lab.measure_row = ~0u;

    // Captures from an older render target size are useless (and the passes would index out of bounds).
    if (!lab.progressive_image.id().is_empty())
    {
        auto const size = render_context.gpu_context->device.image_info(lab.progressive_image.id()).value().size;
        if (size.x != render_target_size.x || size.y != render_target_size.y)
        {
            if (rt_lab_test_running(lab))
            {
                rt_lab_test_end(render_context);
                lab.measurement = {};
            }
            rt_lab_destroy_capture_images(render_context);
            lab.rebuild_requested = true;
        }
    }
    if (lab.post_stabilization_enabled != lab.post_stabilization_built)
    {
        lab.rebuild_requested = true; // graph shape
    }
    bool const force_rebuild = lab.rebuild_requested;
    lab.rebuild_requested = false;

    // Apply the preset. The master switch lives in render_data; a test forces the lab (and RTGI) on and runs the
    // progressive preset in its reference phases, the active (tested) one otherwise, with the split history forced on
    // (convergence error metric, error prediction).
    RtLabTestSetup const & setup = lab.running;
    bool const testing = rt_lab_test_running(lab);
    bool const reference_phase = lab.phase == RtLabTestPhase::REFERENCE || lab.phase == RtLabTestPhase::MOVE_REFERENCE;
    bool const tested_phase = lab.phase == RtLabTestPhase::RUN || lab.phase == RtLabTestPhase::MOVE_WARMUP || lab.phase == RtLabTestPhase::MOVE_TESTED;
    bool const last_frame_of_phase =
        (lab.phase == RtLabTestPhase::REFERENCE && lab.phase_frame + 1 >= setup.reference_frames) ||
        (lab.phase == RtLabTestPhase::RUN && lab.phase_frame + 1 >= setup.frames) ||
        (lab.phase == RtLabTestPhase::MOVE_WARMUP && lab.phase_frame + 1 >= setup.move_warmup_frames) ||
        (lab.phase == RtLabTestPhase::MOVE_REFERENCE && lab.phase_frame + 1 >= setup.move_reference_frames) ||
        lab.phase == RtLabTestPhase::MOVE_TESTED;
    // Phases that start from a reset (MOVE_TESTED reprojects the restored history).
    bool const resets = lab.phase == RtLabTestPhase::REFERENCE || lab.phase == RtLabTestPhase::RUN ||
                        lab.phase == RtLabTestPhase::MOVE_WARMUP || lab.phase == RtLabTestPhase::MOVE_REFERENCE;
    i32 const preset = reference_phase ? RT_LAB_PRESET_PROGRESSIVE : lab.active_preset;
    i32 const enabled = testing ? 1 : render_data.rt_lab_settings.enabled;
    lab.backup_custom_history = lab.phase == RtLabTestPhase::MOVE_WARMUP || lab.phase == RtLabTestPhase::MOVE_TESTED;
    render_data.rt_lab_settings = lab.presets[preset];
    render_data.rt_lab_settings.enabled = enabled;
    render_data.rt_lab_settings.reset_history = resets && lab.phase_frame == 0 ? 1 : 0;
    render_data.rt_lab_settings.sample_count_input_clamp = lab.sample_count_input_clamp;
    render_data.rt_lab_settings.sample_count_debug_max = lab.sample_count_debug_max;
    if (tested_phase)
    {
        render_data.rt_lab_settings.split_history_enabled = 1;
    }
    if (testing)
    {
        render_data.rtgi_settings.enabled = 1;
    }
    // Temporal shock: flip the point light culling inside the shock window of the run, restore it after the test.
    bool const shock_configured = testing && setup.shock_frames > 0;
    if (shock_configured && !lab.shock_active)
    {
        lab.shock_saved_cull_point_lights = render_data.light_settings.cull_all_point_lights != 0;
        lab.shock_active = true;
    }
    if (lab.shock_active)
    {
        bool cull = lab.shock_saved_cull_point_lights;
        if (!shock_configured)
        {
            lab.shock_active = false;
        }
        else if (lab.phase == RtLabTestPhase::RUN && lab.phase_frame >= setup.shock_start && lab.phase_frame < setup.shock_start + setup.shock_frames)
        {
            cull = !cull;
        }
        render_data.light_settings.cull_all_point_lights = cull ? 1 : 0;
    }
    bool const static_reference = rt_lab_test_has_reference(setup) && !setup.move.enabled;
    switch (lab.phase)
    {
        case RtLabTestPhase::REFERENCE:
        {
            lab.capture_target = last_frame_of_phase ? RT_LAB_CAPTURE_PROGRESSIVE : RT_LAB_CAPTURE_NONE;
            break;
        }
        case RtLabTestPhase::RUN:
        {
            lab.measure_row = static_cast<u32>(lab.phase_frame);
            lab.capture_target = static_reference && last_frame_of_phase ? RT_LAB_CAPTURE_CUSTOM : RT_LAB_CAPTURE_NONE;
            break;
        }
        case RtLabTestPhase::MOVE_WARMUP:
        case RtLabTestPhase::MOVE_TESTED:
        {
            if (last_frame_of_phase)
            {
                lab.capture_target = RT_LAB_CAPTURE_CUSTOM;
                lab.save_custom_history = true;
            }
            lab.restore_custom_history = lab.phase == RtLabTestPhase::MOVE_TESTED;
            break;
        }
        case RtLabTestPhase::MOVE_REFERENCE:
        {
            if (last_frame_of_phase)
            {
                lab.capture_target = RT_LAB_CAPTURE_PROGRESSIVE;
                lab.analyze_step = static_cast<u32>(lab.move_step);
            }
            break;
        }
        default: break;
    }
    if (testing)
    {
        lab.phase_frame += 1;
    }

    // PGI update rate override (RT Lab "Force Full PGI Update").
    bool const force_pgi = lab.force_full_pgi_update && render_data.rt_lab_settings.enabled != 0;
    if (force_pgi && !lab.pgi_update_rate_overridden)
    {
        lab.saved_pgi_update_rate = render_data.pgi_settings.update_rate;
        lab.pgi_update_rate_overridden = true;
    }
    else if (!force_pgi && lab.pgi_update_rate_overridden)
    {
        render_data.pgi_settings.update_rate = lab.saved_pgi_update_rate;
        lab.pgi_update_rate_overridden = false;
    }
    if (force_pgi)
    {
        render_data.pgi_settings.update_rate = PGI_UPDATE_RATE_FULL;
    }

    // Analysis readback: continuous after a static reference, tagged with its step for every move reference step.
    auto const & readback = render_context.general_readback;
    bool const move_reference_running = rt_lab_test_move_phase(lab.phase) || (lab.phase == RtLabTestPhase::COLLECTING && setup.move.enabled);
    if ((lab.live_analysis || move_reference_running) && readback.rt_lab_convergence_pixels > 0)
    {
        lab.live_result = rt_lab_convergence_result_from_readback(readback);
        rt_lab_update_histogram_auto_range(lab);
        if (move_reference_running && readback.rt_lab_convergence_step != ~0u)
        {
            lab.measurement.reference = lab.live_result;
            lab.measurement.reference_steps.push_back(RtLabConvergenceStep{
                .step = static_cast<i32>(readback.rt_lab_convergence_step),
                .pixels = lab.live_result.pixels,
                .mse = lab.live_result.mse,
                .bias = lab.live_result.bias,
                .mean_abs_error = lab.live_result.mean_abs_error,
                .rms_predicted = lab.live_result.rms_predicted,
                .rms_actual = lab.live_result.rms_actual,
            });
        }
    }
    return force_rebuild;
}

///
/// === Callbacks ===
///

// Pointer style push (attach = device address of the attachment blob), push.size = size_image's size, one thread per
// size_image pixel.
template <typename Push>
static void rt_lab_dispatch(daxa::TaskInterface ti, RenderContext * render_context, Push push, daxa::TaskImageAttachmentIndex size_image, u32 block_size, u32 render_time, std::string const & pipeline_name)
{
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, render_time);
    auto const size = ti.info(size_image).value().size;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(pipeline_name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    push.attach = alloc.device_address;
    push.size = {size.x, size.y};
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(size.x, block_size), round_up_div(size.y, block_size), 1});
}

static void rt_lab_gradient_trace_callback(daxa::TaskInterface ti, RenderContext * render_context, daxa::Extent3D lab_size, bool per_pixel)
{
    auto const & AT = RtLabGradientTraceH::Info::AT;
    RtLabGradientTracePush push = {};
    push.lab_size = {lab_size.x, lab_size.y};
    push.per_pixel = per_pixel ? 1u : 0u;
    // One thread per record: per cell, or per lab pixel (all base rays).
    rt_lab_dispatch(ti, render_context, push, AT.gradient_record_data, RT_LAB_RAYS_X, RenderTimes::index<"RT_LAB", "GRADIENT_TRACE">(), rt_lab_gradient_trace_compile_info().name);
}

static void rt_lab_gradient_blur_callback(daxa::TaskInterface ti, RenderContext * render_context, bool vertical, daxa::Extent3D lab_size, bool per_pixel)
{
    auto const & AT = RtLabGradientBlurH::Info::AT;
    RtLabGradientBlurPush push = {};
    push.vertical = vertical ? 1u : 0u;
    push.per_pixel = per_pixel ? 1u : 0u;
    push.lab_size = {lab_size.x, lab_size.y};
    u32 const render_time = vertical ? RenderTimes::index<"RT_LAB", "GRADIENT_BLUR_V">() : RenderTimes::index<"RT_LAB", "GRADIENT_BLUR_H">();
    rt_lab_dispatch(ti, render_context, push, vertical ? AT.temporal_gradient : AT.blur_sums_dst, RT_LAB_RAYS_X, render_time, rt_lab_gradient_blur_compile_info().name);
}

static void rt_lab_gradient_dilate_callback(daxa::TaskInterface ti, RenderContext * render_context, bool vertical, daxa::Extent3D lab_size)
{
    auto const & AT = RtLabGradientDilateH::Info::AT;
    RtLabGradientDilatePush push = {};
    push.vertical = vertical ? 1u : 0u;
    push.lab_size = {lab_size.x, lab_size.y};
    u32 const render_time = vertical ? RenderTimes::index<"RT_LAB", "GRADIENT_DILATE_V">() : RenderTimes::index<"RT_LAB", "GRADIENT_DILATE_H">();
    rt_lab_dispatch(ti, render_context, push, AT.dst, RT_LAB_RAYS_X, render_time, rt_lab_gradient_dilate_compile_info().name);
}

static void rt_lab_demand_callback(daxa::TaskInterface ti, RenderContext * render_context, bool history_bound, bool gradient_bound)
{
    auto const & AT = RtLabDemandH::Info::AT;
    RtLabDemandPush push = {};
    push.history_bound = history_bound ? 1u : 0u;
    push.gradient_bound = gradient_bound ? 1u : 0u;
    rt_lab_dispatch(ti, render_context, push, AT.ray_demand, RT_LAB_RAYS_X, RenderTimes::index<"RT_LAB", "RAY_DEMAND">(), rt_lab_demand_compile_info().name);
}

static void rt_lab_distribute_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 ray_capacity)
{
    auto const & AT = RtLabDistributeH::Info::AT;
    RtLabDistributePush push = {};
    push.ray_capacity = ray_capacity;
    rt_lab_dispatch(ti, render_context, push, AT.ray_alloc, RT_LAB_RAYS_X, RenderTimes::index<"RT_LAB", "RAY_DISTRIBUTE">(), rt_lab_distribute_compile_info().name);
}

// One thread per ray list entry: dispatched over the whole capacity, entries past this frame's rays exit.
static void rt_lab_trace_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 ray_capacity, bool ray_lengths_bound)
{
    auto const & AT = RtLabTraceH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RT_LAB", "TRACE">());
    auto const size = ti.info(AT.lab_depth).value().size;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_trace_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLabTracePush push = {};
    push.attach = alloc.device_address;
    push.size = {size.x, size.y};
    push.ray_capacity = ray_capacity;
    push.ray_lengths_bound = ray_lengths_bound ? 1u : 0u;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(ray_capacity, static_cast<u32>(RT_LAB_TRACE_THREADS)), 1, 1});
}

static void rt_lab_ray_filter_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 ray_capacity)
{
    auto const & AT = RtLabRayFilterH::Info::AT;
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RT_LAB", "RAY_FILTER">());
    auto const size = ti.info(AT.lab_depth).value().size;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_ray_filter_compile_info().name));
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    RtLabRayFilterPush push = {};
    push.attach = alloc.device_address;
    push.size = {size.x, size.y};
    push.ray_capacity = ray_capacity;
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(ray_capacity, static_cast<u32>(RT_LAB_TRACE_THREADS)), 1, 1});
}

static void rt_lab_gather_callback(daxa::TaskInterface ti, RenderContext * render_context, bool gradient_enabled, bool filtered_bound, bool gradient_per_pixel, bool raw_luminance_bound)
{
    auto const & AT = RtLabGatherH::Info::AT;
    RtLabGatherPush push = {};
    push.raw_luminance_bound = raw_luminance_bound ? 1u : 0u;
    push.filtered_bound = filtered_bound ? 1u : 0u;
    push.gradient_enabled = gradient_enabled ? 1u : 0u;
    push.gradient_per_pixel = gradient_per_pixel ? 1u : 0u;
    rt_lab_dispatch(ti, render_context, push, AT.lab_raw_diffuse, RT_LAB_RAYS_X, RenderTimes::index<"RT_LAB", "RAY_GATHER">(), rt_lab_gather_compile_info().name);
}

static void rt_lab_anti_lag_downsample_callback(daxa::TaskInterface ti, RenderContext * render_context, daxa::Extent3D lab_size)
{
    auto const & AT = RtLabAntiLagDownsampleH::Info::AT;
    RtLabAntiLagDownsamplePush push = {};
    push.lab_size = {lab_size.x, lab_size.y};
    rt_lab_dispatch(ti, render_context, push, AT.cell_sums, RT_LAB_ANTI_LAG_X, RenderTimes::index<"RT_LAB", "ANTI_LAG_DOWNSAMPLE">(), rt_lab_anti_lag_downsample_compile_info().name);
}

static void rt_lab_anti_lag_blur_callback(daxa::TaskInterface ti, RenderContext * render_context, bool vertical, daxa::Extent3D lab_size, bool gradient_bound)
{
    auto const & AT = RtLabAntiLagBlurH::Info::AT;
    RtLabAntiLagBlurPush push = {};
    push.vertical = vertical ? 1u : 0u;
    push.gradient_bound = gradient_bound ? 1u : 0u;
    push.lab_size = {lab_size.x, lab_size.y};
    u32 const render_time = vertical ? RenderTimes::index<"RT_LAB", "ANTI_LAG_BLUR_V">() : RenderTimes::index<"RT_LAB", "ANTI_LAG_BLUR_H">();
    rt_lab_dispatch(ti, render_context, push, vertical ? AT.history : AT.blur_dst, RT_LAB_ANTI_LAG_X, render_time, rt_lab_anti_lag_blur_compile_info().name);
}

static void rt_lab_temporal_callback(daxa::TaskInterface ti, RenderContext * render_context, bool split_enabled, bool sh_enabled, bool gradient_bound, bool anti_lag_bound, bool raw_luminance_bound, bool ray_length_bound)
{
    auto const & AT = RtLabTemporalH::Info::AT;
    RtLabTemporalPush push = {};
    push.ray_length_bound = ray_length_bound ? 1u : 0u;
    push.raw_luminance_bound = raw_luminance_bound ? 1u : 0u;
    push.anti_lag_bound = anti_lag_bound ? 1u : 0u;
    push.split_enabled = split_enabled ? 1u : 0u;
    push.sh_enabled = sh_enabled ? 1u : 0u;
    push.gradient_bound = gradient_bound ? 1u : 0u;
    rt_lab_dispatch(ti, render_context, push, AT.accumulated, RT_LAB_TEMPORAL_X, RenderTimes::index<"RT_LAB", "TEMPORAL">(), rt_lab_temporal_compile_info().name);
}

// Firefly energy weighting of a spatial chain's first pass: the setting is on and the temporal pass accumulated the raw
// luminance (ray filter on).
static auto rt_lab_firefly_weight(RenderContext * render_context, u32 pass, bool raw_luminance_accumulated) -> u32
{
    return pass == 0 && raw_luminance_accumulated && render_context->render_data.rt_lab_settings.firefly_weight_enabled != 0 ? 1u : 0u;
}

static void rt_lab_blur_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass, bool split_enabled, bool variance_bound, u32 gradient_mode, bool last_pass, bool raw_luminance_accumulated, bool ray_length_bound)
{
    auto const & AT = RtLabBlurH::Info::AT;
    RtLabBlurPush push = {};
    push.ray_length_bound = ray_length_bound ? 1u : 0u;
    push.firefly_weight = rt_lab_firefly_weight(render_context, pass, raw_luminance_accumulated);
    push.gradient_mode = gradient_mode;
    push.last_pass = last_pass ? 1u : 0u;
    push.split_enabled = split_enabled ? 1u : 0u;
    push.variance_bound = variance_bound ? 1u : 0u;
    push.clamp_input_count = pass == 0 ? 1u : 0u;
    push.direction = pass == 0 ? daxa_i32vec2{1, 0} : daxa_i32vec2{0, 1};
    u32 const render_time = pass == 0 ? RenderTimes::index<"RT_LAB", "BLUR_HORIZONTAL">() : RenderTimes::index<"RT_LAB", "BLUR_VERTICAL">();
    rt_lab_dispatch(ti, render_context, push, AT.dst, RT_LAB_BLUR_X, render_time, rt_lab_blur_compile_info().name);
}

static void rt_lab_atrous_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass, bool split_enabled, bool variance_bound, u32 gradient_mode, bool last_pass, bool raw_luminance_accumulated, bool ray_length_bound)
{
    auto const & AT = RtLabBlurH::Info::AT;
    RtLabBlurPush push = {};
    push.ray_length_bound = ray_length_bound ? 1u : 0u;
    push.firefly_weight = rt_lab_firefly_weight(render_context, pass, raw_luminance_accumulated);
    push.gradient_mode = gradient_mode;
    push.last_pass = last_pass ? 1u : 0u;
    push.split_enabled = split_enabled ? 1u : 0u;
    push.variance_bound = variance_bound ? 1u : 0u;
    push.step = 1 << pass;
    push.clamp_input_count = pass == 0 ? 1u : 0u;
    u32 const render_time = [&]() -> u32
    {
        switch (pass)
        {
            case 1: return RenderTimes::index<"RT_LAB", "ATROUS1">();
            case 2: return RenderTimes::index<"RT_LAB", "ATROUS2">();
            case 3: return RenderTimes::index<"RT_LAB", "ATROUS3">();
            case 4: return RenderTimes::index<"RT_LAB", "ATROUS4">();
            case 5: return RenderTimes::index<"RT_LAB", "ATROUS5">();
            case 6: return RenderTimes::index<"RT_LAB", "ATROUS6">();
            case 7: return RenderTimes::index<"RT_LAB", "ATROUS7">();
            default: return RenderTimes::index<"RT_LAB", "ATROUS0">();
        }
    }();
    rt_lab_dispatch(ti, render_context, push, AT.dst, RT_LAB_BLUR_X, render_time, rt_lab_atrous_compile_info().name);
}

static void rt_lab_coarse_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 pass, bool split_enabled, bool raw_luminance_accumulated)
{
    auto const & AT = RtLabCoarseH::Info::AT;
    RtLabCoarsePush push = {};
    push.firefly_weight = rt_lab_firefly_weight(render_context, pass, raw_luminance_accumulated);
    push.split_enabled = split_enabled ? 1u : 0u;
    push.step = 1 << pass;
    push.clamp_input_count = pass == 0 ? 1u : 0u;
    u32 const render_time = [&]() -> u32
    {
        switch (pass)
        {
            case 1: return RenderTimes::index<"RT_LAB", "COARSE1">();
            case 2: return RenderTimes::index<"RT_LAB", "COARSE2">();
            case 3: return RenderTimes::index<"RT_LAB", "COARSE3">();
            case 4: return RenderTimes::index<"RT_LAB", "COARSE4">();
            case 5: return RenderTimes::index<"RT_LAB", "COARSE5">();
            case 6: return RenderTimes::index<"RT_LAB", "COARSE6">();
            case 7: return RenderTimes::index<"RT_LAB", "COARSE7">();
            default: return RenderTimes::index<"RT_LAB", "COARSE0">();
        }
    }();
    rt_lab_dispatch(ti, render_context, push, AT.dst_sh_y, RT_LAB_COARSE_X, render_time, rt_lab_coarse_compile_info().name);
}

static void rt_lab_resolve_callback(daxa::TaskInterface ti, RenderContext * render_context, bool split_enabled, bool spatial_ran, bool coarse_enabled, bool stabilize)
{
    auto const & AT = RtLabResolveH::Info::AT;
    RtLabResolvePush push = {};
    push.stabilize = 0u;
    if (stabilize)
    {
        push.stabilize = RT_LAB_STABILIZE_ENABLED | RT_LAB_STABILIZE_VARIANCE_BOUND;
        if (render_context->rt_lab.post_stabilization_temporal_variance)
        {
            push.stabilize |= RT_LAB_STABILIZE_VARIANCE_CLAMP;
        }
    }
    push.stabilize_alpha = render_context->rt_lab.post_stabilization_alpha;
    push.stabilize_gamma = render_context->rt_lab.post_stabilization_gamma;
    push.stabilize_radius = static_cast<u32>(std::clamp(render_context->rt_lab.post_stabilization_radius, 1, RT_LAB_STABILIZE_MAX_RADIUS));
    push.stabilize_disable_clamp = render_context->rt_lab.post_stabilization_clamp ? 0u : 1u;
    push.split_enabled = split_enabled ? 1u : 0u;
    push.coarse_enabled = coarse_enabled ? 1u : 0u;
    push.clamp_input_count = spatial_ran ? 0u : 1u;
    rt_lab_dispatch(ti, render_context, push, AT.resolved, RT_LAB_RESOLVE_X, RenderTimes::index<"RT_LAB", "RESOLVE">(), rt_lab_resolve_compile_info().name);
}

// Only dispatches on the capture frames of a test's reference comparison (RtLabState::capture_target).
static void rt_lab_convergence_capture_callback(daxa::TaskInterface ti, RenderContext * render_context, bool split_enabled)
{
    if (render_context->rt_lab.capture_target == RT_LAB_CAPTURE_NONE)
    {
        return;
    }
    auto const & AT = RtLabConvergenceCaptureH::Info::AT;
    RtLabConvergenceCapturePush push = {};
    push.target = render_context->rt_lab.capture_target;
    push.split_enabled = split_enabled ? 1u : 0u;
    rt_lab_dispatch(ti, render_context, push, AT.resolved, RT_LAB_CONVERGENCE_X, RenderTimes::index<"RT_LAB", "CONVERGENCE_CAPTURE">(), rt_lab_convergence_capture_compile_info().name);
}

// Runs every frame after a static reference comparison (error stats readback + debug views), and on the progressive
// capture frame of every move reference step (tagged with the step).
static void rt_lab_convergence_analyze_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & lab = render_context->rt_lab;
    if (!lab.live_analysis && lab.analyze_step == ~0u)
    {
        return;
    }
    auto const & AT = RtLabConvergenceAnalyzeH::Info::AT;
    RtLabConvergenceAnalyzePush push = {};
    push.step = lab.analyze_step;
    push.histogram_range = lab.histogram_range;
    if (lab.histogram_auto_range && lab.histogram_auto_range_value > 0.0f)
    {
        push.histogram_range = lab.histogram_auto_range_value;
    }
    rt_lab_dispatch(ti, render_context, push, AT.view_cam_depth, RT_LAB_CONVERGENCE_X, RenderTimes::index<"RT_LAB", "CONVERGENCE_ANALYZE">(), rt_lab_convergence_analyze_compile_info().name);
}

// Measurement passes: only dispatch while a test run records this frame (and only for its checked metrics; the store
// pass always copies the row, unrecorded metrics stay empty).
static void rt_lab_measure_accumulate_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & lab = render_context->rt_lab;
    if (lab.measure_row == ~0u || (lab.measurement.metrics & RT_LAB_TEST_METRIC_HISTOGRAMS) == 0)
    {
        return;
    }
    auto gpu_timer = render_context->render_times.scoped_gpu_timer(ti.recorder, RenderTimes::index<"RT_LAB", "MEASURE">());
    auto const image_size = render_context->render_data.settings.render_target_size;
    RtLabMeasureAccumulatePush push = {};
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    push.attach = alloc.device_address;
    push.size = {round_up_div(image_size.x, RT_LAB_MEASURE_TILE), round_up_div(image_size.y, RT_LAB_MEASURE_TILE)};
    push.frame = lab.measure_row;
    push.window = static_cast<f32>(lab.measurement.window);
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_measure_accumulate_compile_info().name));
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(push.size.x, RT_LAB_MEASURE_X), round_up_div(push.size.y, RT_LAB_MEASURE_Y), 1});
}

static void rt_lab_measure_lag_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & lab = render_context->rt_lab;
    if (lab.measure_row == ~0u || lab.measurement.lag_tiles_x <= 0)
    {
        return;
    }
    RtLabMeasureLagPush push = {};
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    push.attach = alloc.device_address;
    push.tiles = {static_cast<u32>(lab.measurement.lag_tiles_x), static_cast<u32>(lab.measurement.lag_tiles_y)};
    push.row = lab.measure_row;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_measure_lag_compile_info().name));
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(push.tiles.x, static_cast<u32>(RT_LAB_MEASURE_X)), round_up_div(push.tiles.y, static_cast<u32>(RT_LAB_MEASURE_Y)), 1});
}

static void rt_lab_measure_store_callback(daxa::TaskInterface ti, RenderContext * render_context)
{
    auto const & lab = render_context->rt_lab;
    if (lab.measure_row == ~0u)
    {
        return;
    }
    RtLabMeasureStorePush push = {};
    auto alloc = ti.allocator->allocate(ti.attachment_shader_blob.size()).value();
    std::memcpy(alloc.host_address, ti.attachment_shader_blob.data(), ti.attachment_shader_blob.size());
    push.attach = alloc.device_address;
    push.row = lab.measure_row;
    ti.recorder.set_pipeline(*render_context->gpu_context->compute_pipelines.at(rt_lab_measure_store_compile_info().name));
    ti.recorder.push_constant(push);
    ti.recorder.dispatch({round_up_div(static_cast<u32>(RT_LAB_MEASURE_ROW_SIZE), static_cast<u32>(RT_LAB_MEASURE_STORE_X)), 1, 1});
}

///
/// === Task graph ===
///

auto tasks_rt_lab_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult
{
    auto & render_context = info.render_context;
    auto const & settings = render_context.render_data.rt_lab_settings;
    auto const render_target_size = render_context.render_data.settings.render_target_size;
    u32 const lab_div = settings.half_res ? 2u : 1u;
    auto const lab_size = daxa::Extent3D{render_target_size.x / lab_div, render_target_size.y / lab_div, 1};
    auto const full_size = daxa::Extent3D{render_target_size.x, render_target_size.y, 1};

    // History backups (move reference, tested preset graphs only): every persistent history image gets a backup copy
    // that survives the graph rebuilds of the preset switches. Restored into .previous() right after the image is
    // created, before its first use (MOVE_TESTED frames), saved from .current() after the resolve (tested capture
    // frames). The copies only run when requested this frame (RtLabState::restore_custom_history / save_custom_history).
    auto & lab_state = render_context.rt_lab;
    bool const backup_histories = lab_state.backup_custom_history;
    std::vector<std::pair<daxa::TaskImageView, daxa::TaskImageView>> history_saves = {}; // (current, backup)
    auto create_history_image = [&](daxa::TaskImageInfo const & image_info) -> daxa::TaskImageView
    {
        auto image = info.tg.create_task_image(image_info);
        if (!backup_histories)
        {
            return image;
        }
        std::string const name = std::string(image_info.name);
        auto [entry, inserted] = lab_state.history_backups.try_emplace(name, RtLabState::HistoryBackup{
            .image = daxa::ExternalTaskImage{{.name = name + "_backup"}},
        });
        auto & backup = entry->second;
        if (!backup.image.id().is_empty() && (backup.format != image_info.format || backup.size.x != image_info.size.x || backup.size.y != image_info.size.y))
        {
            render_context.gpu_context->device.destroy_image(backup.image.id());
            backup.image.set_image(daxa::ImageId{});
        }
        if (backup.image.id().is_empty())
        {
            backup.image.set_image(render_context.gpu_context->device.create_image({
                .format = image_info.format,
                .size = image_info.size,
                .usage = daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::TRANSFER_DST,
                .name = name + "_backup",
            }));
            backup.format = image_info.format;
            backup.size = image_info.size;
        }
        info.tg.register_image(backup.image);
        daxa::TaskImageView const backup_view = backup.image.view();
        daxa::TaskImageView const previous = image.previous();
        info.tg.add_task(daxa::InlineTask::Transfer(name + "_restore")
            .reads(backup_view)
            .writes(previous)
            .executes([&render_context, backup_view, previous](daxa::TaskInterface ti)
            {
                if (!render_context.rt_lab.restore_custom_history)
                {
                    return;
                }
                ti.recorder.copy_image_to_image({
                    .src_image = ti.id(backup_view),
                    .dst_image = ti.id(previous),
                    .extent = ti.info(backup_view).value().size,
                });
            }));
        history_saves.push_back({image.current(), backup_view});
        return image;
    };

    // Lab resolution geometry, written by the demand pass. Double buffered: the temporal pass reprojects against last frame's.
    auto lab_depth = create_history_image({
        .format = daxa::Format::R32_SFLOAT,
        .size = lab_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "rt_lab_depth_persistent",
    });
    auto lab_face_normal = create_history_image({
        .format = daxa::Format::R32_UINT,
        .size = lab_size,
        .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
        .name = "rt_lab_face_normal_persistent",
    });
    auto lab_raw_diffuse = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = lab_size,
        .name = "rt_lab_raw_diffuse",
    });

    // The coarse fallback filters RTGI style SH-Y + CoCg from the temporal pass (needs temporal + spatial).
    bool const coarse_enabled = settings.temporal_enabled != 0 && settings.spatial_enabled != 0 && settings.coarse_enabled != 0;
    daxa::TaskImageView coarse_sh_y_input = daxa::NullTaskImage;
    daxa::TaskImageView coarse_cocg_count_input = daxa::NullTaskImage;

    // Main temporal history (created here: the demand pass reads last frame's history length).
    // "Smart": fp16 short history. "Progressive": fp32 long accumulation (up to progressive_max_samples).
    bool const progressive = settings.temporal_progressive != 0;
    auto const history_format = progressive ? daxa::Format::R32G32B32A32_SFLOAT : daxa::Format::R16G16B16A16_SFLOAT;
    daxa::TaskImageView history_previous = daxa::NullTaskImage;
    daxa::TaskImageView history_current = daxa::NullTaskImage;
    // Per sample variance (temporal pass output), created here too: the demand pass reads last frame's (variance laziness).
    daxa::TaskImageView variance_image = daxa::NullTaskImage;
    if (settings.temporal_enabled)
    {
        variance_image = create_history_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = lab_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = "rt_lab_variance_persistent",
        });
        auto history = create_history_image({
            .format = history_format,
            .size = lab_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = progressive ? "rt_lab_history_progressive_persistent" : "rt_lab_history_smart_persistent",
        });
        history_previous = history.previous();
        history_current = history.current();
    }

    // Ray redistribution (rt_lab_rays.inl): demand -> distribute -> trace (ray list) -> gather.
    u32 const ray_capacity = lab_size.x * lab_size.y * rt_lab_ray_capacity_per_pixel(settings);
    auto ray_counters = info.tg.create_task_buffer({
        .size = sizeof(RtLabRayCounters),
        .name = "rt_lab_ray_counters",
    });
    auto ray_list = info.tg.create_task_buffer({
        .size = sizeof(u32) * ray_capacity,
        .name = "rt_lab_ray_list",
    });
    auto rays = info.tg.create_task_buffer({
        .size = sizeof(RtLabRay) * ray_capacity,
        .name = "rt_lab_rays",
    });
    // Ray length guide: hit distance per ray (temporal 3x3 means, fine spatial guide). Graph shape: mode != 0, temporal.
    bool const ray_length_guide = settings.ray_length_guide_mode != 0 && settings.temporal_enabled != 0;
    daxa::TaskBufferView ray_lengths = daxa::NullTaskBuffer;
    if (ray_length_guide)
    {
        ray_lengths = info.tg.create_task_buffer({
            .size = sizeof(f32) * ray_capacity,
            .name = "rt_lab_ray_lengths",
        });
    }
    auto ray_demand = info.tg.create_task_image({
        .format = daxa::Format::R32_UINT,
        .size = lab_size,
        .name = "rt_lab_ray_demand",
    });
    auto ray_alloc = info.tg.create_task_image({
        .format = daxa::Format::R32G32_UINT,
        .size = lab_size,
        .name = "rt_lab_ray_alloc",
    });
    info.tg.clear_buffer({.buffer = ray_counters, .name = "rt_lab_ray_counters_clear"});

    // Temporal gradient (A-SVGF style, rt_lab_rays.inl): re-trace last frame's records -> blur -> lambda, consumed by the
    // demand + temporal pass; the gather pass selects and writes this frame's records.
    bool const gradient_enabled = settings.temporal_enabled != 0 && settings.temporal_gradient_enabled != 0;
    auto const gradient_size = daxa::Extent3D{round_up_div(lab_size.x, static_cast<u32>(RT_LAB_GRADIENT_CELL)), round_up_div(lab_size.y, static_cast<u32>(RT_LAB_GRADIENT_CELL)), 1};
    // All base rays: one record per lab pixel, else one per cell.
    bool const gradient_per_pixel = gradient_enabled && settings.temporal_gradient_all_base_rays != 0;
    // Dense base ray records: per 8x8 tile one slot per base pixel (rt_lab_gradient_base_slots, same formula).
    u32 const gradient_base_slots = std::clamp(static_cast<u32>(std::round(std::clamp(settings.min_ray_budget, 0.0f, 1.0f) * 64.0f)), 1u, 64u);
    auto const record_size = gradient_per_pixel
        ? daxa::Extent3D{round_up_div(lab_size.x, static_cast<u32>(RT_LAB_RAYS_X)) * gradient_base_slots, round_up_div(lab_size.y, static_cast<u32>(RT_LAB_RAYS_Y)), 1}
        : gradient_size;
    daxa::TaskImageView gradient_record_origin = daxa::NullTaskImage;
    daxa::TaskImageView gradient_record_surface = daxa::NullTaskImage;
    daxa::TaskImageView gradient_record_data = daxa::NullTaskImage;
    daxa::TaskImageView temporal_gradient = daxa::NullTaskImage;
    if (gradient_enabled)
    {
        auto record_origin = create_history_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = record_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = "rt_lab_gradient_record_origin_persistent",
        });
        auto record_surface = create_history_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = record_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = "rt_lab_gradient_record_surface_persistent",
        });
        auto record_data = create_history_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = record_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = "rt_lab_gradient_record_data_persistent",
        });
        gradient_record_origin = record_origin.current();
        gradient_record_surface = record_surface.current();
        gradient_record_data = record_data.current();
        auto gradient_samples = info.tg.create_task_image({
            .format = daxa::Format::R32_UINT,
            .size = gradient_size,
            .name = "rt_lab_gradient_samples",
        });
        temporal_gradient = info.tg.create_task_image({
            .format = daxa::Format::R32G32B32A32_UINT, // lambda, cell depth, cell normal (edge aware upscale)
            .size = gradient_size,
            .name = "rt_lab_temporal_gradient",
        });
        info.tg.clear_image({.view = gradient_samples, .name = "rt_lab_gradient_samples_clear"});
        auto gradient_delta_sum = info.tg.create_task_image({
            .format = daxa::Format::R32_UINT,
            .size = gradient_size,
            .name = "rt_lab_gradient_delta_sum",
        });
        auto gradient_max_sum = info.tg.create_task_image({
            .format = daxa::Format::R32_UINT,
            .size = gradient_size,
            .name = "rt_lab_gradient_max_sum",
        });
        auto gradient_count_sum = info.tg.create_task_image({
            .format = daxa::Format::R32_UINT,
            .size = gradient_size,
            .name = "rt_lab_gradient_count_sum",
        });
        info.tg.clear_image({.view = gradient_delta_sum, .name = "rt_lab_gradient_delta_sum_clear"});
        info.tg.clear_image({.view = gradient_max_sum, .name = "rt_lab_gradient_max_sum_clear"});
        info.tg.clear_image({.view = gradient_count_sum, .name = "rt_lab_gradient_count_sum_clear"});
        auto gradient_old_sum = info.tg.create_task_image({
            .format = daxa::Format::R32_UINT,
            .size = gradient_size,
            .name = "rt_lab_gradient_old_sum",
        });
        info.tg.clear_image({.view = gradient_old_sum, .name = "rt_lab_gradient_old_sum_clear"});
        info.tg.add_task(daxa::HeadTask<RtLabGradientTraceH::Info>()
            .head_views(RtLabGradientTraceH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .view_cam_depth = info.view_cam_depth,
                .view_cam_face_normals = info.view_cam_face_normals,
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                .gradient_record_origin = record_origin.previous(),
                .gradient_record_surface = record_surface.previous(),
                .gradient_record_data = record_data.previous(),
                .gradient_samples = gradient_samples,
                .gradient_delta_sum = gradient_delta_sum,
                .gradient_max_sum = gradient_max_sum,
                .gradient_count_sum = gradient_count_sum,
                .gradient_old_sum = gradient_old_sum,
                .mesh_instances = info.mesh_instances,
                .sky = info.sky,
                .sky_transmittance = info.sky_transmittance,
                .light_mask_volume = info.light_mask_volume,
                .pgi_color = info.pgi_color,
                .pgi_visibility = info.pgi_visibility,
                .pgi_info = info.pgi_info,
                .pgi_requests = info.pgi_requests,
                .tlas = info.tlas,
                .vsm_globals = info.vsm_globals,
                .vsm_point_lights = info.vsm_point_lights,
                .vsm_spot_lights = info.vsm_spot_lights,
                .vsm_memory_block = info.vsm_memory_block,
                .vsm_point_spot_page_table = info.vsm_point_spot_page_table,
            })
            .executes(rt_lab_gradient_trace_callback, &render_context, lab_size, gradient_per_pixel));
        auto gradient_history = info.tg.create_task_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT, // smoothed sums + cell depth (temporal_gradient_history_alpha)
            .size = gradient_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = "rt_lab_gradient_history_persistent",
        });
        auto gradient_blur_moments = info.tg.create_task_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT, // sum w^2 delta^2, sum w^2 delta old, sum w^2 old^2 (Kalman)
            .size = gradient_size,
            .name = "rt_lab_gradient_blur_moments",
        });
        auto gradient_blur_sums = info.tg.create_task_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT, // sum w delta, sum w max, sum w samples
            .size = gradient_size,
            .name = "rt_lab_gradient_blur_sums",
        });
        info.tg.add_task(daxa::HeadTask<RtLabGradientBlurH::Info>("RtLabGradientBlurHorizontal")
            .head_views(RtLabGradientBlurH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .view_cam_depth = info.view_cam_depth,
                .view_cam_face_normals = info.view_cam_face_normals,
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                .gradient_samples = gradient_samples,
                .gradient_delta_sum = gradient_delta_sum,
                .gradient_max_sum = gradient_max_sum,
                .gradient_count_sum = gradient_count_sum,
                .blur_sums_dst = gradient_blur_sums,
                .blur_sums = daxa::NullTaskImage,
                .temporal_gradient = daxa::NullTaskImage,
                .gradient_history = daxa::NullTaskImage,
                .gradient_history_dst = daxa::NullTaskImage,
                .gradient_old_sum = gradient_old_sum,
                .blur_moments_dst = gradient_blur_moments,
                .blur_moments = daxa::NullTaskImage,
            })
            .executes(rt_lab_gradient_blur_callback, &render_context, false, lab_size, gradient_per_pixel));
        // Dilation on: the blur writes an intermediate, the dilation (H, V) writes temporal_gradient (read by the consumers).
        bool const dilate = settings.temporal_gradient_dilation_radius > 0;
        daxa::TaskImageView gradient_blurred = temporal_gradient;
        daxa::TaskImageView gradient_dilate_h = daxa::NullTaskImage;
        if (dilate)
        {
            gradient_blurred = info.tg.create_task_image({
                .format = daxa::Format::R32G32B32A32_UINT,
                .size = gradient_size,
                .name = "rt_lab_temporal_gradient_blurred",
            });
            gradient_dilate_h = info.tg.create_task_image({
                .format = daxa::Format::R32G32B32A32_UINT,
                .size = gradient_size,
                .name = "rt_lab_temporal_gradient_dilate_h",
            });
        }
        info.tg.add_task(daxa::HeadTask<RtLabGradientBlurH::Info>("RtLabGradientBlurVertical")
            .head_views(RtLabGradientBlurH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .view_cam_depth = info.view_cam_depth,
                .view_cam_face_normals = info.view_cam_face_normals,
                .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                .gradient_samples = daxa::NullTaskImage,
                .gradient_delta_sum = daxa::NullTaskImage,
                .gradient_max_sum = daxa::NullTaskImage,
                .gradient_count_sum = daxa::NullTaskImage,
                .blur_sums_dst = daxa::NullTaskImage,
                .blur_sums = gradient_blur_sums,
                .temporal_gradient = gradient_blurred,
                .gradient_history = gradient_history.previous(),
                .gradient_history_dst = gradient_history.current(),
                .gradient_old_sum = daxa::NullTaskImage,
                .blur_moments_dst = daxa::NullTaskImage,
                .blur_moments = gradient_blur_moments,
            })
            .executes(rt_lab_gradient_blur_callback, &render_context, true, lab_size, gradient_per_pixel));
        if (dilate)
        {
            info.tg.add_task(daxa::HeadTask<RtLabGradientDilateH::Info>("RtLabGradientDilateHorizontal")
                .head_views(RtLabGradientDilateH::Info::Views{
                    .globals = render_context.tgpu_render_data.view(),
                    .src = gradient_blurred,
                    .dst = gradient_dilate_h,
                })
                .executes(rt_lab_gradient_dilate_callback, &render_context, false, lab_size));
            info.tg.add_task(daxa::HeadTask<RtLabGradientDilateH::Info>("RtLabGradientDilateVertical")
                .head_views(RtLabGradientDilateH::Info::Views{
                    .globals = render_context.tgpu_render_data.view(),
                    .src = gradient_dilate_h,
                    .dst = temporal_gradient,
                })
                .executes(rt_lab_gradient_dilate_callback, &render_context, true, lab_size));
        }
    }

    info.tg.add_task(daxa::HeadTask<RtLabDemandH::Info>()
        .head_views(RtLabDemandH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .view_cam_depth = info.view_cam_depth,
            .view_cam_face_normals = info.view_cam_face_normals,
            .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
            .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .lab_depth_history = lab_depth.previous(),
            .lab_face_normal_history = lab_face_normal.previous(),
            .history = history_previous,
            .ray_demand = ray_demand,
            .ray_counters = ray_counters,
            .debug_image = info.debug_image,
            .temporal_gradient = temporal_gradient,
            .variance_history = variance_image != daxa::NullTaskImage ? variance_image.previous() : daxa::NullTaskImage,
        })
        .executes(rt_lab_demand_callback, &render_context, settings.temporal_enabled != 0, gradient_enabled));
    info.tg.add_task(daxa::HeadTask<RtLabDistributeH::Info>()
        .head_views(RtLabDistributeH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .ray_demand = ray_demand,
            .ray_counters = ray_counters,
            .ray_list = ray_list,
            .ray_alloc = ray_alloc,
        })
        .executes(rt_lab_distribute_callback, &render_context, ray_capacity));

    info.tg.add_task(daxa::HeadTask<RtLabTraceH::Info>()
        .head_views(RtLabTraceH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .ray_counters = ray_counters,
            .ray_list = ray_list,
            .ray_alloc = ray_alloc,
            .rays = rays,
            .mesh_instances = info.mesh_instances,
            .sky = info.sky,
            .sky_transmittance = info.sky_transmittance,
            .light_mask_volume = info.light_mask_volume,
            .pgi_color = info.pgi_color,
            .pgi_visibility = info.pgi_visibility,
            .pgi_info = info.pgi_info,
            .pgi_requests = info.pgi_requests,
            .tlas = info.tlas,
            .vsm_globals = info.vsm_globals,
            .vsm_point_lights = info.vsm_point_lights,
            .vsm_spot_lights = info.vsm_spot_lights,
            .vsm_memory_block = info.vsm_memory_block,
            .vsm_point_spot_page_table = info.vsm_point_spot_page_table,
            .ray_lengths = ray_length_guide ? ray_lengths : daxa::NullTaskBuffer,
        })
        .executes(rt_lab_trace_callback, &render_context, ray_capacity, ray_length_guide));

    // Ray filter (RtLabSettings::ray_filter_*): clamped rays for the pixel means, temporal SH and coarse clamp; the
    // temporal gradient selection keeps the raw rays.
    daxa::TaskBufferView filtered_rays = rays;
    bool const ray_filter = settings.ray_filter_mode != RT_LAB_RAY_FILTER_OFF;
    // Firefly energy weighting: the raw mean luminance, accumulated by the temporal pass (lab_variance .a). Always with
    // the ray filter + temporal, so the checkbox needs no rebuild.
    bool const raw_luminance_accumulated = ray_filter && settings.temporal_enabled != 0;
    daxa::TaskImageView lab_raw_luminance = daxa::NullTaskImage;
    if (raw_luminance_accumulated)
    {
        lab_raw_luminance = info.tg.create_task_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = lab_size,
            .name = "rt_lab_raw_luminance",
        });
    }
    if (ray_filter)
    {
        filtered_rays = info.tg.create_task_buffer({
            .size = sizeof(RtLabRay) * ray_capacity,
            .name = "rt_lab_filtered_rays",
        });
        info.tg.add_task(daxa::HeadTask<RtLabRayFilterH::Info>()
            .head_views(RtLabRayFilterH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .lab_depth = lab_depth.current(),
                .lab_face_normal = lab_face_normal.current(),
                .ray_counters = ray_counters,
                .ray_list = ray_list,
                .ray_alloc = ray_alloc,
                .rays = rays,
                .filtered_rays = filtered_rays,
            })
            .executes(rt_lab_ray_filter_callback, &render_context, ray_capacity));
    }

    info.tg.add_task(daxa::HeadTask<RtLabGatherH::Info>()
        .head_views(RtLabGatherH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .ray_alloc = ray_alloc,
            .rays = rays,
            .lab_raw_diffuse = lab_raw_diffuse,
            .lab_depth = gradient_enabled ? lab_depth.current() : daxa::NullTaskImage,
            .lab_face_normal = gradient_enabled ? lab_face_normal.current() : daxa::NullTaskImage,
            .gradient_record_origin = gradient_record_origin,
            .gradient_record_surface = gradient_record_surface,
            .gradient_record_data = gradient_record_data,
            .filtered_rays = ray_filter ? filtered_rays : daxa::NullTaskBuffer,
            .lab_raw_luminance = lab_raw_luminance,
        })
        .executes(rt_lab_gather_callback, &render_context, gradient_enabled, ray_filter, gradient_per_pixel, raw_luminance_accumulated));


    daxa::TaskImageView lab_diffuse = lab_raw_diffuse;
    // Split history error estimate: the signed half difference (lab res) follows the diffuse through the spatial passes
    // and the resolve with the same weights -> post-spatial estimate at full res. Null without split history.
    bool const split_enabled = settings.temporal_enabled && settings.split_history_enabled;
    daxa::TaskImageView lab_split_difference = daxa::NullTaskImage;
    // Per sample variance from the temporal pass (variance guide of the fine spatial filter), null without temporal.
    daxa::TaskImageView lab_variance = daxa::NullTaskImage;
    // Ray length guide: accumulated 3x3 ray length means (temporal output), null when off.
    daxa::TaskImageView lab_ray_length = daxa::NullTaskImage;

    if (settings.temporal_enabled)
    {
        auto variance = variance_image;
        lab_variance = variance.current();
        daxa::TaskImageView sh_y_history = daxa::NullTaskImage;
        daxa::TaskImageView sh_y_accumulated = daxa::NullTaskImage;
        daxa::TaskImageView cocg_history = daxa::NullTaskImage;
        daxa::TaskImageView cocg_accumulated = daxa::NullTaskImage;
        if (coarse_enabled)
        {
            auto sh_y = create_history_image({
                .format = history_format,
                .size = lab_size,
                .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
                .name = progressive ? "rt_lab_sh_y_progressive_persistent" : "rt_lab_sh_y_smart_persistent",
            });
            auto cocg = create_history_image({
                .format = history_format,
                .size = lab_size,
                .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
                .name = progressive ? "rt_lab_cocg_progressive_persistent" : "rt_lab_cocg_smart_persistent",
            });
            sh_y_history = sh_y.previous();
            sh_y_accumulated = sh_y.current();
            cocg_history = cocg.previous();
            cocg_accumulated = cocg.current();
            coarse_sh_y_input = sh_y_accumulated;
            coarse_cocg_count_input = cocg_accumulated;
        }
        // Split (A/B) histories, same format as the main history.
        daxa::TaskImageView split_a_history = daxa::NullTaskImage;
        daxa::TaskImageView split_b_history = daxa::NullTaskImage;
        daxa::TaskImageView split_a_accumulated = daxa::NullTaskImage;
        daxa::TaskImageView split_b_accumulated = daxa::NullTaskImage;
        if (settings.split_history_enabled)
        {
            auto split_a = create_history_image({
                .format = history_format,
                .size = lab_size,
                .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
                .name = progressive ? "rt_lab_split_a_progressive_persistent" : "rt_lab_split_a_smart_persistent",
            });
            auto split_b = create_history_image({
                .format = history_format,
                .size = lab_size,
                .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
                .name = progressive ? "rt_lab_split_b_progressive_persistent" : "rt_lab_split_b_smart_persistent",
            });
            split_a_history = split_a.previous();
            split_b_history = split_b.previous();
            split_a_accumulated = split_a.current();
            split_b_accumulated = split_b.current();
            lab_split_difference = info.tg.create_task_image({
                .format = daxa::Format::R16G16_SFLOAT,
                .size = lab_size,
                .name = "rt_lab_split_difference",
            });
        }
        // Coarse temporal anti-lag (rt_lab_anti_lag.inl): cell moments -> edge aware blur -> coarse temporal statistics.
        daxa::TaskImageView coarse_anti_lag = daxa::NullTaskImage;
        if ((settings.coarse_anti_lag_enabled & RT_LAB_ANTI_LAG_MODE_MASK) != RT_LAB_ANTI_LAG_OFF)
        {
            auto const anti_lag_size = daxa::Extent3D{round_up_div(lab_size.x, static_cast<u32>(RT_LAB_GRADIENT_CELL)), round_up_div(lab_size.y, static_cast<u32>(RT_LAB_GRADIENT_CELL)), 1};
            auto anti_lag_history = create_history_image({
                .format = daxa::Format::R32G32B32A32_SFLOAT,
                .size = anti_lag_size,
                .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
                .name = "rt_lab_coarse_anti_lag_persistent",
            });
            auto cell_sums = info.tg.create_task_image({
                .format = daxa::Format::R32G32B32A32_SFLOAT,
                .size = anti_lag_size,
                .name = "rt_lab_coarse_anti_lag_cell_sums",
            });
            auto blur_sums = info.tg.create_task_image({
                .format = daxa::Format::R32G32B32A32_SFLOAT,
                .size = anti_lag_size,
                .name = "rt_lab_coarse_anti_lag_blur_sums",
            });
            auto anti_lag_fast_history = create_history_image({
                .format = daxa::Format::R32G32B32A32_SFLOAT,
                .size = anti_lag_size,
                .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
                .name = "rt_lab_coarse_anti_lag_fast_persistent",
            });
            info.tg.add_task(daxa::HeadTask<RtLabAntiLagDownsampleH::Info>()
                .head_views(RtLabAntiLagDownsampleH::Info::Views{
                    .globals = render_context.tgpu_render_data.view(),
                    .ray_alloc = ray_alloc,
                    .rays = filtered_rays,
                    .cell_sums = cell_sums,
                })
                .executes(rt_lab_anti_lag_downsample_callback, &render_context, lab_size));
            info.tg.add_task(daxa::HeadTask<RtLabAntiLagBlurH::Info>("RtLabAntiLagBlurHorizontal")
                .head_views(RtLabAntiLagBlurH::Info::Views{
                    .globals = render_context.tgpu_render_data.view(),
                    .view_cam_depth = info.view_cam_depth,
                    .view_cam_face_normals = info.view_cam_face_normals,
                    .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                    .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                    .blur_src = cell_sums,
                    .blur_dst = blur_sums,
                    .history_previous = daxa::NullTaskImage,
                    .history = daxa::NullTaskImage,
                    .debug_image = daxa::NullTaskImage,
                    .fast_history_previous = daxa::NullTaskImage,
                    .fast_history = daxa::NullTaskImage,
                    .temporal_gradient = daxa::NullTaskImage,
                })
                .executes(rt_lab_anti_lag_blur_callback, &render_context, false, lab_size, false));
            info.tg.add_task(daxa::HeadTask<RtLabAntiLagBlurH::Info>("RtLabAntiLagBlurVertical")
                .head_views(RtLabAntiLagBlurH::Info::Views{
                    .globals = render_context.tgpu_render_data.view(),
                    .view_cam_depth = info.view_cam_depth,
                    .view_cam_face_normals = info.view_cam_face_normals,
                    .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
                    .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
                    .blur_src = blur_sums,
                    .blur_dst = daxa::NullTaskImage,
                    .history_previous = anti_lag_history.previous(),
                    .history = anti_lag_history.current(),
                    .debug_image = info.debug_image,
                    .fast_history_previous = anti_lag_fast_history.previous(),
                    .fast_history = anti_lag_fast_history.current(),
                    .temporal_gradient = temporal_gradient,
                })
                .executes(rt_lab_anti_lag_blur_callback, &render_context, true, lab_size, gradient_enabled));
            coarse_anti_lag = anti_lag_history.current();
        }
        daxa::TaskImageView ray_length_history = daxa::NullTaskImage;
        if (ray_length_guide)
        {
            auto ray_length = create_history_image({
                .format = daxa::Format::R16G16_SFLOAT,
                .size = lab_size,
                .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
                .name = "rt_lab_ray_length_persistent",
            });
            ray_length_history = ray_length.previous();
            lab_ray_length = ray_length.current();
        }
        info.tg.add_task(daxa::HeadTask<RtLabTemporalH::Info>()
            .head_views(RtLabTemporalH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .lab_depth = lab_depth.current(),
                .lab_face_normal = lab_face_normal.current(),
                .lab_depth_history = lab_depth.previous(),
                .lab_face_normal_history = lab_face_normal.previous(),
                .lab_raw_diffuse = lab_raw_diffuse,
                .history = history_previous,
                .accumulated = history_current,
                .split_a_history = split_a_history,
                .split_b_history = split_b_history,
                .split_a_accumulated = split_a_accumulated,
                .split_b_accumulated = split_b_accumulated,
                .split_difference = lab_split_difference,
                .variance_history = variance.previous(),
                .variance = variance.current(),
                .ray_alloc = ray_alloc,
                .rays = filtered_rays,
                .sh_y_history = sh_y_history,
                .sh_y_accumulated = sh_y_accumulated,
                .cocg_history = cocg_history,
                .cocg_accumulated = cocg_accumulated,
                .temporal_gradient = temporal_gradient,
                .coarse_anti_lag = coarse_anti_lag,
                .lab_raw_luminance = lab_raw_luminance,
                .ray_lengths = ray_length_guide ? ray_lengths : daxa::NullTaskBuffer,
                .ray_length_history = ray_length_history,
                .ray_length_accumulated = lab_ray_length,
            })
            .executes(rt_lab_temporal_callback, &render_context, settings.split_history_enabled != 0, coarse_enabled, gradient_enabled,
                coarse_anti_lag != daxa::NullTaskImage, raw_luminance_accumulated, ray_length_guide));
        lab_diffuse = history_current;
    }

    // Coarse spatial fallback output (SH-Y, CoCg + count), null when off.
    daxa::TaskImageView coarse_split_input = daxa::NullTaskImage;
    daxa::TaskImageView lab_coarse_sh_y = daxa::NullTaskImage;
    daxa::TaskImageView lab_coarse_cocg_count = daxa::NullTaskImage;
    daxa::TaskImageView lab_coarse_split_difference = daxa::NullTaskImage;
    if (settings.spatial_enabled)
    {
        // Separable gaussian: horizontal + vertical. A-trous: atrous_passes passes, step 1, 2, 4, 8, ...
        bool const atrous = settings.spatial_mode == 1;
        u32 const passes = atrous ? static_cast<u32>(std::clamp(settings.atrous_passes, 1, RT_LAB_ATROUS_MAX_PASSES)) : 2u;
        // The coarse fallback filters the same temporal output (as SH-Y + CoCg).
        coarse_split_input = lab_split_difference;
        daxa::TaskImageView src = lab_diffuse;
        daxa::TaskImageView split_src = lab_split_difference;
        // Gradient aware variance guide: the first pass fits the gradient, the later ones filter it (rt_lab_blur.hlsl).
        bool const gradient = settings.variance_guide_gradient != 0 && lab_variance != daxa::NullTaskImage;
        daxa::TaskImageView gradient_x_src = daxa::NullTaskImage;
        daxa::TaskImageView gradient_y_src = daxa::NullTaskImage;
        for (u32 pass = 0; pass < passes; ++pass)
        {
            std::string const name = atrous ? fmt::format("rt_lab_atrous{}", pass) : (pass == 0 ? "rt_lab_blur_horizontal" : "rt_lab_blur_vertical");
            auto dst = info.tg.create_task_image({
                .format = daxa::Format::R16G16B16A16_SFLOAT,
                .size = lab_size,
                .name = name,
            });
            daxa::TaskImageView split_dst = daxa::NullTaskImage;
            if (split_enabled)
            {
                split_dst = info.tg.create_task_image({
                    .format = daxa::Format::R16G16_SFLOAT,
                    .size = lab_size,
                    .name = name + "_split_difference",
                });
            }
            daxa::TaskImageView gradient_x_dst = daxa::NullTaskImage;
            daxa::TaskImageView gradient_y_dst = daxa::NullTaskImage;
            if (gradient)
            {
                // f32: the .a variance is a squared radiance (f16 overflows).
                gradient_x_dst = info.tg.create_task_image({.format = daxa::Format::R32G32B32A32_SFLOAT, .size = lab_size, .name = name + "_gradient_x"});
                gradient_y_dst = info.tg.create_task_image({.format = daxa::Format::R32G32B32A32_SFLOAT, .size = lab_size, .name = name + "_gradient_y"});
            }
            u32 const gradient_mode = !gradient ? RT_LAB_GRADIENT_OFF : (pass == 0 ? RT_LAB_GRADIENT_FIT : RT_LAB_GRADIENT_CARRY);
            bool const last_pass = pass + 1 == passes;
            auto task = daxa::HeadTask<RtLabBlurH::Info>(atrous ? fmt::format("RtLabAtrous{}", pass) : (pass == 0 ? "RtLabBlurHorizontal" : "RtLabBlurVertical"))
                .head_views(RtLabBlurH::Info::Views{
                    .globals = render_context.tgpu_render_data.view(),
                    .lab_depth = lab_depth.current(),
                    .lab_face_normal = lab_face_normal.current(),
                    .src = src,
                    .dst = dst,
                    .split_src = split_src,
                    .split_dst = split_dst,
                    .lab_variance = lab_variance,
                    .gradient_x_src = gradient_x_src,
                    .gradient_y_src = gradient_y_src,
                    .gradient_x_dst = gradient_x_dst,
                    .gradient_y_dst = gradient_y_dst,
                    .lab_ray_length = lab_ray_length,
                    .debug_image = info.debug_image,
                });
            if (atrous)
            {
                info.tg.add_task(std::move(task).executes(rt_lab_atrous_callback, &render_context, pass, split_enabled, lab_variance != daxa::NullTaskImage, gradient_mode, last_pass, raw_luminance_accumulated, lab_ray_length != daxa::NullTaskImage));
            }
            else
            {
                info.tg.add_task(std::move(task).executes(rt_lab_blur_callback, &render_context, pass, split_enabled, lab_variance != daxa::NullTaskImage, gradient_mode, last_pass, raw_luminance_accumulated, lab_ray_length != daxa::NullTaskImage));
            }
            src = dst;
            split_src = split_dst;
            gradient_x_src = gradient_x_dst;
            gradient_y_src = gradient_y_dst;
        }
        lab_diffuse = src;
        lab_split_difference = split_src;

        // Coarse spatial fallback: a-trous with the sloppy coarse edge stopping, blended in at the resolve.
        if (coarse_enabled)
        {
            u32 const coarse_passes = static_cast<u32>(std::clamp(settings.coarse_atrous_passes, 1, RT_LAB_ATROUS_MAX_PASSES));
            daxa::TaskImageView coarse_sh_y_src = coarse_sh_y_input;
            daxa::TaskImageView coarse_cocg_count_src = coarse_cocg_count_input;
            daxa::TaskImageView coarse_split_src = coarse_split_input;
            for (u32 pass = 0; pass < coarse_passes; ++pass)
            {
                std::string const name = fmt::format("rt_lab_coarse{}", pass);
                auto dst_sh_y = info.tg.create_task_image({
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .size = lab_size,
                    .name = name + "_sh_y",
                });
                auto dst_cocg_count = info.tg.create_task_image({
                    .format = daxa::Format::R16G16B16A16_SFLOAT,
                    .size = lab_size,
                    .name = name + "_cocg_count",
                });
                daxa::TaskImageView split_dst = daxa::NullTaskImage;
                if (split_enabled)
                {
                    split_dst = info.tg.create_task_image({
                        .format = daxa::Format::R16G16_SFLOAT,
                        .size = lab_size,
                        .name = name + "_split_difference",
                    });
                }
                info.tg.add_task(daxa::HeadTask<RtLabCoarseH::Info>(fmt::format("RtLabCoarse{}", pass))
                    .head_views(RtLabCoarseH::Info::Views{
                        .globals = render_context.tgpu_render_data.view(),
                        .lab_depth = lab_depth.current(),
                        .lab_face_normal = lab_face_normal.current(),
                        .src_sh_y = coarse_sh_y_src,
                        .src_cocg_count = coarse_cocg_count_src,
                        .dst_sh_y = dst_sh_y,
                        .dst_cocg_count = dst_cocg_count,
                        .split_src = coarse_split_src,
                        .split_dst = split_dst,
                        .lab_variance = lab_variance,
                    })
                    .executes(rt_lab_coarse_callback, &render_context, pass, split_enabled, raw_luminance_accumulated));
                coarse_sh_y_src = dst_sh_y;
                coarse_cocg_count_src = dst_cocg_count;
                coarse_split_src = split_dst;
            }
            lab_coarse_sh_y = coarse_sh_y_src;
            lab_coarse_cocg_count = coarse_cocg_count_src;
            lab_coarse_split_difference = coarse_split_src;
        }
    }

    auto resolved = info.tg.create_task_image({
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = full_size,
        .name = "rt_lab_resolved_diffuse",
    });
    daxa::TaskImageView split_difference_resolved = daxa::NullTaskImage;
    if (split_enabled)
    {
        split_difference_resolved = info.tg.create_task_image({
            .format = daxa::Format::R16G16_SFLOAT,
            .size = full_size,
            .name = "rt_lab_split_difference_resolved",
        });
    }
    // Post stabilization runs inside the resolve (rt_lab_resolve.hlsl): its history, null when off.
    daxa::TaskImageView stabilize_history = daxa::NullTaskImage;
    daxa::TaskImageView stabilize_history_out = daxa::NullTaskImage;
    daxa::TaskImageView stabilize_variance = daxa::NullTaskImage;
    daxa::TaskImageView stabilize_variance_out = daxa::NullTaskImage;
    if (lab_state.post_stabilization_enabled)
    {
        // f16 .a holds the linear depth: relative precision ~1e-3, plenty for the 2 pixel width plane test.
        auto history = create_history_image({
            .format = daxa::Format::R16G16B16A16_SFLOAT,
            .size = full_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = "rt_lab_post_stabilize_persistent",
        });
        stabilize_history = history.previous();
        stabilize_history_out = history.current();
        // Short temporal variance history of the resolved input (packed f16 mean / std dev YCoCg + frames, see rt_lab_resolve.inl).
        auto variance = create_history_image({
            .format = daxa::Format::R32G32B32A32_UINT,
            .size = full_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = "rt_lab_post_stabilize_variance_persistent",
        });
        stabilize_variance = variance.previous();
        stabilize_variance_out = variance.current();
    }
    info.tg.add_task(daxa::HeadTask<RtLabResolveH::Info>()
        .head_views(RtLabResolveH::Info::Views{
            .globals = render_context.tgpu_render_data.view(),
            .debug_image = info.debug_image,
            .view_cam_depth = info.view_cam_depth,
            .view_cam_face_normals = info.view_cam_face_normals,
            .lab_depth = lab_depth.current(),
            .lab_face_normal = lab_face_normal.current(),
            .lab_diffuse = lab_diffuse,
            .resolved = resolved,
            .lab_split_difference = lab_split_difference,
            .split_difference_resolved = split_difference_resolved,
            .lab_coarse_sh_y = lab_coarse_sh_y,
            .lab_coarse_cocg_count = lab_coarse_cocg_count,
            .lab_coarse_split_difference = lab_coarse_split_difference,
            .stabilize_history = stabilize_history,
            .stabilize_history_out = stabilize_history_out,
            .stabilize_variance = stabilize_variance,
            .stabilize_variance_out = stabilize_variance_out,
        })
        .executes(rt_lab_resolve_callback, &render_context, split_enabled, settings.spatial_enabled != 0, coarse_enabled,
            stabilize_history != daxa::NullTaskImage));

    // Custom history backups: save this frame's histories (see create_history_image).
    for (auto const & [current, backup_view] : history_saves)
    {
        info.tg.add_task(daxa::InlineTask::Transfer("rt_lab_history_save")
            .reads(current)
            .writes(backup_view)
            .executes([&render_context, current, backup_view](daxa::TaskInterface ti)
            {
                if (!render_context.rt_lab.save_custom_history)
                {
                    return;
                }
                ti.recorder.copy_image_to_image({
                    .src_image = ti.id(current),
                    .dst_image = ti.id(backup_view),
                    .extent = ti.info(current).value().size,
                });
            }));
    }

    // Test measurement: per frame tile histograms + lag tiles of the final output (only once a test created the buffers;
    // the callbacks skip their dispatch outside of a run).
    if (!lab_state.measure_frame_histogram.id().is_empty() && split_enabled)
    {
        info.tg.register_buffer(lab_state.measure_frame_histogram);
        info.tg.register_buffer(lab_state.measure_run_histograms);
        info.tg.register_buffer(lab_state.measure_lag_run);
        // (mean, variance, frames, linear depth) per full res pixel, reprojected every frame (rt_lab_measure.hlsl).
        auto luma_statistics = info.tg.create_task_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = full_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = "rt_lab_measure_luma_statistics_persistent",
        });
        // (luma t-1, luma t-2, frames, linear depth) per full res pixel for the jitter metric, reprojected the same way.
        auto jitter_state = info.tg.create_task_image({
            .format = daxa::Format::R32G32B32A32_SFLOAT,
            .size = full_size,
            .lifetime_type = daxa::TaskResourceLifetimeType::PERSISTENT_DOUBLE_BUFFER,
            .name = "rt_lab_measure_jitter_persistent",
        });
        info.tg.clear_buffer({.buffer = lab_state.measure_frame_histogram.view(), .name = "rt_lab_measure_frame_histogram_clear"});
        info.tg.add_task(daxa::HeadTask<RtLabMeasureAccumulateH::Info>()
            .head_views(RtLabMeasureAccumulateH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .view_cam_depth = info.view_cam_depth,
                .view_cam_face_normals = info.view_cam_face_normals,
                .resolved = resolved,
                .split_difference_resolved = split_difference_resolved,
                .luma_statistics_history = luma_statistics.previous(),
                .luma_statistics = luma_statistics.current(),
                .jitter_history = jitter_state.previous(),
                .jitter_state = jitter_state.current(),
                .frame_histogram = lab_state.measure_frame_histogram.view(),
            })
            .executes(rt_lab_measure_accumulate_callback, &render_context));
        info.tg.add_task(daxa::HeadTask<RtLabMeasureStoreH::Info>()
            .head_views(RtLabMeasureStoreH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .frame_histogram = lab_state.measure_frame_histogram.view(),
                .run_histograms = lab_state.measure_run_histograms.view(),
            })
            .executes(rt_lab_measure_store_callback, &render_context));
        info.tg.add_task(daxa::HeadTask<RtLabMeasureLagH::Info>()
            .head_views(RtLabMeasureLagH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .view_cam_depth = info.view_cam_depth,
                .resolved = resolved,
                .lag_run = lab_state.measure_lag_run.view(),
            })
            .executes(rt_lab_measure_lag_callback, &render_context));
    }

    // Reference comparison: last lab tasks. Only in the graph once the capture images exist (a test with a reference
    // creates them and rebuilds the graph); both callbacks skip their dispatch when there is nothing to do.
    auto & lab = lab_state;
    if (!lab.progressive_image.id().is_empty())
    {
        info.tg.register_image(lab.progressive_image);
        info.tg.register_image(lab.custom_image);
        info.tg.register_image(lab.predicted_error_image);
        info.tg.add_task(daxa::HeadTask<RtLabConvergenceCaptureH::Info>()
            .head_views(RtLabConvergenceCaptureH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .resolved = resolved,
                .progressive = lab.progressive_image.view(),
                .custom = lab.custom_image.view(),
                .split_difference = split_difference_resolved,
                .predicted_error = lab.predicted_error_image.view(),
            })
            .executes(rt_lab_convergence_capture_callback, &render_context, split_enabled));
        info.tg.add_task(daxa::HeadTask<RtLabConvergenceAnalyzeH::Info>()
            .head_views(RtLabConvergenceAnalyzeH::Info::Views{
                .globals = render_context.tgpu_render_data.view(),
                .debug_image = info.debug_image,
                .view_cam_depth = info.view_cam_depth,
                .progressive = lab.progressive_image.view(),
                .custom = lab.custom_image.view(),
                .predicted_error = lab.predicted_error_image.view(),
            })
            .executes(rt_lab_convergence_analyze_callback, &render_context));
    }

    return TasksRtgiMainResult{
        .opaque_diffuse = resolved,
        .opaque_specular = daxa::NullTaskImage,
    };
}
