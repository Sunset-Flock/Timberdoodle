#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../scene_renderer_context.hpp"

/// 
/// Typical denoisers do one of these three things:
///     1. spatial -> temporal (simplest to get good results with strong caveats (disocclusion shimmer etc))
///         * uncommon, only used fast paced games (doom)
///         * also quite intuitive
///         * very easy to tune
///         * as the spatial filter goes before the temporal, a accumulation to the ground truth can be slow
///             * games dont resolve to ground truth anyways, when keeping it blurry (like for indirect diffuse) this doesn't matter at all.
///         * allows for fast stochastic spatial filtering as its temporally stabilized later
///             * allows for very stable performance scaling no matter the input
///             * much more sample efficient, can use disc gaussian disc importance sampling -> all samples have weight 1
///             * very efficient scaling to large filter radii, simply keep the sample count and increase disc size.
///         * generally best temporal reaction time
///             * very important for fast paced games
///             * scene changes and blurry resolve is very fast
///             * as the spatial filter is before the temporal, the fast history can be very short and reliable
///         * has shimmering issues in motion
///             * as the spatial filter works on raw data, it will not resolve a stable value for small disocclusions or reduces sample counts from reprojecting edges.
///         * has ugly banding in disocclusions
///             * temporal filter will get stuck on temporal slices of the past in each frame for a bit until it resolves
///         * generally better performance with good quality (if artifacts are acceptable)
///     2. temporal -> spatial
///         * most common, used in slow games (cyberpunk, assassins creed)
///         * very intuitive to work with
///         * can resolve to a ground truth the fastest as it accumulates a "true" history
///         * poor temporal reaction speed
///             * as no spatial pass happens before temporal integration, fast history has to be quite long (8 frames at least)
///             * even with a longer fast history its much more noisy, can not be trusted a lot, clamping is much slower
///         * requires very expensive and fully smooth post spatial filtering
///             * especially wide filtering ultra expensive, poor perf scaling on high variance
///         * very high quality resolve as spatial filter can hide all pervious instabilities
///     3. prev frame spatial -> temporal -> spatial (very hard to tune, basically black magic. Tends to overblur a ton)
///         * very arcane to tune
///         * fast, similar to spatial -> temporal, can use sloppy stochastic spatial filters
///         * recurrent nature makes it VERY blurry
///         * okish stable in motion as the spatial filter following the temporal have blurred history but the spatial filter is noisy
///         * resolve is VERY ugly in my opinion, it overblurrs A LOT, not acceptable
///
/// Tidos denoiser approach:
///     * fast spatial -> temporal -> cleanup spatial
///         * combines advantages of both the first two approaches
///         * result of prefilter temporally integrated -> can use fast stochastic sample efficient blur
///         * temporal gets spatially integrated values -> reliable fast history -> good temporal reaction time
///         * image already pre integrated -> post filter can be really small and smooth -> cleans all remaining artifacts such as perma shimmer
///         * post filter allows for temporarily altering the result without polluting the temporal history 
///             * for example: go from gauss to box blur for stability in a few frames after disocclusion
///         * naturally integrates with the upscale later, as upscaling really is also a spatial blur filter.
///


///
/// === Pipeline compile infos ===
///

inline auto rtgi_trace_diffuse_compile_info() -> daxa::RayTracingPipelineCompileInfo2
{
    auto file = daxa::ShaderFile{"./src/rendering/rtgi/rtgi_trace_diffuse.hlsl"};
    return daxa::RayTracingPipelineCompileInfo2{
        .ray_gen_infos = {{.source = file, .entry_point = "ray_gen", .language = daxa::ShaderLanguage::SLANG}},
        .any_hit_infos = {{.source = file, .entry_point = "any_hit", .language = daxa::ShaderLanguage::SLANG}},
        .closest_hit_infos = {{.source = daxa::ShaderFile{"./src/rendering/rtgi/rtgi_trace_diffuse_shading.hlsl"}, .entry_point = "closest_hit", .language = daxa::ShaderLanguage::SLANG}},
        .miss_hit_infos = {{.source = daxa::ShaderFile{"./src/rendering/rtgi/rtgi_trace_diffuse_shading.hlsl"}, .entry_point = "miss", .language = daxa::ShaderLanguage::SLANG}},
        .shader_groups_infos = {
            // Gen Group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::GENERAL, .general_shader_index = 0},
            // Miss group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::GENERAL, .general_shader_index = 3},
            // Hit group
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP, .closest_hit_shader_index = 2},
            daxa::RayTracingShaderGroupInfo{.type = daxa::ShaderGroup::TRIANGLES_HIT_GROUP, .closest_hit_shader_index = 2, .any_hit_shader_index = 1},
        },
        .max_ray_recursion_depth = 2,
        .name = "Trace RTGI",
    };
}

MAKE_COMPUTE_COMPILE_INFO(rtgi_temporal_compile_info, "./src/rendering/rtgi/rtgi_temporal.hlsl", "entry_reproject_halfres")
MAKE_COMPUTE_COMPILE_INFO(rtgi_pre_filter_prepare_compile_info, "./src/rendering/rtgi/rtgi_pre_filter.hlsl", "entry_prepare")
MAKE_COMPUTE_COMPILE_INFO(rtgi_pre_filter_apply_compile_info, "./src/rendering/rtgi/rtgi_pre_filter.hlsl", "entry_apply")
MAKE_COMPUTE_COMPILE_INFO(rtgi_pre_blur_compile_info, "./src/rendering/rtgi/rtgi_pre_blur.hlsl", "entry_adaptive_blur")
MAKE_COMPUTE_COMPILE_INFO(rtgi_post_blur_compile_info, "./src/rendering/rtgi/rtgi_post_blur.hlsl", "entry_post_blur")
MAKE_COMPUTE_COMPILE_INFO(rtgi_upscale_diffuse_compile_info, "./src/rendering/rtgi/rtgi_upscale.hlsl", "entry_upscale_diffuse")

struct TasksRtgiInfo
{
    daxa::TaskGraph & tg;
    RenderContext & render_context;

    daxa::TaskBufferView globals = {};
    daxa::TaskImageView debug_image = {};
    daxa::TaskImageView clocks_image = {};
    daxa::TaskImageView view_cam_half_res_depth = {};
    daxa::TaskImageView view_cam_half_res_face_normals = {};
    daxa::TaskImageView view_cam_depth = {};
    daxa::TaskImageView view_cam_face_normals = {};
    daxa::TaskImageView view_camera_detail_normal_image = {};
    daxa::TaskImageView depth_history = {};
    daxa::TaskBufferView meshlet_instances = {};
    daxa::TaskBufferView mesh_instances = {};
    daxa::TaskImageView sky = {};
    daxa::TaskImageView sky_transmittance = {};
    daxa::TaskImageView light_mask_volume = {};
    daxa::TaskImageView pgi_irradiance = {};
    daxa::TaskImageView pgi_visibility = {};
    daxa::TaskImageView pgi_info = {};
    daxa::TaskImageView pgi_requests = {};
    daxa::TaskTlasView tlas = {};
    daxa::TaskBufferView vsm_globals = {};
    daxa::TaskBufferView vsm_point_lights = {};
    daxa::TaskBufferView vsm_spot_lights = {};
    daxa::TaskImageView vsm_memory_block = {};
    daxa::TaskImageView vsm_point_spot_page_table = {};
};
struct TasksRtgiMainResult
{
    daxa::TaskImageView opaque_diffuse = {};
};
auto tasks_rtgi_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult;