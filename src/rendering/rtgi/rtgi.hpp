#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../scene_renderer_context.hpp"

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

MAKE_COMPUTE_COMPILE_INFO(rtgi_reproject_diffuse_compile_info, "./src/rendering/rtgi/rtgi_reproject.hlsl", "entry_reproject_halfres")
MAKE_COMPUTE_COMPILE_INFO(rtgi_pre_blur_flatten_compile_info, "./src/rendering/rtgi/rtgi_pre_blur.hlsl", "entry_flatten")
MAKE_COMPUTE_COMPILE_INFO(rtgi_pre_blur_prepare_compile_info, "./src/rendering/rtgi/rtgi_pre_blur.hlsl", "entry_prepare")
MAKE_COMPUTE_COMPILE_INFO(rtgi_pre_blur_apply_compile_info, "./src/rendering/rtgi/rtgi_pre_blur.hlsl", "entry_apply")
MAKE_COMPUTE_COMPILE_INFO(rtgi_adaptive_blur_diffuse_compile_info, "./src/rendering/rtgi/rtgi_adaptive_blur.hlsl", "entry_blur_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_upscale_diffuse_compile_info, "./src/rendering/rtgi/rtgi_upscale.hlsl", "entry_upscale_diffuse")
MAKE_COMPUTE_COMPILE_INFO(rtgi_diffuse_temporal_stabilization_compile_info, "./src/rendering/rtgi/rtgi_reproject.hlsl", "entry_temporal_stabilization")

///
/// === Persistent Images ===
///

inline auto rtgi_create_diffuse_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R16G16B16A16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi diffuse history image",
    };
}

inline auto rtgi_create_diffuse2_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R16G16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi diffuse2 history image",
    };
}

inline auto rtgi_create_depth_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R32_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi depth history image",
    };
}

inline auto rtgi_create_samplecnt_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R16_SFLOAT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi samplecnt history image",
    };
}

inline auto rtgi_create_face_normal_history_image_info(RenderContext * render_context) -> daxa::ImageInfo
{
    return daxa::ImageInfo{
        .format = daxa::Format::R32_UINT,
        .size = {
            render_context->render_data.settings.render_target_size.x / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            render_context->render_data.settings.render_target_size.y / RTGI_DIFFUSE_PIXEL_SCALE_DIV,
            1,
        },
        .usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::TRANSFER_SRC | daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE,
        .name = "rtgi face normal history image",
    };
}

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
    daxa::TaskImageView rtgi_depth_history = {};
    daxa::TaskImageView rtgi_samplecnt_history = {};
    daxa::TaskImageView rtgi_face_normal_history = {};
    daxa::TaskImageView rtgi_full_color_history = {};
    daxa::TaskImageView rtgi_full_statistics_history = {};
    daxa::TaskImageView rtgi_full_face_normal_history = {};
    daxa::TaskImageView rtgi_full_samplecount_history = {};
};
struct TasksRtgiMainResult
{
    daxa::TaskImageView opaque_diffuse = {};
};
auto tasks_rtgi_main(TasksRtgiInfo const & info) -> TasksRtgiMainResult;