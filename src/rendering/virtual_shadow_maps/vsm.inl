#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>
#include "../../shader_shared/vsm_shared.inl"
#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"

#define MARK_REQUIRED_PAGES_X_DISPATCH 16
#define MARK_REQUIRED_PAGES_Y_DISPATCH 16

DAXA_DECL_TASK_HEAD_BEGIN(MarkRequiredPagesH, 8)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_BufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE, daxa_BufferPtr(AllocationCount), vsm_allocation_count)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_WRITE, daxa_BufferPtr(AllocationRequest), vsm_allocation_requests)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(VSMClipProjection), vsm_clip_projections)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_SAMPLED, REGULAR_2D, depth)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D_ARRAY, vsm_page_table)
DAXA_TH_IMAGE_ID(COMPUTE_SHADER_STORAGE_READ_WRITE, REGULAR_2D, vsm_meta_memory_table)
DAXA_DECL_TASK_HEAD_END

#if __cplusplus
#include "../tasks/misc.hpp"
#include "vsm_state.hpp"
#include "../scene_renderer_context.hpp"

inline daxa::ComputePipelineCompileInfo vsm_mark_required_pages_pipeline_compile_info()
{
    return {
        .shader_info = daxa::ShaderCompileInfo{
            .source = daxa::ShaderFile{"./src/rendering/virtual_shadow_maps/mark_required_pages.glsl"}},
        .push_constant_size = static_cast<u32>(sizeof(MarkRequiredPagesH::AttachmentShaderBlob)),
        .name = std::string{MarkRequiredPagesH::NAME}};
}

struct MarkRequiredPagesTask : MarkRequiredPagesH::Task
{
    AttachmentViews views = {};
    RenderContext * render_context = {};

    void callback(daxa::TaskInterface ti)
    {
        auto const depth_resolution = render_context->gpuctx->device.info_image(ti.get(AT.depth).ids[0]).value().size;
        auto const dispatch_size = u32vec2{
            (depth_resolution.x + MARK_REQUIRED_PAGES_X_DISPATCH - 1) / MARK_REQUIRED_PAGES_X_DISPATCH,
            (depth_resolution.y + MARK_REQUIRED_PAGES_Y_DISPATCH - 1) / MARK_REQUIRED_PAGES_Y_DISPATCH,
        };
        ti.recorder.set_pipeline(*render_context->gpuctx->compute_pipelines.at(vsm_mark_required_pages_pipeline_compile_info().name));
        MarkRequiredPagesH::AttachmentShaderBlob push = {};
        assign_blob(push, ti.attachment_shader_blob);
        ti.recorder.push_constant(push);
        ti.recorder.dispatch({.x = dispatch_size.x, .y = dispatch_size.y});
    }
};

struct TaskDrawVSMsInfo
{
    RenderContext * render_context = {};
    daxa::TaskGraph * tg = {};
    VSMState * vsm_state = {};
    daxa::TaskImageView depth = {};
};

inline void task_draw_vsms(TaskDrawVSMsInfo const & info)
{
    info.tg->add_task({
        .attachments = {
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, info.vsm_state->clip_projections),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, info.vsm_state->free_wrapped_pages_info),
            daxa::inl_attachment(daxa::TaskBufferAccess::TRANSFER_WRITE, info.vsm_state->globals),
        },
        .task = [info](daxa::TaskInterface ti)
        {
            allocate_fill_copy(ti, info.vsm_state->clip_projections_cpu, ti.get(info.vsm_state->clip_projections));
            allocate_fill_copy(ti, info.vsm_state->free_wrapped_pages_info_cpu, ti.get(info.vsm_state->free_wrapped_pages_info));
            allocate_fill_copy(ti, info.vsm_state->globals_cpu, ti.get(info.vsm_state->globals));
        },
    });

    info.tg->add_task(MarkRequiredPagesTask{
        .views = std::array{
            daxa::attachment_view(MarkRequiredPagesH::AT.globals, info.render_context->tgpu_render_data),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_globals, info.vsm_state->globals),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_allocation_count, info.vsm_state->allocation_count),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_allocation_requests, info.vsm_state->allocation_requests),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_clip_projections, info.vsm_state->clip_projections),
            daxa::attachment_view(MarkRequiredPagesH::AT.depth, info.depth),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_page_table, info.vsm_state->page_table),
            daxa::attachment_view(MarkRequiredPagesH::AT.vsm_meta_memory_table, info.vsm_state->meta_memory_table),
        },
        .render_context = info.render_context,
    });
}

struct CameraController;
struct GetVSMProjectionsInfo
{
    CameraController const * camera_info = {};
    f32vec3 sun_direction = {};
    f32 clip_0_scale = {};
    f32 clip_0_near = {};
    f32 clip_0_far = {};
    f32 clip_0_height_offset = {};

    ShaderDebugDrawContext * debug_context = {};
};

inline auto get_vsm_projections(GetVSMProjectionsInfo const & info) -> std::array<VSMClipProjection, VSM_CLIP_LEVELS>
{
    std::array<VSMClipProjection, VSM_CLIP_LEVELS> clip_projections = {};
    auto const default_vsm_pos = glm::vec3{0.0, 0.0, 0.0};
    auto const default_vsm_up = glm::vec3{0.0, 0.0, 1.0};
    auto const default_vsm_forward = -info.sun_direction;
    auto const default_vsm_view = glm::lookAt(default_vsm_pos, default_vsm_forward, default_vsm_up);

    auto calculate_clip_projection = [&info](i32 clip) -> glm::mat4x4
    {
        auto const clip_scale = std::pow(2.0f, s_cast<f32>(clip));
        auto clip_projection = glm::ortho(
            -info.clip_0_scale * clip_scale, // left
            info.clip_0_scale * clip_scale,  // right
            -info.clip_0_scale * clip_scale, // bottom
            info.clip_0_scale * clip_scale,  // top
            info.clip_0_near * clip_scale,   // near
            info.clip_0_far * clip_scale     // far
        );
        // Switch from OpenGL default to Vulkan default (invert the Y clip coordinate)
        clip_projection[1][1] *= -1.0;
        return clip_projection;
    };
    auto const target_camera_position = glm::vec4(info.camera_info->position, 1.0);
    auto const uv_page_size = s_cast<f32>(VSM_PAGE_SIZE) / s_cast<f32>(VSM_TEXTURE_RESOLUTION);
    // NDC space is [-1, 1] but uv space is [0, 1], PAGE_SIZE / TEXTURE_RESOLUTION gives us the page size in uv space
    // thus we need to multiply by two to get the page size in ndc coordinates
    auto const ndc_page_size = uv_page_size * 2.0f;

    for (i32 clip = 0; clip < VSM_CLIP_LEVELS; clip++)
    {
        auto const curr_clip_proj = calculate_clip_projection(clip);
        auto const clip_projection_view = curr_clip_proj * default_vsm_view;

        // Project the target position into VSM ndc coordinates and calculate a page alligned position
        auto const clip_projected_target_pos = clip_projection_view * target_camera_position;
        auto const ndc_target_pos = glm::vec3(clip_projected_target_pos) / clip_projected_target_pos.w;
        auto const ndc_page_scaled_target_pos = glm::vec2(ndc_target_pos) / ndc_page_size;
        auto const ndc_page_scaled_aligned_target_pos = glm::vec2(glm::ceil(ndc_page_scaled_target_pos));

        // Here we calculate the offsets that will be applied per page in the clip level
        // This is used to virtually offset the depth of each page so that we can actually snap the vsm position to the camera position
        auto const near_offset_ndc_u_in_world = glm::inverse(clip_projection_view) * glm::vec4(ndc_page_size, 0.0, 0.0, 1.0);
        auto const near_offset_ndc_v_in_world = glm::inverse(clip_projection_view) * glm::vec4(0.0, ndc_page_size, 0.0, 1.0);

        // Inverse projection from ndc -> world does not account for near plane offset, thus we need to add it manually
        // we simply shift the position in the oppposite of view direction by near plane distance
        auto const curr_clip_scale = std::pow(2.0f, s_cast<f32>(clip));
        auto const curr_clip_near = info.clip_0_near * curr_clip_scale;
        auto const ndc_u_in_world = glm::vec3(near_offset_ndc_u_in_world) + curr_clip_near * -default_vsm_forward;
        auto const ndc_v_in_world = glm::vec3(near_offset_ndc_v_in_world) + curr_clip_near * -default_vsm_forward;

        // Calculate the actual per page world space offsets
        f32 const u_offset_scale = ndc_u_in_world.z / default_vsm_forward.z;
        auto const u_offset_vector = u_offset_scale * -default_vsm_forward;

        f32 const v_offset_scale = ndc_v_in_world.z / default_vsm_forward.z;
        auto const v_offset_vector = v_offset_scale * -default_vsm_forward;

        // Get the per page offsets on a world space xy plane
        auto const xy_plane_ndc_u_in_world = ndc_u_in_world + u_offset_vector;
        auto const xy_plane_ndc_v_in_world = ndc_v_in_world + v_offset_vector;

        // Clip position on the xy world plane
        auto const clip_xy_plane_world_position = glm::vec3(
            ndc_page_scaled_aligned_target_pos.x * xy_plane_ndc_u_in_world +
            ndc_page_scaled_aligned_target_pos.y * xy_plane_ndc_v_in_world);

        // Clip offset from the xy plane - essentially clip_xy_plane_world_position gives us the position on a world xy plane positioned
        // at the height 0. We want to shift the clip camera up so that it observes the player position from the above. The height from
        // which the camera observes this player should be set according to the info.height_offset
        auto const view_offset_scale = s_cast<i32>(std::floor(info.camera_info->position.z / -default_vsm_forward.z) + info.clip_0_height_offset);
        auto const view_offset = s_cast<f32>(view_offset_scale) * -default_vsm_forward;
        auto const clip_position = clip_xy_plane_world_position + view_offset;

        auto const origin_shift = (clip_projection_view * glm::vec4(0.0, 0.0, 0.0, 1.0)).z;
        auto const page_u_depth_offset = (clip_projection_view * glm::vec4(u_offset_vector, 1.0)).z - origin_shift;
        auto const page_v_depth_offset = (clip_projection_view * glm::vec4(v_offset_vector, 1.0)).z - origin_shift;

        auto const final_clip_view = glm::lookAt(clip_position, clip_position + glm::normalize(default_vsm_forward), default_vsm_up);

        clip_projections.at(clip) = VSMClipProjection{
            .height_offset = view_offset_scale,
            .depth_page_offset = {page_u_depth_offset, page_v_depth_offset},
            .page_offset = {
                -s_cast<daxa_i32>(ndc_page_scaled_aligned_target_pos.x),
                -s_cast<daxa_i32>(ndc_page_scaled_aligned_target_pos.y),
            },
            .view = std::bit_cast<daxa_f32mat4x4>(final_clip_view),
            .projection = std::bit_cast<daxa_f32mat4x4>(curr_clip_proj),
            .projection_view = std::bit_cast<daxa_f32mat4x4>(curr_clip_proj * final_clip_view),
            .inv_projection_view = std::bit_cast<daxa_f32mat4x4>(glm::inverse(curr_clip_proj * final_clip_view)),
        };
    }
    return clip_projections;
}

struct DebugDrawClipFrustiInfo
{
    std::span<VSMClipProjection const> clip_projections;
    bool draw_individual_pages = {};
    ShaderDebugDrawContext * debug_context = {};
    f32vec3 vsm_view_direction = {};
};

inline void debug_draw_clip_fusti(DebugDrawClipFrustiInfo const & info)
{
    static constexpr std::array offsets = {
        glm::ivec2(-1, 1), glm::ivec2(-1, -1), glm::ivec2(1, -1), glm::ivec2(1, 1),
        glm::ivec2(-1, 1), glm::ivec2(-1, -1), glm::ivec2(1, -1), glm::ivec2(1, 1)};

    for (auto const & clip_projection : info.clip_projections)
    {
        auto const left_right_size = std::abs((1.0f / std::bit_cast<glm::mat4x4>(clip_projection.projection)[0][0])) * 2.0f;
        auto const top_bottom_size = std::abs((1.0f / std::bit_cast<glm::mat4x4>(clip_projection.projection)[1][1])) * 2.0f;
        auto const near_far_size = (1.0f / std::bit_cast<glm::mat4x4>(clip_projection.projection)[2][2]) * 2.0f;
        auto const page_size = glm::vec2(left_right_size / VSM_PAGE_TABLE_RESOLUTION, top_bottom_size / VSM_PAGE_TABLE_RESOLUTION);

        auto const page_proj = glm::ortho(
            -page_size.x / 2.0f,
            page_size.x / 2.0f,
            -page_size.y / 2.0f,
            page_size.y / 2.0f,
            1.0f,
            100.0f);
        if (info.draw_individual_pages)
        {
            auto const uv_page_size = s_cast<f32>(VSM_PAGE_SIZE) / s_cast<f32>(VSM_TEXTURE_RESOLUTION);
            for (i32 page_u_index = 0; page_u_index < VSM_PAGE_TABLE_RESOLUTION; page_u_index++)
            {
                for (i32 page_v_index = 0; page_v_index < VSM_PAGE_TABLE_RESOLUTION; page_v_index++)
                {
                    auto const corner_virtual_uv = uv_page_size * glm::vec2(page_u_index, page_v_index);
                    auto const page_center_virtual_uv_offset = glm::vec2(uv_page_size * 0.5f);
                    auto const virtual_uv = corner_virtual_uv + page_center_virtual_uv_offset;

                    auto const page_index = glm::ivec2(virtual_uv * s_cast<f32>(VSM_PAGE_TABLE_RESOLUTION));
                    f32 const depth =
                        ((VSM_PAGE_TABLE_RESOLUTION - 1) - page_index.x) * clip_projection.depth_page_offset.x +
                        ((VSM_PAGE_TABLE_RESOLUTION - 1) - page_index.y) * clip_projection.depth_page_offset.y;
                    auto const virtual_page_ndc = (virtual_uv * 2.0f) - glm::vec2(1.0f);
                    auto const page_ndc_position = glm::vec4(virtual_page_ndc, -depth, 1.0);
                    auto const new_position = std::bit_cast<glm::mat4x4>(clip_projection.inv_projection_view) * page_ndc_position;

                    auto const page_view = glm::lookAt(glm::vec3(new_position), glm::vec3(new_position) + info.vsm_view_direction, {0.0, 0.0, 1.0});
                    auto const page_inv_projection_view = glm::inverse(page_proj * page_view);

                    ShaderDebugBoxDraw box_draw = {};
                    box_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
                    box_draw.color = daxa_f32vec3{0.0, 0.0, 1.0};
                    for (i32 i = 0; i < 8; i++)
                    {
                        auto const ndc_pos = glm::vec4(offsets[i], i < 4 ? 0.0f : 1.0f, 1.0f);
                        auto const world_pos = page_inv_projection_view * ndc_pos;
                        box_draw.vertices[i] = {world_pos.x, world_pos.y, world_pos.z};
                    }
                    info.debug_context->cpu_debug_box_draws.push_back(box_draw);
                }
            }
        }
        else
        {
            ShaderDebugBoxDraw box_draw = {};
            box_draw.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
            box_draw.color = daxa_f32vec3{0.0, 0.0, 1.0};
            for (i32 i = 0; i < 8; i++)
            {
                auto const ndc_pos = glm::vec4(offsets[i], i < 4 ? 0.0f : 1.0f, 1.0f);
                auto const world_pos = std::bit_cast<glm::mat4x4>(clip_projection.inv_projection_view) * ndc_pos;
                box_draw.vertices[i] = {world_pos.x, world_pos.y, world_pos.z};
            }
            info.debug_context->cpu_debug_box_draws.push_back(box_draw);
        }
    }
}
#endif //__cplusplus