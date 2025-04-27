#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "cull_meshes.inl"
#include "analyze_visbuffer.inl"
#include "gen_hiz.inl"
#include "select_first_pass_meshlets.inl"
#include "draw_visbuffer.hpp"

#include "../scene_renderer_context.hpp"

namespace raster_visbuf
{
    auto create_visbuffer(daxa::TaskGraph & tg, RenderContext const & render_context, char const * name = "view_camera_visbuffer") -> daxa::TaskImageView;

    auto create_atomic_visbuffer(daxa::TaskGraph & tg, RenderContext const & render_context) -> daxa::TaskImageView;

    auto create_depth(daxa::TaskGraph & tg, RenderContext const & render_context, char const * name = "main_camera_depth") -> daxa::TaskImageView;

    struct TaskDrawVisbufferAllInfo
    {
        daxa::TaskGraph & tg;
        std::unique_ptr<RenderContext> & render_context;
        Scene * scene;
        daxa::TaskBufferView meshlet_instances;
        daxa::TaskBufferView meshlet_instances_last_frame;
        daxa::TaskBufferView visible_meshlet_instances;
        daxa::TaskImageView debug_image;
        daxa::TaskImageView depth_history;
    };
    struct TaskDrawVisbufferAllOut
    {
        daxa::TaskImageView main_camera_visbuffer;
        daxa::TaskImageView main_camera_depth;
        daxa::TaskImageView view_camera_visbuffer; //
        daxa::TaskImageView view_camera_depth;     // View Camera is either observer or main camera
        daxa::TaskImageView view_camera_overdraw;  //
    };
    auto task_draw_visbuffer_all(TaskDrawVisbufferAllInfo const info) -> TaskDrawVisbufferAllOut;
} // namespace raster_visbuf