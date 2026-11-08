#include "rt_lab_gradient_oct.hpp"
#include "rt_lab_internal.hpp"

#include "rt_lab_gradient_oct.inl"
#include "rt_lab_rays.inl" // RT_LAB_RAYS_X

// The history scale consumes the per pixel result, the debug views show the pyramid.
static bool rt_lab_gradient_oct_resolve_needed(RenderContext const & render_context)
{
    i32 const mode = render_context.render_data.settings.debug_draw_mode;
    return render_context.render_data.rt_lab_settings.temporal_gradient2_scale_history != 0 ||
           (mode >= DEBUG_DRAW_MODE_RT_LAB_GRADIENT2_CHANGE && mode <= DEBUG_DRAW_MODE_RT_LAB_GRADIENT2_BIAS);
}

static bool rt_lab_gradient_oct_active(RenderContext const & render_context)
{
    i32 const mode = render_context.render_data.settings.debug_draw_mode;
    return rt_lab_gradient_oct_resolve_needed(render_context) ||
           (mode >= DEBUG_DRAW_MODE_RT_LAB_GRADIENT_OCT_MAP0 && mode <= DEBUG_DRAW_MODE_RT_LAB_GRADIENT_OCT_RAW_DIFFERENCE2); // contiguous (shared.inl)
}

// Level of the pass: area_pixels = lab pixels per area, src_size = size of the source image (cells, positions or maps).
static void rt_lab_gradient_oct_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 mode, daxa::Extent3D src_size, u32 area_pixels, u32 lab_height)
{
    if (!rt_lab_gradient_oct_active(*render_context))
    {
        return;
    }
    auto const & AT = RtLabGradientOctH::Info::AT;
    RtLabGradientOctPush push = {};
    push.mode = mode;
    push.src_size = {src_size.x, src_size.y};
    push.area_pixels = area_pixels;
    push.lab_height = lab_height;
    push.denoise = render_context->render_data.rt_lab_settings.temporal_gradient_oct_denoise != 0 ? 1u : 0u;
    push.distance_scale = render_context->render_data.rt_lab_settings.temporal_gradient_oct_distance_scale;
    bool const positions = mode == RT_LAB_GRADIENT_OCT_MODE_POSITIONS_CELLS || mode == RT_LAB_GRADIENT_OCT_MODE_POSITIONS_AREAS;
    rt_lab_dispatch(ti, render_context, push, positions ? AT.dst_positions : AT.dst_maps, RT_LAB_GRADIENT_OCT, RenderTimes::index<"RT_LAB", "GRADIENT_OCT">(), rt_lab_gradient_oct_compile_info().name);
}

static void rt_lab_gradient_oct_resolve_callback(daxa::TaskInterface ti, RenderContext * render_context, u32 lab_height)
{
    if (!rt_lab_gradient_oct_active(*render_context))
    {
        return;
    }
    auto const & AT = RtLabGradientOctResolveH::Info::AT;
    RtLabGradientOctResolvePush push = {};
    push.lab_height = lab_height;
    push.resolve = rt_lab_gradient_oct_resolve_needed(*render_context) ? 1u : 0u;
    push.distance_scale = render_context->render_data.rt_lab_settings.temporal_gradient_oct_distance_scale;
    rt_lab_dispatch(ti, render_context, push, AT.gradient2, RT_LAB_RAYS_X, RenderTimes::index<"RT_LAB", "GRADIENT_OCT">(), rt_lab_gradient_oct_resolve_compile_info().name);
}

auto rt_lab_gradient_oct_add_tasks(TasksRtgiInfo const & info, daxa::TaskImageView oct_samples, daxa::TaskImageView oct_positions, daxa::Extent3D cell_size, daxa::Extent3D lab_size) -> daxa::TaskImageView
{
    u32 const area_cells = RT_LAB_GRADIENT_OCT_AREA;
    u32 const octs = RT_LAB_GRADIENT_OCT;
    u32 const lab_height = lab_size.y;
    // Areas per level, maps = octs x octs texels per area.
    std::array<daxa::Extent3D, 3> areas = {};
    std::array<daxa::Extent3D, 3> maps = {};
    std::array<u32, 3> area_pixels = {};
    daxa::Extent3D previous = cell_size;
    u32 pixels = static_cast<u32>(RT_LAB_GRADIENT_CELL);
    for (u32 level = 0; level < 3; ++level)
    {
        areas[level] = {round_up_div(previous.x, area_cells), round_up_div(previous.y, area_cells), 1};
        maps[level] = {areas[level].x * octs, areas[level].y * octs, 1};
        pixels *= area_cells;
        area_pixels[level] = pixels;
        previous = areas[level];
    }
    auto create = [&](daxa::Extent3D size, char const * name, daxa::Format format = daxa::Format::R32G32B32A32_SFLOAT)
    {
        return info.tg.create_task_image({.format = format, .size = size, .name = name});
    };
    std::array<daxa::TaskImageView, 3> positions = {
        create(areas[0], "rt_lab_gradient_oct_positions0"),
        create(areas[1], "rt_lab_gradient_oct_positions1"),
        create(areas[2], "rt_lab_gradient_oct_positions2"),
    };
    std::array<daxa::TaskImageView, 3> sums = {
        create(maps[0], "rt_lab_gradient_oct_sums0"),
        create(maps[1], "rt_lab_gradient_oct_sums1"),
        create(maps[2], "rt_lab_gradient_oct_sums2"),
    };
    std::array<daxa::TaskImageView, 3> moments = {
        create(maps[0], "rt_lab_gradient_oct_moments0"),
        create(maps[1], "rt_lab_gradient_oct_moments1"),
        create(maps[2], "rt_lab_gradient_oct_moments2"),
    };
    std::array<daxa::TaskImageView, 3> denoised = {
        create(maps[0], "rt_lab_gradient_oct_denoised0"),
        create(maps[1], "rt_lab_gradient_oct_denoised1"),
        create(maps[2], "rt_lab_gradient_oct_denoised2"),
    };
    std::array<daxa::TaskImageView, 3> denoised_moments = {
        create(maps[0], "rt_lab_gradient_oct_denoised_moments0"),
        create(maps[1], "rt_lab_gradient_oct_denoised_moments1"),
        create(maps[2], "rt_lab_gradient_oct_denoised_moments2"),
    };
    std::array<daxa::TaskImageView, 3> convolved = {
        create(maps[0], "rt_lab_gradient_oct_convolved0"),
        create(maps[1], "rt_lab_gradient_oct_convolved1"),
        create(maps[2], "rt_lab_gradient_oct_convolved2"),
    };
    daxa::TaskImageView gradient2 = create(lab_size, "rt_lab_gradient2", daxa::Format::R16G16B16A16_SFLOAT);

    struct PassViews
    {
        daxa::TaskImageView cells = daxa::NullTaskImage;
        daxa::TaskImageView maps = daxa::NullTaskImage;
        daxa::TaskImageView moments = daxa::NullTaskImage;
        daxa::TaskImageView source_positions = daxa::NullTaskImage;
        daxa::TaskImageView area_positions = daxa::NullTaskImage;
        daxa::TaskImageView dst_maps = daxa::NullTaskImage;
        daxa::TaskImageView dst_moments = daxa::NullTaskImage;
        daxa::TaskImageView dst_positions = daxa::NullTaskImage;
    };
    auto add_pass = [&](char const * name, u32 mode, PassViews const & views, daxa::Extent3D source_size, u32 level)
    {
        info.tg.add_task(daxa::HeadTask<RtLabGradientOctH::Info>(name)
            .head_views(RtLabGradientOctH::Info::Views{
                .globals = info.render_context.tgpu_render_data.view(),
                .src_cells = views.cells,
                .src_maps = views.maps,
                .src_moments = views.moments,
                .src_positions = views.source_positions,
                .area_positions = views.area_positions,
                .dst_maps = views.dst_maps,
                .dst_moments = views.dst_moments,
                .dst_positions = views.dst_positions,
            })
            .executes(rt_lab_gradient_oct_callback, &info.render_context, mode, source_size, area_pixels[level], lab_height));
    };
    char const * const position_names[3] = {"RtLabGradientOctPositions0", "RtLabGradientOctPositions1", "RtLabGradientOctPositions2"};
    char const * const bin_names[3] = {"RtLabGradientOctBin0", "RtLabGradientOctBin1", "RtLabGradientOctBin2"};
    char const * const denoise_names[3] = {"RtLabGradientOctDenoise0", "RtLabGradientOctDenoise1", "RtLabGradientOctDenoise2"};
    char const * const convolve_names[3] = {"RtLabGradientOctConvolve0", "RtLabGradientOctConvolve1", "RtLabGradientOctConvolve2"};
    for (u32 level = 0; level < 3; ++level)
    {
        // Positions: the smooth majority of the children, then the maps binned with distance weights, denoised, convolved.
        PassViews position_pass = {};
        position_pass.source_positions = level == 0 ? oct_positions : positions[level - 1];
        position_pass.dst_positions = positions[level];
        add_pass(position_names[level], level == 0 ? RT_LAB_GRADIENT_OCT_MODE_POSITIONS_CELLS : RT_LAB_GRADIENT_OCT_MODE_POSITIONS_AREAS, position_pass,
            level == 0 ? cell_size : areas[level - 1], level);

        PassViews bin_pass = {};
        bin_pass.cells = level == 0 ? oct_samples : daxa::NullTaskImage;
        bin_pass.maps = level == 0 ? daxa::NullTaskImage : sums[level - 1];
        bin_pass.moments = level == 0 ? daxa::NullTaskImage : moments[level - 1];
        bin_pass.source_positions = level == 0 ? oct_positions : positions[level - 1];
        bin_pass.area_positions = positions[level];
        bin_pass.dst_maps = sums[level];
        bin_pass.dst_moments = moments[level];
        add_pass(bin_names[level], level == 0 ? RT_LAB_GRADIENT_OCT_MODE_BIN_CELLS : RT_LAB_GRADIENT_OCT_MODE_BIN_MAPS, bin_pass,
            level == 0 ? cell_size : maps[level - 1], level);

        PassViews denoise_pass = {};
        denoise_pass.maps = sums[level];
        denoise_pass.moments = moments[level];
        denoise_pass.area_positions = positions[level];
        denoise_pass.dst_maps = denoised[level];
        denoise_pass.dst_moments = denoised_moments[level];
        add_pass(denoise_names[level], RT_LAB_GRADIENT_OCT_MODE_DENOISE, denoise_pass, maps[level], level);

        PassViews convolve_pass = {};
        convolve_pass.maps = denoised[level];
        convolve_pass.moments = denoised_moments[level];
        convolve_pass.dst_maps = convolved[level];
        add_pass(convolve_names[level], RT_LAB_GRADIENT_OCT_MODE_CONVOLVE, convolve_pass, maps[level], level);
    }

    info.tg.add_task(daxa::HeadTask<RtLabGradientOctResolveH::Info>()
        .head_views(RtLabGradientOctResolveH::Info::Views{
            .globals = info.render_context.tgpu_render_data.view(),
            .view_cam_depth = info.view_cam_depth,
            .view_cam_face_normals = info.view_cam_face_normals,
            .view_cam_half_res_depth = info.view_cam_half_res_depth.current(),
            .view_cam_half_res_face_normals = info.view_cam_half_res_face_normals.current(),
            .convolved0 = convolved[0],
            .convolved1 = convolved[1],
            .convolved2 = convolved[2],
            .positions0 = positions[0],
            .positions1 = positions[1],
            .positions2 = positions[2],
            .sums0 = denoised[0],
            .sums1 = denoised[1],
            .sums2 = denoised[2],
            .gradient2 = gradient2,
            .debug_image = info.debug_image,
        })
        .executes(rt_lab_gradient_oct_resolve_callback, &info.render_context, lab_height));
    return gradient2;
}
