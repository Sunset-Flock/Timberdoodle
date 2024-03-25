#pragma once

#include "dvmaa.inl"

namespace dvmaa
{
    inline auto create_dvmaa_ms_visbuffer(daxa::TaskGraph tg, RenderContext const & render_context) -> daxa::TaskImageView
    {
        return tg.create_transient_image({
            .format = daxa::Format::R32_UINT,
            .size = { 
                render_context.render_data.settings.render_target_size.x,
                render_context.render_data.settings.render_target_size.y,
                1,
            },
            .sample_count = 4,
            .name = "dvmaa ms visbuffer",
        });
    }

    inline auto create_dvmaa_ms_depth(daxa::TaskGraph tg, RenderContext const & render_context) -> daxa::TaskImageView
    {
        return tg.create_transient_image({
            .format = daxa::Format::D32_SFLOAT,
            .size = { 
                render_context.render_data.settings.render_target_size.x,
                render_context.render_data.settings.render_target_size.y,
                1,
            },
            .sample_count = 4,
            .name = "dvmaa ms depth",
        });
    }

    // Creates depth image with Sfloat instead of Dfloat.
    inline auto create_dvmaa_depth(daxa::TaskGraph tg, RenderContext const & render_context) -> daxa::TaskImageView
    {
        return tg.create_transient_image({
            .format = daxa::Format::R32_SFLOAT,
            .size = { 
                render_context.render_data.settings.render_target_size.x,
                render_context.render_data.settings.render_target_size.y,
                1,
            },
            .name = "dvmaa depth",
        });
    }
}