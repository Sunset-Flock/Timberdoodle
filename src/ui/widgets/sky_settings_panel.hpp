#pragma once

#include <imgui.h>
#include "../ui_shared.hpp"
#include "../../rendering/scene_renderer_context.hpp"

// Sun + atmosphere settings block, drawn with plain ImGui and ImDrawList.

namespace tido
{
    namespace ui
    {
        void draw_sky_settings_panel(RenderContext & render_context);
    } // namespace ui
} // namespace tido
