#pragma once
#include <array>
#include <imgui.h>
#include "../timberdoodle.hpp"

namespace tido
{
    namespace ui
    {
        enum struct ICONS
        {
            CHEVRON_UP = 0,
            CHEVRON_DOWN,
            CHEVRON_RIGHT,
            MESH,
            MESHGROUP,
            PLUS,
            MINUS,
            CAMERA,
            LIGHT,
            MATERIAL,
            COLLECTION,
            SUN,
            SIZE,
        };

        static inline auto const tab_alt_0 = ImVec4(0.22, 0.22, 0.22, 1.0f);
        static inline auto const tab_alt_1 = ImVec4(0.13, 0.13, 0.13, 1.0f);

        static inline auto const bg_0 = ImVec4(0.09, 0.09, 0.09, 1.0f);
        static inline auto const bg_1 = ImVec4(0.14, 0.14, 0.14, 1.0f);
        static inline auto const bg_2 = ImVec4(0.16, 0.16, 0.16, 1.0f);
        static inline auto const bg_3 = ImVec4(0.18, 0.18, 0.18, 1.0f);
        static inline auto const bg_4 = ImVec4(0.22, 0.22, 0.22, 1.0f);
        static inline auto const bg_5 = ImVec4(0.27, 0.27, 0.27, 1.0f);

        static inline auto const alt_1 = ImVec4(0.22, 0.22, 0.22, 1.0f);
        static inline auto const alt_2 = ImVec4(0.32, 0.32, 0.32, 1.0f);

        static inline auto const hovered_1 = ImVec4(0.28, 0.28, 0.28, 0.5f);

        static inline auto const select_blue_1 = ImVec4(0.05, 0.29, 0.65, 1.0f);
        static inline auto const select_orange_1 = ImVec4(0.75, 0.30, 0.14, 1.0f);

        struct SceneInterfaceState
        {
            u32 picked_entity = {};
        };
    } // namespace ui
} // namespace tido