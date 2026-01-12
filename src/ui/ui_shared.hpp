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

        static inline auto const tab_alt_0 = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
        static inline auto const tab_alt_1 = ImVec4(0.13f, 0.13f, 0.13f, 1.0f);

        static inline auto const bg_0 = ImVec4(0.09f, 0.09f, 0.09f, 1.0f);
        static inline auto const bg_1 = ImVec4(0.14f, 0.14f, 0.14f, 1.0f);
        static inline auto const bg_2 = ImVec4(0.16f, 0.16f, 0.16f, 1.0f);
        static inline auto const bg_3 = ImVec4(0.18f, 0.18f, 0.18f, 1.0f);
        static inline auto const bg_4 = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
        static inline auto const bg_5 = ImVec4(0.27f, 0.27f, 0.27f, 1.0f);

        static inline auto const alt_1 = ImVec4(0.22f, 0.22f, 0.22f, 1.0f);
        static inline auto const alt_2 = ImVec4(0.32f, 0.32f, 0.32f, 1.0f);

        static inline auto const hovered_1 = ImVec4(0.28f, 0.28f, 0.28f, 0.5f);

        static inline auto const select_blue_1 = ImVec4(0.05f, 0.29f, 0.65f, 1.0f);
        static inline auto const select_orange_1 = ImVec4(0.75f, 0.30f, 0.14f, 1.0f);

        struct SceneInterfaceState
        {
            u32 picked_entity = ~0u;
            u32 picked_mesh_in_meshgroup = {};
            u32 picked_mesh = {};
            u32 picked_meshlet_in_mesh = {};
            u32 picked_triangle_in_meshlet = {};
        };
    } // namespace ui
} // namespace tido