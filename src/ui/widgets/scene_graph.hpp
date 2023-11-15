#pragma once
#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>
#include "../ui_shared.hpp"
#include "../../timberdoodle.hpp"
#include "../../scene/scene.hpp"

namespace tido
{
    namespace ui
    {
        enum struct RetNodeState
        {
            OPEN,
            CLOSED,
            ERROR
        };

        struct SceneGraph
        {
            public:
                f32 icon_text_spacing = {};
                f32 icon_size = {};
                f32 indent = {};
                SceneGraph() = default;
                SceneGraph(daxa::ImGuiRenderer * renderer, std::vector<daxa::ImageId> const * icons, daxa::SamplerId linear_sampler);
                SceneGraph(SceneGraph const &) = delete;
                SceneGraph(SceneGraph const &&) = delete;
                void begin();
                void end();
                auto add_node(RenderEntity const & entity, Scene const & scene) -> RetNodeState;
                void add_level();
                void remove_level();

            private:
                enum struct LeafType
                {
                    MATERIAL,
                    MESH,
                    CAMERA,
                    LIGHT
                };
                ImGuiContext *context = {};
                ImGuiTable *table = {};
                ImGuiWindow *window = {};
                daxa::ImGuiRenderer * renderer = {};
                daxa::SamplerId linear_sampler = {};
                std::vector<daxa::ImageId> const * icons = {};

                ImGuiID selected_id = {};

                auto get_cell_bounds() -> ImRect;
                auto add_meshgroup_node(RenderEntity const & entity, Scene const & scene) -> RetNodeState;
                auto add_leaf_node(std::string uuid, ICONS icon) -> RetNodeState;
                auto add_inner_node(std::string uuid, ICONS icon = ICONS::SIZE) -> RetNodeState;
        };
    }
}