#pragma once
#include <daxa/utils/imgui.hpp>
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>
#include "../ui_shared.hpp"
#include "../../timberdoodle.hpp"

namespace tido
{
    namespace ui
    {
        enum struct NodeType
        {
            MESH,
            INNER,
            UNKNOWN
        };

        enum struct RetNodeState
        {
            OPEN,
            CLOSED,
            ERROR
        };

        struct SceneGraph
        {
            public:
                SceneGraph() = default;
                SceneGraph(daxa::ImGuiRenderer * renderer, std::vector<daxa::ImageId> const * icons, daxa::SamplerId linear_sampler);
                SceneGraph(SceneGraph const &) = delete;
                SceneGraph(SceneGraph const &&) = delete;
                void begin();
                void end();
                auto add_node(NodeType type, std::string uuid) -> RetNodeState;
                void add_level();
                void remove_level();

            private:
                ImGuiContext *context = {};
                ImGuiTable *table = {};
                ImGuiWindow *window = {};
                daxa::ImGuiRenderer * renderer = {};
                daxa::SamplerId linear_sampler = {};
                std::vector<daxa::ImageId> const * icons = {};

                ImGuiID selected_id = {};
                f32 per_level_indent = {};

                auto get_cell_bounds() -> ImRect;
                auto add_leaf_node(std::string uuid, NodeType type) -> RetNodeState;
                auto add_inner_node(std::string uuid) -> RetNodeState;
        };
    }
}