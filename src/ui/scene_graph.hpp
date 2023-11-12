#pragma once
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imgui.h>
#include "../timberdoodle.hpp"

namespace tido
{
    namespace ui
    {
        enum struct NodeType
        {
            LEAF,
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

                auto get_cell_bounds() -> ImRect;
                auto add_leaf_node(std::string uuid) -> RetNodeState;
                auto add_inner_node(std::string uuid) -> RetNodeState;
        };
    }
}