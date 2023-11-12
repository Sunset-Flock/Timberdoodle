#include "ui.hpp"

UIEngine::UIEngine(Window &window)
{
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(window.glfw_handle, true);
}

void UIEngine::main_update(Settings &settings, Scene const & scene)
{
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("Widgets"))
        {
            ImGui::MenuItem("Settings", NULL, &widget_settings);
            ImGui::MenuItem("Renderer Statistics", NULL, &widget_renderer_statistics);
            ImGui::MenuItem("Scene Hierarchy", NULL, &widget_scene_hierarchy);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
    if (widget_settings)
    {
        if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            ImGui::InputScalarN("resolution", ImGuiDataType_U32, &settings.render_target_size, 2);
            ImGui::End();
        }
    }
    if (widget_renderer_statistics)
    {
        if (ImGui::Begin("Renderer Statistics", nullptr, ImGuiWindowFlags_NoCollapse))
        {
            ImGui::Text("fps: 100");
            ImGui::End();
        }
    }
    if(widget_scene_hierarchy)
    {
        draw_scenegraph(scene);
    }
    ImGui::Render();
}

void UIEngine::draw_scenegraph(Scene const & scene)
{
    ImGui::Begin("Scene Hierarchy", nullptr, ImGuiWindowFlags_NoCollapse);
    scene_graph.begin();

    struct StackEntry
    {
        RenderEntityId id;
        bool is_first_child;
        bool is_last_child;
        u32 accumulated_indent;
    };
    // ROOT - first/last - indent and accumulate
    //  INNER - first - indent
    //    LEAF - first - indent and unindent
    //    INNER - last - indent and accumulate
    //      LEAF - first/last - indent once and unindent twice (one accumulated from parent INNER )
    //  INNER - middle - indent and unindent
    //  INNER - last - indent and unindent twice (one accumulated from ROOT)
    std::vector<StackEntry> entity_graph_stack;
    entity_graph_stack.push_back({scene._scene_file_manifest.at(0).root_render_entity, true, true});

    while(!entity_graph_stack.empty())
    {
        auto const top_stack_entry = entity_graph_stack.back();
        entity_graph_stack.pop_back();

        auto const &entity = *scene._render_entities.slot(top_stack_entry.id);
        if(top_stack_entry.is_first_child) 
        {
            scene_graph.add_level();
        }
        NodeType type = entity.first_child.has_value() ? NodeType::INNER : NodeType::LEAF;
        std::string const uuid = fmt::format("{}_{}_{}", entity.name, top_stack_entry.id.index, top_stack_entry.id.version);
        RetNodeState const result = scene_graph.add_node(type, uuid);

        // Check if we are both the last child and we will add no more of our own children to the stack
        bool const should_remove_indentation = top_stack_entry.is_last_child && result == RetNodeState::CLOSED;
        // If that is the case remove our own indentation + the accumulated indentation from our parents
        if(should_remove_indentation) 
        {
            // Remove all the previous indentations that were accumulated as well as our own
            for(u32 pop_count = 0; pop_count < top_stack_entry.accumulated_indent + 1; pop_count++)
            {
                scene_graph.remove_level();
            }
        } 
        // If this node was not opened on previous frame or it is a leaf (aka has no more children)
        // it is CLOSED (maybe COLLAPSED is a better name) and we don't want to continue to the next entry on stack
        if(result == RetNodeState::CLOSED) { continue; }

        // Otherwise we find collect all of our children into a vector
        /// NOTE: We collect into a vector because we want our children to be processed in the order
        //        of first -> last, but if we just pushed them directly to the stack we would have the inverse order
        //        as the first child will be pushed first and latter children would be stacked on top of it
        RenderEntityId curr_child_index = entity.first_child.value();
        std::vector<StackEntry> child_entries = {};
        while(true)
        {
            auto const * curr_child = scene._render_entities.slot(curr_child_index);
            bool const is_first_child = child_entries.empty();
            bool const is_last_child = !curr_child->next_sibling.has_value();
            // If the curenntly processed (pop from stack) node was a last child and we are adding
            // it's own last child we should accumulate the indentation so that the leaf then knows
            // how many levels of indentation it should remove
            bool const should_accumulate_remove_indent = top_stack_entry.is_last_child && is_last_child;
            u32 const accumulated_indent = should_accumulate_remove_indent ? top_stack_entry.accumulated_indent + 1 : 0;
            child_entries.push_back({
                .id = curr_child_index,
                .is_first_child = is_first_child,
                .is_last_child = is_last_child,
                .accumulated_indent = accumulated_indent
            });
            if(is_last_child) { break; }
            // Move on to the next child
            curr_child_index = curr_child->next_sibling.value();
        }
        entity_graph_stack.reserve(entity_graph_stack.size() + child_entries.size());
        entity_graph_stack.insert(entity_graph_stack.end(), child_entries.rbegin(), child_entries.rend());
    }

    scene_graph.end();
    ImGui::End(); // Scene graph widget window

    ImGui::ShowDemoWindow();
}