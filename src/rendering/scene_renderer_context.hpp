#pragma once

#include <string>
#include <daxa/utils/imgui.hpp>

#include "../window.hpp"
#include "../scene/scene.hpp"
#include "../scene/asset_processor.hpp"

#include "../shader_shared/asset.inl"
#include "../shader_shared/geometry_pipeline.inl"

#include "../gpu_context.hpp"

struct SceneRendererContext
{
    std::array<std::vector<MeshDrawTuple>, 2> opaque_draw_lists = {};
    daxa::TaskBuffer opaque_draw_list_buffer = daxa::TaskBuffer{{.name = "opaque draw lists"}};
};