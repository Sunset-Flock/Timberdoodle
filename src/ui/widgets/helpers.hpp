#pragma once

#include <imgui.h>
#include <imgui_internal.h>
#include "../ui_shared.hpp"

struct State
{
    bool hovered = {};
    bool pressed = {};
    bool held = {};
};

inline auto button_like_behavior = [](ImRect bb, ImGuiID id) -> State
{
    State ret_state = {};
    ImGuiButtonFlags const flags = ImGuiButtonFlags_AllowOverlap | ImGuiButtonFlags_PressedOnClick;
    ret_state.pressed = ImGui::ButtonBehavior(bb, id, &ret_state.hovered, &ret_state.held, flags);
    return ret_state;
};

static constexpr inline auto icon_to_color(ICONS icon) -> ImVec4
{
    switch(icon)
    {
        case ICONS::CAMERA:     [[fallthrough]];
        case ICONS::COLLECTION: [[fallthrough]];
        case ICONS::LIGHT:
            return ImVec4(1.00f, 0.62f, 0.37f, 1.00f);
        case ICONS::MESH:       [[fallthrough]];
        case ICONS::MESHGROUP:
            return ImVec4(0.40f, 1.00f, 0.52f, 1.00f);
        case ICONS::MATERIAL:
            return ImVec4(1.00f, 0.37f, 0.37f, 1.00f);
        default:
            return ImGui::GetStyleColorVec4(ImGuiCol_Text);
    }
}