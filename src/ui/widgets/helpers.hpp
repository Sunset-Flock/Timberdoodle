#pragma once

#include <imgui.h>
#include <imgui_internal.h>

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