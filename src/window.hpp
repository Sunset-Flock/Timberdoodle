#pragma once

#include <array>

#include "timberdoodle.hpp"
using namespace tido::types;

#include <GLFW/glfw3.h>

struct WindowState
{
    bool b_close_requested = {};
    bool b_focused = true;
    std::array<bool, 5> mouse_button_down_old = {};
    std::array<bool, 5> mouse_button_down = {};
    std::array<bool, 512> key_down = {};
    std::array<bool, 512> key_down_old = {};
    i32 old_cursor_pos_x = {};
    i32 old_cursor_pos_y = {};
    i32 cursor_change_x = {};
    i32 cursor_change_y = {};
};

using Key = i32;
using Button = i32;

struct Window
{
    Window(i32 width, i32 height, std::string_view name);
    Window(Window &&) = default;
    ~Window();

    bool update(f32 deltaTime);

    auto key_pressed(Key key) const -> bool;
    auto key_just_pressed(Key key) const -> bool;
    auto key_just_released(Key key) const -> bool;

    auto button_pressed(Button button) const -> bool;
    auto button_just_pressed(Button button) const -> bool;
    auto button_just_released(Button button) const -> bool;

    auto scroll_x() const -> f32;
    auto scroll_y() const -> f32;

    auto cursor_x() const -> i32;
    auto get_cursor_x() const -> i32;
    auto get_cursor_y() const -> i32;
    auto get_cursor_change_x() const -> i32;
    auto get_cursor_change_y() const -> i32;
    auto is_cursor_over_window() const -> bool;
    void capture_cursor();
    void release_cursor();
    auto is_cursor_captured() const -> bool;

    auto is_focused() const -> bool;

    void set_width(u32 width);
    void set_height(u32 height);
    auto get_width() const -> u32;
    auto get_height() const -> u32;

    void set_name(std::string name);
    auto get_name() -> std::string const &;

    i32vec2 size = {};
    std::unique_ptr<WindowState> window_state = {};
    u32 glfw_window_id = {};
    bool cursor_captured = {};
    std::string name = {};
    GLFWwindow *glfw_handle = {};
    i32 cursor_pos_change_x = {};
    i32 cursor_pos_change_y = {};
};