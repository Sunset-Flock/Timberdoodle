#include "window.hpp"


#if defined(_WIN32)
#include <dwmapi.h>
#ifndef DWMWA_USE_IMMERSIVE_DARK_MODE
#define DWMWA_USE_IMMERSIVE_DARK_MODE 20
#endif // DWMWA_USE_IMMERSIVE_DARK_MODE
#endif // defined(_WIN32)

#include <GLFW/glfw3native.h>

using namespace tido::types;

void close_callback(GLFWwindow *window)
{
    WindowState *self = reinterpret_cast<WindowState *>(glfwGetWindowUserPointer(window));
    self->b_close_requested = true;
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == -1)
        return;
    WindowState *self = reinterpret_cast<WindowState *>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS)
    {
        self->key_down[key] = true;
    }
    else if (action == GLFW_RELEASE)
    {
        self->key_down[key] = false;
    }
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    WindowState *self = reinterpret_cast<WindowState *>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS)
    {
        self->mouse_button_down[button] = true;
    }
    else if (action == GLFW_RELEASE)
    {
        self->mouse_button_down[button] = false;
    }
}

void cursor_move_callback(GLFWwindow *window, double xpos, double ypos)
{
    WindowState *self = reinterpret_cast<WindowState *>(glfwGetWindowUserPointer(window));
    self->cursor_change_x = static_cast<i32>(std::floor(xpos)) - self->old_cursor_pos_x;
    self->cursor_change_y = static_cast<i32>(std::floor(ypos)) - self->old_cursor_pos_y;
}

void window_focus_callback(GLFWwindow *window, int focused)
{
    WindowState *self = reinterpret_cast<WindowState *>(glfwGetWindowUserPointer(window));
    self->b_focused = focused;
}

Window::Window(i32 width, i32 height, std::string_view name)
    : size{width, height},
        name{name},
        glfw_handle{
            [=]()
            {
                glfwInit();
                glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
                return glfwCreateWindow(width, height, name.data(), nullptr, nullptr);
            }()},
        window_state{std::make_unique<WindowState>()}
{
    glfwSetWindowUserPointer(this->glfw_handle, window_state.get());

    glfwSetWindowCloseCallback(this->glfw_handle, close_callback);
    glfwSetKeyCallback(this->glfw_handle, key_callback);
    glfwSetMouseButtonCallback(this->glfw_handle, mouse_button_callback);
    glfwSetCursorPosCallback(this->glfw_handle, cursor_move_callback);
    glfwSetWindowFocusCallback(this->glfw_handle, window_focus_callback);
/// NOTE: This makes the borders of the window dark mode on win 10 and 11
#if defined(_WIN32)
    {
        auto hwnd = r_cast<HWND>(glfw_handle);
        BOOL value = true;
        DwmSetWindowAttribute(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, &value, sizeof(value));
        auto is_windows11_or_greater = []() -> bool
        {
            using Fn_RtlGetVersion = void(WINAPI *)(OSVERSIONINFOEX *);
            Fn_RtlGetVersion fn_RtlGetVersion = nullptr;
            auto ntdll_dll = LoadLibrary(TEXT("ntdll.dll"));
            if (ntdll_dll)
                fn_RtlGetVersion = (Fn_RtlGetVersion)GetProcAddress(ntdll_dll, "RtlGetVersion");
            auto version_info = OSVERSIONINFOEX{};
            version_info.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);
            fn_RtlGetVersion(&version_info);
            FreeLibrary(ntdll_dll);
            return version_info.dwMajorVersion >= 10 && version_info.dwMinorVersion >= 0 && version_info.dwBuildNumber >= 22000;
        };
        if (!is_windows11_or_greater())
        {
            MSG msg{.hwnd = hwnd, .message = WM_NCACTIVATE, .wParam = FALSE, .lParam = 0};
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            msg.wParam = TRUE;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
#endif //_WIN32
}

Window::~Window()
{
    glfwDestroyWindow(this->glfw_handle);
    glfwTerminate();
}

const std::string &Window::get_name()
{
    return this->name;
}

bool Window::is_focused() const
{
    return this->window_state->b_focused;
}

// keys

bool Window::key_pressed(Key key) const
{
    return window_state->key_down[key];
}

bool Window::key_just_pressed(Key key) const
{
    return !this->window_state->key_down_old[key] && this->window_state->key_down[key];
}

bool Window::key_just_released(Key key) const
{
    return this->window_state->key_down_old[key] && !this->window_state->key_down[key];
}

// buttons

bool Window::button_pressed(Button button) const
{
    return this->window_state->mouse_button_down[button];
}

bool Window::button_just_pressed(Button button) const
{
    return !this->window_state->mouse_button_down_old[button] && this->window_state->mouse_button_down[button];
}

bool Window::button_just_released(Button button) const
{
    return this->window_state->mouse_button_down_old[button] && !this->window_state->mouse_button_down[button];
}

// cursor

i32 Window::get_cursor_x() const
{
    double x, y;
    glfwGetCursorPos(this->glfw_handle, &x, &y);
    return static_cast<i32>(std::floor(x));
}

i32 Window::get_cursor_y() const
{
    double x, y;
    glfwGetCursorPos(this->glfw_handle, &x, &y);
    return static_cast<i32>(std::floor(y));
}

i32 Window::get_cursor_change_x() const
{
    return this->window_state->cursor_change_x;
}

i32 Window::get_cursor_change_y() const
{
    return this->window_state->cursor_change_y;
}

bool Window::is_cursor_over_window() const
{
    double x, y;
    glfwGetCursorPos(this->glfw_handle, &x, &y);
    i32 width, height;
    glfwGetWindowSize(this->glfw_handle, &width, &height);
    return x >= 0 && x <= width && y >= 0 && y <= height;
}

void Window::capture_cursor()
{
    glfwSetInputMode(this->glfw_handle, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void Window::release_cursor()
{
    glfwSetInputMode(this->glfw_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

bool Window::is_cursor_captured() const
{
    return glfwGetInputMode(this->glfw_handle, GLFW_CURSOR) == GLFW_CURSOR_DISABLED;
}

bool Window::update(f32 deltaTime)
{
    this->window_state->key_down_old = this->window_state->key_down;
    this->window_state->mouse_button_down_old = this->window_state->mouse_button_down;
    this->window_state->old_cursor_pos_x = this->get_cursor_x();
    this->window_state->old_cursor_pos_y = this->get_cursor_y();
    this->window_state->cursor_change_x = {};
    this->window_state->cursor_change_y = {};

    glfwPollEvents();
    if (this->is_cursor_captured())
    {
        glfwSetCursorPos(this->glfw_handle, -10000, -10000);
    }
    return this->window_state->b_close_requested;
}

void Window::set_width(u32 width)
{
    i32 oldW, oldH;
    glfwGetWindowSize(this->glfw_handle, &oldW, &oldH);
    glfwSetWindowSize(this->glfw_handle, width, oldH);
}

void Window::set_height(u32 height)
{
    i32 oldW, oldH;
    glfwGetWindowSize(this->glfw_handle, &oldW, &oldH);
    glfwSetWindowSize(this->glfw_handle, oldW, height);
}

u32 Window::get_width() const
{
    i32 w, h;
    glfwGetWindowSize(this->glfw_handle, &w, &h);
    return w;
}

u32 Window::get_height() const
{
    i32 w, h;
    glfwGetWindowSize(this->glfw_handle, &w, &h);
    return h;
}