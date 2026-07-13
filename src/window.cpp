#include "window.hpp"


#if defined(_WIN32)
#include <dwmapi.h>
#include <shobjidl.h>
#ifndef DWMWA_USE_IMMERSIVE_DARK_MODE
#define DWMWA_USE_IMMERSIVE_DARK_MODE 20
#endif // DWMWA_USE_IMMERSIVE_DARK_MODE
#endif // defined(_WIN32)

#include <vector>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

using namespace tido::types;

void close_callback(GLFWwindow *window)
{
    WindowState *self = reinterpret_cast<WindowState *>(glfwGetWindowUserPointer(window));
    self->b_close_requested = true;
}

void key_callback(GLFWwindow *window, int key, [[maybe_unused]]int scancode, int action, [[maybe_unused]]int mods)
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

void mouse_button_callback(GLFWwindow *window, int button, int action, [[maybe_unused]]int mods)
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
        auto hwnd = s_cast<HWND>(glfwGetWin32Window(glfw_handle));
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

bool Window::update([[maybe_unused]]f32 deltaTime)
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

namespace
{
    auto utf8_to_wide_string(std::string_view const utf8_string) -> std::wstring
    {
        if (utf8_string.empty())
        {
            return {};
        }
        i32 const wide_char_count = MultiByteToWideChar(CP_UTF8, 0, utf8_string.data(), static_cast<i32>(utf8_string.size()), nullptr, 0);
        std::wstring wide_string(static_cast<usize>(wide_char_count), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, utf8_string.data(), static_cast<i32>(utf8_string.size()), wide_string.data(), wide_char_count);
        return wide_string;
    }

    auto wide_string_to_utf8(PCWSTR const wide_string) -> std::string
    {
        if (wide_string == nullptr)
        {
            return {};
        }
        i32 const utf8_byte_count_with_null = WideCharToMultiByte(CP_UTF8, 0, wide_string, -1, nullptr, 0, nullptr, nullptr);
        if (utf8_byte_count_with_null <= 0)
        {
            return {};
        }
        std::string utf8_string(static_cast<usize>(utf8_byte_count_with_null - 1), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wide_string, -1, utf8_string.data(), utf8_byte_count_with_null, nullptr, nullptr);
        return utf8_string;
    }

    auto strip_trailing_path_separator(std::wstring path) -> std::wstring
    {
        while (!path.empty() && (path.back() == L'\\' || path.back() == L'/'))
        {
            path.pop_back();
        }
        return path;
    }

    // Blocks IFileDialog navigation to any folder outside of a fixed root, so browsing for an asset
    // cannot wander into unrelated parts of the filesystem. Ref-counted manually since it is handed to
    // COM via IFileDialog::Advise instead of being created through CoCreateInstance.
    struct RootRestrictedFileDialogEvents final : IFileDialogEvents
    {
        std::wstring root_path;
        ULONG ref_count = 1;

        explicit RootRestrictedFileDialogEvents(std::wstring root_path) : root_path{strip_trailing_path_separator(std::move(root_path))} {}

        HRESULT __stdcall QueryInterface(REFIID riid, void ** object_out) override
        {
            if (riid == IID_IUnknown || riid == IID_IFileDialogEvents)
            {
                *object_out = static_cast<IFileDialogEvents *>(this);
                AddRef();
                return S_OK;
            }
            *object_out = nullptr;
            return E_NOINTERFACE;
        }

        ULONG __stdcall AddRef() override
        {
            return ++ref_count;
        }

        ULONG __stdcall Release() override
        {
            ULONG const new_ref_count = --ref_count;
            if (new_ref_count == 0)
            {
                delete this;
            }
            return new_ref_count;
        }

        HRESULT __stdcall OnFolderChanging(IFileDialog *, IShellItem * candidate_folder) override
        {
            PWSTR candidate_path_raw = nullptr;
            if (FAILED(candidate_folder->GetDisplayName(SIGDN_FILESYSPATH, &candidate_path_raw)))
            {
                // Non-filesystem locations (e.g. "This PC", network locations) have no comparable path; reject them.
                return E_ACCESSDENIED;
            }
            std::wstring const candidate_path{candidate_path_raw};
            CoTaskMemFree(candidate_path_raw);

            bool is_within_root = candidate_path.size() >= root_path.size() &&
                                   CompareStringOrdinal(
                                       candidate_path.c_str(), static_cast<i32>(root_path.size()),
                                       root_path.c_str(), static_cast<i32>(root_path.size()),
                                       TRUE) == CSTR_EQUAL;
            if (is_within_root && candidate_path.size() > root_path.size())
            {
                wchar_t const boundary_char = candidate_path[root_path.size()];
                is_within_root = boundary_char == L'\\' || boundary_char == L'/';
            }
            return is_within_root ? S_OK : E_ACCESSDENIED;
        }

        HRESULT __stdcall OnFileOk(IFileDialog *) override { return S_OK; }
        HRESULT __stdcall OnFolderChange(IFileDialog *) override { return S_OK; }
        HRESULT __stdcall OnSelectionChange(IFileDialog *) override { return S_OK; }
        HRESULT __stdcall OnShareViolation(IFileDialog *, IShellItem *, FDE_SHAREVIOLATION_RESPONSE *) override { return S_OK; }
        HRESULT __stdcall OnTypeChange(IFileDialog *) override { return S_OK; }
        HRESULT __stdcall OnOverwrite(IFileDialog *, IShellItem *, FDE_OVERWRITE_RESPONSE *) override { return S_OK; }
    };
}

// `filter` follows the legacy GetOpenFileName format: a run of null-separated "Description", "*.ext" pairs
// terminated by an extra trailing null (e.g. "GLTF\0*.gltf\0"). std::string_view truncates at the first
// embedded null, so the pairs are walked directly off filter.data() as a raw C string instead.
std::string open_file_dialog(std::string_view const filter, std::string_view const initial_dir)
{
    std::string result;

    HRESULT const co_initialize_result = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(co_initialize_result) && co_initialize_result != RPC_E_CHANGED_MODE)
    {
        return result;
    }
    bool const should_uninitialize_com = SUCCEEDED(co_initialize_result);

    IFileOpenDialog * file_open_dialog = nullptr;
    HRESULT const create_instance_result = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&file_open_dialog));
    if (SUCCEEDED(create_instance_result))
    {
        // Wide string storage must outlive the COMDLG_FILTERSPEC array, which only holds pointers into it.
        std::vector<std::wstring> filter_descriptions = {};
        std::vector<std::wstring> filter_specs = {};
        char const * filter_cursor = filter.data();
        if (filter_cursor != nullptr)
        {
            while (*filter_cursor != '\0')
            {
                std::string_view const description{filter_cursor};
                filter_cursor += description.size() + 1;
                std::string_view const spec{filter_cursor};
                filter_cursor += spec.size() + 1;
                filter_descriptions.push_back(utf8_to_wide_string(description));
                filter_specs.push_back(utf8_to_wide_string(spec));
            }
        }
        if (!filter_descriptions.empty())
        {
            std::vector<COMDLG_FILTERSPEC> filter_spec_array = {};
            filter_spec_array.reserve(filter_descriptions.size());
            for (usize filter_index = 0; filter_index < filter_descriptions.size(); ++filter_index)
            {
                filter_spec_array.push_back(COMDLG_FILTERSPEC{filter_descriptions[filter_index].c_str(), filter_specs[filter_index].c_str()});
            }
            file_open_dialog->SetFileTypes(static_cast<UINT>(filter_spec_array.size()), filter_spec_array.data());
        }

        // Advise() AddRefs the events object, so it stays alive as long as the dialog holds a reference to it.
        RootRestrictedFileDialogEvents * root_restriction_events = nullptr;
        DWORD root_restriction_cookie = 0;
        if (!initial_dir.empty())
        {
            std::wstring const initial_dir_wide = utf8_to_wide_string(initial_dir);
            IShellItem * initial_dir_item = nullptr;
            if (SUCCEEDED(SHCreateItemFromParsingName(initial_dir_wide.c_str(), nullptr, IID_PPV_ARGS(&initial_dir_item))))
            {
                file_open_dialog->SetFolder(initial_dir_item);

                // Restrict browsing to initial_dir and below (e.g. the assets root), so the dialog cannot
                // navigate to unrelated parts of the filesystem when picking an asset to load.
                PWSTR initial_dir_path_raw = nullptr;
                if (SUCCEEDED(initial_dir_item->GetDisplayName(SIGDN_FILESYSPATH, &initial_dir_path_raw)))
                {
                    root_restriction_events = new RootRestrictedFileDialogEvents(std::wstring{initial_dir_path_raw});
                    CoTaskMemFree(initial_dir_path_raw);
                    if (FAILED(file_open_dialog->Advise(root_restriction_events, &root_restriction_cookie)))
                    {
                        root_restriction_events->Release();
                        root_restriction_events = nullptr;
                    }
                }
                initial_dir_item->Release();
            }
        }

        // FOS_FORCEFILESYSTEM restricts results to real filesystem paths, matching GetOpenFileName's behavior.
        // Unlike GetOpenFileName, IFileOpenDialog never changes the process's current working directory.
        DWORD dialog_options = 0;
        file_open_dialog->GetOptions(&dialog_options);
        file_open_dialog->SetOptions(dialog_options | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST | FOS_FORCEFILESYSTEM);

        if (SUCCEEDED(file_open_dialog->Show(nullptr)))
        {
            IShellItem * result_item = nullptr;
            if (SUCCEEDED(file_open_dialog->GetResult(&result_item)))
            {
                PWSTR result_path = nullptr;
                if (SUCCEEDED(result_item->GetDisplayName(SIGDN_FILESYSPATH, &result_path)))
                {
                    result = wide_string_to_utf8(result_path);
                    CoTaskMemFree(result_path);
                }
                result_item->Release();
            }
        }

        if (root_restriction_events != nullptr)
        {
            file_open_dialog->Unadvise(root_restriction_cookie);
            root_restriction_events->Release();
        }

        file_open_dialog->Release();
    }

    if (should_uninitialize_com)
    {
        CoUninitialize();
    }

    return result;
}

