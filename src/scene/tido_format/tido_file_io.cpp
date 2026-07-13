#include "tido_file_io.hpp"

// NOMINMAX/WIN32_LEAN_AND_MEAN are scoped to this translation unit only, so <Windows.h>'s min/max macros
// never leak into the rest of the codebase (which uses std::min/std::max freely).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <algorithm>
#include <chrono>
#include <thread>

namespace
{
struct Win32Handle
{
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~Win32Handle()
    {
        if (handle != INVALID_HANDLE_VALUE) { CloseHandle(handle); }
    }
};
} // namespace

auto tido_write_file_exclusive(std::filesystem::path const & path, void const * data, usize size) -> bool
{
    DWORD retry_delay_milliseconds = 1;
    while (true)
    {
        Win32Handle file = {.handle = CreateFileW(
            path.c_str(),
            GENERIC_WRITE,
            0, // deny read and write to everyone else while the write is in flight
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr)};

        if (file.handle != INVALID_HANDLE_VALUE)
        {
            DWORD bytes_written = 0;
            BOOL const ok = WriteFile(file.handle, data, s_cast<DWORD>(size), &bytes_written, nullptr);
            return ok && bytes_written == s_cast<DWORD>(size);
        }

        DWORD const last_error = GetLastError();
        if (last_error != ERROR_SHARING_VIOLATION)
        {
            return false; // Not contention (bad path, permissions, ...) - do not spin on this.
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_milliseconds));
        retry_delay_milliseconds = std::min<DWORD>(retry_delay_milliseconds * 2, 16);
    }
}

auto tido_read_file_shared(std::filesystem::path const & path) -> std::optional<std::vector<std::byte>>
{
    Win32Handle file = {.handle = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ, // other readers may open concurrently, writers may not
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr)};
    if (file.handle == INVALID_HANDLE_VALUE) { return std::nullopt; }

    LARGE_INTEGER file_size = {};
    if (!GetFileSizeEx(file.handle, &file_size)) { return std::nullopt; }

    std::vector<std::byte> data(s_cast<usize>(file_size.QuadPart));
    DWORD bytes_read = 0;
    BOOL const ok = ReadFile(file.handle, data.data(), s_cast<DWORD>(data.size()), &bytes_read, nullptr);
    if (!ok || bytes_read != data.size()) { return std::nullopt; }
    return data;
}
