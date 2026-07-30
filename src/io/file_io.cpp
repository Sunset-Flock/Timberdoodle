#include "file_io.hpp"

// NOMINMAX/WIN32_LEAN_AND_MEAN are scoped to this translation unit only, so <Windows.h>'s min/max macros
// never leak into the rest of the codebase (which uses std::min/std::max freely).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

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

auto open_error_to_file_io_error(DWORD last_error) -> FileIoResult
{
    switch (last_error)
    {
        case ERROR_FILE_NOT_FOUND:
        case ERROR_PATH_NOT_FOUND:      return FileIoResult::NOT_FOUND;
        case ERROR_SHARING_VIOLATION:
        case ERROR_LOCK_VIOLATION:      return FileIoResult::LOCKED;
        default:                        return FileIoResult::IO_FAILED;
    }
}
} // namespace

auto read_file(std::filesystem::path const & path, std::optional<ByteSlice> slice) -> std::pair<FileIoResult, std::vector<std::byte>>
{
    Win32Handle file = {.handle = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ, // other readers may open concurrently, writers may not
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr)};
    if (file.handle == INVALID_HANDLE_VALUE) { return {open_error_to_file_io_error(GetLastError()), {}}; }

    LARGE_INTEGER file_size = {};
    if (!GetFileSizeEx(file.handle, &file_size)) { return {FileIoResult::IO_FAILED, {}}; }

    // A whole-file read can only learn its length here - which is why a whole-file location stores none.
    ByteSlice const read_slice = slice.value_or(ByteSlice{.byte_offset = 0, .byte_length = s_cast<u64>(file_size.QuadPart)});
    if (read_slice.byte_offset + read_slice.byte_length > s_cast<u64>(file_size.QuadPart)) { return {FileIoResult::OUT_OF_BOUNDS, {}}; }

    LARGE_INTEGER seek_target = {};
    seek_target.QuadPart = s_cast<LONGLONG>(read_slice.byte_offset);
    if (!SetFilePointerEx(file.handle, seek_target, nullptr, FILE_BEGIN)) { return {FileIoResult::IO_FAILED, {}}; }

    std::vector<std::byte> data(s_cast<usize>(read_slice.byte_length));
    DWORD bytes_read = 0;
    BOOL const ok = ReadFile(file.handle, data.data(), s_cast<DWORD>(data.size()), &bytes_read, nullptr);
    if (!ok || bytes_read != data.size()) { return {FileIoResult::IO_FAILED, {}}; }
    return {FileIoResult::SUCCESS, data};
}

auto write_file(std::filesystem::path const & path, void const * data, usize size) -> FileIoResult
{
    Win32Handle file = {.handle = CreateFileW(
        path.c_str(),
        GENERIC_WRITE,
        0, // deny read and write to everyone else while the write is in flight
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        nullptr)};
    if (file.handle == INVALID_HANDLE_VALUE) { return open_error_to_file_io_error(GetLastError()); }

    DWORD bytes_written = 0;
    BOOL const ok = WriteFile(file.handle, data, s_cast<DWORD>(size), &bytes_written, nullptr);
    if (!ok || bytes_written != s_cast<DWORD>(size)) { return FileIoResult::IO_FAILED; }
    return FileIoResult::SUCCESS;
}

auto rename_file(std::filesystem::path const & from, std::filesystem::path const & to) -> FileIoResult
{
    if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING)) { return open_error_to_file_io_error(GetLastError()); }
    return FileIoResult::SUCCESS;
}

auto read_file_modified_time(std::filesystem::path const & path) -> std::optional<i64>
{
    std::error_code ec = {};
    auto const write_time = std::filesystem::last_write_time(path, ec);
    if (ec) { return std::nullopt; }
    return write_time.time_since_epoch().count();
}