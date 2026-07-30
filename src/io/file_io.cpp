#include "file_io.hpp"

#include <algorithm>
#include <limits>

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

auto FileReader::open(std::filesystem::path const & path) -> std::pair<FileIoResult, FileReader>
{
    HANDLE const handle = CreateFileW(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ, // other readers may open concurrently, writers may not
        nullptr,
        OPEN_EXISTING,
        FILE_FLAG_SEQUENTIAL_SCAN, // consumers walk the file front to back, rewinding at most a few times
        nullptr);
    if (handle == INVALID_HANDLE_VALUE) { return {open_error_to_file_io_error(GetLastError()), FileReader{}}; }

    LARGE_INTEGER file_size = {};
    if (!GetFileSizeEx(handle, &file_size))
    {
        CloseHandle(handle);
        return {FileIoResult::IO_FAILED, FileReader{}};
    }

    FileReader reader = {};
    reader.file_handle = handle;
    reader.total_byte_size = s_cast<u64>(file_size.QuadPart);
    return {FileIoResult::SUCCESS, std::move(reader)};
}

FileReader::~FileReader()
{
    if (file_handle != nullptr) { CloseHandle(file_handle); }
}

FileReader::FileReader(FileReader && other) noexcept
    : file_handle{other.file_handle},
      total_byte_size{other.total_byte_size},
      current_byte_offset{other.current_byte_offset}
{
    other.file_handle = nullptr;
}

auto FileReader::operator=(FileReader && other) noexcept -> FileReader &
{
    if (this == &other) { return *this; }
    if (file_handle != nullptr) { CloseHandle(file_handle); }
    file_handle = other.file_handle;
    total_byte_size = other.total_byte_size;
    current_byte_offset = other.current_byte_offset;
    other.file_handle = nullptr;
    return *this;
}

auto FileReader::read_into(void * destination, u64 byte_count) -> FileIoResult
{
    if (file_handle == nullptr) { return FileIoResult::IO_FAILED; }
    // Phrased as a subtraction so a byte_count near the u64 ceiling cannot wrap the bound it is checked against.
    if (byte_count > total_byte_size - current_byte_offset) { return FileIoResult::OUT_OF_BOUNDS; }

    // ReadFile counts bytes in a DWORD, so anything past 4 GiB has to be issued as several reads.
    auto * destination_bytes = s_cast<std::byte *>(destination);
    u64 remaining_byte_count = byte_count;
    while (remaining_byte_count > 0)
    {
        DWORD const request_byte_count = s_cast<DWORD>(std::min(remaining_byte_count, s_cast<u64>(std::numeric_limits<DWORD>::max())));
        DWORD bytes_read = 0;
        BOOL const ok = ReadFile(file_handle, destination_bytes, request_byte_count, &bytes_read, nullptr);
        if (!ok || bytes_read != request_byte_count) { return FileIoResult::IO_FAILED; }
        destination_bytes += bytes_read;
        remaining_byte_count -= bytes_read;
        current_byte_offset += bytes_read;
    }
    return FileIoResult::SUCCESS;
}

auto FileReader::seek(u64 byte_offset) -> FileIoResult
{
    if (file_handle == nullptr) { return FileIoResult::IO_FAILED; }
    if (byte_offset > total_byte_size) { return FileIoResult::OUT_OF_BOUNDS; }

    LARGE_INTEGER seek_target = {};
    seek_target.QuadPart = s_cast<LONGLONG>(byte_offset);
    if (!SetFilePointerEx(file_handle, seek_target, nullptr, FILE_BEGIN)) { return FileIoResult::IO_FAILED; }
    current_byte_offset = byte_offset;
    return FileIoResult::SUCCESS;
}

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