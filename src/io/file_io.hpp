#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <variant>
#include <vector>

#include "../timberdoodle.hpp"
using namespace tido::types;

struct ByteSlice
{
    u64 byte_offset = {};
    u64 byte_length = {};
};

// Where a source asset's bytes live. A whole-file source carries no length at all rather than a resolved one,
// so a stored location can never go stale against a file that grew or shrank.
struct SourceLocation
{
    std::filesystem::path file = {};
    std::optional<ByteSlice> slice = {};   // nullopt reads the whole file
};

enum struct FileIoResult
{
    SUCCESS,
    NOT_FOUND,
    LOCKED,
    OUT_OF_BOUNDS,
    IO_FAILED,
};

// Keeps a handle open across reads, so a consumer that only touches part of a file never materializes the
// rest. Use read_file when the whole slice is wanted at once - this exists for consumers that pull their own
// bytes incrementally and would otherwise force a full copy just to be fed.
struct FileReader
{
    static auto open(std::filesystem::path const & path) -> std::pair<FileIoResult, FileReader>;

    FileReader() = default;
    ~FileReader();
    FileReader(FileReader const &) = delete;
    auto operator=(FileReader const &) -> FileReader & = delete;
    FileReader(FileReader && other) noexcept;
    auto operator=(FileReader && other) noexcept -> FileReader &;

    // Reads past EOF fails rather than reading short, leaving destination untouched.
    auto read_into(void * destination, u64 byte_count) -> FileIoResult;
    auto seek(u64 byte_offset) -> FileIoResult;

    auto file_byte_size() const -> u64 { return total_byte_size; }
    auto read_byte_offset() const -> u64 { return current_byte_offset; }

private:
    void * file_handle = nullptr; // Win32 HANDLE, kept untyped so <Windows.h> stays out of this header
    u64 total_byte_size = {};
    u64 current_byte_offset = {};
};

// No slice reads the whole file; a slice past EOF fails rather than reading short.
auto read_file(std::filesystem::path const & path, std::optional<ByteSlice> slice = {}) -> std::pair<FileIoResult, std::vector<std::byte>>;
auto write_file(std::filesystem::path const & path, void const * data, usize size) -> FileIoResult;
// Moves from onto to, replacing an existing file there. A reader sees either the old file or the new one.
auto rename_file(std::filesystem::path const & from, std::filesystem::path const & to) -> FileIoResult;

auto read_file_modified_time(std::filesystem::path const & path) -> std::optional<i64>;
