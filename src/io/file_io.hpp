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

// No slice reads the whole file; a slice past EOF fails rather than reading short.
auto read_file(std::filesystem::path const & path, std::optional<ByteSlice> slice = {}) -> std::pair<FileIoResult, std::vector<std::byte>>;
auto write_file(std::filesystem::path const & path, void const * data, usize size) -> FileIoResult;
// Moves from onto to, replacing an existing file there. A reader sees either the old file or the new one.
auto rename_file(std::filesystem::path const & from, std::filesystem::path const & to) -> FileIoResult;

auto read_file_modified_time(std::filesystem::path const & path) -> std::optional<i64>;
