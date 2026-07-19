#pragma once

#include <cstddef>
#include <filesystem>
#include <variant>
#include <vector>

#include "../timberdoodle.hpp"
using namespace tido::types;

struct FileByteRange
{
    std::filesystem::path file = {};
    u64 byte_offset = {};
    u64 byte_length = {};
};

enum struct FileIoResult
{
    SUCCESS,
    NOT_FOUND,
    LOCKED,
    OUT_OF_BOUNDS, 
    IO_FAILED,
};

auto read_file_byte_range(FileByteRange const & range) -> std::pair<FileIoResult, std::vector<std::byte>>;
auto read_file_shared(std::filesystem::path const & path) -> std::pair<FileIoResult, std::vector<std::byte>>;
auto write_file_exclusive(std::filesystem::path const & path, void const * data, usize size) -> FileIoResult;
