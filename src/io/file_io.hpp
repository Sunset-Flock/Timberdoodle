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
auto read_file(std::filesystem::path const & path) -> std::pair<FileIoResult, std::vector<std::byte>>;
auto write_file(std::filesystem::path const & path, void const * data, usize size) -> FileIoResult;
auto write_file_byte_range(std::filesystem::path const & path, u64 byte_offset, void const * data, usize size) -> FileIoResult;

auto read_file_modified_time(std::filesystem::path const & path) -> std::optional<i64>;
