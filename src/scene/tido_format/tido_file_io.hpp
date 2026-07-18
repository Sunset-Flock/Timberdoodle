#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <vector>

#include "../../timberdoodle.hpp"
using namespace tido::types;

// Onverwrites file, fully replacing its contents.
// Retries while the file is held open (by a concurrent writer, or a shared-read handle) until the write succeeds.
auto tido_write_file_exclusive(std::filesystem::path const & path, void const * data, usize size) -> bool;

// Reads the whole file through a single-attempt shared-read open.
// nullopt if it does not exist or someone else currently holds it open for writing.
auto tido_read_file_shared(std::filesystem::path const & path) -> std::optional<std::vector<std::byte>>;
