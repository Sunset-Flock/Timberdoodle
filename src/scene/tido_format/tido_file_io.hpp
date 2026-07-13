#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <vector>

#include "../../timberdoodle.hpp"
using namespace tido::types;

/// --- .tido_bin exclusive-write / shared-read I/O (Windows) ---
/// A cook worker writes a .tido_bin on a worker thread while the Streamer may read that same file from
/// another thread at the same time; plain std::ofstream/ifstream give no share-mode control over that
/// race. Both sides pick complementary Win32 CreateFileW share modes instead of a mutex: a writer opens
/// the file deny-all and retries on ERROR_SHARING_VIOLATION until no reader or writer holds it; a reader
/// opens FILE_SHARE_READ (so concurrent reads are unaffected by each other) and never retries - losing
/// the race to a writer is an expected, not an error, outcome for that one read.

// Writes `data` (`size` bytes) to `path`, replacing its contents. Retries while the file is held open (by
// a concurrent writer, or a Streamer's shared-read handle) until the write succeeds; returns false only on
// a genuine I/O failure (bad path, permissions, disk full) - callers must not treat contention as failure.
auto tido_write_file_exclusive(std::filesystem::path const & path, void const * data, usize size) -> bool;

// Reads the whole file at `path` through a single-attempt shared-read open; nullopt if it does not exist
// or a cook worker currently holds it open for writing. Callers must NOT retry - a miss here is an
// expected, occasional outcome of the write side's exclusive hold, not a real error.
auto tido_read_file_shared(std::filesystem::path const & path) -> std::optional<std::vector<std::byte>>;
