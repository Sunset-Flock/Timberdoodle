#pragma once
#include <filesystem>

#include "timberdoodle.hpp"
using namespace tido::types;

#include "gpu_context.hpp"

daxa::ImageId load_vdb(std::filesystem::path const & path, daxa::Device & device, bool normalize_sdf, bool remap_channels);