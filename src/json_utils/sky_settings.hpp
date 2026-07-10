#pragma once
#include <filesystem>

#include "../shader_shared/shared.inl"

#include "../timberdoodle.hpp"
using namespace tido::types;

auto load_sky_settings(std::filesystem::path const & path) -> SkySettings;
