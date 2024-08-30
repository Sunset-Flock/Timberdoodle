#pragma once
#include <filesystem>

#include "camera.hpp"
#include "shader_shared/shared.inl"

#include "timberdoodle.hpp"
using namespace tido::types;

auto load_camera_animation(std::filesystem::path const & path) -> std::vector<CameraAnimationKeyframe>;
auto load_sky_settings(std::filesystem::path const & path) -> SkySettings;