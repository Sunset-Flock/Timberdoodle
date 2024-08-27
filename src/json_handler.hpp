#pragma once

#include "camera.hpp"
#include "shared.inl"

#include "timberdoodle.hpp"
using namespace tido::types;

auto load_camera_animation(std::filesystem::path const & path) -> std::vector<CameraAnimationKeyframe>;
auto load_sky_settings(std::filelsytem::path const & path) -> SkySettings;