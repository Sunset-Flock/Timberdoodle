#pragma once
#include <filesystem>
#include <vector>

#include "../camera.hpp"

#include "../timberdoodle.hpp"
using namespace tido::types;

auto load_camera_animation(std::filesystem::path const & path) -> std::vector<CameraAnimationKeyframe>;
void export_camera_animation(std::filesystem::path const & path, std::vector<CameraAnimationKeyframe> const & keyframes);
