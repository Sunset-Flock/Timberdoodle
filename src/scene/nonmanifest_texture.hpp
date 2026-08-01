#pragma once

#include <filesystem>
#include <optional>

#include <daxa/daxa.hpp>

#include "../timberdoodle.hpp"

using namespace tido::types;

/// --- Non-manifest textures ---
/// Engine-owned PNGs that never enter the scene: the renderer's blue-noise arrays and the UI icons. They live
/// under `deps/`, outside the assets root, so `SceneRuntime::request_import` rejects them by design - they are
/// not scene assets, carry no recipe and get no manifest entry. This loads them straight to a GPU image
/// through the same `image_parse` the cook uses, and uploads synchronously because both callers need the
/// image before the first frame.

struct LoadNonManifestTextureInfo
{
    std::filesystem::path filepath = {};
    // Layers > 1 reads `filepath` with its trailing digit replaced by the layer index, into one 2D array.
    u32 layers = 1;
    // PNG carries no authority on this - the caller knows whether the file holds colour or raw data.
    bool load_as_srgb = true;
};

auto load_nonmanifest_texture(daxa::Device & device, LoadNonManifestTextureInfo const & info) -> std::optional<daxa::ImageId>;
