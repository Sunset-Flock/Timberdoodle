#pragma once

#include <vector>
#include <array>
#include <string>
#include <variant>
#include <cstddef>

#include <daxa/daxa.hpp>

#include "../../timberdoodle.hpp"
using namespace tido::types;

/// --- Image Optimizer ---
/// Generic image cook (part 2 of texture loading). Takes the raw image data produced by an importer
/// (decoded pixels or a basis-compressed KTX2 container) and produces the GPU-ready, compressed CPU
/// memory. It does NOT create a daxa image - it returns the optimized memory, which the caller (the
/// streamer / the function adding the texture to the manifest) uploads to make resident. Mirrors the
/// geometry optimizer: raw CPU in -> cooked CPU out, no GPU work, no source-format knowledge.

// How an image is used; determines the block-compression format the optimizer targets.
enum struct TextureMaterialType
{
    NONE,
    DIFFUSE,
    DIFFUSE_OPACITY,
    OPACITY,
    NORMAL,
    ROUGHNESS_METALNESS,
};

// The on-disk/container format of the raw bytes the optimizer is handed.
enum struct ImageFileFormat
{
    PNG,
    KTX2,
    // RAW_PIXELS (already-decoded) to be added together with the non-manifest texture path.
};

// --- Optimizer input ---
// The raw source bytes (as read by an importer) plus all the metadata needed to process them: the
// source format and the target usage (which selects the block-compression format). The optimizer does
// all decoding/transcoding/compression itself; the importer only reads bytes and tags their format.
struct OptimizeImageInfo
{
    std::vector<std::byte> data = {}; // PNG file bytes or KTX2 container bytes
    ImageFileFormat format = {};
    TextureMaterialType type = {};
    std::string name = {};
};

// --- Optimizer output: GPU-ready compressed memory + the info needed to create/upload the image ---
struct CookedImageData
{
    std::vector<std::byte> src_data = {};
    daxa::ImageInfo image_info = {};
    u32 mips_to_copy = {};
    std::array<u64, 16> mip_copy_offsets = {};
    bool compressed_bc5_rg = {};
};

enum struct ImageOptimizeError
{
    FAILED_TO_PROCESS_KTX,
    FAILED_TO_DECODE_PNG,
};

// Turn raw source bytes into GPU-ready compressed memory. Decodes/transcodes/compresses per the
// source format: PNG -> decode (-> BC compression, TODO); KTX2 -> basis transcode to BCn.
auto optimize_image(OptimizeImageInfo const & info) -> std::variant<ImageOptimizeError, CookedImageData>;
