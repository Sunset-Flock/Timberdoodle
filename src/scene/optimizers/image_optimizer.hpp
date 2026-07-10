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
/// Generic image cook (part 2 of texture loading): takes raw image data from an importer (encoded PNG/KTX2
/// bytes) and produces GPU-ready, block-compressed CPU memory (a ProcessedImage) with a full mip chain.
/// CPU-only and format-agnostic.

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

// --- Optimizer output: GPU-ready compressed CPU memory + the info needed to write/upload the image ---
// The image analog of ProcessedMesh: the cooked, in-memory result, before it is written to a .tido.
struct ProcessedImage
{
    std::vector<std::byte> src_data = {};
    daxa::ImageInfo image_info = {};
    u32 mips_to_copy = {};
    std::array<u64, 16> mip_copy_offsets = {};
};

enum struct ImageOptimizeError
{
    FAILED_TO_PROCESS_KTX,
    FAILED_TO_DECODE_PNG,
};

// Turn raw source bytes into GPU-ready cooked CPU memory: PNG is decoded, mipped and BC-compressed; KTX2
// is basis-transcoded to BCn (mips already in the container).
auto process_image(OptimizeImageInfo const & info) -> std::variant<ImageOptimizeError, ProcessedImage>;
