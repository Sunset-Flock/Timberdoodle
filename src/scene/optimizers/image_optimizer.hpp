#pragma once

#include <vector>
#include <array>
#include <string>
#include <variant>
#include <cstddef>

#include <daxa/daxa.hpp>

#include "../../timberdoodle.hpp"
#include "../importers/openvdb_importer.hpp" // VDBGridInfo
#include "tex_compression.hpp"               // Compression
using namespace tido::types;

struct ThreadPool;

/// --- Image Optimizer ---
/// Generic image cook (part 2 of texture loading): takes raw image data from an importer (encoded PNG/KTX2
/// bytes) and produces GPU-ready, block-compressed CPU memory (a ProcessedImage) with a full mip chain.
/// CPU-only and format-agnostic.

// How an image is used; determines the block-compression format the optimizer targets.
enum struct TextureMaterialType
{
    NONE,
    DIFFUSE,
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
// The image analog of ProcessedMesh: the cooked, in-memory result, before it is written to a .tido_bin.
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
    // An OPACITY cook was requested for a source that carries no alpha channel (source color_type/tRNS
    // for PNG, basis component count for KTX2 - not the material's alphaMode).
    SOURCE_HAS_NO_ALPHA,
};

// Turn raw source bytes into GPU-ready cooked CPU memory: PNG is decoded, mipped and BC-compressed; KTX2
// is basis-transcoded to BCn (mips already in the container). One call cooks exactly one artifact for the
// requested type: an OPACITY request compresses the source's alpha channel alone (BC4), a DIFFUSE request
// the color channels (BC7, alpha forced opaque when the source had one - the alpha's source of truth is
// its own OPACITY artifact).
auto process_image(OptimizeImageInfo const & info) -> std::variant<ImageOptimizeError, ProcessedImage>;

// --- Optimizer input: one decoded VDB volume + its cook recipe ---
// grids_data/grid_extents are LoadVDBTask's raw output (grids_data[i] holds grids[i]'s decoded samples,
// fp16 or fp32 per its own convert_to_fp16); grids/target are the recipe: channel order and the
// compression to cook to (BC6, BC1_SDF, or UNDEFINED for uncompressed RGBA16F).
struct OptimizeVolumeInfo
{
    std::vector<std::vector<std::byte>> grids_data = {};
    i32vec3 grid_extents = {};
    std::vector<VDBGridInfo> grids = {};
    Compression target = {};
    std::string name = {};
};

// Cook decoded VDB grids into GPU-ready volume memory: interleaves grids into channels per the recipe (BC6
// needs 3 fp16 grids, uncompressed needs 4; BC1_SDF compresses its single fp32 grid directly after
// remapping it from value_range into [0,1], the range the BC1 scalar compressor requires), then
// BC6/BC1_SDF-compresses via compress_image or leaves the interleave uncompressed. Compression is
// dispatched across threadpool rather than run inline (unlike process_image's per-mip compression) since a
// volume cook is one big task rather than one-per-image.
auto process_volume(OptimizeVolumeInfo const & info, ThreadPool * threadpool) -> ProcessedImage;
