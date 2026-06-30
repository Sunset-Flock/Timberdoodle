#pragma once

#include <filesystem>
#include <cstddef>
#include <vector>
#include <optional>

#include <daxa/daxa.hpp>

#include "../../timberdoodle.hpp"
using namespace tido::types;

// Defined in optimizers/image_optimizer.hpp. Forward-declared here (instead of included) so that
// image_optimizer.hpp can include this header to use TidoTextureCookResult as optimize_image's return
// type without an include cycle. tido_texture.cpp includes image_optimizer.hpp for the full type.
struct CookedImageData;

/// --- .tido texture format (writer) ---
/// The cooked, runtime-ready on-disk form of a texture. See "Texture .tido format.md" in the design
/// notes. A cooked texture is two files sharing one stem:
///   <stem>.tido        - raw texel/block data only, no header. Subresources are laid out mip-major,
///                        coarse-first (all array layers of one mip are contiguous so a single
///                        VkCopyBufferToImage uploads a whole mip level).
///   <stem>.tido_cache  - metadata sidecar: header + descriptor + the subresource offset table that
///                        tells a reader where each subresource lives inside the .tido. (Writing the
///                        cache manifest is a later step; for now the optimizer writes the .tido data
///                        file and returns this metadata so it can be persisted then.)

inline constexpr u32 TIDO_CACHE_VERSION = 1;

// Which kind of streamable blob a .tido_cache describes. Lets a reader verify it opened the right
// file. Extended later (collider, cpu_mesh, audio, ...).
enum struct StreamableBlobType : u8
{
    TEXTURE = 0,
    MESH = 1,
};

// First struct in a .tido_cache. version stays the first field so header changes never break the
// version check (mirrors TidoVolumetricCloudDataHeader's convention).
struct TidoCacheHeader
{
    std::array<char, 4> magic = {'T', 'I', 'D', 'C'};
    u32 version = TIDO_CACHE_VERSION;
    u8 blob_type = s_cast<u8>(StreamableBlobType::TEXTURE);
    std::array<u8, 3> _pad = {};
};

// Fixed texture description following the header in a .tido_cache.
struct TidoTextureDescriptor
{
    u32 format = {};       // daxa::Format == VkFormat (see static_assert below)
    u32 width = {};        // texel width of mip 0
    u32 height = {};       // texel height of mip 0
    u32 depth = {};        // 1 for 2D/array; voxel depth for 3D
    u32 array_layers = {}; // 1 for non-array, 6 for cubemap, N for arrays
    u32 mip_count = {};    // total mip levels including the full-resolution mip 0
};

// One entry per subresource. The flat table is in the same order the subresources are stored in the
// .tido file (mip-major, coarse-first), so the entry for a given (mip, layer) is at
// ((mip_count - 1 - mip) * array_layers + layer), where mip uses the GPU/Vulkan convention
// (mip 0 = finest). The offset is the authoritative location in the .tido file.
struct TidoSubresourceEntry
{
    u64 offset = {};    // byte offset from the start of the .tido data file
    u32 byte_size = {}; // byte size of this subresource
};

// The format field is stored as the raw daxa::Format/VkFormat integer; both are 32-bit.
static_assert(sizeof(daxa::Format) == sizeof(u32), "TidoTextureDescriptor::format assumes a 32-bit daxa::Format");

// True if a descriptor's stored format is the two-channel BC5 normal-map encoding (the shader
// reconstructs Z). Deduced from the cooked format rather than tracked separately through the import.
inline auto tido_format_is_bc5_rg(u32 format) -> bool
{
    daxa::Format const f = std::bit_cast<daxa::Format>(format);
    return f == daxa::Format::BC5_UNORM_BLOCK || f == daxa::Format::BC5_SNORM_BLOCK;
}

// The cooked metadata produced alongside the .tido data file. Persisted into the .tido_cache later.
struct TidoTextureCookResult
{
    TidoTextureDescriptor descriptor = {};
    std::vector<TidoSubresourceEntry> subresources = {}; // size == array_layers * mip_count
    std::filesystem::path tido_path = {};                // the written .tido data file
};

// Default directory the optimizer writes cooked .tido artifacts into (relative to the working dir,
// matching how the other asset paths are resolved).
inline std::filesystem::path const TIDO_ASSET_CACHE_DIR = "tido_asset_cache";

// Writes <cache_dir>/<name>.tido (raw data, mip-major coarse-first) from already-cooked image memory
// and returns its descriptor + subresource offset table. Does NOT write the .tido_cache manifest yet
// (that is aggregated per imported file in a later step). Returns std::nullopt on an IO failure.
auto write_texture_tido(CookedImageData const & cooked, std::filesystem::path const & cache_dir, std::string const & name) -> std::optional<TidoTextureCookResult>;
