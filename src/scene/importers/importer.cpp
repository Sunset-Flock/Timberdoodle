#include "importer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <ranges>
#include <span>

#include <fmt/format.h>

#include "../optimizers/image_processor.hpp"
#include "../optimizers/geometry_optimizer.hpp"
#include "../optimizers/vdb_processor.hpp"
#include "../tido_format/tido_format.hpp"
#include "../tido_format/tido_util.hpp"
#include "../../io/file_io.hpp"
#include "../../json_utils/tido_format.hpp"

// Bumped whenever the image cook's output or .tido_bin layout changes; a cached artifact whose stored
// version differs is treated as stale on re-import. Stamped into every cooked image's metadata header.
static constexpr u32 IMAGE_COOK_VERSION = 1;

// Bumped whenever the mesh cook's output or .tido_bin layout changes; a cached artifact whose stored
// version differs is treated as stale on re-import. Stamped into every cooked mesh's metadata header.
static constexpr u32 MESH_COOK_VERSION = 1;

// Bumped whenever the VDB cook's output or .tido_bin layout changes; a cached artifact whose stored version
// differs is treated as stale on re-import. Stamped into every cooked VDB image's metadata header.
static constexpr u32 VDB_COOK_VERSION = 1;

auto image_identity_key(ImageImporterData const & importer_data) -> u64
{
    std::vector<std::byte> importer_data_as_bytes;
    // Location: the resolved source byte range (path + offset + length); cache_path is routing-only and excluded.
    std::string const path_string = importer_data.source_bytes.file.generic_string();
    tido_append_bytes(importer_data_as_bytes, path_string.data(), path_string.size());
    tido_append_pod(importer_data_as_bytes, importer_data.source_bytes.byte_offset);
    tido_append_pod(importer_data_as_bytes, importer_data.source_bytes.byte_length);

    // Recipe: container format joins the channel mapping and target format so one source blob used under
    // different recipes gets distinct keys.
    tido_append_pod(importer_data_as_bytes, importer_data.container_format);
    for (auto const & mapped_channel : importer_data.channel_mapping)
    {
        importer_data_as_bytes.push_back(static_cast<std::byte>(mapped_channel));
    }
    tido_append_pod(importer_data_as_bytes, importer_data.target_format);

    return tido_fnv1a(importer_data_as_bytes, 0);
}

auto mesh_identity_key(MeshImporterData const & importer_data) -> u64
{
    std::vector<std::byte> importer_data_as_bytes;
    // Location only: each attribute stream's resolved byte range (path + offset + length) and its component
    // type. cache_path is routing-only and excluded; meshes carry no recipe. An absent uvs stream simply
    // contributes nothing, so a mesh with uvs keys differently from one without.
    auto fold_attrib_source = [&](MeshAttribSource const & attrib_source)
    {
        std::string const path_string = attrib_source.range.file.generic_string();
        tido_append_bytes(importer_data_as_bytes, path_string.data(), path_string.size());
        tido_append_pod(importer_data_as_bytes, attrib_source.range.byte_offset);
        tido_append_pod(importer_data_as_bytes, attrib_source.range.byte_length);
        tido_append_pod(importer_data_as_bytes, attrib_source.component_type);
    };
    fold_attrib_source(importer_data.indices);
    fold_attrib_source(importer_data.positions);
    fold_attrib_source(importer_data.normals);
    if (importer_data.uvs.has_value()) { fold_attrib_source(importer_data.uvs.value()); }

    return tido_fnv1a(importer_data_as_bytes, 0);
}

auto vdb_identity_key(VdbImporterData const & importer_data) -> u64
{
    std::vector<std::byte> importer_data_as_bytes;
    // Location: whole-file source range (path + offset + length); cache_path is routing-only and excluded.
    std::string const path_string = importer_data.source_bytes.file.generic_string();
    tido_append_bytes(importer_data_as_bytes, path_string.data(), path_string.size());
    tido_append_pod(importer_data_as_bytes, importer_data.source_bytes.byte_offset);
    tido_append_pod(importer_data_as_bytes, importer_data.source_bytes.byte_length);

    for (auto const & grid : importer_data.grid_names)
    {
        tido_append_bytes(importer_data_as_bytes, grid.data(), grid.size());
    }
    for (auto const & mapped_channel : importer_data.channel_mapping)
    {
        importer_data_as_bytes.push_back(static_cast<std::byte>(mapped_channel));
    }
    tido_append_pod(importer_data_as_bytes, importer_data.target_format);

    return tido_fnv1a(importer_data_as_bytes, 0);
}

namespace
{

// The per-source cache directory an artifact is written into: mirrors the owning source's location under
// the tido_asset_cache tree. cache_path selects the dir; the identity-keyed stem selects the file.
auto asset_cache_dir(std::filesystem::path const & source_path) -> std::filesystem::path
{
    auto const relative = tido_relative_to_assets_root(source_path);
    DBG_ASSERT_TRUE_M(relative.has_value(), "asset_cache_dir: source path must be under the Tido assets root");
    return TIDO_ASSET_CACHE_DIR / (relative.has_value() ? relative->parent_path() : std::filesystem::path{});
}

// The JSON header region (metadata + descriptor, no payload) and payload offset of a .tido_bin, read
// without loading the payload. nullopt if the file is missing, too short, or its preamble is invalid - any
// of which routes the caller to a full cook. The two byte-range reads keep the artifact's bulk off the disk
// for a cache probe.
struct CachedArtifactHeader
{
    std::vector<std::byte> header_region = {};
    u64 file_data_offset = {};
};
auto read_cached_artifact_header(std::filesystem::path const & tido_path) -> std::optional<CachedArtifactHeader>
{
    // Read the preamble alone to learn how many header bytes follow it - keeps the payload off disk.
    auto [preamble_result, preamble_bytes] = read_file_byte_range({.file = tido_path, .byte_offset = 0, .byte_length = TIDO_FILE_PREAMBLE_SIZE});
    if (preamble_result != FileIoResult::SUCCESS || preamble_bytes.size() < sizeof(TidoFilePreamble)) { return std::nullopt; }
    TidoFilePreamble preamble = {};
    std::memcpy(&preamble, preamble_bytes.data(), sizeof(TidoFilePreamble));

    // Read the preamble + header region (still no payload), then validate and slice out the header region.
    u64 const header_end = TIDO_FILE_PREAMBLE_SIZE + preamble.header_byte_length;
    auto [header_result, header_bytes] = read_file_byte_range({.file = tido_path, .byte_offset = 0, .byte_length = header_end});
    if (header_result != FileIoResult::SUCCESS) { return std::nullopt; }

    std::optional<std::span<std::byte const>> const region = tido_header_region(header_bytes);
    if (!region.has_value()) { return std::nullopt; }

    return CachedArtifactHeader{
        .header_region = std::vector<std::byte>(region->begin(), region->end()),
        .file_data_offset = header_end,
    };
}

// Run every chunk of a one-shot task synchronously on the calling thread. The cook already runs on a
// ThreadPool worker; dispatching back into the pool and blocking on it could starve it, so the downsample /
// remap / compress passes execute inline instead.
void run_task_inline(Task & task)
{
    for (u32 chunk_index = 0; chunk_index < task.chunk_count; ++chunk_index)
    {
        task.callback(chunk_index, EXTERNAL_THREAD_INDEX);
    }
}

// Drives one asset cook end to end: derive the identity key + cache path, try the mtime and content cache
// tiers, and on a miss read the sources, cook, and write the .tido_bin. Everything type-specific (which
// sources to read, how to cook, which descriptor/streamer types) lives in the Policy; this skeleton is the
// same for images and meshes.
template<typename Policy>
struct CookTask final : Task
{
    using ImporterData = typename Policy::ImporterData;
    using Descriptor = typename Policy::Descriptor;
    using StreamerData = typename Policy::StreamerData;

    Importer * importer = {};
    ImporterData importer_data = {};
    u32 manifest_index = {};

    CookTask(Importer * importer, ImporterData importer_data, u32 manifest_index)
        : importer{importer}, importer_data{std::move(importer_data)}, manifest_index{manifest_index}
    {
        chunk_count = 1;
    }

    void cook()
    {
        u64 const identity_key = Policy::identity_key(importer_data);
        std::string const artifact_name = Policy::artifact_source_name(importer_data);
        std::filesystem::path const cache_dir = asset_cache_dir(importer_data.cache_path);
        std::filesystem::path const tido_path = tido_artifact_path(cache_dir, artifact_name, identity_key);
        i64 const current_mtime = Policy::current_mtime(importer_data);

        // Fast path: reuse an existing artifact whose stored metadata still matches, skipping the recook.
        std::optional<CachedArtifactHeader> const cached = read_cached_artifact_header(tido_path);
        std::optional<std::pair<TidoMetadataHash, Descriptor>> cached_header = {};
        if (cached.has_value()) { cached_header = Policy::read_header(cached->header_region); }
        bool const cache_usable = cached_header.has_value()
            && cached_header->first.version == Policy::COOK_VERSION
            && cached_header->first.cache_key == identity_key;

        auto push_streamer_result = [&](StreamerData streamer_data)
        {
            importer->push_result(ImporterTaskResult{.data = ImporterTaskResult::CookedAsset{
                .streamer_data = std::move(streamer_data),
                .manifest_index = manifest_index,
            }});
        };

        // Both cache tiers reuse the on-disk artifact untouched.
        auto serve_cached = [&]()
        {
            DEBUG_MSG(fmt::format("[{}] cache hit '{}' -> '{}'", Policy::LOG_TAG, artifact_name, tido_path.string()));
            push_streamer_result(StreamerData{
                .descriptor = cached_header->second,
                .bin_source = tido_path,
                .file_data_offset = cached->file_data_offset,
            });
        };

        // Miss path: cook the freshly-read sources and write a new artifact, stamping the cache identity.
        auto cook_and_write = [&](typename Policy::ReadState const & read_state, u64 content_hash)
        {
            std::vector<std::byte> payload = {};
            std::optional<Descriptor> const descriptor = Policy::do_cook(importer_data, read_state, payload);
            if (!descriptor.has_value())
            {
                DEBUG_MSG(fmt::format("[ERROR][{}] failed to cook '{}'", Policy::LOG_TAG, artifact_name));
                return;
            }

            WriteTidoFileInfo const write_info = {
                .destination_folder = cache_dir,
                .name = artifact_name,
                .metadata_hash = TidoMetadataHash{
                    .cache_key = identity_key,
                    .source_mtime_at_bake = current_mtime,
                    .content_hash = content_hash,
                    .version = Policy::COOK_VERSION,
                },
                .data = payload,
            };
            std::optional<StreamerData> const streamer_data = Policy::write_artifact(write_info, descriptor.value());
            if (!streamer_data.has_value())
            {
                DEBUG_MSG(fmt::format("[WARN][{}] failed to write .tido_bin for '{}'", Policy::LOG_TAG, artifact_name));
                return;
            }

            DEBUG_MSG(fmt::format("[{}] cooked '{}' ({}) -> '{}'", Policy::LOG_TAG, artifact_name,
                Policy::cooked_detail(descriptor.value()), streamer_data->bin_source.string()));
            push_streamer_result(streamer_data.value());
        };

        // Tier 1 (mtime): sources untouched since the bake - reuse without reading them at all.
        if (cache_usable && cached_header->first.source_mtime_at_bake == current_mtime)
        {
            serve_cached();
            return;
        }

        // Past the mtime tier every remaining path reads the sources - the content tier hashes them to confirm
        // the artifact is still valid, and a fresh cook consumes them. The hash is over the loaded bytes
        // (before parse); the read state owns the buffers do_cook interprets.
        typename Policy::ReadState read_state = {};
        std::optional<u64> const content_hash = Policy::read_and_hash(importer_data, read_state);
        if (!content_hash.has_value())
        {
            DEBUG_MSG(fmt::format("[ERROR][{}] failed to read source bytes for '{}'", Policy::LOG_TAG, artifact_name));
            return;
        }

        // Tier 2 (content): the mtime moved but the bytes are unchanged, so the artifact is still valid. Patch
        // the stale bake mtime in place so later imports hit the cheap mtime tier. The mtime is fixed-width in
        // the header, so the patch can't move the payload offset; a failed patch just leaves it for next time.
        if (cache_usable && cached_header->first.content_hash == content_hash.value())
        {
            TidoMetadataHash refreshed_hash = cached_header->first;
            refreshed_hash.source_mtime_at_bake = current_mtime;
            if (!try_patch_tido_metadata_hash(tido_path, cached_header->first, refreshed_hash))
            {
                DEBUG_MSG(fmt::format("[WARN][{}] failed to refresh bake mtime for '{}'", Policy::LOG_TAG, artifact_name));
            }
            serve_cached();
            return;
        }

        // Tier 3 (miss): no usable artifact - cook and write one.
        cook_and_write(read_state, content_hash.value());
    }

    void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        cook();
        importer->notify();
    }
};

// Cook an already-decoded uncompressed source (2D or 3D) into the recipe's target format: box-filter the mip
// chain, then produce the target per level - block-compress, channel-remap, or pass through when the source
// already matches. channel_mapping selects/orders source channels into the target (dst d <- src mapping[d]);
// mip_count == 1 skips mip generation. Returns the cooked image ready for write_tido_image.
auto cook_uncompressed_source(TidoImageWithData decoded, std::span<u8 const> channel_mapping, daxa::Format target_format, u32 mip_count) -> std::optional<TidoImageWithData>
{
    u32 const width = decoded.descriptor.info.size.x;
    u32 const height = decoded.descriptor.info.size.y;
    u32 const depth = decoded.descriptor.info.size.z;
    u32 const dimensions = decoded.descriptor.info.dimensions;

    FormatInfo const target_block = get_info_from_format(target_format);

    FormatInfo decoded_info = get_info_from_format(decoded.descriptor.info.format);
    if (decoded_info.is_srgb != target_block.is_srgb)
    {
        // Vulkan names sRGB only for 8-bit UNORM, so the tag cannot be carried on a wider source.
        if (target_block.is_srgb && (decoded_info.channel_byte_size != 1 || decoded_info.numeric_type != FormatNumericType::UNORM))
        {
            DEBUG_MSG(fmt::format("[ERROR][cook_uncompressed_source] an sRGB target needs an 8-bit UNORM source, got {} byte channels",
                decoded_info.channel_byte_size));
            return std::nullopt;
        }
        decoded_info.is_srgb = target_block.is_srgb;
        decoded.descriptor.info.format = get_format_from_info(decoded_info);
    }

    image_resize_for_mipmaps(decoded, mip_count);

    // The remap destination is sized from channel_mapping, so a mapping that does not fit the target would
    // index source channels that are not there. Recipes are producer data, so this stays a runtime check.
    bool const is_block_target = target_block.block_width > 1;
    bool const channel_count_fits = is_block_target
        ? (!channel_mapping.empty() && channel_mapping.size() <= target_block.channel_count)
        : (channel_mapping.size() == target_block.channel_count);
    if (!channel_count_fits)
    {
        DEBUG_MSG(fmt::format("[ERROR][cook_uncompressed_source] channel mapping of {} channels does not fit a {} channel target",
            channel_mapping.size(), target_block.channel_count));
        return std::nullopt;
    }

    // The mapping indexes the decoded source's channels; an index past them would read out of bounds in the
    // remap. Recipes are producer data, so this stays a runtime check.
    for (u8 const source_channel : channel_mapping)
    {
        if (source_channel >= decoded_info.channel_count)
        {
            DEBUG_MSG(fmt::format("[ERROR][cook_uncompressed_source] channel mapping indexes channel {} of a {} channel source",
                source_channel, decoded_info.channel_count));
            return std::nullopt;
        }
    }

    bool const channel_mapping_is_identity = std::ranges::equal(channel_mapping, std::views::iota(0u, target_block.channel_count));

    enum struct MipEncoding
    {
        PASS_THROUGH,     // decoded mip already in the target layout - copy it verbatim
        CHANNEL_REMAP,    // uncompressed target - remap the recipe's channels straight into it
        BLOCK_COMPRESS,   // BC target - remap into the uncompressed BC source layout, then compress
    };

    MipEncoding const mip_encoding = [&]()
    {
        if (target_block.block_width > 1) { return MipEncoding::BLOCK_COMPRESS; }
        if (decoded.descriptor.info.format == target_format && channel_mapping_is_identity) { return MipEncoding::PASS_THROUGH; }
        return MipEncoding::CHANNEL_REMAP;
    }();

    // Lay out the cooked mip chain in the target format: image_resize_for_mipmaps fills the (block-aware)
    // subresource offsets/sizes and allocates the payload. Start from an empty (0-mip) image so it builds the
    // whole chain instead of appending to existing levels.
    TidoImageWithData cooked = {};
    cooked.descriptor.info = {
        .format = target_format,
        .dimensions = dimensions,
        .size = {width, height, depth},
        .mip_level_count = 0,
        .array_layer_count = 1,
    };
    image_resize_for_mipmaps(cooked, mip_count);

    // Encode one filtered mip into the target format: pass it through when the source already matches,
    // otherwise remap the recipe's channels - straight into an uncompressed target, or into the BC source
    // layout which is then compressed into the block target.
    auto encode_mip = [&](std::span<std::byte const> src_pixels, std::span<std::byte> dst_pixels, u32vec3 mip_dimensions)
    {
        u64 const texel_count = s_cast<u64>(mip_dimensions.x) * mip_dimensions.y * mip_dimensions.z;
        switch (mip_encoding)
        {
            case MipEncoding::PASS_THROUGH:
            {
                std::copy(src_pixels.begin(), src_pixels.end(), dst_pixels.begin());
                return;
            }
            case MipEncoding::CHANNEL_REMAP:
            {
                run_task_inline(*remap_channels(RemapChannelsInfo{
                    .src_data = src_pixels,
                    .texel_count = texel_count,
                    .format = decoded.descriptor.info.format,
                    .channel_mapping = channel_mapping,
                    .dst_format = target_format,
                    .dst_data = dst_pixels,
                }));
                return;
            }
            case MipEncoding::BLOCK_COMPRESS:
            {
                // Remap the recipe's channels into a tightly packed BC source.
                FormatInfo bc_source_info = target_block;
                bc_source_info.channel_count = s_cast<u32>(channel_mapping.size());
                daxa::Format const bc_source_format = get_format_from_info(bc_source_info);
                std::vector<std::byte> bc_source(s_cast<usize>(texel_count) * get_info_from_format(bc_source_format).block_byte_size);

                run_task_inline(*remap_channels(RemapChannelsInfo{
                    .src_data = src_pixels,
                    .texel_count = texel_count,
                    .format = decoded.descriptor.info.format,
                    .channel_mapping = channel_mapping,
                    .dst_format = bc_source_format,
                    .dst_data = bc_source,
                }));
                run_task_inline(*compress_image(CreateCompressedImageInfo{
                    .src_data = bc_source,
                    .image_dimensions = mip_dimensions,
                    .target_format = target_format,
                    .source_format = bc_source_format,
                    .dst_data = dst_pixels,
                }));
                return;
            }
            default: DBG_ASSERT_TRUE_M(false, "cook_uncompressed_source: unhandled mip encoding");
        }
    };

    auto const & cooked_subresources = cooked.descriptor.subresources;
    auto const & decoded_subresources = decoded.descriptor.subresources;

    for (u32 mip = 0; mip < mip_count; ++mip)
    {
        u32 const mip_width = std::max(1u, width >> mip);
        u32 const mip_height = std::max(1u, height >> mip);
        u32 const mip_depth = std::max(1u, depth >> mip);
        // Derive the next mip from this one in the decoded format, so a 16-bit source is box-filtered at full
        // precision and only narrowed per-mip in the encode below. This mip's texels stay intact for that.
        if (mip + 1 < mip_count)
        {
            run_task_inline(*downsample_image(DownsampleImageInfo{
                .src_data = std::span<std::byte const>(decoded.data).subspan(decoded_subresources[mip].offset, decoded_subresources[mip].byte_size),
                .dst_data = std::span<std::byte>(decoded.data).subspan(decoded_subresources[mip + 1].offset, decoded_subresources[mip + 1].byte_size),
                .src_dimensions = {mip_width, mip_height, mip_depth},
                .format = decoded.descriptor.info.format,
            }));
        }

        encode_mip(
            std::span<std::byte const>(decoded.data).subspan(decoded_subresources[mip].offset, decoded_subresources[mip].byte_size),
            std::span<std::byte>(cooked.data).subspan(cooked_subresources[mip].offset, cooked_subresources[mip].byte_size),
            {mip_width, mip_height, mip_depth});
    }
    return cooked;
}

// Decode a PNG source and cook its full 2D mip chain into the recipe's target format.
auto cook_png_source(std::span<std::byte const> source_bytes, ImageImporterData const & importer_data) -> std::optional<TidoImageWithData>
{
    auto parse_result = image_parse(ImageParseInfo{.src_data = source_bytes, .source_format = importer_data.container_format});
    if (std::holds_alternative<ImageProcessResult>(parse_result)) { return std::nullopt; }
    TidoImageWithData decoded = std::move(std::get<TidoImageWithData>(parse_result));

    u32 const width = decoded.descriptor.info.size.x;
    u32 const height = decoded.descriptor.info.size.y;
    u32 const mip_count = s_cast<u32>(std::log2(std::max(width, height))) + 1;
    return cook_uncompressed_source(std::move(decoded), importer_data.channel_mapping, importer_data.target_format, mip_count);
}

// Transcode a Basis-compressed KTX2 source straight to the recipe's BCn target (mips already present).
auto cook_ktx_source(std::span<std::byte const> source_bytes, ImageImporterData const & importer_data) -> std::optional<TidoImageWithData>
{
    auto transcode_result = image_transcode(ImageTranscodeInfo{
        .src_data = source_bytes,
        .source_format = importer_data.container_format,
        .target_format = importer_data.target_format,
        .channel_mapping = importer_data.channel_mapping,
    });
    if (std::holds_alternative<ImageProcessResult>(transcode_result)) { return std::nullopt; }
    return std::move(std::get<TidoImageWithData>(transcode_result));
}

// Cooks one image asset: read its resolved source range, decode/transcode + BC-compress per the baked
// recipe, and write the .tido_bin artifact keyed by its identity into the per-source cache dir.
struct ImageCookPolicy
{
    using ImporterData = ImageImporterData;
    using Descriptor = TidoImageDescriptor;
    using StreamerData = ImageStreamerData;
    struct ReadState { std::vector<std::byte> source_bytes = {}; };

    static constexpr u32 COOK_VERSION = IMAGE_COOK_VERSION;
    static constexpr char const * LOG_TAG = "ImageCookTask";

    static auto identity_key(ImporterData const & importer_data) -> u64 { return image_identity_key(importer_data); }
    static auto read_header(std::span<std::byte const> header_region) -> std::optional<std::pair<TidoMetadataHash, Descriptor>> { return read_tido_image_header_data(header_region); }
    static auto write_artifact(WriteTidoFileInfo const & write_info, Descriptor const & descriptor) -> std::optional<StreamerData> { return write_tido_image(write_info, descriptor); }
    static auto artifact_source_name(ImporterData const & importer_data) -> std::string { return importer_data.source_bytes.file.stem().string(); }
    static auto current_mtime(ImporterData const & importer_data) -> i64 { return read_file_modified_time(importer_data.source_bytes.file).value_or(0); }

    static auto read_and_hash(ImporterData const & importer_data, ReadState & read_state) -> std::optional<u64>
    {
        auto [read_result, source_bytes] = read_file_byte_range(importer_data.source_bytes);
        if (read_result != FileIoResult::SUCCESS) { return std::nullopt; }
        // Content hash over the loaded encoded bytes (before parse), stamped into the artifact header.
        u64 const content_hash = tido_fnv1a(std::span<std::byte const>(source_bytes));
        read_state.source_bytes = std::move(source_bytes);
        return content_hash;
    }

    static auto do_cook(ImporterData const & importer_data, ReadState const & read_state, std::vector<std::byte> & payload) -> std::optional<Descriptor>
    {
        std::optional<TidoImageWithData> cooked = {};
        switch (importer_data.container_format)
        {
            case ImageFileFormat::PNG:  cooked = cook_png_source(read_state.source_bytes, importer_data); break;
            case ImageFileFormat::KTX2: cooked = cook_ktx_source(read_state.source_bytes, importer_data); break;
            default: DBG_ASSERT_TRUE_M(false, "ImageCookPolicy: unhandled container format"); return std::nullopt;
        }
        if (!cooked.has_value()) { return std::nullopt; }
        payload = std::move(cooked->data);
        return std::move(cooked->descriptor);
    }

    static auto cooked_detail(Descriptor const & descriptor) -> std::string
    {
        return fmt::format("{}x{}, {} mips", descriptor.info.size.x, descriptor.info.size.y, descriptor.info.mip_level_count);
    }
};

// Pack a cooked mesh into one contiguous payload plus its descriptor: each LOD's arrays are appended
// back-to-back in the SAME order make_resident_mesh walks the GPU mesh buffer, so the streamer can memcpy
// a LOD blob straight into a BDA buffer and wire the sub-pointers from the stored counts. glm::vec3 / vec2
// are bit-identical to daxa_f32vec3 / vec2, so the vertex arrays append byte-for-byte.
auto pack_processed_mesh(ProcessedMesh const & processed, std::vector<std::byte> & payload) -> TidoMeshDescriptor
{
    TidoMeshDescriptor descriptor = {};
    descriptor.lods.resize(processed.lod_count);
    for (u32 lod = 0; lod < processed.lod_count; ++lod)
    {
        ProcessedMeshLod const & cooked = processed.lods[lod];
        bool const lod_has_uv = !cooked.vertex_uvs.empty();

        u64 const blob_offset = payload.size();
        tido_append_array(payload, cooked.meshlets);
        tido_append_array(payload, cooked.meshlet_bounds);
        tido_append_array(payload, cooked.meshlet_aabbs);
        tido_append_array(payload, cooked.micro_indices);
        tido_append_array(payload, cooked.indirect_vertices);
        tido_append_array(payload, cooked.primitive_indices);
        tido_append_array(payload, cooked.vertex_positions);
        if (lod_has_uv) { tido_append_array(payload, cooked.vertex_uvs); }
        tido_append_array(payload, cooked.vertex_normals);

        descriptor.lods[lod] = {
            .offset = blob_offset,
            .byte_size = payload.size() - blob_offset,
            .aabb = cooked.aabb,
            .bounding_sphere = cooked.bounding_sphere,
            .lod_error = cooked.lod_error,
            .vertex_count = cooked.vertex_count,
            .primitive_count = cooked.primitive_count,
            .meshlet_count = s_cast<u32>(cooked.meshlets.size()),
            .micro_indices_count = s_cast<u32>(cooked.micro_indices.size()),
            .indirect_vertices_count = s_cast<u32>(cooked.indirect_vertices.size()),
            .primitive_indices_count = s_cast<u32>(cooked.primitive_indices.size()),
            .has_uv = lod_has_uv ? 1u : 0u,
        };
    }
    return descriptor;
}

// Cooks one mesh asset: read its resolved attribute-source ranges, interpret + validate them (mesh_parse),
// optimize into the runtime meshlet form (optimize_mesh), and write the .tido_bin artifact keyed by its
// identity into the per-source cache dir.
struct MeshCookPolicy
{
    using Descriptor = TidoMeshDescriptor;

    using ImporterData = MeshImporterData;
    using StreamerData = MeshStreamerData;
    struct ReadState
    {
        std::vector<std::byte> indices_bytes = {};
        std::vector<std::byte> positions_bytes = {};
        std::vector<std::byte> normals_bytes = {};
        std::vector<std::byte> uvs_bytes = {};
    };

    static constexpr u32 COOK_VERSION = MESH_COOK_VERSION;
    static constexpr char const * LOG_TAG = "MeshCookTask";

    static auto identity_key(ImporterData const & importer_data) -> u64 { return mesh_identity_key(importer_data); }
    static auto read_header(std::span<std::byte const> header_region) -> std::optional<std::pair<TidoMetadataHash, Descriptor>> { return read_tido_mesh_header_data(header_region); }
    static auto write_artifact(WriteTidoFileInfo const & write_info, Descriptor const & descriptor) -> std::optional<StreamerData> { return write_tido_mesh(write_info, descriptor); }
    static auto artifact_source_name(ImporterData const & importer_data) -> std::string { return importer_data.indices.range.file.stem().string(); }

    // The attribute streams usually share one .bin/.glb; the newest across them all is the bake stamp.
    static auto current_mtime(ImporterData const & importer_data) -> i64
    {
        i64 newest = read_file_modified_time(importer_data.indices.range.file).value_or(0);
        newest = std::max(newest, read_file_modified_time(importer_data.positions.range.file).value_or(0)); 
        newest = std::max(newest, read_file_modified_time(importer_data.normals.range.file).value_or(0)); 
        newest = std::max(newest, importer_data.uvs.has_value() ? read_file_modified_time(importer_data.uvs->range.file).value_or(0) : 0); 
        return newest;
    }

    static auto read_and_hash(ImporterData const & importer_data, ReadState & read_state) -> std::optional<u64>
    {
        auto [indices_read, indices_bytes] = read_file_byte_range(importer_data.indices.range);
        auto [positions_read, positions_bytes] = read_file_byte_range(importer_data.positions.range);
        auto [normals_read, normals_bytes] = read_file_byte_range(importer_data.normals.range);
        std::pair<FileIoResult, std::vector<std::byte>> uvs_read = {FileIoResult::SUCCESS, {}};
        if (importer_data.uvs.has_value()) { uvs_read = read_file_byte_range(importer_data.uvs->range); }
        if (indices_read != FileIoResult::SUCCESS || positions_read != FileIoResult::SUCCESS ||
            normals_read != FileIoResult::SUCCESS || uvs_read.first != FileIoResult::SUCCESS)
        {
            return std::nullopt;
        }

        // Content hash over the loaded raw geometry bytes (before parse), chained across the streams and
        // stamped into the artifact header.
        u64 content_hash = tido_fnv1a(std::span<std::byte const>(indices_bytes));
        content_hash = tido_fnv1a(std::span<std::byte const>(positions_bytes), content_hash);
        content_hash = tido_fnv1a(std::span<std::byte const>(normals_bytes), content_hash);
        if (importer_data.uvs.has_value()) { content_hash = tido_fnv1a(std::span<std::byte const>(uvs_read.second), content_hash); }

        read_state.indices_bytes = std::move(indices_bytes);
        read_state.positions_bytes = std::move(positions_bytes);
        read_state.normals_bytes = std::move(normals_bytes);
        read_state.uvs_bytes = std::move(uvs_read.second);
        return content_hash;
    }

    static auto do_cook(ImporterData const & importer_data, ReadState const & read_state, std::vector<std::byte> & payload) -> std::optional<Descriptor>
    {
        MeshParseInfo const parse_info = {
            .indices = {.data = read_state.indices_bytes, .component_type = importer_data.indices.component_type},
            .positions = {.data = read_state.positions_bytes, .component_type = importer_data.positions.component_type},
            .normals = {.data = read_state.normals_bytes, .component_type = importer_data.normals.component_type},
            .uvs = {.data = read_state.uvs_bytes, .component_type = importer_data.uvs.has_value() ? importer_data.uvs->component_type : ComponentType{}},
            .vertex_count = importer_data.vertex_count,
            .index_count = importer_data.index_count,
        };
        std::optional<RawMesh> raw = mesh_parse(parse_info);
        if (!raw.has_value()) { return std::nullopt; }

        ProcessedMesh const processed = optimize_mesh(raw.value());
        return pack_processed_mesh(processed, payload);
    }

    static auto cooked_detail(Descriptor const & descriptor) -> std::string
    {
        return fmt::format("{} LODs", descriptor.lods.size());
    }
};

// Cooks one VDB asset into a 3D image: read the whole .vdb, densify its grids into an interleaved fp32 volume
// (vdb_parse), then run the shared uncompressed cook tail into the recipe's target format. Reuses the image
// artifact (write_tido_image / ImageStreamerData), keyed by its VDB identity.
struct VdbCookPolicy
{
    using ImporterData = VdbImporterData;
    using Descriptor = TidoImageDescriptor;
    using StreamerData = ImageStreamerData;
    struct ReadState { std::vector<std::byte> source_bytes = {}; };

    static constexpr u32 COOK_VERSION = VDB_COOK_VERSION;
    static constexpr char const * LOG_TAG = "VdbCookTask";

    static auto identity_key(ImporterData const & importer_data) -> u64 { return vdb_identity_key(importer_data); }
    static auto read_header(std::span<std::byte const> header_region) -> std::optional<std::pair<TidoMetadataHash, Descriptor>> { return read_tido_image_header_data(header_region); }
    static auto write_artifact(WriteTidoFileInfo const & write_info, Descriptor const & descriptor) -> std::optional<StreamerData> { return write_tido_image(write_info, descriptor); }
    static auto artifact_source_name(ImporterData const & importer_data) -> std::string { return importer_data.source_bytes.file.stem().string(); }
    static auto current_mtime(ImporterData const & importer_data) -> i64 { return read_file_modified_time(importer_data.source_bytes.file).value_or(0); }

    static auto read_and_hash(ImporterData const & importer_data, ReadState & read_state) -> std::optional<u64>
    {
        auto [read_result, source_bytes] = read_file_byte_range(importer_data.source_bytes);
        if (read_result != FileIoResult::SUCCESS) { return std::nullopt; }
        // Content hash over the whole loaded .vdb (before parse), stamped into the artifact header.
        u64 const content_hash = tido_fnv1a(std::span<std::byte const>(source_bytes));
        read_state.source_bytes = std::move(source_bytes);
        return content_hash;
    }

    static auto do_cook(ImporterData const & importer_data, ReadState const & read_state, std::vector<std::byte> & payload) -> std::optional<Descriptor>
    {
        std::optional<TidoImageWithData> decoded = vdb_parse(VdbParseInfo{.src_data = read_state.source_bytes, .grid_names = importer_data.grid_names});
        if (!decoded.has_value()) { return std::nullopt; }

        // vdb_parse densifies one channel per grid, in order; the recipe's channel_mapping then permutes those
        // into the target, whose format decides the final precision/layout via remap/compress.
        // VDB defers mip generation for now - a single level.
        std::optional<TidoImageWithData> cooked = cook_uncompressed_source(std::move(decoded.value()), importer_data.channel_mapping, importer_data.target_format, 1);
        if (!cooked.has_value()) { return std::nullopt; }
        payload = std::move(cooked->data);
        return std::move(cooked->descriptor);
    }

    static auto cooked_detail(Descriptor const & descriptor) -> std::string
    {
        return fmt::format("{}x{}x{}, {} mips", descriptor.info.size.x, descriptor.info.size.y, descriptor.info.size.z, descriptor.info.mip_level_count);
    }
};

using ImageCookTask = CookTask<ImageCookPolicy>;
using MeshCookTask = CookTask<MeshCookPolicy>;
using VdbCookTask = CookTask<VdbCookPolicy>;

// Dispatches the fastgltf-free asset cooks (image and mesh) from their already-resolved ImporterData,
// consuming those tasks. Scene-parse tasks are left in place for the gltf backend.
void dispatch_asset_cooks(Importer & importer, std::vector<ImporterTask> & tasks)
{
    auto consume_cook_task = [&](ImporterTask & task) -> bool
    {
        if (auto * import_image = std::get_if<ImporterTask::ImportImageAsset>(&task.data))
        {
            importer.thread_pool->async_dispatch(
                std::make_shared<ImageCookTask>(&importer, std::move(import_image->importer_data), import_image->image_manifest_index),
                TaskPriority::LOW);
            return true;
        }
        if (auto * import_mesh = std::get_if<ImporterTask::ImportMeshAsset>(&task.data))
        {
            importer.thread_pool->async_dispatch(
                std::make_shared<MeshCookTask>(&importer, std::move(import_mesh->importer_data), import_mesh->mesh_manifest_index),
                TaskPriority::LOW);
            return true;
        }
        if (auto * import_vdb = std::get_if<ImporterTask::ImportVdbAsset>(&task.data))
        {
            importer.thread_pool->async_dispatch(
                std::make_shared<VdbCookTask>(&importer, std::move(import_vdb->importer_data), import_vdb->image_manifest_index),
                TaskPriority::LOW);
            return true;
        }
        return false;
    };
    std::erase_if(tasks, consume_cook_task);
}
} // namespace

Importer::Importer(ThreadPool * thread_pool)
    : thread_pool{thread_pool}
{
    _thread = std::thread([this]() { thread_main(); });
}

Importer::~Importer()
{
    stop();
}

void Importer::stop()
{
    {
        std::lock_guard<std::mutex> lock{_queue_mutex};
        _stop_requested = true;
    }
    _wake_signal.notify_all();
    if (_thread.joinable())
    {
        _thread.join();
    }
}

void Importer::push_tasks(std::span<ImporterTask> tasks)
{
    {
        std::lock_guard<std::mutex> lock{_queue_mutex};
        if (_stop_requested)
        {
            return;
        }
        _task_queue.insert(_task_queue.end(), std::make_move_iterator(tasks.begin()), std::make_move_iterator(tasks.end()));
    }
    _wake_signal.notify_one();
}

auto Importer::pop_results() -> std::vector<ImporterTaskResult>
{
    std::lock_guard<std::mutex> lock{_result_queue_mutex};
    std::vector<ImporterTaskResult> results = std::move(_result_queue);
    _result_queue.clear();
    return results;
}

void Importer::push_result(ImporterTaskResult result)
{
    std::lock_guard<std::mutex> lock{_result_queue_mutex};
    _result_queue.push_back(std::move(result));
}

void Importer::notify()
{
    {
        std::lock_guard<std::mutex> lock{_queue_mutex};
        _upkeep_requested = true;
    }
    _wake_signal.notify_one();
}

void Importer::thread_main()
{
    for (;;)
    {
        std::vector<ImporterTask> tasks = {};
        {
            std::unique_lock<std::mutex> lock{_queue_mutex};
            _wake_signal.wait(lock, [&]() { return _stop_requested || !_task_queue.empty() || _upkeep_requested; });
            if (_stop_requested)
            {
                return; // Still-queued tasks are deliberately dropped on shutdown.
            }
            tasks = std::move(_task_queue);
            _task_queue.clear();
            _upkeep_requested = false;
        }

        // TODO(saky): TEMP HACK - Fix once threadpool has proper task priorities
        std::sort(tasks.begin(), tasks.end(), [](ImporterTask const & a, ImporterTask const & b) {
            auto get_priority = [](ImporterTask const & task) -> u32 {
                if (std::holds_alternative<ImporterTask::ImportScene>(task.data)) { return 0; }
                if (std::holds_alternative<ImporterTask::ImportMeshAsset>(task.data)) { return 1; }
                if (std::holds_alternative<ImporterTask::ImportImageAsset>(task.data)) { return 2; }
                if (std::holds_alternative<ImporterTask::ImportVdbAsset>(task.data)) { return 2; }
                return 3;
            };
            return get_priority(a) < get_priority(b);
        });

        for(auto const & task : tasks)
        {
            if (std::holds_alternative<ImporterTask::ImportImageAsset>(task.data))
            {
                auto const & import_image = std::get<ImporterTask::ImportImageAsset>(task.data);
                DEBUG_MSG(fmt::format("[Importer] dispatching image cook for '{}'", import_image.importer_data.source_bytes.file.string()));
            }
            else if (std::holds_alternative<ImporterTask::ImportMeshAsset>(task.data))
            {
                auto const & import_mesh = std::get<ImporterTask::ImportMeshAsset>(task.data);
                DEBUG_MSG(fmt::format("[Importer] dispatching mesh cook for '{}'", import_mesh.importer_data.indices.range.file.string()));
            }
            else if (std::holds_alternative<ImporterTask::ImportVdbAsset>(task.data))
            {
                auto const & import_vdb = std::get<ImporterTask::ImportVdbAsset>(task.data);
                DEBUG_MSG(fmt::format("[Importer] dispatching vdb cook for '{}'", import_vdb.importer_data.source_bytes.file.string()));
            }
            else if (std::holds_alternative<ImporterTask::ImportScene>(task.data))
            {
                auto const & import_scene = std::get<ImporterTask::ImportScene>(task.data);
                DEBUG_MSG(fmt::format("[Importer] dispatching scene parse for '{}'", import_scene.path.string()));
            }
        }
        // Generic, backend-agnostic asset cooks first (they act on resolved ImporterData); the gltf backend
        // then consumes what's left - the scene-parse tasks.
        dispatch_asset_cooks(*this, tasks);
        dispatch_scene_parses(*this, tasks);
        DBG_ASSERT_TRUE_M(tasks.empty(), "An ImporterTask was left unconsumed - no importer handles its provenance");
    }
}
