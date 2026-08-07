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

// Bumped whenever the image cook's output or .tido_bin layout changes; folded into the artifact key, so a bump
// makes every image artifact cooked by an earlier version unreachable instead of silently stale.
static constexpr u32 IMAGE_COOK_VERSION = 1;

// Bumped whenever the mesh cook's output or .tido_bin layout changes; folded into the artifact key, so a bump
// makes every mesh artifact cooked by an earlier version unreachable instead of silently stale.
static constexpr u32 MESH_COOK_VERSION = 1;

// Bumped whenever the VDB cook's output or .tido_bin layout changes; folded into the artifact key, so a bump
// makes every VDB artifact cooked by an earlier version unreachable instead of silently stale.
static constexpr u32 VDB_COOK_VERSION = 1;

namespace
{

// Folded into every artifact key so the flat store's key space stays disjoint per asset type - the cook
// version constants alone do not distinguish an image from a mesh.
enum struct ArtifactKind : u32
{
    IMAGE = 1,
    MESH = 2,
    VDB = 3,
};

// Length-prefixes a variable-length recipe field, so two different splits of the same concatenated bytes
// cannot fold into the same key.
void append_recipe_field(std::vector<std::byte> & key_bytes, void const * data, usize size)
{
    tido_append_pod(key_bytes, s_cast<u64>(size));
    tido_append_bytes(key_bytes, data, size);
}

// Identity is a property of what the asset is, not of where it was found: two byte-identical sources produce
// one artifact however many files, containers or slots they arrived through. Nothing about the location is
// folded in, so slot renumbering and re-export to a new path are non-events.
auto image_artifact_key(ImageImporterData const & importer_data, u64 content_hash) -> u64
{
    std::vector<std::byte> key_bytes;
    tido_append_pod(key_bytes, ArtifactKind::IMAGE);
    tido_append_pod(key_bytes, IMAGE_COOK_VERSION);
    tido_append_pod(key_bytes, content_hash);

    // Recipe: container format joins the channel mapping and target format so one source blob used under
    // different recipes gets distinct keys.
    tido_append_pod(key_bytes, importer_data.container_format);
    append_recipe_field(key_bytes, importer_data.channel_mapping.data(), importer_data.channel_mapping.size());
    tido_append_pod(key_bytes, importer_data.target_format);

    return tido_fnv1a(key_bytes, 0);
}

auto mesh_artifact_key(MeshImporterData const & importer_data, u64 content_hash) -> u64
{
    std::vector<std::byte> key_bytes;
    tido_append_pod(key_bytes, ArtifactKind::MESH);
    tido_append_pod(key_bytes, MESH_COOK_VERSION);
    tido_append_pod(key_bytes, content_hash);

    // Meshes carry no cook recipe, but the component types and counts decide how the hashed bytes are read,
    // and one blob can be a valid read under more than one interpretation - those must not share a key.
    tido_append_pod(key_bytes, importer_data.indices.component_type);
    tido_append_pod(key_bytes, importer_data.positions.component_type);
    tido_append_pod(key_bytes, importer_data.normals.component_type);
    // An absent uvs stream contributes nothing to the content hash, so the flag is what separates it from an
    // empty one.
    tido_append_pod(key_bytes, importer_data.uvs.has_value());
    if (importer_data.uvs.has_value()) { tido_append_pod(key_bytes, importer_data.uvs->component_type); }
    tido_append_pod(key_bytes, importer_data.vertex_count);
    tido_append_pod(key_bytes, importer_data.index_count);

    return tido_fnv1a(key_bytes, 0);
}

auto vdb_artifact_key(VdbImporterData const & importer_data, u64 content_hash) -> u64
{
    std::vector<std::byte> key_bytes;
    tido_append_pod(key_bytes, ArtifactKind::VDB);
    tido_append_pod(key_bytes, VDB_COOK_VERSION);
    tido_append_pod(key_bytes, content_hash);

    // Recipe: the selected grids join the channel mapping and target format, so one .vdb cooked under
    // different grid selections or formats gets distinct keys.
    tido_append_pod(key_bytes, s_cast<u64>(importer_data.grid_names.size()));
    for (auto const & grid : importer_data.grid_names) { append_recipe_field(key_bytes, grid.data(), grid.size()); }
    append_recipe_field(key_bytes, importer_data.channel_mapping.data(), importer_data.channel_mapping.size());
    tido_append_pod(key_bytes, importer_data.target_format);

    return tido_fnv1a(key_bytes, 0);
}

// The JSON header region (metadata + descriptor, no payload) and payload offset of a .tido_bin, read
// without loading the payload. nullopt if the file is missing, too short, or its preamble is invalid - any
// of which routes the caller to a full cook. The two byte-range reads keep the artifact's bulk off the disk
// for a cache probe.
struct CachedArtifactHeaderData
{
    std::vector<std::byte> header_region = {};
    u64 file_data_offset = {};
};
auto read_cached_artifact_header(std::filesystem::path const & tido_path) -> std::optional<CachedArtifactHeaderData>
{
    // Read the preamble to find the header region length.
    auto [read_preamble_result, preamble_bytes] = read_file(tido_path, ByteSlice{.byte_offset = 0, .byte_length = TIDO_FILE_PREAMBLE_SIZE});
    if (read_preamble_result != FileIoResult::SUCCESS) { return std::nullopt; }

    std::optional<TidoFilePreamble> const preamble = tido_parse_preamble(preamble_bytes);
    if (!preamble.has_value()) { return std::nullopt; }

    // Read the json header region.
    auto [read_json_header_result, json_header_bytes] = read_file(tido_path, ByteSlice{.byte_offset = TIDO_FILE_PREAMBLE_SIZE, .byte_length = preamble->header_byte_length});
    if (read_json_header_result != FileIoResult::SUCCESS) { return std::nullopt; }

    return CachedArtifactHeaderData{
        .header_region = std::vector<std::byte>(json_header_bytes.begin(), json_header_bytes.end()),
        .file_data_offset = TIDO_FILE_PREAMBLE_SIZE + preamble->header_byte_length,
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

// Drives one asset cook end to end: read and hash the sources, derive the artifact key from that hash, and
// either serve the artifact already sitting at the key or cook and write one. Everything type-specific (which
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
    SourceSlot slot = {};

    CookTask(Importer * importer, ImporterData importer_data, SourceSlot slot)
        : importer{importer}, importer_data{std::move(importer_data)}, slot{slot}
    {
        chunk_count = 1;
    }

    void cook()
    {
        std::string const artifact_name = Policy::artifact_source_name(importer_data);

        // Every probe hashes the sources first: the key is derived from the bytes, so with location out of it
        // nothing cheaper can even name the artifact. The hash streams through a bounded buffer rather than
        // materializing the sources, because a hit never looks at them.
        std::optional<u64> const probe_hash = Policy::hash_sources(importer_data);
        if (!probe_hash.has_value())
        {
            DEBUG_MSG(fmt::format("[ERROR][{}] failed to hash source bytes for '{}'", Policy::LOG_TAG, artifact_name));
            importer->push_cook_failed<StreamerData>(slot);
            return;
        }

        u64 const artifact_key = Policy::artifact_key(importer_data, probe_hash.value());
        std::filesystem::path const tido_path = tido_artifact_path(TIDO_ASSET_CACHE_DIR, artifact_key);

        std::optional<CachedArtifactHeaderData> const cached_header_data = read_cached_artifact_header(tido_path);
        std::optional<std::pair<TidoMetadataHash, Descriptor>> cached_header = {};
        if (cached_header_data.has_value()) { cached_header = Policy::parse_artifact_header(cached_header_data->header_region); }
        // The key already folds the content hash and the cook version, so an artifact that parses at this path
        // is by construction current; the stored fields are re-checked only to catch a foreign file.
        bool const cache_usable = cached_header.has_value()
            && cached_header->first.version == Policy::COOK_VERSION
            && cached_header->first.cache_key == artifact_key;

        auto push_streamer_result = [&](StreamerData streamer_data)
        {
            importer->push_cook_finished(std::move(streamer_data), slot);
        };

        if (cache_usable)
        {
            DEBUG_MSG(fmt::format("[{}] cache hit '{}' -> '{}'", Policy::LOG_TAG, artifact_name, tido_path.string()));
            push_streamer_result(StreamerData{
                .descriptor = cached_header->second,
                .bin_source = tido_path,
                .file_data_offset = cached_header_data->file_data_offset,
            });
            return;
        }

        // Only a miss materializes the sources, and it re-hashes what it actually loaded rather than trusting
        // the probe: a source edited in between would otherwise store a cook of the new bytes under the old
        // bytes' key, where another source holding the old content would later be served it.
        typename Policy::ReadState read_state = {};
        std::optional<u64> const content_hash = Policy::read_and_hash(importer_data, read_state);
        if (!content_hash.has_value())
        {
            DEBUG_MSG(fmt::format("[ERROR][{}] failed to read source bytes for '{}'", Policy::LOG_TAG, artifact_name));
            importer->push_cook_failed<StreamerData>(slot);
            return;
        }
        // Equal to artifact_key unless the source moved between the probe and the read, which just costs a
        // cook of something that may already exist - the write is content-addressed either way.
        u64 const cooked_artifact_key = Policy::artifact_key(importer_data, content_hash.value());

        std::vector<std::byte> payload = {};
        std::optional<Descriptor> const descriptor = Policy::do_cook(importer_data, read_state, payload);
        if (!descriptor.has_value())
        {
            DEBUG_MSG(fmt::format("[ERROR][{}] failed to cook '{}'", Policy::LOG_TAG, artifact_name));
            importer->push_cook_failed<StreamerData>(slot);
            return;
        }

        WriteTidoFileInfo const write_info = {
            .store_dir = TIDO_ASSET_CACHE_DIR,
            .metadata_hash = TidoMetadataHash{
                .cache_key = cooked_artifact_key,
                .source_mtime_at_bake = Policy::current_mtime(importer_data),
                .content_hash = content_hash.value(),
                .version = Policy::COOK_VERSION,
                .name = tido_sanitize_stem(artifact_name),
            },
            .data = payload,
        };
        std::optional<StreamerData> const streamer_data = Policy::write_artifact(write_info, descriptor.value());
        if (!streamer_data.has_value())
        {
            DEBUG_MSG(fmt::format("[WARN][{}] failed to write .tido_bin for '{}'", Policy::LOG_TAG, artifact_name));
            importer->push_cook_failed<StreamerData>(slot);
            return;
        }

        DEBUG_MSG(fmt::format("[{}] cooked '{}' ({}) -> '{}'", Policy::LOG_TAG, artifact_name,
            Policy::cooked_detail(descriptor.value()), streamer_data->bin_source.string()));
        push_streamer_result(streamer_data.value());
    }

    void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override
    {
        cook();
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
                // BC1 is an RGB codec, so a single channel mapped into it is the custom scalar SDF encoding
                // rather than a colour block. compress_image selects that encoder off an fp32 source.
                bool const is_bc1_target =
                    target_format == daxa::Format::BC1_RGB_UNORM_BLOCK || target_format == daxa::Format::BC1_RGB_SRGB_BLOCK ||
                    target_format == daxa::Format::BC1_RGBA_UNORM_BLOCK || target_format == daxa::Format::BC1_RGBA_SRGB_BLOCK;
                if (is_bc1_target && bc_source_info.channel_count == 1)
                {
                    bc_source_info.channel_byte_size = sizeof(f32);
                    bc_source_info.numeric_type = FormatNumericType::SFLOAT;
                }
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
// recipe, and write the .tido_bin artifact into the flat store under its content-derived key.
struct ImageCookPolicy
{
    using ImporterData = ImageImporterData;
    using Descriptor = TidoImageDescriptor;
    using StreamerData = ImageStreamerData;
    struct ReadState { std::vector<std::byte> source_bytes = {}; };

    static constexpr u32 COOK_VERSION = IMAGE_COOK_VERSION;
    static constexpr char const * LOG_TAG = "ImageCookTask";

    static auto artifact_key(ImporterData const & importer_data, u64 content_hash) -> u64 { return image_artifact_key(importer_data, content_hash); }
    static auto parse_artifact_header(std::span<std::byte const> header_region) -> std::optional<std::pair<TidoMetadataHash, Descriptor>> { return parse_tido_image_header_data(header_region); }
    static auto write_artifact(WriteTidoFileInfo const & write_info, Descriptor const & descriptor) -> std::optional<StreamerData> { return write_tido_image(write_info, descriptor); }
    static auto artifact_source_name(ImporterData const & importer_data) -> std::string { return importer_data.source_location.file.stem().string(); }
    static auto current_mtime(ImporterData const & importer_data) -> i64 { return read_file_modified_time(importer_data.source_location.file).value_or(0); }

    static auto hash_sources(ImporterData const & importer_data) -> std::optional<u64>
    {
        return tido_hash_file(importer_data.source_location.file, importer_data.source_location.slice);
    }

    static auto read_and_hash(ImporterData const & importer_data, ReadState & read_state) -> std::optional<u64>
    {
        auto [read_result, source_bytes] = read_file(importer_data.source_location.file, importer_data.source_location.slice);
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
// optimize into the runtime meshlet form (optimize_mesh), and write the .tido_bin artifact into the flat
// store under its content-derived key.
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

    static auto artifact_key(ImporterData const & importer_data, u64 content_hash) -> u64 { return mesh_artifact_key(importer_data, content_hash); }
    static auto parse_artifact_header(std::span<std::byte const> header_region) -> std::optional<std::pair<TidoMetadataHash, Descriptor>> { return parse_tido_mesh_header_data(header_region); }
    static auto write_artifact(WriteTidoFileInfo const & write_info, Descriptor const & descriptor) -> std::optional<StreamerData> { return write_tido_mesh(write_info, descriptor); }
    static auto artifact_source_name(ImporterData const & importer_data) -> std::string { return importer_data.indices.location.file.stem().string(); }

    // The attribute streams usually share one .bin/.glb; the newest across them all is the bake stamp.
    static auto current_mtime(ImporterData const & importer_data) -> i64
    {
        i64 newest = read_file_modified_time(importer_data.indices.location.file).value_or(0);
        newest = std::max(newest, read_file_modified_time(importer_data.positions.location.file).value_or(0));
        newest = std::max(newest, read_file_modified_time(importer_data.normals.location.file).value_or(0));
        newest = std::max(newest, importer_data.uvs.has_value() ? read_file_modified_time(importer_data.uvs->location.file).value_or(0) : 0);
        return newest;
    }

    // Chains the streams in the same order read_and_hash does - the two must agree byte for byte, or the probe
    // names a different artifact than the write and every run recooks.
    static auto hash_sources(ImporterData const & importer_data) -> std::optional<u64>
    {
        std::optional<u64> hash = tido_hash_file(importer_data.indices.location.file, importer_data.indices.location.slice);
        if (hash.has_value()) { hash = tido_hash_file(importer_data.positions.location.file, importer_data.positions.location.slice, hash.value()); }
        if (hash.has_value()) { hash = tido_hash_file(importer_data.normals.location.file, importer_data.normals.location.slice, hash.value()); }
        if (hash.has_value() && importer_data.uvs.has_value()) { hash = tido_hash_file(importer_data.uvs->location.file, importer_data.uvs->location.slice, hash.value()); }
        return hash;
    }

    static auto read_and_hash(ImporterData const & importer_data, ReadState & read_state) -> std::optional<u64>
    {
        auto [indices_read, indices_bytes] = read_file(importer_data.indices.location.file, importer_data.indices.location.slice);
        auto [positions_read, positions_bytes] = read_file(importer_data.positions.location.file, importer_data.positions.location.slice);
        auto [normals_read, normals_bytes] = read_file(importer_data.normals.location.file, importer_data.normals.location.slice);
        std::pair<FileIoResult, std::vector<std::byte>> uvs_read = {FileIoResult::SUCCESS, {}};
        if (importer_data.uvs.has_value()) { uvs_read = read_file(importer_data.uvs->location.file, importer_data.uvs->location.slice); }
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

    static auto artifact_key(ImporterData const & importer_data, u64 content_hash) -> u64 { return vdb_artifact_key(importer_data, content_hash); }
    static auto parse_artifact_header(std::span<std::byte const> header_region) -> std::optional<std::pair<TidoMetadataHash, Descriptor>> { return parse_tido_image_header_data(header_region); }
    static auto write_artifact(WriteTidoFileInfo const & write_info, Descriptor const & descriptor) -> std::optional<StreamerData> { return write_tido_image(write_info, descriptor); }
    static auto artifact_source_name(ImporterData const & importer_data) -> std::string { return importer_data.source_location.file.stem().string(); }
    static auto current_mtime(ImporterData const & importer_data) -> i64 { return read_file_modified_time(importer_data.source_location.file).value_or(0); }

    static auto hash_sources(ImporterData const & importer_data) -> std::optional<u64>
    {
        return tido_hash_file(importer_data.source_location.file, importer_data.source_location.slice);
    }

    static auto read_and_hash(ImporterData const & importer_data, ReadState & read_state) -> std::optional<u64>
    {
        auto [read_result, source_bytes] = read_file(importer_data.source_location.file, importer_data.source_location.slice);
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

// Dispatches the backend-agnostic asset cooks from their already-resolved ImporterData, consuming those
// tasks. Source-import tasks are left in place for the backends.
void dispatch_asset_cooks(Importer & importer, std::vector<ImporterTask> & tasks)
{
    auto consume_cook_task = [&](ImporterTask & task) -> bool
    {
        if (auto * cook_image = std::get_if<ImporterTask::CookImage>(&task.data))
        {
            // The recipe's alternative decides which cook the slot needs; nothing past this point knows
            // which backend produced it.
            if (auto * image_importer_data = std::get_if<ImageImporterData>(&cook_image->importer_data))
            {
                importer.thread_pool->async_dispatch(
                    std::make_shared<ImageCookTask>(&importer, std::move(*image_importer_data), cook_image->slot),
                    TaskPriority::LOW);
            }
            else if (auto * vdb_importer_data = std::get_if<VdbImporterData>(&cook_image->importer_data))
            {
                importer.thread_pool->async_dispatch(
                    std::make_shared<VdbCookTask>(&importer, std::move(*vdb_importer_data), cook_image->slot),
                    TaskPriority::LOW);
            }
            else
            {
                DBG_ASSERT_TRUE_M(false, "Image cook request has no cook");
            }
            return true;
        }
        if (auto * cook_mesh = std::get_if<ImporterTask::CookMesh>(&task.data))
        {
            importer.thread_pool->async_dispatch(
                std::make_shared<MeshCookTask>(&importer, std::move(cook_mesh->importer_data), cook_mesh->slot),
                TaskPriority::LOW);
            return true;
        }
        return false;
    };
    std::erase_if(tasks, consume_cook_task);
}

// The open list of source backends. A new source kind is a row here plus its dispatch function; nothing else
// in the pipeline distinguishes them.
constexpr SourceBackend SOURCE_BACKENDS[] = {
    {".gltf", dispatch_gltf_source},
    {".glb",  dispatch_gltf_source},
    {".png",  dispatch_standalone_image_source},
    {".ktx2", dispatch_standalone_image_source},
    {".vdb",  dispatch_cloud_volume_source},
};

} // namespace

auto find_source_backend(std::filesystem::path const & path) -> SourceBackend const *
{
    std::string const extension = tido_lowercase_extension(path);
    for (SourceBackend const & backend : SOURCE_BACKENDS)
    {
        if (backend.extension == extension) { return &backend; }
    }
    return nullptr;
}

namespace
{
// The editor's own assets, deliberately outside the Tido Assets root: the sandbox constrains paths a user
// picks, and these are neither picked nor scene content. They are still imported and cooked like anything
// else, so editing one re-cooks it and a cook version bump reaches it.
struct PlaceholderSource
{
    MaterialTextureSlot slot = {};
    char const * path = {};
    ImageSlotRecipe recipe = {};
};

// The fills are neutral so a pending asset reads as plausibly lit rather than announcing itself. Each recipe
// matches the one the real asset in that slot is cooked with, so a stand-in and the artifact replacing it
// are the same kind of image.
auto placeholder_sources() -> std::array<PlaceholderSource, s_cast<usize>(MaterialTextureSlot::COUNT)> const &
{
    static std::array<PlaceholderSource, s_cast<usize>(MaterialTextureSlot::COUNT)> const SOURCES = {
        // Mid grey once the sRGB target decodes it.
        PlaceholderSource{MaterialTextureSlot::DIFFUSE, "editor/placeholders/placeholder_diffuse.png",
            {.channel_mapping = {0, 1, 2}, .target_format = daxa::Format::BC1_RGB_SRGB_BLOCK}},
        // Fully opaque, so nothing is alpha-discarded while it is pending.
        PlaceholderSource{MaterialTextureSlot::OPACITY, "editor/placeholders/placeholder_opacity.png",
            {.channel_mapping = {0}, .target_format = daxa::Format::BC4_UNORM_BLOCK}},
        // Flat +Z. Cooked to a non-sRGB target, so the source stays linear and 128 means 0.5.
        PlaceholderSource{MaterialTextureSlot::NORMAL, "editor/placeholders/placeholder_normal.png",
            {.channel_mapping = {0, 1}, .target_format = daxa::Format::BC5_UNORM_BLOCK}},
        // glTF channel order: mid rough, fully dielectric.
        PlaceholderSource{MaterialTextureSlot::ROUGHNESS_METALNESS, "editor/placeholders/placeholder_rough_metal.png",
            {.channel_mapping = {0, 1, 2, 3}, .target_format = daxa::Format::BC7_UNORM_BLOCK}},
    };
    return SOURCES;
}

// The material's binding for `slot`; the one place the four texture fields are addressed by slot.
auto material_texture_binding(ImporterTaskResult::SceneMetadataBatch::Material & material, MaterialTextureSlot slot)
    -> std::optional<ImporterTaskResult::SceneMetadataBatch::TextureBinding> &
{
    switch (slot)
    {
        case MaterialTextureSlot::DIFFUSE:             return material.diffuse_info;
        case MaterialTextureSlot::OPACITY:             return material.opacity_mask_info;
        case MaterialTextureSlot::NORMAL:              return material.normal_info;
        case MaterialTextureSlot::ROUGHNESS_METALNESS: return material.roughness_metalness_info;
        case MaterialTextureSlot::COUNT:
        default:
            DBG_ASSERT_TRUE_M(false, "material_texture_binding: unhandled material texture slot");
            return material.diffuse_info;
    }
}
} // namespace

Importer::Importer(ThreadPool * thread_pool)
    : thread_pool{thread_pool}
{
    _thread = std::thread([this]() { thread_main(); });
    request_placeholder_imports();
}

void Importer::request_placeholder_imports()
{
    std::vector<ImporterTask> tasks = {};
    for (PlaceholderSource const & source : placeholder_sources())
    {
        tasks.push_back(ImporterTask{.data = ImporterTask::ImportSource{
            .path = source.path,
            .image_recipe = source.recipe,
        }});
    }
    push_tasks(tasks);
}

auto Importer::placeholders_ready() const -> bool
{
    for (usize slot = 0; slot < _placeholder_manifest_indices.size(); ++slot)
    {
        if (!_placeholder_manifest_indices.at(slot).has_value() && !_placeholder_failed.at(slot)) { return false; }
    }
    return true;
}

void Importer::fail_placeholder_slot(MaterialTextureSlot slot, std::string_view reason)
{
    if (_placeholder_failed.at(s_cast<usize>(slot))) { return; }
    _placeholder_failed.at(s_cast<usize>(slot)) = true;
    DEBUG_MSG(fmt::format("[ERROR][Importer] placeholder for material texture slot {} unavailable: {}. That slot gets no stand-in.",
        s_cast<u32>(slot), reason));
    release_parked_source_imports();
}

void Importer::release_parked_source_imports()
{
    if (!placeholders_ready() || _parked_source_imports.empty()) { return; }
    DEBUG_MSG(fmt::format("[Importer] placeholders settled; releasing {} held source imports", _parked_source_imports.size()));
    std::vector<ImporterTask> released = std::move(_parked_source_imports);
    _parked_source_imports.clear();
    push_tasks(released);
}

void Importer::dispatch_source_imports(std::vector<ImporterTask> & tasks)
{
    auto consume_source_task = [&](ImporterTask & task) -> bool
    {
        auto * import_source = std::get_if<ImporterTask::ImportSource>(&task.data);
        if (import_source == nullptr) { return false; }

        std::optional<MaterialTextureSlot> const placeholder_slot = [&]() -> std::optional<MaterialTextureSlot>
        {
            for (PlaceholderSource const & source : placeholder_sources())
            {
                if (import_source->path == std::filesystem::path{source.path}) { return source.slot; }
            }
            return std::nullopt;
        }();

        // A material is published bound to its stand-ins, so a source publishing before those entries exist
        // would have nothing to bind and no later chance to pick one up.
        if (!placeholder_slot.has_value() && !placeholders_ready())
        {
            _parked_source_imports.push_back(std::move(task));
            return true;
        }

        SourceBackend const * backend = find_source_backend(import_source->path);
        // request_import rejects an unclaimed extension before it ever reaches the queue.
        DBG_ASSERT_TRUE_M(backend != nullptr, "ImportSource for an extension no backend claims");
        if (backend == nullptr)
        {
            // A placeholder that cannot even be dispatched would otherwise hold every later import forever,
            // so it is settled as failed instead: its slot simply gets no stand-in.
            if (placeholder_slot.has_value()) { fail_placeholder_slot(placeholder_slot.value(), "no source backend claims its extension"); }
            return true;
        }

        DEBUG_MSG(fmt::format("[Importer] dispatching source import for '{}'", import_source->path.string()));
        SourceImportRequest request = register_source_import(import_source->path);
        request.image_recipe = std::move(import_source->image_recipe);
        if (placeholder_slot.has_value()) { _placeholder_source_indices.at(s_cast<usize>(placeholder_slot.value())) = request.source_index; }
        backend->dispatch(*this, request);
        return true;
    };
    std::erase_if(tasks, consume_source_task);
}

auto Importer::placeholder_slot_for_source(u32 source_index) const -> std::optional<MaterialTextureSlot>
{
    for (usize slot = 0; slot < _placeholder_source_indices.size(); ++slot)
    {
        if (_placeholder_source_indices.at(slot) == source_index) { return s_cast<MaterialTextureSlot>(slot); }
    }
    return std::nullopt;
}

auto Importer::placeholder_for_slot(MaterialTextureSlot slot) const -> std::optional<u32>
{
    if (slot == MaterialTextureSlot::COUNT) { return std::nullopt; }
    return _placeholder_manifest_indices.at(s_cast<usize>(slot));
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

auto Importer::register_source_import(std::filesystem::path const & path) -> SourceImportRequest
{
    auto const [registered, inserted] = _source_indices.try_emplace(path, s_cast<u32>(_sources.size()));
    if (inserted)
    {
        _sources.push_back(ImportedSource{.path = path});
    }
    u32 const source_index = registered->second;
    DEBUG_MSG(fmt::format("[Importer] source {} generation {} '{}'", source_index, _sources.at(source_index).generation, path.string()));
    return SourceImportRequest{
        .path = path,
        .source_index = source_index,
        .generation = _sources.at(source_index).generation,
    };
}

void Importer::publish_source_import(SourceImportResult import_result)
{
    ImporterTask task = {.data = ImporterTask::PublishImport{.import_result = std::move(import_result)}};
    push_tasks(std::span{&task, 1});
}

void Importer::publish_import_on_importer_thread(SourceImportResult import_result)
{
    using SceneBatch = ImporterTaskResult::SceneMetadataBatch;

    u32 const source_index = import_result.batch.source_index;
    u32 const batch_id = _next_batch_id++;
    import_result.batch.batch_id = batch_id;

    ImportedSource::PendingCooks pending = {
        .batch_id = batch_id,
        .generation = _sources.at(source_index).generation,
    };

    std::vector<ImporterTask> cook_tasks = {};
    cook_tasks.reserve(import_result.image_cooks.size() + import_result.mesh_cooks.size());

    // A cook is named by its index in these lists, which is what the target it left behind is indexed by -
    // that pairing is what lets a finished cook be resolved without ever having known a manifest index.
    for (u32 slot_index = 0; slot_index < s_cast<u32>(import_result.image_cooks.size()); ++slot_index)
    {
        ImageCookRequest & request = import_result.image_cooks[slot_index];
        // Bind every consuming slot to its stand-in, so a material is published sampling something real.
        // A slot with no stand-in available stays empty, which reads as "no such texture".
        for (MaterialTextureConsumer const & consumer : request.target.consumers)
        {
            std::optional<u32> const stand_in = placeholder_for_slot(consumer.slot);
            if (!stand_in.has_value()) { continue; }
            material_texture_binding(import_result.batch.materials.at(consumer.batch_material_index), consumer.slot) =
                SceneBatch::TextureBinding{.image = {.kind = SceneBatch::SceneRef::Kind::MANIFEST_ENTRY, .index = stand_in.value()}};
        }
        cook_tasks.push_back(ImporterTask{.data = ImporterTask::CookImage{
            .importer_data = std::move(request.importer_data),
            .slot = {.source_index = source_index, .batch_id = batch_id, .slot_index = slot_index},
        }});
        pending.image_cook_targets.push_back(std::move(request.target));
    }
    for (u32 slot_index = 0; slot_index < s_cast<u32>(import_result.mesh_cooks.size()); ++slot_index)
    {
        MeshCookRequest & request = import_result.mesh_cooks[slot_index];
        cook_tasks.push_back(ImporterTask{.data = ImporterTask::CookMesh{
            .importer_data = std::move(request.importer_data),
            .slot = {.source_index = source_index, .batch_id = batch_id, .slot_index = slot_index},
        }});
        pending.mesh_cook_targets.push_back(request.batch_mesh_index);
    }
    pending.outstanding_cooks = s_cast<u32>(cook_tasks.size());
    // Copied before the batch is moved away, which is the whole reason this runs before the push below.
    pending.materials = import_result.batch.materials;
    pending.mesh_lod_groups = import_result.batch.mesh_lod_groups;

    _sources.at(source_index).pending_cooks.push_back(std::move(pending));
    push_result(ImporterTaskResult{.data = std::move(import_result.batch)});
    push_tasks(cook_tasks);
}

auto Importer::find_pending_cooks(SourceSlot slot) -> ImportedSource::PendingCooks *
{
    std::vector<ImportedSource::PendingCooks> & publishes = _sources.at(slot.source_index).pending_cooks;
    auto const found = std::ranges::find_if(publishes,
        [&](ImportedSource::PendingCooks const & candidate) { return candidate.batch_id == slot.batch_id; });
    return found == publishes.end() ? nullptr : &*found;
}

void Importer::push_cook_finished(std::variant<ImageStreamerData, MeshStreamerData> streamer_data, SourceSlot slot)
{
    push_cook_result(std::move(streamer_data), slot, false);
}

void Importer::push_cook_result(std::variant<ImageStreamerData, MeshStreamerData> streamer_data, SourceSlot slot, bool failed)
{
    ImporterTask task = {.data = ImporterTask::CookFinished{.streamer_data = std::move(streamer_data), .slot = slot, .failed = failed}};
    push_tasks(std::span{&task, 1});
}

void Importer::resolve_cook(ImporterTask::CookFinished cook,
    std::unordered_map<u32, ImporterTaskResult::SceneMetadataBatch> & cook_batches)
{
    using SceneBatch = ImporterTaskResult::SceneMetadataBatch;

    ImportedSource::PendingCooks * pending = find_pending_cooks(cook.slot);
    // A publish is retired only once every cook it queued has resolved, so a live cook always finds one.
    DBG_ASSERT_TRUE_M(pending != nullptr, "A finished cook naming a publish that has been retired");
    if (pending == nullptr) { return; }

    // A cook the source has already moved past carries an artifact of content that is no longer there.
    if (pending->generation != _sources.at(cook.slot.source_index).generation)
    {
        pending->outstanding_cooks -= 1;
        return;
    }
    // A failed cook has nothing to say to the engine - its entry simply never becomes streamable - but it
    // still has to be retired here, and before the park below, since it needs nothing the publish is waiting
    // for. Resolved ahead of everything else because a placeholder that never settles holds every later
    // import for the rest of the process.
    if (cook.failed)
    {
        pending->outstanding_cooks -= 1;
        std::optional<MaterialTextureSlot> const placeholder_slot = placeholder_slot_for_source(cook.slot.source_index);
        if (placeholder_slot.has_value()) { fail_placeholder_slot(placeholder_slot.value(), "its cook failed"); }
        return;
    }
    // Nothing has applied this publish yet, which is the common case: a cache hit finishes long before the
    // engine's next poll can report where its entries landed.
    if (!pending->result.has_value())
    {
        _parked_cooks.push_back(std::move(cook));
        return;
    }

    auto const [iterator, inserted] = cook_batches.try_emplace(cook.slot.source_index);
    SceneBatch & batch = iterator->second;
    if (inserted)
    {
        batch.source_index = cook.slot.source_index;
        batch.batch_id = _next_batch_id++;
    }

    if (auto * mesh_streamer_data = std::get_if<MeshStreamerData>(&cook.streamer_data))
    {
        u32 const batch_mesh_index = pending->mesh_cook_targets.at(cook.slot.slot_index);
        // Re-emitted whole, because a modification carries every producer-owned field. The stored record's
        // references were normalized to manifest entries when the result arrived, so it is safe to send.
        SceneBatch::MeshLodGroup mesh = pending->mesh_lod_groups.at(batch_mesh_index);
        mesh.manifest_index = pending->result->mesh_manifest_indices.at(batch_mesh_index);
        mesh.streamer_data = std::move(*mesh_streamer_data);
        batch.mesh_lod_groups.push_back(std::move(mesh));
        pending->outstanding_cooks -= 1;
        return;
    }

    ImageCookTarget const & target = pending->image_cook_targets.at(cook.slot.slot_index);
    auto & image_streamer_data = std::get<ImageStreamerData>(cook.streamer_data);

    if (target.batch_image_index.has_value())
    {
        // Fills an entry the publish created empty because something referenced it from the parse.
        batch.images.push_back(SceneBatch::Image{
            .manifest_index = pending->result->image_manifest_indices.at(target.batch_image_index.value()),
            .name = target.name,
            .streamer_data = std::move(image_streamer_data),
        });
        pending->outstanding_cooks -= 1;
        return;
    }

    // A material still holding an unresolved rebind would be re-emitted carrying a batch element reference
    // that means nothing here, silently reverting the slot it names to its stand-in.
    bool const consumer_is_blocked = std::ranges::any_of(target.consumers, [&](MaterialTextureConsumer const & consumer)
    {
        return std::ranges::any_of(pending->pending_rebinds,
            [&](ImportedSource::PendingCooks::PendingRebind const & rebind) { return rebind.batch_material_index == consumer.batch_material_index; });
    });
    if (consumer_is_blocked)
    {
        _parked_cooks.push_back(std::move(cook));
        return;
    }

    u32 const batch_image_index = s_cast<u32>(batch.images.size());
    batch.images.push_back(SceneBatch::Image{
        .name = target.name,
        .streamer_data = std::move(image_streamer_data),
    });
    for (MaterialTextureConsumer const & consumer : target.consumers)
    {
        SceneBatch::Material material = pending->materials.at(consumer.batch_material_index);
        material.manifest_index = pending->result->material_manifest_indices.at(consumer.batch_material_index);
        material_texture_binding(material, consumer.slot) = SceneBatch::TextureBinding{
            .image = {.kind = SceneBatch::SceneRef::Kind::BATCH_ELEMENT, .index = batch_image_index},
        };
        batch.materials.push_back(std::move(material));

        pending->pending_rebinds.push_back(ImportedSource::PendingCooks::PendingRebind{
            .batch_material_index = consumer.batch_material_index,
            .slot = consumer.slot,
            .image_batch_id = batch.batch_id,
            .batch_image_index = batch_image_index,
        });
    }
    pending->outstanding_cooks -= 1;
}

void Importer::resolve_pending_rebinds(u32 source_index, ImporterTaskResult::AppliedBatch const & applied)
{
    using SceneBatch = ImporterTaskResult::SceneMetadataBatch;
    for (ImportedSource::PendingCooks & publish : _sources.at(source_index).pending_cooks)
    {
        std::erase_if(publish.pending_rebinds, [&](ImportedSource::PendingCooks::PendingRebind const & rebind)
        {
            if (rebind.image_batch_id != applied.batch_id) { return false; }
            material_texture_binding(publish.materials.at(rebind.batch_material_index), rebind.slot) =
                SceneBatch::TextureBinding{.image = {
                    .kind = SceneBatch::SceneRef::Kind::MANIFEST_ENTRY,
                    .index = applied.image_manifest_indices.at(rebind.batch_image_index),
                }};
            return true;
        });
    }
}

void Importer::retire_settled_publishes()
{
    for (ImportedSource & source : _sources)
    {
        std::erase_if(source.pending_cooks, [](ImportedSource::PendingCooks const & publish)
        {
            return publish.result.has_value() && publish.outstanding_cooks == 0 && publish.pending_rebinds.empty();
        });
    }
}

void Importer::publish_cook_batches(std::unordered_map<u32, ImporterTaskResult::SceneMetadataBatch> & cook_batches)
{
    for (auto & [source_index, batch] : cook_batches)
    {
        push_result(ImporterTaskResult{.data = std::move(batch)});
    }
    cook_batches.clear();
}

void Importer::record_applied_batch(ImporterTaskResult::AppliedBatch applied)
{
    using SceneBatch = ImporterTaskResult::SceneMetadataBatch;
    u32 const source_index = applied.source_index;

    // Rebinds already sent name their image as an element of this batch; this is where that becomes a
    // manifest index the row can keep re-emitting.
    resolve_pending_rebinds(source_index, applied);

    // A placeholder's entry is created by the batch carrying its finished cook, so this is where the editor
    // learns which entry stands in for a material texture slot. Its source is a single whole-file image, so
    // that batch's only creation is the one wanted.
    std::optional<MaterialTextureSlot> const placeholder_slot = placeholder_slot_for_source(source_index);
    if (placeholder_slot.has_value() && !applied.image_manifest_indices.empty())
    {
        std::optional<u32> & placeholder_manifest_index = _placeholder_manifest_indices.at(s_cast<usize>(placeholder_slot.value()));
        if (!placeholder_manifest_index.has_value()) { placeholder_manifest_index = applied.image_manifest_indices.front(); }
    }

    ImportedSource::PendingCooks * pending = find_pending_cooks(SourceSlot{.source_index = source_index, .batch_id = applied.batch_id});
    if (pending != nullptr)
    {
        // Normalize the stored records now that the entries they name exist: a batch element index means
        // nothing in the later batch that re-emits them.
        for (SceneBatch::MeshLodGroup & mesh : pending->mesh_lod_groups)
        {
            if (!mesh.material.has_value() || mesh.material->kind != SceneBatch::SceneRef::Kind::BATCH_ELEMENT) { continue; }
            mesh.material = SceneBatch::SceneRef{
                .kind = SceneBatch::SceneRef::Kind::MANIFEST_ENTRY,
                .index = applied.material_manifest_indices.at(mesh.material->index),
            };
        }
        DEBUG_MSG(fmt::format("[Importer] source {} batch {} applied: {} image, {} material, {} mesh entries",
            source_index, applied.batch_id, applied.image_manifest_indices.size(),
            applied.material_manifest_indices.size(), applied.mesh_manifest_indices.size()));
        pending->result = std::move(applied);
    }

    // Everything parked can now be retried; resolve_cook re-parks whatever still cannot proceed.
    std::vector<ImporterTask::CookFinished> parked = std::move(_parked_cooks);
    _parked_cooks.clear();
    std::unordered_map<u32, SceneBatch> cook_batches = {};
    for (ImporterTask::CookFinished & cook : parked)
    {
        resolve_cook(std::move(cook), cook_batches);
    }
    publish_cook_batches(cook_batches);

    retire_settled_publishes();

    // Requeued rather than dispatched here: dispatch_source_imports is the one place that decides what a
    // source import does, and it runs on the next drain of this same thread.
    release_parked_source_imports();
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

void Importer::thread_main()
{
    for (;;)
    {
        std::vector<ImporterTask> tasks = {};
        {
            std::unique_lock<std::mutex> lock{_queue_mutex};
            _wake_signal.wait(lock, [&]() { return _stop_requested || !_task_queue.empty(); });
            if (_stop_requested)
            {
                return; // Still-queued tasks are deliberately dropped on shutdown.
            }
            tasks = std::move(_task_queue);
            _task_queue.clear();
        }

        // Bookkeeping before work. A mapping landing in the same drain as cooks it resolves has to be
        // recorded first, or those cooks park for a frame they did not need to.
        std::erase_if(tasks, [&](ImporterTask & task) -> bool
        {
            auto * batch_applied = std::get_if<ImporterTask::BatchApplied>(&task.data);
            if (batch_applied == nullptr) { return false; }
            record_applied_batch(std::move(batch_applied->applied));
            return true;
        });
        std::erase_if(tasks, [&](ImporterTask & task) -> bool
        {
            auto * publish = std::get_if<ImporterTask::PublishImport>(&task.data);
            if (publish == nullptr) { return false; }
            publish_import_on_importer_thread(std::move(publish->import_result));
            return true;
        });
        // Accumulated across the drain so a cold cook of a large source produces one batch per source per
        // drain rather than one per finished cook.
        std::unordered_map<u32, ImporterTaskResult::SceneMetadataBatch> cook_batches = {};
        std::erase_if(tasks, [&](ImporterTask & task) -> bool
        {
            auto * cook_finished = std::get_if<ImporterTask::CookFinished>(&task.data);
            if (cook_finished == nullptr) { return false; }
            resolve_cook(std::move(*cook_finished), cook_batches);
            return true;
        });
        publish_cook_batches(cook_batches);
        retire_settled_publishes();

        // TODO(saky): TEMP HACK - Fix once threadpool has proper task priorities
        std::sort(tasks.begin(), tasks.end(), [](ImporterTask const & a, ImporterTask const & b) {
            auto get_priority = [](ImporterTask const & task) -> u32 {
                if (std::holds_alternative<ImporterTask::ImportSource>(task.data)) { return 0; }
                if (std::holds_alternative<ImporterTask::CookMesh>(task.data)) { return 1; }
                if (std::holds_alternative<ImporterTask::CookImage>(task.data)) { return 2; }
                return 3;
            };
            return get_priority(a) < get_priority(b);
        });

        for(auto const & task : tasks)
        {
            if (auto const * cook_image = std::get_if<ImporterTask::CookImage>(&task.data))
            {
                std::filesystem::path const & source_file = std::holds_alternative<ImageImporterData>(cook_image->importer_data)
                    ? std::get<ImageImporterData>(cook_image->importer_data).source_location.file
                    : std::get<VdbImporterData>(cook_image->importer_data).source_location.file;
                DEBUG_MSG(fmt::format("[Importer] dispatching image cook for '{}'", source_file.string()));
            }
            else if (auto const * cook_mesh = std::get_if<ImporterTask::CookMesh>(&task.data))
            {
                DEBUG_MSG(fmt::format("[Importer] dispatching mesh cook for '{}'", cook_mesh->importer_data.indices.location.file.string()));
            }
        }

        // Generic, backend-agnostic asset cooks first (they act on resolved ImporterData); the source backends
        // then consume what's left - the sources still to be resolved into slots.
        dispatch_asset_cooks(*this, tasks);
        dispatch_source_imports(tasks);
        DBG_ASSERT_TRUE_M(tasks.empty(), "An ImporterTask was left unconsumed - no importer handles its provenance");
    }
}
