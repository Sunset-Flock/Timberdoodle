#include "cook.hpp"

#include <algorithm>
#include <cmath>
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

// Drives one asset cook end to end: read and hash the sources, derive the artifact key from that hash, and
// either serve the artifact already sitting at the key or cook and write one. Everything type-specific (which
// sources to read, how to cook, which descriptor/streamer types) lives in the Policy; this skeleton is the
// same for images and meshes. The result is left on the task itself - nothing here knows the Importer exists.
template<typename Policy>
struct PolicyCookTask final : CookTask
{
    using ImporterData = typename Policy::ImporterData;
    using Descriptor = typename Policy::Descriptor;
    using StreamerData = typename Policy::StreamerData;

    ImporterData importer_data = {};

    PolicyCookTask(ImporterData importer_data, SourceIndentifier cook_slot)
        : importer_data{std::move(importer_data)}
    {
        identifier = cook_slot;
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

        if (cache_usable)
        {
            DEBUG_MSG(fmt::format("[{}] cache hit '{}' -> '{}'", Policy::LOG_TAG, artifact_name, tido_path.string()));
            streamer_data = StreamerData{
                .descriptor = cached_header->second,
                .bin_source = tido_path,
                .file_data_offset = cached_header_data->file_data_offset,
            };
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
        std::optional<StreamerData> const cooked_streamer_data = Policy::write_artifact(write_info, descriptor.value());
        if (!cooked_streamer_data.has_value())
        {
            DEBUG_MSG(fmt::format("[WARN][{}] failed to write .tido_bin for '{}'", Policy::LOG_TAG, artifact_name));
            return;
        }

        DEBUG_MSG(fmt::format("[{}] cooked '{}' ({}) -> '{}'", Policy::LOG_TAG, artifact_name,
            Policy::cooked_detail(descriptor.value()), cooked_streamer_data->bin_source.string()));
        streamer_data = cooked_streamer_data.value();
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
    using ReadState = std::vector<std::byte>;

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
        read_state = std::move(source_bytes);
        return content_hash;
    }

    static auto do_cook(ImporterData const & importer_data, ReadState const & read_state, std::vector<std::byte> & payload) -> std::optional<Descriptor>
    {
        std::optional<TidoImageWithData> cooked = {};
        switch (importer_data.container_format)
        {
            case ImageFileFormat::PNG:  cooked = cook_png_source(read_state, importer_data); break;
            case ImageFileFormat::KTX2: cooked = cook_ktx_source(read_state, importer_data); break;
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
    using ReadState = std::vector<std::byte>;

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
        read_state = std::move(source_bytes);
        return content_hash;
    }

    static auto do_cook(ImporterData const & importer_data, ReadState const & read_state, std::vector<std::byte> & payload) -> std::optional<Descriptor>
    {
        std::optional<TidoImageWithData> decoded = vdb_parse(VdbParseInfo{.src_data = read_state, .grid_names = importer_data.grid_names});
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

using ImageCookTask = PolicyCookTask<ImageCookPolicy>;
using MeshCookTask = PolicyCookTask<MeshCookPolicy>;
using VdbCookTask = PolicyCookTask<VdbCookPolicy>;

} // namespace

auto cook_image(ImageImporterData importer_data, SourceIndentifier slot) -> std::shared_ptr<CookTask>
{
    return std::make_shared<ImageCookTask>(std::move(importer_data), slot);
}

auto cook_mesh(MeshImporterData importer_data, SourceIndentifier slot) -> std::shared_ptr<CookTask>
{
    return std::make_shared<MeshCookTask>(std::move(importer_data), slot);
}

auto cook_vdb(VdbImporterData importer_data, SourceIndentifier slot) -> std::shared_ptr<CookTask>
{
    return std::make_shared<VdbCookTask>(std::move(importer_data), slot);
}
