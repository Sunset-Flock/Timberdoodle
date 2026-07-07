#include "tido_mesh.hpp"

#include <fstream>
#include <vector>

#include "tido_util.hpp"
#include "../optimizers/geometry_optimizer.hpp" // full ProcessedMesh (forward-declared in the header)

// glm::vec3 / vec2 are bit-identical to daxa_f32vec3 / vec2, so appending the cooked vertex arrays
// byte-for-byte (tido_append_array) matches the GPU mesh-buffer layout exactly.

auto write_mesh_tido(ProcessedMesh const & processed, std::filesystem::path const & cache_dir, std::string const & name, u64 cache_key) -> std::optional<TidoMeshCookResult>
{
    std::error_code ec = {};
    std::filesystem::create_directories(cache_dir, ec); // ignore "already exists"; the open below reports real failures

    TidoMeshCookResult result = {};
    result.cache_key = cache_key;
    result.descriptor.lod_count = processed.lod_count;

    // Build the .tido payload: one contiguous blob per LOD, the LOD's arrays packed back-to-back in the
    // SAME order make_resident_mesh packs the GPU mesh buffer (so the streamer can memcpy a blob straight
    // into a BDA buffer and wire the sub-pointers from the stored counts).
    std::vector<std::byte> payload = {};
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

        result.lods[lod] = {
            .blob_offset = blob_offset,
            .blob_byte_size = payload.size() - blob_offset,
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

    // Stem disambiguator is the mesh's unique source-identity key (gltf mesh/primitive index), NOT a
    // content hash: two distinct primitives with byte-identical geometry must NOT share a .tido path,
    // or the parallel cook tasks race on a single file (trunc-open while another reads). The key is also
    // deterministic from the source, so a re-cook overwrites the same file rather than orphaning it.
    std::string const stem = tido_stem(name, cache_key);
    std::filesystem::path const tido_path = cache_dir / (stem + ".tido");

    std::ofstream ofs{tido_path, std::ios::binary | std::ios::trunc};
    if (!ofs) { return std::nullopt; }
    ofs.write(r_cast<char const *>(payload.data()), s_cast<std::streamsize>(payload.size()));
    if (!ofs.good()) { return std::nullopt; }

    result.tido_path = tido_path;
    return result;
}
