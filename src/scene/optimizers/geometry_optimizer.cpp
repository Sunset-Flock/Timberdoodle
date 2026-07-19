#include "geometry_optimizer.hpp"

#include <cmath>
#include <bit>
#include <cstring>
#include <limits>
#include <array>

#include <meshoptimizer.h>

// Credit: https://github.com/zeux/meshoptimizer/blob/20787cc054fa0e46584c2791139c1878aa899d78/src/clusterizer.cpp#L715
static auto compute_bounding_sphere(glm::vec3 const * points, size_t count) -> BoundingSphere
{
    assert(count > 0);

    // find extremum points along all 3 axes; for each axis we get a pair of points with min/max coordinates
    size_t pmin[3] = {0, 0, 0};
    size_t pmax[3] = {0, 0, 0};

    for (size_t i = 0; i < count; ++i)
    {
        float const * p = &points[i].x;

        for (int axis = 0; axis < 3; ++axis)
        {
            pmin[axis] = (p[axis] < points[pmin[axis]][axis]) ? i : pmin[axis];
            pmax[axis] = (p[axis] > points[pmax[axis]][axis]) ? i : pmax[axis];
        }
    }

    // find the pair of points with largest distance
    float paxisd2 = 0;
    int paxis = 0;

    for (int axis = 0; axis < 3; ++axis)
    {
        float const * p1 = &points[pmin[axis]].x;
        float const * p2 = &points[pmax[axis]].x;

        float d2 = (p2[0] - p1[0]) * (p2[0] - p1[0]) + (p2[1] - p1[1]) * (p2[1] - p1[1]) + (p2[2] - p1[2]) * (p2[2] - p1[2]);

        if (d2 > paxisd2)
        {
            paxisd2 = d2;
            paxis = axis;
        }
    }

    // use the longest segment as the initial sphere diameter
    float const * p1 = &points[pmin[paxis]].x;
    float const * p2 = &points[pmax[paxis]].x;

    float center[3] = {(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2};
    float radius = sqrtf(paxisd2) / 2;

    // iteratively adjust the sphere up until all points fit
    for (size_t i = 0; i < count; ++i)
    {
        float const * p = &points[i].x;
        float d2 = (p[0] - center[0]) * (p[0] - center[0]) + (p[1] - center[1]) * (p[1] - center[1]) + (p[2] - center[2]) * (p[2] - center[2]);

        if (d2 > radius * radius)
        {
            float d = sqrtf(d2);
            assert(d > 0);

            float k = 0.5f + (radius / d) / 2;

            center[0] = center[0] * k + p[0] * (1 - k);
            center[1] = center[1] * k + p[1] * (1 - k);
            center[2] = center[2] * k + p[2] * (1 - k);
            radius = (radius + d) / 2;
        }
    }

    BoundingSphere ret = {};
    ret.center.x = center[0];
    ret.center.y = center[1];
    ret.center.z = center[2];
    ret.radius = radius;
    return ret;
}

auto mesh_parse(MeshParseInfo const & info) -> std::optional<RawMesh>
{
    // Indices must be an integral scalar stream; positions/normals/uvs must be F32.
    u64 index_element_size = 0;
    switch (info.indices.component_type)
    {
        case ComponentType::U16: index_element_size = sizeof(u16); break;
        case ComponentType::U32: index_element_size = sizeof(u32); break;
        case ComponentType::F32: return std::nullopt;
    }
    if (info.positions.component_type != ComponentType::F32 || info.normals.component_type != ComponentType::F32)
    {
        return std::nullopt;
    }

    // Tightly-packed streams: byte length must match the shared counts times the element size.
    bool const has_uvs = !info.uvs.data.empty();
    if (info.indices.data.size() != info.index_count * index_element_size ||
        info.positions.data.size() != info.vertex_count * sizeof(glm::vec3) ||
        info.normals.data.size() != info.vertex_count * sizeof(glm::vec3))
    {
        return std::nullopt;
    }
    if (has_uvs && (info.uvs.component_type != ComponentType::F32 || info.uvs.data.size() != info.vertex_count * sizeof(glm::vec2)))
    {
        return std::nullopt;
    }

    RawMesh raw = {};

    raw.indices.resize(info.index_count);
    if (info.indices.component_type == ComponentType::U16)
    {
        // memcpy through a narrow buffer first - the source bytes are not guaranteed u16-aligned.
        std::vector<u16> narrow_indices(info.index_count);
        std::memcpy(narrow_indices.data(), info.indices.data.data(), info.indices.data.size());
        for (u32 index = 0; index < info.index_count; ++index)
        {
            raw.indices[index] = s_cast<u32>(narrow_indices[index]);
        }
    }
    else
    {
        std::memcpy(raw.indices.data(), info.indices.data.data(), info.indices.data.size());
    }

    raw.positions.resize(info.vertex_count);
    std::memcpy(raw.positions.data(), info.positions.data.data(), info.positions.data.size());

    raw.normals.resize(info.vertex_count);
    std::memcpy(raw.normals.data(), info.normals.data.data(), info.normals.data.size());

    if (has_uvs)
    {
        raw.uvs.resize(info.vertex_count);
        std::memcpy(raw.uvs.data(), info.uvs.data.data(), info.uvs.data.size());
    }

    return raw;
}

auto optimize_mesh(RawMesh const & raw) -> ProcessedMesh
{
    std::vector<u32> lod0_index_buffer = raw.indices;
    std::vector<glm::vec3> vert_positions = raw.positions;
    std::vector<glm::vec3> vert_normals = raw.normals;
    bool const has_uv = !raw.uvs.empty();
    std::vector<glm::vec2> vert_texcoord0 = has_uv ? raw.uvs : std::vector<glm::vec2>(vert_positions.size());
    u64 const vertex_count = vert_positions.size();

    /// NOTE: Generate meshlets:
    constexpr usize MAX_VERTICES = MAX_VERTICES_PER_MESHLET;
    constexpr usize MAX_TRIANGLES = MAX_TRIANGLES_PER_MESHLET;
    // No clue what cone culling is.
    constexpr float CONE_WEIGHT = 1.0f;
    // TODO: Make this optimization optional!
    {
        std::vector<u32> optimized_indices(lod0_index_buffer.size());
        meshopt_optimizeVertexCache(optimized_indices.data(), lod0_index_buffer.data(), lod0_index_buffer.size(), vertex_count);
        lod0_index_buffer = std::move(optimized_indices);
    }

    ProcessedMesh processed = {};

    /// ===== Calculate Normalized Vertex Distance =====
    // - When simplifying/ generating lods mesh optimizer takes into account position error AND attribute error (in our case normals)
    // - We need to give the attributes a meaningful weight to mix position and normal errorion
    // - The smaller the triangles, the less the normals matter.
    // - A good estimation for visual impact of normals is their distortion multiplied with the average vertex distance
    // - This is because the normals will not change the visuals past vertex distance, as each vertex normal can only effect the space between two vertices.
    // - We calculate the average vertex distance for lod0 and then estimate the vertex distance for other lods with a heuristic to speed it up.
    f32 modelspace_average_vertex_distance = 0.0f;
    glm::vec3 vertices_max = vert_positions[lod0_index_buffer[0]];
    glm::vec3 vertices_min = vert_positions[lod0_index_buffer[0]];
    for (u32 tri = 0; tri < lod0_index_buffer.size() / 3; ++tri)
    {
        glm::vec3 c0 = vert_positions[lod0_index_buffer[tri * 3 + 0]];
        glm::vec3 c1 = vert_positions[lod0_index_buffer[tri * 3 + 1]];
        glm::vec3 c2 = vert_positions[lod0_index_buffer[tri * 3 + 2]];
        vertices_max = glm::max(glm::max(vertices_max, c0), glm::max(c1, c2));
        vertices_min = glm::min(glm::min(vertices_min, c0), glm::min(c1, c2));
        f32 const e0_dst = glm::length(c0 - c1);
        f32 const e1_dst = glm::length(c1 - c2);
        f32 const e2_dst = glm::length(c2 - c0);
        f32 const max_edge = std::max(std::max(e0_dst, e1_dst), e2_dst);
        modelspace_average_vertex_distance += max_edge;
    }
    modelspace_average_vertex_distance /= static_cast<f32>(lod0_index_buffer.size() / 3ull);
    glm::vec3 const vertex_bounds_size = vertices_max - vertices_min;
    f32 const vertex_bounds_scale = std::max(vertex_bounds_size.x, std::max(vertex_bounds_size.y, vertex_bounds_size.z));
    f32 const normalized_average_vertex_distance = modelspace_average_vertex_distance / vertex_bounds_scale;
    f32 const lod0_average_vertex_distance = normalized_average_vertex_distance;
    /// ===== Calculate Normalized Vertex Distance =====

    std::vector<f32> attributes_normals_uvs = {};
    attributes_normals_uvs.resize(vertex_count * 5);
    for (u32 i = 0; i < vertex_count; ++i)
    {
        attributes_normals_uvs[i * 5 + 0] = vert_normals[i].x;
        attributes_normals_uvs[i * 5 + 1] = vert_normals[i].y;
        attributes_normals_uvs[i * 5 + 2] = vert_normals[i].z;
        if (has_uv)
        {
            attributes_normals_uvs[i * 5 + 3] = vert_texcoord0[i].x;
            attributes_normals_uvs[i * 5 + 4] = vert_texcoord0[i].y;
        }
    }

    std::vector<daxa::u32> prev_lod_index_buffer = {};
    for (u32 lod = 0; lod < MAX_MESHES_PER_LOD_GROUP; ++lod)
    {
        std::vector<u32> simplified_indices = {};
        std::vector<u32> * index_buffer = {};
        f32 lod_error = 0.0f;
        if (lod == 0)
        {
            index_buffer = &lod0_index_buffer;
        }
        else
        {
#pragma region LOD_GENERATION
            const u32 lod_index_count = round_up_div(s_cast<u32>(prev_lod_index_buffer.size()), 3 * 2) * 3u;
            simplified_indices.resize(prev_lod_index_buffer.size(), 0u); // Mesh optimizer needs them to be this large for some reason....
            index_buffer = &simplified_indices;
            f32 target_error = std::numeric_limits<f32>::max();
            f32 max_acceptable_error = 0.5f;
            // TODO: Only enable this for meshes that really need it!
            // It completely prevents foliage optimization and we desperately need foliage optimization!
            // It worsenes performance a lot
            // It should only be on for things that need it like street tiles or planes.
            u32 options = meshopt_SimplifyLockBorder;
            f32 result_error = {};

            /// INFO:       Meshoptimizer changed how the attributes are weighted
            ///             Must determine a new scaling for normals and or modify meshopt for optimal normal weighting.
            ///             For now use fixed weights (they work much nicer with recent changes)
            /// ===== Estimate Average Vertex Distance For LOD ====
            // We assume a simplification rate that halves triangles from lod to lod.
            // In this case, the average vertex distance increases at a rate of sqrt(2) per lod.
            // This gives us this vertex distance estimation function: lod_vertex_distance * sqrt(2)^lod
            // This is intuitive when thinking of merging two equirectangular triangles into one,
            //         x --                              x --
            //      x  x  x  len: sqrt(2)    ==>      x     x  len: sqrt(2)
            //   x     x    x --             ==>   x          x --
            // xxxxxxxxxxxxxxxxx                 xxxxxxxxxxxxxxxxx
            // |    len: 1     |                 |    len: 1     |
            // In this case the two longest edges before simplification are len sqrt(2)
            // The longest edge of the simplified triangle is len 2.
            [[maybe_unused]] f32 const lod_average_normalized_vertex_distance = lod0_average_vertex_distance * s_cast<f32>(std::pow(sqrt(2.0f), lod));
            // - We bias the weight towards the normal a little here with a factor of 2
            // - Typically normals are a little more important for visual error than position as they effect the shading more.
            f32 const MESH_LOD_GEN_NORMAL_IMPORTANCE_FACTOR = 3.0f;
            f32 const lod_normal_weight = MESH_LOD_GEN_NORMAL_IMPORTANCE_FACTOR;
            /// ===== Estimate Average Vertex Distance For LOD ====

            // ===== TexCoord Weight =====
            // Some meshes use complex uvs to save on texture memory space.
            // One such complex uvs would be to MIRROR the uv on some vertex.
            // Such vertices often lie on a flat plane between tringles
            // If meshoptimizer does not know about uvs, it will simply remove these vertices.
            // This is because these verts are usually of very low geometric value.
            //
            // Give uvs a small weight to that extreme uv distortions are prevented in simplification.
            f32 const TEXCOORD_WEIGHT = 1.0f;

            f32 attribute_weights[] = {lod_normal_weight, lod_normal_weight, lod_normal_weight, TEXCOORD_WEIGHT, TEXCOORD_WEIGHT};
            u64 result_index_count = meshopt_simplifyWithAttributes(
                index_buffer->data(), prev_lod_index_buffer.data(), prev_lod_index_buffer.size(),
                &vert_positions.data()->x, vert_positions.size(), sizeof(glm::vec3),
                attributes_normals_uvs.data(), sizeof(f32) * 5, attribute_weights, 5, nullptr,
                lod_index_count, target_error, options, &result_error);
            result_error *= (1.0f / (1.0f + MESH_LOD_GEN_NORMAL_IMPORTANCE_FACTOR)); // renormalize error based on normal importance boost
            lod_error = processed.lods[lod - 1].lod_error + result_error;
            if (result_index_count > (lod_index_count + lod_index_count / 2) || result_index_count < 12 || result_error > max_acceptable_error)
            {
                break;
            }
            index_buffer->resize(result_index_count);
#pragma endregion
        }
        prev_lod_index_buffer = *index_buffer;

#pragma region MESH OPTIMIZATION
        std::vector<u32> vertex_remap = {};
        std::vector<u32> remapped_index_buffer = {};
        vertex_remap.resize(vert_positions.size());
        remapped_index_buffer.resize(index_buffer->size());

        usize unique_vertices = meshopt_optimizeVertexFetchRemap(vertex_remap.data(), index_buffer->data(), index_buffer->size(), vert_positions.size());
        std::vector<glm::vec3> remapped_vert_positions = {};
        std::vector<glm::vec3> remapped_vert_normals = {};
        std::vector<glm::vec2> remapped_vert_texcoord0 = {};
        remapped_vert_positions.resize(unique_vertices);
        remapped_vert_normals.resize(unique_vertices);
        if (has_uv) { remapped_vert_texcoord0.resize(unique_vertices); }

        meshopt_remapIndexBuffer(remapped_index_buffer.data(), index_buffer->data(), index_buffer->size(), vertex_remap.data());
        index_buffer = &remapped_index_buffer;
        meshopt_remapVertexBuffer(remapped_vert_positions.data(), &vert_positions[0].x, vert_positions.size(), sizeof(glm::vec3), vertex_remap.data());
        meshopt_remapVertexBuffer(remapped_vert_normals.data(), &vert_normals[0].x, vert_normals.size(), sizeof(glm::vec3), vertex_remap.data());
        if (has_uv) { meshopt_remapVertexBuffer(remapped_vert_texcoord0.data(), &vert_texcoord0[0].x, vert_texcoord0.size(), sizeof(glm::vec2), vertex_remap.data()); }

        std::vector<u32> optimized_indices = {};
        optimized_indices.resize(index_buffer->size());
        meshopt_optimizeVertexCache(optimized_indices.data(), index_buffer->data(), index_buffer->size(), unique_vertices);
        index_buffer = &optimized_indices;

        std::vector<glm::vec3> & optimized_vert_positions = remapped_vert_positions;
        std::vector<glm::vec3> & optimized_vert_normals = remapped_vert_normals;
        std::vector<glm::vec2> & optimized_vert_texcoord0 = remapped_vert_texcoord0;
#pragma endregion

#pragma region MESHLET GENERATION
        size_t max_meshlets = meshopt_buildMeshletsBound(index_buffer->size(), MAX_VERTICES, MAX_TRIANGLES);
        std::vector<meshopt_Meshlet> meshlets(max_meshlets);
        std::vector<u32> meshlet_indirect_vertices(max_meshlets * MAX_VERTICES);
        std::vector<u8> meshlet_micro_indices(max_meshlets * MAX_TRIANGLES * 3);
        size_t meshlet_count = meshopt_buildMeshlets(
            meshlets.data(),
            meshlet_indirect_vertices.data(),
            meshlet_micro_indices.data(),
            index_buffer->data(),
            index_buffer->size(),
            r_cast<float *>(optimized_vert_positions.data()),
            s_cast<usize>(unique_vertices),
            sizeof(glm::vec3),
            MAX_VERTICES,
            MAX_TRIANGLES,
            CONE_WEIGHT);
        // TODO: Compute OBBs
        std::vector<BoundingSphere> meshlet_bounds(meshlet_count);
        std::vector<AABB> meshlet_aabbs(meshlet_count);
        glm::vec3 mesh_min_pos;
        glm::vec3 mesh_max_pos;
        for (size_t meshlet_index = 0; meshlet_index < meshlet_count; ++meshlet_index)
        {
            meshopt_Bounds raw_bounds = meshopt_computeMeshletBounds(
                &meshlet_indirect_vertices[meshlets[meshlet_index].vertex_offset],
                &meshlet_micro_indices[meshlets[meshlet_index].triangle_offset],
                meshlets[meshlet_index].triangle_count,
                r_cast<float *>(optimized_vert_positions.data()),
                s_cast<usize>(unique_vertices),
                sizeof(glm::vec3));
            meshlet_bounds[meshlet_index].center.x = raw_bounds.center[0];
            meshlet_bounds[meshlet_index].center.y = raw_bounds.center[1];
            meshlet_bounds[meshlet_index].center.z = raw_bounds.center[2];
            meshlet_bounds[meshlet_index].radius = raw_bounds.radius;

            glm::vec3 min_pos = optimized_vert_positions[meshlet_indirect_vertices[meshlets[meshlet_index].vertex_offset]];
            glm::vec3 max_pos = optimized_vert_positions[meshlet_indirect_vertices[meshlets[meshlet_index].vertex_offset]];

            if (meshlet_index == 0)
            {
                mesh_min_pos = optimized_vert_positions[meshlet_indirect_vertices[meshlets[0].vertex_offset]];
                mesh_max_pos = optimized_vert_positions[meshlet_indirect_vertices[meshlets[0].vertex_offset]];
            }

            for (u32 vertex_index = 1; vertex_index < meshlets[meshlet_index].vertex_count; ++vertex_index)
            {
                glm::vec3 pos = optimized_vert_positions[meshlet_indirect_vertices[meshlets[meshlet_index].vertex_offset + vertex_index]];
                min_pos = glm::min(min_pos, pos);
                max_pos = glm::max(max_pos, pos);
            }
            mesh_min_pos = glm::min(mesh_min_pos, min_pos);
            mesh_max_pos = glm::max(mesh_max_pos, max_pos);

            meshlet_aabbs[meshlet_index].center = std::bit_cast<daxa_f32vec3>((max_pos + min_pos) * 0.5f);
            meshlet_aabbs[meshlet_index].size = std::bit_cast<daxa_f32vec3>(max_pos - min_pos);
        }
        AABB mesh_aabb;
        mesh_aabb.center = std::bit_cast<daxa_f32vec3>((mesh_max_pos + mesh_min_pos) * 0.5f);
        mesh_aabb.size = std::bit_cast<daxa_f32vec3>(mesh_max_pos - mesh_min_pos);

        // Trimm array sizes.
        meshopt_Meshlet const & last = meshlets[meshlet_count - 1];
        meshlet_indirect_vertices.resize(last.vertex_offset + last.vertex_count);
        meshlet_micro_indices.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
        meshlets.resize(meshlet_count);
#pragma endregion

        // Pack the cooked LOD into CPU memory. The scene packs these arrays into a single GPU buffer
        // at upload time (mirroring the GPUMesh BDA layout).
        ProcessedMeshLod & out = processed.lods[lod];
        out.lod_error = lod_error;
        out.aabb = mesh_aabb;
        out.bounding_sphere = compute_bounding_sphere(optimized_vert_positions.data(), optimized_vert_positions.size());
        out.vertex_count = s_cast<u32>(unique_vertices);
        out.primitive_count = s_cast<u32>(index_buffer->size() / 3);

        out.meshlets.resize(meshlet_count);
        for (size_t i = 0; i < meshlet_count; ++i)
        {
            // meshopt_Meshlet is layout-compatible with Meshlet; copy fields explicitly for clarity.
            out.meshlets[i].indirect_vertex_offset = meshlets[i].vertex_offset;
            out.meshlets[i].micro_indices_offset = meshlets[i].triangle_offset;
            out.meshlets[i].vertex_count = meshlets[i].vertex_count;
            out.meshlets[i].triangle_count = meshlets[i].triangle_count;
        }
        out.meshlet_bounds = std::move(meshlet_bounds);
        out.meshlet_aabbs = std::move(meshlet_aabbs);

        while ((meshlet_micro_indices.size() % 4) != 0)
        {
            meshlet_micro_indices.push_back({});
        }
        out.micro_indices = std::move(meshlet_micro_indices);
        out.indirect_vertices = std::move(meshlet_indirect_vertices);
        out.primitive_indices.assign(index_buffer->begin(), index_buffer->end());
        out.vertex_positions = std::move(optimized_vert_positions);
        if (has_uv) { out.vertex_uvs = std::move(optimized_vert_texcoord0); }
        out.vertex_normals = std::move(optimized_vert_normals);

        processed.lod_count += 1;
    }

    return processed;
}
