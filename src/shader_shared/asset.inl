#pragma once

#include <daxa/daxa.inl>
#include "shared.inl"

#define INVALID_MESHLET_INDEX (~(0u))

// MUST never be greater then 124!
#define MAX_TRIANGLES_PER_MESHLET (64)

// MUST never be greater then MAX_TRIANGLES_PER_MESHLET * 3!
#define MAX_VERTICES_PER_MESHLET (64)

#define ENTITY_MESHLET_VISIBILITY_ARENA_SIZE (1<<20)
#define ENTITY_MESHLET_VISIBILITY_ARENA_BIT_SIZE (ENTITY_MESHLET_VISIBILITY_ARENA_SIZE * 8)
#define ENTITY_MESHLET_VISIBILITY_ARENA_UINT_SIZE (ENTITY_MESHLET_VISIBILITY_ARENA_SIZE / 4)

#define MAX_MESHES_PER_MESHGROUP 7

#if defined(DAXA_SHADER) && DAXA_SHADER
uint triangle_mask_bit_from_triangle_index(uint triangle_index)
{
    #if MAX_TRIANGLES_PER_MESHLET > 64
        return 1u << (triangle_index >> 2u);
    #elif MAX_TRIANGLES_PER_MESHLET > 32
        return 1u << (triangle_index >> 1u);
    #else
        return 1u << triangle_index;
    #endif
}
#endif // #if defined(DAXA_SHADER) && DAXA_SHADER

// Used to tell threads in the meshlet cull dispatch what to work on.
struct MeshletCullIndirectArg
{
    daxa_u32 entity_index;
    daxa_u32 material_index;
    daxa_u32 meshlet_indices_offset;
    daxa_u32 mesh_index;
    daxa_u32 in_meshgroup_index; 
};
DAXA_DECL_BUFFER_PTR(MeshletCullIndirectArg)

// Table is set up in write command of cull_meshes.glsl.
struct MeshletCullIndirectArgTable
{
    daxa_RWBufferPtr(MeshletCullIndirectArg) indirect_arg_ptrs[32];
    daxa_u32 indirect_arg_counts[32];
};
DAXA_DECL_BUFFER_PTR(MeshletCullIndirectArgTable)

// !!NEEDS TO BE ABI COMPATIBLE WITH meshopt_Meshlet!!
struct Meshlet
{
    // Offset into the meshs vertex index array.
    daxa_u32 indirect_vertex_offset;
    // Equivalent to meshoptimizers triangle_offset.
    // Renamed the field for more clarity.
    daxa_u32 micro_indices_offset;
    daxa_u32 vertex_count;
    daxa_u32 triangle_count;
};
DAXA_DECL_BUFFER_PTR(Meshlet)

struct MeshletInstance
{
    daxa_u32 entity_index;
    daxa_u32 material_index;        // Can pack more data into this
    daxa_u32 meshlet_index;         // Can pack more data into this
    daxa_u32 mesh_index;
    daxa_u32 in_meshgroup_index; 
};

struct PackedMeshletInstance
{
    daxa_u32vec4 value;
};
DAXA_DECL_BUFFER_PTR_ALIGN(PackedMeshletInstance, 16) // Aligned for faster loads and stores.

SHARED_FUNCTION PackedMeshletInstance pack_meshlet_instance(MeshletInstance meshlet_instance)
{
    PackedMeshletInstance ret;
    ret.value.x = meshlet_instance.entity_index;
    ret.value.y = meshlet_instance.material_index;
    ret.value.z = meshlet_instance.meshlet_index; 
    ret.value.w = (meshlet_instance.mesh_index << 3) | (meshlet_instance.in_meshgroup_index & 0x7);
    return ret;
}

SHARED_FUNCTION MeshletInstance unpack_meshlet_instance(PackedMeshletInstance packed_meshlet_instance)
{
    MeshletInstance ret;
    ret.entity_index = packed_meshlet_instance.value.x;
    ret.material_index = packed_meshlet_instance.value.y;
    ret.meshlet_index = packed_meshlet_instance.value.z;
    ret.mesh_index = packed_meshlet_instance.value.w >> 3;
    ret.in_meshgroup_index = packed_meshlet_instance.value.w & 0x7;
    return ret;
}

struct BoundingSphere
{
    daxa_f32vec3 center;
    daxa_f32 radius;
};
DAXA_DECL_BUFFER_PTR(BoundingSphere)

#if 0
#define DEBUG_VERTEX_ID 1
void encode_vertex_id(daxa_u32 instantiated_meshlet_index, daxa_u32 triangle_index, daxa_u32 triangle_corner, out daxa_u32 vertex_id)
{
    #if DEBUG_VERTEX_ID
    vertex_id = instantiated_meshlet_index * 10000 + triangle_index * 10 + triangle_corner;
    #else
    vertex_id = (instantiated_meshlet_index << 9) | (triangle_index << 2) | triangle_corner;
    #endif
}
void decode_vertex_id(daxa_u32 vertex_id, out daxa_u32 instantiated_meshlet_index, out daxa_u32 triangle_index, out daxa_u32 triangle_corner)
{
    #if DEBUG_VERTEX_ID
    instantiated_meshlet_index = vertex_id / 10000;
    triangle_index = (vertex_id / 10) % 1000;
    triangle_corner = vertex_id % 10;
    #else
    instantiated_meshlet_index = vertex_id >> 9;
    triangle_index = (vertex_id >> 2) & 0x3F;
    triangle_corner = vertex_id & 0x3f;
    #endif
}
#endif // #if defined(DAXA_SHADER)

// TODO:
// REMOVE GPUMesh In favor of GPUMeshDescriptor
struct GPUMesh
{
    daxa_BufferId mesh_buffer;
    daxa_u32 material_index;
    daxa_u32 meshlet_count;
    daxa_u32 vertex_count;
    daxa_BufferPtr(Meshlet) meshlets;
    daxa_BufferPtr(BoundingSphere) meshlet_bounds;
    daxa_BufferPtr(daxa_u32) micro_indices;
    daxa_BufferPtr(daxa_u32) indirect_vertices;
    daxa_BufferPtr(daxa_f32vec3) vertex_positions;
    daxa_BufferPtr(daxa_f32vec2) vertex_uvs;
};
DAXA_DECL_BUFFER_PTR_ALIGN(GPUMesh, 8)

struct GPUMeshDescriptor
{
    daxa_BufferId mesh_buffer;
    daxa_u32 material_index;
    daxa_u32 meshlet_count;
    daxa_u32 vertex_count;
    daxa_u32 offset_meshlets;          // daxa_BufferPtr(Meshlet)
    daxa_u32 offset_meshlet_bounds;    // daxa_BufferPtr(BoundingSphere)
    daxa_u32 offset_micro_indices;     // daxa_BufferPtr(daxa_u32)
    daxa_u32 offset_indirect_vertices; // daxa_BufferPtr(daxa_u32)
    daxa_u32 offset_vertex_positions;  // daxa_BufferPtr(daxa_f32vec3)
};
DAXA_DECL_BUFFER_PTR_ALIGN(GPUMeshDescriptor, 8)

#if DAXA_SHADER
uint get_micro_index(daxa_BufferPtr(daxa_u32) micro_indices, daxa_u32 index_offset)
{
    uint pack_index = index_offset / 4;
    uint index_pack = deref(micro_indices[pack_index]);
    uint in_pack_offset = index_offset % 4;
    uint in_pack_shift = in_pack_offset * 8;
    return (index_pack >> in_pack_shift) & 0xFF;
}
#endif // #if defined(DAXA_SHADER)

struct GPUMeshGroup
{
    daxa_u32 mesh_manifest_indices[MAX_MESHES_PER_MESHGROUP];
    daxa_u32 count;
};
DAXA_DECL_BUFFER_PTR(GPUMeshGroup)

struct MeshletInstances
{
    daxa_u32 first_count;
    daxa_u32 second_count;
    PackedMeshletInstance meshlets[MAX_MESHLET_INSTANCES];
};
DAXA_DECL_BUFFER_PTR(MeshletInstances)

struct VisibleMeshletList
{
    daxa_u32 count;
    daxa_u32 meshlet_ids[MAX_MESHLET_INSTANCES];
};
DAXA_DECL_BUFFER_PTR(VisibleMeshletList)

struct EntityMeshletVisibilityBitfieldOffsets
{
    daxa_u32 mesh_bitfield_offset[MAX_MESHES_PER_MESHGROUP];
    daxa_u32 padd;
};
DAXA_DECL_BUFFER_PTR(EntityMeshletVisibilityBitfieldOffsets)

#define ENT_MESHLET_VIS_OFFSET_UNALLOCATED (~0u)
#define ENT_MESHLET_VIS_OFFSET_EMPTY ((~0u) ^ 1u)

#if defined(__cplusplus)
static_assert(ENT_MESHLET_VIS_OFFSET_EMPTY != ENT_MESHLET_VIS_OFFSET_UNALLOCATED);
#endif

#if defined(DAXA_SHADER) && DAXA_SHADER
DAXA_DECL_BUFFER_REFERENCE EntityMeshletVisibilityBitfieldOffsetsView
{
    queuefamilycoherent daxa_u32 back_offset;
    queuefamilycoherent EntityMeshletVisibilityBitfieldOffsets entity_offsets[];
};
#endif

#define INVALID_MANIFEST_INDEX (~0u)