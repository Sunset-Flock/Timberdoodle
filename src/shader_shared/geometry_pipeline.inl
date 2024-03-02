#pragma once

/// --- Goal of file ---
/// - should contain anything needed only for forward drawing
/// - should NOT contain things required for shading or post processing

#include <daxa/daxa.inl>

#include "shared.inl"
#include "geometry.inl"

/// --- Mesh Instance Draw List Begin ---

#define OPAQUE_DRAW_LIST_SOLID 0
#define OPAQUE_DRAW_LIST_MASKED 1
#define OPAQUE_DRAW_LIST_COUNT 2

struct MeshDrawTuple
{
    daxa_u32 entity_index;
    daxa_u32 mesh_index;
    daxa_u32 in_mesh_group_index;
};
DAXA_DECL_BUFFER_PTR_ALIGN(MeshDrawTuple, 4);

struct OpaqueMeshDrawListBufferHead
{
    daxa_u32 list_sizes[2];
    daxa_BufferPtr(MeshDrawTuple) mesh_draw_tuples[2];
}; 
DAXA_DECL_BUFFER_PTR_ALIGN(OpaqueMeshDrawListBufferHead, 8)

#if defined(__cplusplus)
#include <span>
inline auto make_opaque_draw_list_buffer_head(daxa::DeviceAddress address, std::array<std::span<MeshDrawTuple>, 2> draw_lists) -> OpaqueMeshDrawListBufferHead
{
    OpaqueMeshDrawListBufferHead ret = {};
    ret.list_sizes[0] = static_cast<daxa::u32>(draw_lists[0].size());
    ret.list_sizes[1] = static_cast<daxa::u32>(draw_lists[1].size());
    ret.mesh_draw_tuples[0] = address + sizeof(OpaqueMeshDrawListBufferHead);
    ret.mesh_draw_tuples[1] = address + sizeof(OpaqueMeshDrawListBufferHead) + sizeof(MeshDrawTuple) * draw_lists[0].size();
    return ret;
}
inline auto get_opaque_draw_list_buffer_size() -> daxa::usize
{
    return sizeof(OpaqueMeshDrawListBufferHead) + sizeof(MeshDrawTuple) * MAX_MESH_INSTANCES * 2;
}
#endif // #if defined(__cplusplus)

/// NOTE: In the future we want a TransparentMeshDrawListBufferHead, that has a much larger array for custom material permutations.

#if !defined(__cplusplus)
DAXA_DECL_BUFFER_REFERENCE_ALIGN(4) U32ArenaBufferRef
{
    daxa_u32 offsets_section_size;
    daxa_u32 bitfield_section_size;
    daxa_u32 uints[];
};
#endif // #if !defined(__cplusplus)

#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID (~0u)
#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED (~0u ^ 1u)
#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_DEBUG (~0u ^ 2u)

#define FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE (1u<<20u)

/// --- Mesh Instance Draw List End ---


/// --- Culling Arguments ---

struct MeshletCullIndirectArg
{
    daxa_u32 meshlet_indices_offset;
    daxa_u32 entity_index;
    daxa_u32 material_index;
    daxa_u32 mesh_index;
    daxa_u32 in_mesh_group_index; 
    // Usually identical to buckets meshlet per arg count.
    // Can be lower when a bigger bucket is used to cull the rest.
    // For example bucket 1<<4 used to cull 10 meshlets.
    daxa_u32 meshlet_count;
};
DAXA_DECL_BUFFER_PTR(MeshletCullIndirectArg)

// Table is set up in write command of cull_meshes.glsl.
struct MeshletCullArgBucketsBufferHead
{
    DispatchIndirectStruct commands[32];
    daxa_RWBufferPtr(MeshletCullIndirectArg) indirect_arg_ptrs[32];
    daxa_u32 indirect_arg_counts[32];
};
DAXA_DECL_BUFFER_PTR(MeshletCullArgBucketsBufferHead)

#if __cplusplus
inline auto meshlet_cull_arg_bucket_size(daxa_u32 max_meshes, daxa_u32 max_meshlets, daxa_u32 bucket) -> daxa_u32
{
    // round_up(div(max_meshlets,pow(2,i)))
    daxa_u32 const args_needed_for_max_meshlets_this_bucket = (max_meshlets + ((1 << bucket) - 1) ) >> bucket;
    // Min with max_meshes, as each mesh can write up most one arg into each bucket!
    return std::min(max_meshes, args_needed_for_max_meshlets_this_bucket) * static_cast<daxa_u32>(sizeof(MeshletCullIndirectArg));
}

inline auto meshlet_cull_arg_buckets_buffer_size(daxa_u32 max_meshes, daxa_u32 max_meshlets) -> daxa_u32
{
    daxa_u32 worst_case_size = {};
    for (daxa_u32 i = 0; i < 32; ++i)
    {
        worst_case_size += meshlet_cull_arg_bucket_size(max_meshes, max_meshlets, i);
    }
    return worst_case_size + static_cast<daxa_u32>(sizeof(MeshletCullArgBucketsBufferHead));
}
inline auto meshlet_cull_arg_buckets_buffer_make_head(daxa_u32 max_meshes, daxa_u32 max_meshlets, daxa_u64 address) -> MeshletCullArgBucketsBufferHead
{
    MeshletCullArgBucketsBufferHead ret = {};
    daxa_u32 current_buffer_offset = static_cast<daxa_u32>(sizeof(MeshletCullArgBucketsBufferHead));
    for (daxa_u32 i = 0; i < 32; ++i)
    {
        ret.commands[i] = { 0, 1, 1 };
        ret.indirect_arg_ptrs[i] = static_cast<daxa_u64>(current_buffer_offset) + address;
        current_buffer_offset += meshlet_cull_arg_bucket_size(max_meshes, max_meshlets, i);
    }
    return ret;
}
#endif

/// --- End Culling Arguments ---

/// --- Analyze Visbuffer Results Begin ---

// TODO: Convert into buffer head.
struct VisibleMeshletList
{
    daxa_u32 count;
    daxa_u32 meshlet_ids[MAX_MESHLET_INSTANCES];
};
DAXA_DECL_BUFFER_PTR(VisibleMeshletList)

/// --- Analyze Visbuffer Results End ---