#pragma once

/// --- Goal of file ---
/// - should contain anything needed only for forward drawing
/// - should NOT contain things required for shading or post processing

#include "daxa/daxa.inl"

#include "shared.inl"
#include "geometry.inl"

/// --- Mesh Instance Draw List Begin ---

#define DRAW_LIST_OPAQUE 0
#define DRAW_LIST_MASKED 1
#define DRAW_LIST_TYPES 2

struct MeshDrawList
{
    daxa_u32 count;
    daxa_RWBufferPtr(daxa_u32) instances;
};

#define MESH_INSTANCE_FLAG_OPAQUE (1 << 0)
#define MESH_INSTANCE_FLAG_MASKED (1 << 1)

struct MeshInstance
{
    daxa_u32 entity_index;
    daxa_u32 mesh_index;
    daxa_u32 in_mesh_group_index;
    daxa_u32 flags;
};
DAXA_DECL_BUFFER_PTR_ALIGN(MeshInstance, 4);

struct OpaqueMeshInstancesBufferHead
{   
    daxa_u32 count;
    daxa_BufferPtr(MeshInstance) instances;
    MeshDrawList draw_lists[2];
}; 
DAXA_DECL_BUFFER_PTR_ALIGN(OpaqueMeshInstancesBufferHead, 8)

#if defined(__cplusplus)
#include <span>
inline void fill_opaque_draw_list_buffer_head(daxa::DeviceAddress address, uint8_t* host_address, std::array<std::span<MeshInstance const>, 2> draw_lists)
{
    OpaqueMeshInstancesBufferHead ret = {};
    auto device_address_back_offset = address + sizeof(OpaqueMeshInstancesBufferHead);
    ret.instances = device_address_back_offset;
    device_address_back_offset += sizeof(MeshInstance) * MAX_MESH_INSTANCES;
    for (uint32_t draw_list = 0; draw_list < DRAW_LIST_TYPES; ++draw_list)
    {
        ret.draw_lists[draw_list].instances = device_address_back_offset;
        device_address_back_offset += sizeof(daxa_u32) * MAX_MESH_INSTANCES;
        for (uint32_t element = 0; element < draw_lists[draw_list].size(); ++element)
        {
            uint32_t mesh_instance_index = ret.count++;
            uint32_t draw_list_element_index = ret.draw_lists[draw_list].count++;
            MeshInstance mesh_instance = (draw_lists[draw_list])[element];
            mesh_instance.flags = mesh_instance.flags | (draw_list == DRAW_LIST_OPAQUE ? MESH_INSTANCE_FLAG_OPAQUE : 0);
            reinterpret_cast<MeshInstance*>(host_address + sizeof(OpaqueMeshInstancesBufferHead))[mesh_instance_index] = mesh_instance;
            reinterpret_cast<uint32_t*>(
                host_address + sizeof(OpaqueMeshInstancesBufferHead) + 
                sizeof(MeshInstance) * MAX_MESH_INSTANCES + 
                sizeof(uint32_t) * MAX_MESH_INSTANCES * draw_list)[element] = mesh_instance_index;
        }
    }
    *reinterpret_cast<OpaqueMeshInstancesBufferHead*>(host_address) = ret;
}
inline auto get_opaque_draw_list_buffer_size() -> daxa::usize
{
    return 
        sizeof(OpaqueMeshInstancesBufferHead) + 
        sizeof(MeshInstance) * MAX_MESH_INSTANCES +
        sizeof(uint32_t) * MAX_MESH_INSTANCES * 2;
}
#endif // #if defined(__cplusplus)

/// NOTE: In the future we want a TransparentMeshDrawListBufferHead, that has a much larger array for custom material permutations.

#if defined(DAXA_SHADER)
#if (DAXA_SHADERLANG == DAXA_SHADERLANG_GLSL)

DAXA_DECL_BUFFER_REFERENCE_ALIGN(4) U32ArenaBufferRef
{
    daxa_u32 offsets_section_size;
    daxa_u32 bitfield_section_size;
    daxa_u32 uints[];
};

#else

#define U32ArenaBufferRef U32ArenaBuffer*

struct U32ArenaBuffer
{
    daxa_u32 offsets_section_size;
    daxa_u32 bitfield_section_size;
    daxa_u32 uints[1];
};

#endif 
#endif // #if !defined(__cplusplus)

#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_INVALID (~0u)
#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_LOCKED (~0u ^ 1u)
#define FIRST_PASS_MESHLET_BITFIELD_OFFSET_DEBUG (~0u ^ 2u)

#define FIRST_OPAQUE_PASS_BITFIELD_ARENA_U32_SIZE (1u<<22u)

/// --- Mesh Instance Draw List End ---

/// --- Analyze Visbuffer Results Begin ---

// TODO: Convert into buffer head.
#if !defined(__cplusplus)
struct VisibleMeshletList
{
    daxa_u32 count;
    daxa_u32vec3 padd;
    daxa_u32 meshlet_ids[1];
};
DAXA_DECL_BUFFER_PTR(VisibleMeshletList)
#endif

struct VisibleMeshesList
{
    daxa_u32 count;
    daxa_u32vec3 padd;
    daxa_u32 mesh_ids[MAX_MESH_INSTANCES];
};
DAXA_DECL_BUFFER_PTR(VisibleMeshesList)

/// --- Analyze Visbuffer Results End ---