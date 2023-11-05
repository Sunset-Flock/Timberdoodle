#pragma once

#include <daxa/daxa.inl>

#define SHADER_GLOBALS_SLOT 0

#define MAX_SURFACE_RES_X 3840
#define MAX_SURFACE_RES_Y 2160

#define MAX_INSTANTIATED_MESHES 100000
#define MAX_MESHLET_INSTANCES 1000000
#define VISIBLE_ENTITY_MESHLETS_BITFIELD_SCRATCH 1000000
#define MAX_DRAWN_TRIANGLES (MAX_SURFACE_RES_X * MAX_SURFACE_RES_Y)
#define MAX_DRAWN_MESHES 100000
#define TRIANGLE_SIZE 12
#define WARP_SIZE 32
#define MAX_ENTITY_COUNT (1u << 20u)
#define MESH_SHADER_WORKGROUP_X 32
#define ENABLE_MESHLET_CULLING 1
#define ENABLE_TRIANGLE_CULLING 1
#define ENABLE_SHADER_PRINT_DEBUG 1
#define COMPILE_IN_MESH_SHADER 0

#if __cplusplus
#define SHADER_ONLY(x)
#else
#define SHADER_ONLY(x) x
#endif

struct Settings
{
    daxa_u32vec2 render_target_size;
    daxa_f32vec2 render_target_size_inv;
    daxa_u32 enable_mesh_shader;
    daxa_u32 enable_observer;
    daxa_i32 observer_show_pass;
#if __cplusplus
    auto operator==(Settings const &other) const -> bool 
    {
        return std::memcmp(this, &other, sizeof(Settings)) == 0;
    }
    auto operator!=(Settings const &other) const -> bool
    {
        return std::memcmp(this, &other, sizeof(Settings)) != 0;
    }
    Settings()
        : render_target_size{ 16, 16 },
        render_target_size_inv{ 1.0f / this->render_target_size.x, 1.0f / this->render_target_size.y },
        enable_mesh_shader{ 0 },
        enable_observer{ 0 },
        observer_show_pass{ 0 }
    {
    }
#endif
};

struct GlobalSamplers
{
    daxa_SamplerId linear_clamp;
    daxa_SamplerId nearest_clamp;
};

struct ShaderGlobals
{
    daxa_f32mat4x4 camera_view;
    daxa_f32mat4x4 camera_projection;
    daxa_f32mat4x4 camera_view_projection;
    daxa_f32vec3 camera_pos;
    daxa_f32vec3 camera_up;
    daxa_f32vec3 camera_near_plane_normal;
    daxa_f32vec3 camera_left_plane_normal;
    daxa_f32vec3 camera_right_plane_normal;
    daxa_f32vec3 camera_top_plane_normal;
    daxa_f32vec3 camera_bottom_plane_normal;
    daxa_f32mat4x4 observer_camera_view;
    daxa_f32mat4x4 observer_camera_projection;
    daxa_f32mat4x4 observer_camera_view_projection;
    daxa_f32vec3 observer_camera_pos;
    daxa_f32vec3 observer_camera_up;
    daxa_u32 frame_index;
    daxa_f32 delta_time;
    Settings settings;
    GlobalSamplers samplers;
};
DAXA_DECL_BUFFER_PTR(ShaderGlobals)

DAXA_DECL_UNIFORM_BUFFER(SHADER_GLOBALS_SLOT) ShaderGlobalsBlock
{
    ShaderGlobals globals;
};

#if DAXA_SHADER
#define my_sizeof(T) uint64_t(daxa_BufferPtr(T)(daxa_u64(0)) + 1)
#endif

#if defined(__cplusplus)
#define SHARED_FUNCTION inline
#define SHARED_FUNCTION_INOUT(X) X&
#else
#define SHARED_FUNCTION
#define SHARED_FUNCTION_INOUT(X) inout X
#endif

SHARED_FUNCTION daxa_u32 round_up_to_multiple(daxa_u32 value, daxa_u32 multiple_of)
{
    return ((value + multiple_of - 1) / multiple_of) * multiple_of;
}

SHARED_FUNCTION daxa_u32 round_up_div(daxa_u32 value, daxa_u32 div)
{
    return (value + div - 1) / div;
}

#define ENABLE_TASK_USES(STRUCT, NAME)

struct DrawIndexedIndirectStruct
{
    daxa_u32 index_count;
    daxa_u32 instance_count;
    daxa_u32 first_index;
    daxa_u32 vertex_offset;
    daxa_u32 first_instance;
};
DAXA_DECL_BUFFER_PTR(DrawIndexedIndirectStruct)

struct DrawIndirectStruct
{
    daxa_u32 vertex_count;
    daxa_u32 instance_count;
    daxa_u32 first_vertex;
    daxa_u32 first_instance;
};
DAXA_DECL_BUFFER_PTR(DrawIndirectStruct)

struct DispatchIndirectStruct
{
    daxa_u32 x;
    daxa_u32 y;
    daxa_u32 z;
};
DAXA_DECL_BUFFER_PTR(DispatchIndirectStruct)

#define BUFFER_COMPUTE_READ(NAME, TYPE) DAXA_TASK_USE_BUFFER(NAME, daxa_BufferPtr(TYPE), COMPUTE_SHADER_READ)
#define BUFFER_COMPUTE_WRITE(NAME, TYPE) DAXA_TASK_USE_BUFFER(NAME, daxa_RWBufferPtr(TYPE), COMPUTE_SHADER_WRITE)