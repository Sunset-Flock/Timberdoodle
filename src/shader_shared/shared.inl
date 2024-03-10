#pragma once

#include "daxa/daxa.inl"

#define SHADER_GLOBALS_SLOT 0

#define MAX_SURFACE_RES_X 3840
#define MAX_SURFACE_RES_Y 2160

#define MAX_INSTANTIATED_MESHES 100000
#define MAX_MESHLET_INSTANCES 1000000
#define MAX_MESH_INSTANCES 100000
#define VISIBLE_ENTITY_MESHLETS_BITFIELD_SCRATCH 1000000
#define MAX_DRAWN_TRIANGLES (MAX_SURFACE_RES_X * MAX_SURFACE_RES_Y)
#define TRIANGLE_SIZE 12
#define WARP_SIZE 32
#define MAX_ENTITY_COUNT (1u << 20u)
#define MAX_MATERIAL_COUNT (1u << 8u)
#define MESH_SHADER_WORKGROUP_X 32
#define ENABLE_MESHLET_CULLING 1
#define ENABLE_TRIANGLE_CULLING 1
#define ENABLE_SHADER_PRINT_DEBUG 1
#define COMPILE_IN_MESH_SHADER 0
#define CULLING_DEBUG_DRAWS 1

#if defined(__cplusplus)
#include <glm/glm.hpp>
#define glmsf32vec2 glm::vec2
#define glmsf32vec3 glm::vec3
#define glmsf32vec4 glm::vec4
#define glmsf32mat4 glm::mat4
#else
#define glmsf32vec2 daxa_f32vec2
#define glmsf32vec3 daxa_f32vec3
#define glmsf32vec4 daxa_f32vec4
#define glmsf32mat4 daxa_f32mat4x4
#endif

#if defined(__cplusplus)
#define SHADER_ONLY(x)
#define HOST_ONLY(x) x
#else
#define SHADER_ONLY(x) x
#define HOST_ONLY(x)
#endif

#define PROFILE_LAYER_COUNT 2
// An atmosphere layer density which can be calculated as:
//   density = exp_term * exp(exp_scale * h) + linear_term * h + constant_term,
struct DensityProfileLayer
{
    daxa_f32 layer_width;
    daxa_f32 exp_term;
    daxa_f32 exp_scale;
    daxa_f32 lin_term;
    daxa_f32 const_term;
};
DAXA_DECL_BUFFER_PTR(DensityProfileLayer)

struct SkySettings
{
    daxa_u32vec2 transmittance_dimensions;
    daxa_u32vec2 multiscattering_dimensions;
    daxa_u32vec2 sky_dimensions;

    // =============== Atmosphere =====================
    daxa_f32 sun_brightness;
    daxa_f32vec3 sun_direction;

    daxa_f32 atmosphere_bottom;
    daxa_f32 atmosphere_top;

    daxa_f32vec3 mie_scattering;
    daxa_f32vec3 mie_extinction;
    daxa_f32 mie_scale_height;
    daxa_f32 mie_phase_function_g;
    DensityProfileLayer mie_density[PROFILE_LAYER_COUNT];
    daxa_BufferPtr(DensityProfileLayer) mie_density_ptr; 

    daxa_f32vec3 rayleigh_scattering;
    daxa_f32 rayleigh_scale_height;
    DensityProfileLayer rayleigh_density[PROFILE_LAYER_COUNT];
    daxa_BufferPtr(DensityProfileLayer) rayleigh_density_ptr; 

    daxa_f32vec3 absorption_extinction;
    DensityProfileLayer absorption_density[PROFILE_LAYER_COUNT];
    daxa_BufferPtr(DensityProfileLayer) absorption_density_ptr;
#if defined(__cplusplus)
    auto operator==(SkySettings const & other) const -> bool
    {
        return std::memcmp(this, &other, sizeof(SkySettings)) == 0;
    }
    auto operator!=(SkySettings const & other) const -> bool
    {
        return std::memcmp(this, &other, sizeof(SkySettings)) != 0;
    }
    struct ResolutionChangedFlags
    {
        bool transmittance_changed = {};
        bool multiscattering_changed = {};
        bool sky_changed = {};
    };
    auto resolutions_changed(SkySettings const & other) const -> ResolutionChangedFlags
    {
        bool const transmittance_changed =
            other.transmittance_dimensions.x != transmittance_dimensions.x ||
            other.transmittance_dimensions.y != transmittance_dimensions.y;
        bool const multiscattering_changed =
            other.multiscattering_dimensions.x != multiscattering_dimensions.x ||
            other.multiscattering_dimensions.y != multiscattering_dimensions.y;
        bool const sky_changed =
            other.sky_dimensions.x != sky_dimensions.x ||
            other.sky_dimensions.y != sky_dimensions.y;
        return {transmittance_changed, multiscattering_changed, sky_changed};
    }
#endif
};
DAXA_DECL_BUFFER_PTR_ALIGN(SkySettings, 8)

struct Settings
{
    daxa_u32vec2 render_target_size;
    daxa_f32vec2 render_target_size_inv;
    daxa_u32 enable_mesh_shader;
    daxa_u32 draw_from_observer;
    daxa_i32 observer_show_pass;
#if defined(__cplusplus)
    auto operator==(Settings const & other) const -> bool
    {
        return std::memcmp(this, &other, sizeof(Settings)) == 0;
    }
    auto operator!=(Settings const & other) const -> bool
    {
        return std::memcmp(this, &other, sizeof(Settings)) != 0;
    }
    Settings()
        : render_target_size{16, 16},
          render_target_size_inv{1.0f / this->render_target_size.x, 1.0f / this->render_target_size.y},
          enable_mesh_shader{0},
          draw_from_observer{0},
          observer_show_pass{0}
    {
    }
#endif
};

struct PostprocessSettings
{
    daxa_f32 min_luminance_log2;
    daxa_f32 max_luminance_log2;
    daxa_f32 luminance_adaption_tau;
    daxa_f32 exposure_bias;
    daxa_f32 calibration;
    daxa_f32 sensor_sensitivity;
    daxa_f32 luminance_log2_range;
    daxa_f32 inv_luminance_log2_range;
#if defined(__cplusplus)
    PostprocessSettings()
        : min_luminance_log2{std::log2(0.0002f)},
          max_luminance_log2{std::log2(4096.0f)},
          luminance_adaption_tau{1.0f},
          exposure_bias{1.0f},
          calibration{12.5f},
          sensor_sensitivity{100.0f},
          luminance_log2_range{max_luminance_log2 - min_luminance_log2},
          inv_luminance_log2_range{1.0f / (max_luminance_log2 - min_luminance_log2)}
    {
    }
#endif
};
DAXA_DECL_BUFFER_PTR(PostprocessSettings)

struct GlobalSamplers
{
    daxa_SamplerId linear_clamp;
    daxa_SamplerId linear_repeat;
    daxa_SamplerId nearest_clamp;
    daxa_SamplerId linear_repeat_ani;
};

struct CameraInfo
{
    glmsf32mat4 view;
    glmsf32mat4 inv_view;
    glmsf32mat4 proj;
    glmsf32mat4 inv_proj;
    glmsf32mat4 view_proj;
    glmsf32mat4 inv_view_proj;
    glmsf32vec3 position;
    glmsf32vec3 up;
    glmsf32vec3 near_plane_normal;
    glmsf32vec3 left_plane_normal;
    glmsf32vec3 right_plane_normal;
    glmsf32vec3 top_plane_normal;
    glmsf32vec3 bottom_plane_normal;
    daxa_u32vec2 screen_size;
    daxa_f32vec2 inv_screen_size;
    daxa_f32 near_plane;
};

#if DAXA_SHADER
#define my_sizeof(T) uint64_t(daxa_BufferPtr(T)(daxa_u64(0)) + 1)

daxa_f32mat4x4 mat_4x3_to_4x4(daxa_f32mat4x3 in_mat)
{
    return daxa_f32mat4x4(
        daxa_f32vec4(in_mat[0], 0.0),
        daxa_f32vec4(in_mat[1], 0.0),
        daxa_f32vec4(in_mat[2], 0.0),
        daxa_f32vec4(in_mat[3], 1.0));
}
#endif

#if defined(__cplusplus)
#define SHARED_FUNCTION inline
#define SHARED_FUNCTION_INOUT(X) X &
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

#define PASS0_DRAW_VISIBLE_LAST_FRAME 0
#define PASS1_DRAW_POST_CULL 1
#define PASS2_OBSERVER_DRAW_VISIBLE_LAST_FRAME 2
#define PASS3_OBSERVER_DRAW_POST_CULLED 3
#define PASS4_OBSERVER_DRAW_ALL 4