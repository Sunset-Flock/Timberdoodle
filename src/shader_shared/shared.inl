#pragma once

#include "daxa/daxa.inl"

#define MAX_SURFACE_RES_X 3840
#define MAX_SURFACE_RES_Y 2160

#define MAX_MESHLET_INSTANCES (1u << 24u) // Roughly 16.8 million
//150000
#define MAX_MESH_INSTANCES 100000 
#define WARP_SIZE 32
#define MAX_ENTITIES (1u << 20u)
#define MAX_MATERIALS (1u << 8u)
#define MAX_MESHES 10000
#define MESH_SHADER_WORKGROUP_X 32
#define ENABLE_TRIANGLE_CULLING 0
#define ENABLE_SHADER_PRINT_DEBUG 1
#define COMPILE_IN_MESH_SHADER 1
#define COMPILE_IN_SLANG 1
#define CULLING_DEBUG_DRAWS 1
#define SHADER_DEBUG_VISBUFFER 0

#if defined(__cplusplus)
#define GLM_FORCE_DEPTH_ZERO_TO_ONE 1
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
#define HOST_ONLY(...) __VA_ARGS__
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
    SkySettings() : transmittance_dimensions{1,1}, multiscattering_dimensions{1,1}, sky_dimensions{1,1} {}
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

#define AA_MODE_NONE 0
#define AA_MODE_SUPER_SAMPLE 1
#define AA_MODE_DVM 2

struct VSMSettings
{
    daxa_i32 enable;
    daxa_u32 force_clip_level;
    daxa_u32 enable_caching;
    daxa_i32 forced_clip_level;
    daxa_f32 clip_0_frustum_scale;
    daxa_f32 clip_selection_bias;
    daxa_f32 slope_bias;
    daxa_f32 constant_bias;
    daxa_i32 use_simplified_light_matrix;
    daxa_i32 use64bit;
    daxa_u32 sun_moved;
#if defined(__cplusplus)
    VSMSettings()
        : enable{ 1 },
          force_clip_level{ 0 },
          enable_caching{ 1 },
          forced_clip_level{ 0 },
          clip_0_frustum_scale{2.0f},
          clip_selection_bias{0.3f},
          slope_bias{2.0f},
          constant_bias{10.0f},
          use_simplified_light_matrix{0},
          use64bit{0}
    {
    }
#endif
};
DAXA_DECL_BUFFER_PTR_ALIGN(VSMSettings, 4);

#define DEBUG_DRAW_MODE_NONE 0
#define DEBUG_DRAW_MODE_OVERDRAW 1
#define DEBUG_DRAW_MODE_TRIANGLE_INSTANCE_ID 2
#define DEBUG_DRAW_MODE_MESHLET_INSTANCE_ID 3
#define DEBUG_DRAW_MODE_ENTITY_ID 4
#define DEBUG_DRAW_MODE_VSM_OVERDRAW 5
#define DEBUG_DRAW_MODE_VSM_CLIP_LEVEL 6
#define DEBUG_DRAW_MODE_DEBUG_IMAGE 7
#define DEBUG_DRAW_MODE_DEPTH 8
#define DEBUG_DRAW_MODE_ALBEDO 9
#define DEBUG_DRAW_MODE_NORMAL 10
#define DEBUG_DRAW_MODE_LIGHT 11

struct Settings
{
    daxa_u32vec2 render_target_size;
    daxa_f32vec2 render_target_size_inv;
    // Used by occlusion cull textures:
    daxa_u32vec2 next_lower_po2_render_target_size;
    daxa_f32vec2 next_lower_po2_render_target_size_inv;
    daxa_u32vec2 window_size;
    daxa_u32 draw_from_observer;
    daxa_i32 observer_show_pass;
    daxa_i32 anti_aliasing_mode;
    daxa_i32 debug_draw_mode;
    daxa_f32 debug_overdraw_scale;
    daxa_b32 enable_mesh_cull;
    daxa_b32 enable_meshlet_cull;
    daxa_b32 enable_triangle_cull;
    daxa_b32 enable_atomic_visbuffer;
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
          next_lower_po2_render_target_size{render_target_size.x, render_target_size.y},
          next_lower_po2_render_target_size_inv{1.0f / this->render_target_size.x, 1.0f / this->render_target_size.y},
          window_size{16, 16},
          draw_from_observer{0},
          observer_show_pass{0},
          anti_aliasing_mode{AA_MODE_NONE},
          debug_draw_mode{0},
          debug_overdraw_scale{0.1},
          enable_mesh_cull{1},
          enable_meshlet_cull{1},
          enable_triangle_cull{1},
          enable_atomic_visbuffer{0}
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
        : min_luminance_log2{std::log2(4.0f)},
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
    daxa_SamplerId nearest_repeat;
    daxa_SamplerId nearest_clamp;
    daxa_SamplerId linear_repeat_ani;
    daxa_SamplerId nearest_repeat_ani;
    daxa_SamplerId normals;
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
    // vec4 for planes contains normal (xyz) and offset (w).
    glmsf32vec3 near_plane_normal;
    glmsf32vec3 left_plane_normal;
    glmsf32vec3 right_plane_normal;
    glmsf32vec3 top_plane_normal;
    glmsf32vec3 bottom_plane_normal;
    daxa_u32vec2 screen_size;
    daxa_f32vec2 inv_screen_size;
    daxa_f32 near_plane;
    daxa_f32 orthogonal_half_ws_width;
    daxa_b32 is_orthogonal;
};

#if DAXA_SHADER
#define my_sizeof(T) uint64_t(daxa_BufferPtr(T)(daxa_u64(0)) + 1)

daxa_f32mat4x4 mat_4x3_to_4x4(daxa_f32mat4x3 in_mat)
{
#if DAXA_SHADERLANG == DAXA_SHADERLANG_SLANG
    // In slang the indexing is row major!
    // HLSL: RxCmat
    // GLSL: CxRmat
    daxa_f32mat4x4 ret = daxa_f32mat4x4(
        daxa_f32vec4(in_mat[0]),
        daxa_f32vec4(in_mat[1]),
        daxa_f32vec4(in_mat[2]),
        daxa_f32vec4(0.0f,0.0f,0.0f, 1.0f));
#else
    daxa_f32mat4x4 ret = daxa_f32mat4x4(
        daxa_f32vec4(in_mat[0], 0.0),
        daxa_f32vec4(in_mat[1], 0.0),
        daxa_f32vec4(in_mat[2], 0.0),
        daxa_f32vec4(in_mat[3], 1.0));
#endif
    return ret;
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

#include "../shader_lib/glsl_to_slang.glsl"