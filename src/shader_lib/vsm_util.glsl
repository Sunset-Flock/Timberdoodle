#pragma once

#include <daxa/daxa.inl>
#include "shader_lib/glsl_to_slang.glsl"
#include "shader_shared/vsm_shared.inl"
daxa_u32 n_mask(daxa_u32 count) { return ((1 << count) - 1); }

// BIT 31 -> 0 - FREE/ 1 - ALLOCATED
// BIT 30 -> 1 - REQUESTS_ALLOCATION
// BIT 29 -> 1 - ALLOCATION_FAILED
// BIT 28 -> 1 - DIRTY
// BIT 27 -> 1 - VISITED_MARKED

// VSM PAGE TABLE MASKS AND FUNCTIONS
daxa_u32 allocated_mask()           { return 1 << 31; }
daxa_u32 requests_allocation_mask() { return 1 << 30; }
daxa_u32 allocation_failed_mask()   { return 1 << 29; }
daxa_u32 dirty_mask()               { return 1 << 28; }
daxa_u32 visited_marked_mask()      { return 1 << 27; }

bool get_is_allocated(daxa_u32 page_entry)           { return (page_entry & allocated_mask()) != 0; }
bool get_requests_allocation(daxa_u32 page_entry)    { return (page_entry & requests_allocation_mask()) != 0; }
bool get_allocation_failed(daxa_u32 page_entry)      { return (page_entry & allocation_failed_mask()) != 0; }
bool get_is_dirty(daxa_u32 page_entry)               { return (page_entry & dirty_mask()) != 0; }
bool get_is_visited_marked(daxa_u32 page_entry)      { return (page_entry & visited_marked_mask()) != 0; }

// BIT 0 - 7  page entry x coord
// BIT 8 - 15 page entry y coord
daxa_i32vec2 get_meta_coords_from_vsm_entry(daxa_u32 page_entry)
{
    daxa_i32vec2 vsm_page_coordinates = daxa_i32vec2(0,0);
    vsm_page_coordinates.x = daxa_i32((page_entry >> 0) & n_mask(8));
    vsm_page_coordinates.y = daxa_i32((page_entry >> 8) & n_mask(8));

    return vsm_page_coordinates;
}

daxa_u32 pack_meta_coords_to_vsm_entry(daxa_i32vec2 coords)
{
    daxa_u32 packed_coords = 0;
    packed_coords |= (coords.y << 8);
    packed_coords |= (coords.x & n_mask(8));
    return packed_coords;
}

// VSM MEMORY META MASKS AND FUNCTIONS
// BIT 31 -> 0 - FREE/ 1 - ALLOCATED
// BIT 30 -> 1 - VISITED
// BIT 29 -> 1 - IS POINT LIGHT
daxa_u32 meta_memory_allocated_mask()   { return 1 << 31; }
daxa_u32 meta_memory_visited_mask()     { return 1 << 30; }
daxa_u32 meta_memory_point_light_mask() { return 1 << 29; }

bool get_meta_memory_is_allocated(daxa_u32 meta_page_entry){ return (meta_page_entry & meta_memory_allocated_mask()) != 0; }
bool get_meta_memory_is_visited(daxa_u32 meta_page_entry)  { return (meta_page_entry & meta_memory_visited_mask()) != 0; }
bool get_meta_memory_is_point_light(daxa_u32 meta_page_entry) {return (meta_page_entry & meta_memory_point_light_mask()) != 0; }

// BIT 0 - 7   page entry x coord
// BIT 8 - 15  page entry y coord
// BIT 16 - 19 vsm clip level
daxa_i32vec3 get_vsm_coords_from_meta_entry(daxa_u32 page_entry)
{
    daxa_i32vec3 physical_coordinates = daxa_i32vec3(0,0,0);
    physical_coordinates.x = daxa_i32((page_entry >> 0)  & n_mask(8));
    physical_coordinates.y = daxa_i32((page_entry >> 8)  & n_mask(8));
    physical_coordinates.z = daxa_i32((page_entry >> 16) & n_mask(4));

    return physical_coordinates;
}

daxa_u32 pack_vsm_coords_to_meta_entry(daxa_i32vec3 coords)
{
    daxa_u32 packed_coords = 0;
    packed_coords |= ((coords.z & n_mask(4)) << 16);
    packed_coords |= ((coords.y & n_mask(8)) << 8);
    packed_coords |= ((coords.x & n_mask(8)) << 0);
    return packed_coords;
}

struct PointLightCoords
{
    daxa_i32vec2 texel_coords;
    daxa_i32 mip_level;
    daxa_i32 face_index;
    daxa_i32 point_light_index;
};

// BIT 0 - 7 page entry x coord
// BIT 8 - 15 page entry y coord
// BIT 16 - 19 mip level
// BIT 20 - 23 face index
// BIT 24 - 28 point light index
daxa_u32 pack_vsm_point_light_coords_to_meta_entry(const PointLightCoords info)
{
    daxa_u32 packed_coords = 0;
    packed_coords |= ((info.point_light_index  & n_mask(4)) << 24);
    packed_coords |= ((info.face_index  & n_mask(4)) << 20);
    packed_coords |= ((info.mip_level   & n_mask(4)) << 16);
    packed_coords |= ((info.texel_coords.y & n_mask(8)) << 8);
    packed_coords |= ((info.texel_coords.x & n_mask(8)) << 0);
    return packed_coords;
}

PointLightCoords get_vsm_point_light_coords_from_meta_entry(const daxa_u32 page_entry)
{
    PointLightCoords coords;
    coords.texel_coords.x    = daxa_i32((page_entry >> 0)  & n_mask(8));
    coords.texel_coords.y    = daxa_i32((page_entry >> 8)  & n_mask(8));
    coords.mip_level         = daxa_i32((page_entry >> 16) & n_mask(4));
    coords.face_index        = daxa_i32((page_entry >> 20) & n_mask(4));
    coords.point_light_index = daxa_i32((page_entry >> 24) & n_mask(4));
    return coords;
}

struct ClipInfo
{
    daxa_i32 clip_level;
    daxa_f32vec2 clip_depth_uv;
    daxa_f32 clip_depth;
};

daxa_f32vec3 world_space_from_uv(daxa_f32vec2 screen_space_uv, daxa_f32 depth, daxa_f32mat4x4 inv_view_proj)
{
    const daxa_f32vec2 remap_uv = (screen_space_uv * 2.0) - 1.0;
    const daxa_f32vec4 ndc_position = daxa_f32vec4(remap_uv, depth, 1.0);
    const daxa_f32vec4 unprojected_ndc_position = mul(inv_view_proj, ndc_position);

    const daxa_f32vec3 world_position = unprojected_ndc_position.xyz / unprojected_ndc_position.w;
    return world_position;
}

struct ClipFromUVsInfo
{
    // Should be UVs of the center of the texel
    daxa_f32vec2 uv;
    daxa_u32vec2 screen_resolution;
    daxa_f32 depth;
    daxa_f32mat4x4 inv_view_proj;
    daxa_i32 force_clip_level;
    daxa_BufferPtr(VSMClipProjection) clip_projections;
    daxa_BufferPtr(VSMGlobals) vsm_globals;
    daxa_BufferPtr(RenderGlobalData) globals;
};

ClipInfo clip_info_from_uvs(ClipFromUVsInfo info)
{
    daxa_i32 clip_level;
    if(info.force_clip_level == -1)
    {
        const daxa_f32vec2 center_texel_coords = info.uv * info.screen_resolution;
        #if 1
        const daxa_f32vec2 texel_coords = center_texel_coords;
        const daxa_f32vec2 texel_uvs = texel_coords / daxa_f32vec2(info.screen_resolution);
        const daxa_f32vec3 world_space = world_space_from_uv(texel_uvs, info.depth, info.inv_view_proj);

        const daxa_f32 dist = length(world_space - deref(info.globals).camera.position);
        // The shadow camera is not strictly aligned to the player position. Instead it can be up to
        // one page away from the player, thus we must propriately scale the heuristic, to account for this
        const daxa_i32 page_count = (VSM_TEXTURE_RESOLUTION / VSM_PAGE_SIZE);
        const daxa_f32 scale_ratio = daxa_f32(page_count - 2) / daxa_f32(page_count);
        const daxa_f32 base_scale = deref(info.globals).vsm_settings.clip_0_frustum_scale * scale_ratio;
        clip_level = daxa_i32(clamp(ceil(log2((dist / base_scale) * deref(info.globals).vsm_settings.clip_selection_bias)), 0, VSM_CLIP_LEVELS - 1));
        #else 
        const daxa_f32vec2 left_side_texel_coords = center_texel_coords - daxa_f32vec2(0.5, 0.0);
        const daxa_f32vec2 left_side_texel_uvs = left_side_texel_coords / daxa_f32vec2(info.screen_resolution);
        const daxa_f32vec3 left_world_space = world_space_from_uv( left_side_texel_uvs, info.depth, info.inv_view_proj);

        const daxa_f32vec2 right_side_texel_coords = center_texel_coords + daxa_f32vec2(0.5, 0.0);
        const daxa_f32vec2 right_side_texel_uvs = right_side_texel_coords / daxa_f32vec2(info.screen_resolution);
        const daxa_f32vec3 right_world_space = world_space_from_uv( right_side_texel_uvs, info.depth, info.inv_view_proj);

        const daxa_f32 texel_world_size = length(left_world_space - right_world_space);
        const daxa_f32 f_clip_level = log2(texel_world_size / deref(info.vsm_globals).clip_0_texel_world_size) + deref(info.globals).vsm_settings.clip_selection_bias;
        clip_level = max(daxa_i32(ceil(f_clip_level)), 0);
        if(clip_level >= VSM_CLIP_LEVELS) 
        {
            return ClipInfo(clip_level, daxa_f32vec2(0.0));
        }
        #endif

    } 
    else 
    {
        clip_level = info.force_clip_level;
    }

    const daxa_f32vec3 center_world_space = world_space_from_uv( info.uv, info.depth, info.inv_view_proj);
    const daxa_f32vec4 sun_projected_world_position = mul(deref_i(info.clip_projections, clip_level).camera.view_proj, daxa_f32vec4(center_world_space, 1.0)); 
    const daxa_f32vec3 sun_ndc_position = sun_projected_world_position.xyz / sun_projected_world_position.w; 
    const daxa_f32vec2 sun_depth_uv = (sun_ndc_position.xy + daxa_f32vec2(1.0)) / daxa_f32vec2(2.0); 
    return ClipInfo(clip_level, sun_depth_uv, sun_ndc_position.z);
}

daxa_i32vec3 vsm_page_coords_to_wrapped_coords(daxa_i32vec3 page_coords, daxa_BufferPtr(VSMClipProjection) clip_projections)
{
    const daxa_i32vec2 vsm_toroidal_offset = deref_i(clip_projections, page_coords.z).page_offset;
    const daxa_i32vec2 vsm_toroidal_pix_coords = page_coords.xy - vsm_toroidal_offset.xy;
    const daxa_i32vec2 vsm_wrapped_pix_coords = daxa_i32vec2(_mod(vsm_toroidal_pix_coords.xy, daxa_f32vec2(VSM_PAGE_TABLE_RESOLUTION)));
    return daxa_i32vec3(vsm_wrapped_pix_coords, page_coords.z);
}

daxa_i32vec3 vsm_clip_info_to_wrapped_coords(ClipInfo info, daxa_BufferPtr(VSMClipProjection) clip_projections)
{
    if(any(lessThan(info.clip_depth_uv, daxa_f32vec2(0.0))) || any(greaterThanEqual(info.clip_depth_uv, daxa_f32vec2(1.0))))
    {
        return daxa_i32vec3(-1, -1, info.clip_level);
    }
    const daxa_i32vec3 vsm_page_pix_coords = daxa_i32vec3(daxa_i32vec2(floor(info.clip_depth_uv * VSM_PAGE_TABLE_RESOLUTION)), info.clip_level);
    return vsm_page_coords_to_wrapped_coords(vsm_page_pix_coords, clip_projections);
}

daxa_i32vec2 virtual_uv_to_physical_texel(daxa_f32vec2 virtual_uv, daxa_i32vec2 physical_page_coords)
{
    const daxa_i32vec2 virtual_texel_coord = daxa_i32vec2(virtual_uv * VSM_TEXTURE_RESOLUTION);
    const daxa_i32vec2 in_page_texel_coord = daxa_i32vec2(_mod(virtual_texel_coord, daxa_f32(VSM_PAGE_SIZE)));
    const daxa_i32vec2 in_memory_offset = physical_page_coords * VSM_PAGE_SIZE;
    const daxa_i32vec2 memory_texel_coord = in_memory_offset + in_page_texel_coord;
    return memory_texel_coord;
}

daxa_u32 unwrap_vsm_page_from_mask(daxa_i32vec3 vsm_page_coords, daxa_BufferPtr(FreeWrappedPagesInfo) info)
{
    const daxa_u32 linear_index = vsm_page_coords.y * VSM_PAGE_TABLE_RESOLUTION + vsm_page_coords.x;
    const daxa_u32 index_offset = linear_index / 32;
    const daxa_u32 in_uint_offset = daxa_u32(_mod(daxa_f32(linear_index),daxa_f32(32)));
    const daxa_u32 mask_entry = deref_i(info, vsm_page_coords.z).mask[index_offset];
    const daxa_u32 decoded_bit = mask_entry & (1 << in_uint_offset);
    return decoded_bit;
}

#if (DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG)
int cube_face_from_dir(float3 normalized_direction)
{
    const float3 abs_direction = abs(normalized_direction);
    const int coord_idx = abs_direction.x < abs_direction.y ? 
                // y bigger than x -> compare against z
                (abs_direction.y < abs_direction.z ? 2 : 1) :
                // x bigger than y -> compare against z
                (abs_direction.x < abs_direction.z ? 2 : 0);
    const int negative_face_offset = normalized_direction[coord_idx] < 0 ? 1 : 0;
    const int face_idx = coord_idx * 2 + negative_face_offset;
    return face_idx;
}

float3 ray_plane_intersection(float3 ray_direction, float3 ray_origin, float3 plane_normal, float3 plane_origin)
{
    float denom = dot(plane_normal, ray_direction);
    if (abs(denom) > 0.0001f) // your favorite epsilon
    {
        float t = dot((plane_origin - ray_origin), plane_normal) / denom;
        return ray_origin + t * ray_direction;
    }
    return 0.0f;
}

struct VSMPointIndirections
{
    uint mip_level;
    uint face_index;
    uint point_light_index;
    uint mesh_instance_index;
};

uint pack_vsm_point_light_indirections(const VSMPointIndirections indirections)
{
    uint packed_value = 0;
    packed_value |= ((indirections.point_light_index   & n_mask(7)) << 25);
    packed_value |= ((indirections.face_index          & n_mask(4)) << 21);
    packed_value |= ((indirections.mip_level           & n_mask(4)) << 17);
    packed_value |= ((indirections.mesh_instance_index & n_mask(17)) << 0);
    return packed_value;
}

VSMPointIndirections unpack_vsm_point_light_indirections(uint packed_value)
{
    VSMPointIndirections indirections;
    indirections.mesh_instance_index  = daxa_i32((packed_value >> 0)  & n_mask(17));
    indirections.mip_level            = daxa_i32((packed_value >> 17) & n_mask(4));
    indirections.face_index           = daxa_i32((packed_value >> 21) & n_mask(4));
    indirections.point_light_index    = daxa_i32((packed_value >> 25) & n_mask(7));
    return indirections;
}

uint get_vsm_point_page_array_idx(int face_idx, int point_light_idx)
{
    return point_light_idx * 6 + face_idx;
}

struct PointMipInfo
{
    int mip_level;
    int cube_face;
    int2 page_texel_coords;
    float2 page_uvs;
};

PointMipInfo project_into_point_light(
    float depth,
    float3 normal,
    int point_light_idx,
    float2 screen_space_uv,
    RenderGlobalData * globals,
    VSMPointLight * point_lights,
    VSMGlobals * vsm_globals)
{
    const float2 uv_offset = 0.5f * globals->settings.render_target_size_inv.xy;
    screen_space_uv += uv_offset;
    const float4x4 inverse_camera_view_proj = globals->camera.inv_view_proj;
    const float3 frag_ws = world_space_from_uv(screen_space_uv, depth, inverse_camera_view_proj);
    const float3 point_ws = point_lights[point_light_idx].light.position;

    const float3 point_to_frag_norm = normalize(frag_ws - point_ws);

    if(length(frag_ws - point_ws) > point_lights[point_light_idx].light->cutoff)
    {
        return PointMipInfo(6, -1, int2(-1), float2(-2.0f));
    }

    const int face_idx = cube_face_from_dir(point_to_frag_norm);

    // Reprojecting screen space into point light space
    const float2 bottom_right = screen_space_uv + float2(uv_offset.x, uv_offset.y);
    const float3 bottom_right_ws = world_space_from_uv( bottom_right, depth, inverse_camera_view_proj);

    const float2 bottom_left = screen_space_uv + float2(-uv_offset.x, uv_offset.y);
    const float3 bottom_left_ws = world_space_from_uv( bottom_left, depth, inverse_camera_view_proj);

    const float2 top_right = screen_space_uv + float2(uv_offset.x, -uv_offset.y);
    const float3 top_right_ws = world_space_from_uv( top_right, depth, inverse_camera_view_proj);

    const float2 top_left = screen_space_uv + float2(-uv_offset.x, -uv_offset.y);
    const float3 top_left_ws = world_space_from_uv( top_left, depth, inverse_camera_view_proj);


    const float3 bottom_right_real_ws = ray_plane_intersection(normalize(bottom_right_ws - globals->camera.position), globals->camera.position, normal, frag_ws);
    const float3 bottom_left_real_ws = ray_plane_intersection(normalize(bottom_left_ws - globals->camera.position), globals->camera.position, normal, frag_ws);
    const float3 top_right_real_ws = ray_plane_intersection(normalize(top_right_ws - globals->camera.position), globals->camera.position, normal, frag_ws);
    const float3 top_left_real_ws = ray_plane_intersection(normalize(top_left_ws - globals->camera.position), globals->camera.position, normal, frag_ws);

    if( any(bottom_right_real_ws == float3(0.0f)) || 
        any(bottom_left_real_ws == float3(0.0f))  || 
        any(top_right_real_ws == float3(0.0f))    || 
        any(top_left_real_ws == float3(0.0f)))
    {
        return PointMipInfo(6, -1, int2(-1), float2(-2.0f));
    }

    {
        const float4x4 point_view_projection = point_lights[point_light_idx].face_cameras[face_idx].view_proj;

        const float4 bottom_right_side_cs = mul(point_view_projection, float4(bottom_right_real_ws, 1.0f));
        const float4 bottom_left_side_cs = mul(point_view_projection, float4(bottom_left_real_ws, 1.0f));
        const float4 top_right_side_cs = mul(point_view_projection, float4(top_right_real_ws, 1.0f));
        const float4 top_left_side_cs = mul(point_view_projection, float4(top_left_real_ws, 1.0f));

        const float2 bottom_right_ndc = bottom_right_side_cs.xy / bottom_right_side_cs.w;
        const float2 bottom_left_ndc = bottom_left_side_cs.xy / bottom_left_side_cs.w;
        const float2 top_right_ndc = top_right_side_cs.xy / top_right_side_cs.w;
        const float2 top_left_ndc = top_left_side_cs.xy / top_left_side_cs.w;

        // const float max_axis_dist = max(abs(point_left_ndc.x - point_right_ndc.x), abs(point_left_ndc.y - point_right_ndc.y));
        
        const float max_axis_dist = min(length(bottom_right_ndc - top_left_ndc), length(bottom_left_ndc - top_right_ndc));

        const float point_uv_dist = max_axis_dist / 2.0f; // ndc is twice as large as uvs
        const float point_texel_dist = point_uv_dist * VSM_TEXTURE_RESOLUTION;
        const int mip_level = max(int(log2(floor(point_texel_dist))), 0);
        const int clamped_mip_level = clamp(mip_level, 0, 5);

        const float2 middle_ndc = bottom_right_ndc * 0.5f + top_left_ndc * 0.5f;
        const float2 middle_uv = clamp((middle_ndc + 1.0f) * 0.5f, 0.0f, 0.99999f);
        const int2 texel_coord = int2(middle_uv * (VSM_PAGE_TABLE_RESOLUTION / (1 << mip_level)));
        return PointMipInfo(int(clamped_mip_level), int(face_idx), texel_coord, middle_uv);
    }
}
#endif