#pragma once

#include <daxa/daxa.inl>
#include "shader_shared/vsm_shared.inl"
uint n_mask(uint count) { return ((1 << count) - 1); }

// BIT 31 -> 0 - FREE/ 1 - ALLOCATED
// BIT 30 -> 1 - REQUESTS_ALLOCATION
// BIT 29 -> 1 - ALLOCATION_FAILED
// BIT 28 -> 1 - DIRTY
// BIT 27 -> 1 - VISITED_MARKED

// VSM PAGE TABLE MASKS AND FUNCTIONS
uint allocated_mask()           { return 1 << 31; }
uint requests_allocation_mask() { return 1 << 30; }
uint allocation_failed_mask()   { return 1 << 29; }
uint dirty_mask()               { return 1 << 28; }
uint visited_marked_mask()      { return 1 << 27; }

bool get_is_allocated(uint page_entry)        { return (page_entry & allocated_mask()) != 0; }
bool get_requests_allocation(uint page_entry) { return (page_entry & requests_allocation_mask()) != 0; }
bool get_allocation_failed(uint page_entry)   { return (page_entry & allocation_failed_mask()) != 0; }
bool get_is_dirty(uint page_entry)            { return (page_entry & dirty_mask()) != 0; }
bool get_is_visited_marked(uint page_entry)   { return (page_entry & visited_marked_mask()) != 0; }

// BIT 0 - 7  page entry x coord
// BIT 8 - 15 page entry y coord
ivec2 get_meta_coords_from_vsm_entry(uint page_entry)
{
    ivec2 vsm_page_coordinates = ivec2(0,0);
    vsm_page_coordinates.x = int((page_entry >> 0) & n_mask(8));
    vsm_page_coordinates.y = int((page_entry >> 8) & n_mask(8));

    return vsm_page_coordinates;
}

uint pack_meta_coords_to_vsm_entry(ivec2 coords)
{
    uint packed_coords = 0;
    packed_coords |= (coords.y << 8);
    packed_coords |= (coords.x & n_mask(8));
    return packed_coords;
}

// VSM MEMORY META MASKS AND FUNCTIONS
// BIT 31 -> 0 - FREE/ 1 - ALLOCATED
// BIT 30 -> 1 - NEEDS_CLEAR
// BIT 29 -> 1 - VISITED
uint meta_memory_allocated_mask()   { return 1 << 31; }
uint meta_memory_needs_clear_mask() { return 1 << 30; }
uint meta_memory_visited_mask()     { return 1 << 29; }

bool get_meta_memory_is_allocated(uint meta_page_entry){ return (meta_page_entry & meta_memory_allocated_mask()) != 0; }
bool get_meta_memory_needs_clear(uint meta_page_entry) { return (meta_page_entry & meta_memory_needs_clear_mask()) != 0; }
bool get_meta_memory_is_visited(uint meta_page_entry)  { return (meta_page_entry & meta_memory_visited_mask()) != 0; }

// BIT 0 - 7   page entry x coord
// BIT 8 - 15  page entry y coord
// BIT 16 - 19 vsm clip level
ivec3 get_vsm_coords_from_meta_entry(uint page_entry)
{
    ivec3 physical_coordinates = ivec3(0,0,0);
    physical_coordinates.x = int((page_entry >> 0)  & n_mask(8));
    physical_coordinates.y = int((page_entry >> 8)  & n_mask(8));
    physical_coordinates.z = int((page_entry >> 16) & n_mask(4));

    return physical_coordinates;
}

uint pack_vsm_coords_to_meta_entry(ivec3 coords)
{
    uint packed_coords = 0;
    packed_coords |= ((coords.z & n_mask(4)) << 16);
    packed_coords |= ((coords.y & n_mask(8)) << 8);
    packed_coords |= ((coords.x & n_mask(8)) << 0);
    return packed_coords;
}

struct ClipInfo
{
    int clip_level;
    vec2 sun_depth_uv;
};

struct ClipFromUVsInfo
{
    // Should be UVs of the center of the texel
    vec2 uv;
    uvec2 screen_resolution;
    float depth;
    mat4x4 inv_view_proj;
    int force_clip_level;
    daxa_BufferPtr(VSMClipProjection) clip_projections;
    daxa_BufferPtr(VSMGlobals) globals;
};

vec3 world_space_from_uv(vec2 screen_space_uv, float depth, mat4x4 inv_view_proj)
{
    const vec2 remap_uv = (screen_space_uv * 2.0) - 1.0;
    const vec4 ndc_position = daxa_f32vec4(remap_uv, depth, 1.0);
    const vec4 unprojected_ndc_position = inv_view_proj * ndc_position;

    const vec3 world_position = unprojected_ndc_position.xyz / unprojected_ndc_position.w;
    return world_position;
}

ClipInfo clip_info_from_uvs(ClipFromUVsInfo info)
{
    int clip_level;
    if(info.force_clip_level == -1)
    {
        const vec2 center_texel_coords = info.uv * info.screen_resolution;

        const vec2 left_side_texel_coords = center_texel_coords - vec2(0.5, 0.0);
        const vec2 left_side_texel_uvs = left_side_texel_coords / vec2(info.screen_resolution);
        const vec3 left_world_space = world_space_from_uv( left_side_texel_uvs, info.depth, info.inv_view_proj);

        const vec2 right_side_texel_coords = center_texel_coords + vec2(0.5, 0.0);
        const vec2 right_side_texel_uvs = right_side_texel_coords / vec2(info.screen_resolution);
        const vec3 right_world_space = world_space_from_uv( right_side_texel_uvs, info.depth, info.inv_view_proj);

        const float texel_world_size = length(left_world_space - right_world_space);
        clip_level = max(int(ceil(log2(texel_world_size / deref(info.globals).clip_0_texel_world_size))), 0);
        if(clip_level >= VSM_CLIP_LEVELS) 
        {
            return ClipInfo(clip_level, vec2(0.0));
        }
    } 
    else 
    {
        clip_level = info.force_clip_level;
    }

    const vec3 center_world_space = world_space_from_uv( info.uv, info.depth, info.inv_view_proj);
    const vec4 sun_projected_world_position = deref_i(info.clip_projections, clip_level).projection_view * vec4(center_world_space, 1.0);
    const vec3 sun_ndc_position = sun_projected_world_position.xyz / sun_projected_world_position.w;
    const vec2 sun_depth_uv = (sun_ndc_position.xy + vec2(1.0)) / vec2(2.0);
    return ClipInfo(clip_level, sun_depth_uv);
}

ivec3 vsm_page_coords_to_wrapped_coords(ivec3 page_coords, daxa_BufferPtr(VSMClipProjection) clip_projections)
{
    const ivec2 vsm_toroidal_offset = deref_i(clip_projections, page_coords.z).page_offset;
    const ivec2 vsm_toroidal_pix_coords = page_coords.xy - vsm_toroidal_offset.xy;
    if( 
        page_coords.x < 0 ||
        page_coords.x > (VSM_PAGE_TABLE_RESOLUTION - 1) ||
        page_coords.y < 0 ||
        page_coords.y > (VSM_PAGE_TABLE_RESOLUTION - 1))
    {
        return ivec3(-1, -1, page_coords.z);
    }
    const ivec2 vsm_wrapped_pix_coords = ivec2(mod(vsm_toroidal_pix_coords.xy, float(VSM_PAGE_TABLE_RESOLUTION)));
    return ivec3(vsm_wrapped_pix_coords, page_coords.z);
}

ivec3 vsm_clip_info_to_wrapped_coords(ClipInfo info, daxa_BufferPtr(VSMClipProjection) clip_projections)
{
    const ivec3 vsm_page_pix_coords = ivec3(floor(info.sun_depth_uv * VSM_PAGE_TABLE_RESOLUTION), info.clip_level);
    return vsm_page_coords_to_wrapped_coords(vsm_page_pix_coords, clip_projections);
}

ivec2 virtual_uv_to_physical_texel(vec2 virtual_uv, ivec2 physical_page_coords)
{
    const ivec2 virtual_texel_coord = ivec2(virtual_uv * VSM_TEXTURE_RESOLUTION);
    const ivec2 in_page_texel_coord = ivec2(mod(virtual_texel_coord, daxa_f32(VSM_PAGE_SIZE)));
    const ivec2 in_memory_offset = physical_page_coords * VSM_PAGE_SIZE;
    const ivec2 memory_texel_coord = in_memory_offset + in_page_texel_coord;
    return memory_texel_coord;
}
