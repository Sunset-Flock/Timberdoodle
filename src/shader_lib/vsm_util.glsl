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

bool get_is_allocated(daxa_u32 page_entry)        { return (page_entry & allocated_mask()) != 0; }
bool get_requests_allocation(daxa_u32 page_entry) { return (page_entry & requests_allocation_mask()) != 0; }
bool get_allocation_failed(daxa_u32 page_entry)   { return (page_entry & allocation_failed_mask()) != 0; }
bool get_is_dirty(daxa_u32 page_entry)            { return (page_entry & dirty_mask()) != 0; }
bool get_is_visited_marked(daxa_u32 page_entry)   { return (page_entry & visited_marked_mask()) != 0; }

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
// BIT 30 -> 1 - NEEDS_CLEAR
// BIT 29 -> 1 - VISITED
daxa_u32 meta_memory_allocated_mask()   { return 1 << 31; }
daxa_u32 meta_memory_needs_clear_mask() { return 1 << 30; }
daxa_u32 meta_memory_visited_mask()     { return 1 << 29; }

bool get_meta_memory_is_allocated(daxa_u32 meta_page_entry){ return (meta_page_entry & meta_memory_allocated_mask()) != 0; }
bool get_meta_memory_needs_clear(daxa_u32 meta_page_entry) { return (meta_page_entry & meta_memory_needs_clear_mask()) != 0; }
bool get_meta_memory_is_visited(daxa_u32 meta_page_entry)  { return (meta_page_entry & meta_memory_visited_mask()) != 0; }

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

struct ClipInfo
{
    daxa_i32 clip_level;
    daxa_f32vec2 clip_depth_uv;
};

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

daxa_f32 get_page_offset_depth(ClipInfo info, daxa_f32 current_depth, daxa_BufferPtr(VSMClipProjection) clip_projections)
{
    const daxa_i32vec2 non_wrapped_page_coords = daxa_i32vec2(info.clip_depth_uv * VSM_PAGE_TABLE_RESOLUTION);
    const daxa_i32vec2 inverted_page_coords = daxa_i32vec2((VSM_PAGE_TABLE_RESOLUTION - 1) - non_wrapped_page_coords);
    const daxa_f32vec2 per_inv_page_depth_offset = deref_i(clip_projections, info.clip_level).depth_page_offset;
    const daxa_f32 depth_offset = 
        inverted_page_coords.x * per_inv_page_depth_offset.x +
        inverted_page_coords.y * per_inv_page_depth_offset.y;
    return clamp(current_depth + depth_offset, 0.0, 1.0);
}

daxa_f32vec3 world_space_from_uv(daxa_f32vec2 screen_space_uv, daxa_f32 depth, daxa_f32mat4x4 inv_view_proj)
{
    const daxa_f32vec2 remap_uv = (screen_space_uv * 2.0) - 1.0;
    const daxa_f32vec4 ndc_position = daxa_f32vec4(remap_uv, depth, 1.0);
    const daxa_f32vec4 unprojected_ndc_position = mul(inv_view_proj, ndc_position);

    const daxa_f32vec3 world_position = unprojected_ndc_position.xyz / unprojected_ndc_position.w;
    return world_position;
}

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
        clip_level = daxa_i32(clamp(ceil(log2((dist / deref(info.globals).vsm_settings.clip_0_frustum_scale))), 0, VSM_CLIP_LEVELS - 1));
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
    return ClipInfo(clip_level, sun_depth_uv);
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
    const daxa_u32 linear_index = vsm_page_coords.x * VSM_PAGE_TABLE_RESOLUTION + vsm_page_coords.y;
    const daxa_u32 index_offset = linear_index / 32;
    const daxa_u32 in_uint_offset = daxa_u32(_mod(daxa_f32(linear_index),daxa_f32(32)));
    const daxa_u32 mask_entry = deref_i(info, vsm_page_coords.z).mask[index_offset];
    const daxa_u32 decoded_bit = mask_entry & (1 << in_uint_offset);
    return decoded_bit;
}