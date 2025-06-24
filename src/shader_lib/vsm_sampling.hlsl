#pragma once

#include <daxa/daxa.inl>

#include "shader_shared/shared.inl"
#include "vsm_util.glsl"
#include "misc.hlsl"

static const uint PCF_NUM_SAMPLES = 4;
// https://developer.download.nvidia.com/whitepapers/2008/PCSS_Integration.pdf
static const float2 poisson_disk[16] = {
    float2( -0.94201624, -0.39906216 ),
    float2( 0.94558609, -0.76890725 ),
    float2( -0.094184101, -0.92938870 ),
    float2( 0.34495938, 0.29387760 ),
    float2( -0.91588581, 0.45771432 ),
    float2( -0.81544232, -0.87912464 ),
    float2( -0.38277543, 0.27676845 ),
    float2( 0.97484398, 0.75648379 ),
    float2( 0.44323325, -0.97511554 ),
    float2( 0.53742981, -0.47373420 ),
    float2( -0.26496911, -0.41893023 ),
    float2( 0.79197514, 0.19090188 ),
    float2( -0.24188840, 0.99706507 ),
    float2( -0.81409955, 0.91437590 ),
    float2( 0.19984126, 0.78641367 ),
    float2( 0.14383161, -0.14100790 )
};











///
/// POINT LIGHTS
///

float vsm_point_shadow_test(
    Texture2D<float> vsm_memory_block, 
    VSMPointLight* vsm_point_lights, 
    PointMipInfo info, 
    uint vsm_page_entry, 
    float3 world_position, 
    int point_light_idx,
    float point_norm_dot)
{
    let memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);

    const int2 physical_texel_coords = int2(info.page_uvs * (VSM_POINT_SPOT_TEXTURE_RESOLUTION / (1 << int(info.mip_level))));
    const int2 in_page_texel_coords = int2(_mod(physical_texel_coords, float(VSM_PAGE_SIZE)));

    const uint2 in_memory_offset = memory_page_coords * VSM_PAGE_SIZE;
    const uint2 memory_texel_coord = in_memory_offset + in_page_texel_coords;

    let shadow_view = vsm_point_lights[point_light_idx].face_cameras[info.cube_face].view;
    let shadow_proj = vsm_point_lights[point_light_idx].face_cameras[info.cube_face].proj;
    const float4 world_position_in_shadow_vs = mul(shadow_view, float4(world_position, 1.0f));
    const float shadow_vs_offset = 0.07 / abs(point_norm_dot);
    const float4 offset_world_position_in_shadow_vs = float4(world_position_in_shadow_vs.xy, world_position_in_shadow_vs.z + shadow_vs_offset, 1.0f);
    const float4 world_position_in_shadow_cs = mul(shadow_proj, offset_world_position_in_shadow_vs);

    const float depth = world_position_in_shadow_cs.z / world_position_in_shadow_cs.w;
    const float vsm_depth = vsm_memory_block.Load(int3(memory_texel_coord, 0)).r;

    const bool is_in_shadow = depth < vsm_depth;
    return is_in_shadow ? 0.0f : 1.0f;
}

float get_vsm_point_shadow(
    RenderGlobalData* globals, 
    VSMGlobals* vsm_globals,
    Texture2D<float> vsm_memory_block, 
    daxa::RWTexture2DArrayId<daxa_u32>* vsm_point_spot_page_table,
    VSMPointLight* vsm_point_lights, 
    float2 screen_uv, 
    float3 world_normal, 
    int point_light_idx, 
    ScreenSpacePixelWorldFootprint pixel_footprint,
    float point_norm_dot)
{
    PointMipInfo info = project_into_point_light(point_light_idx, pixel_footprint, globals, vsm_point_lights, vsm_globals);
    if(info.cube_face == -1) 
    {
        return float(1.0f);
    }
    info.mip_level = clamp(info.mip_level, 0, 6);

    const float filter_radius = 0.05;
    float sum = 0.0;

    rand_seed(asuint(screen_uv.x + screen_uv.y * 13136.1235f) * globals.frame_index);

    for(int sample = 0; sample < PCF_NUM_SAMPLES; sample++)
    {
        float theta = (rand()) * 2 * PI;
        float r = sqrt(rand());
        let filter_rot_offset =  float2(cos(theta), sin(theta)) * r;

        let level = 0;
        let filter_view_space_offset = float4(filter_rot_offset * filter_radius, 0.0, 0.0);

        let mip_proj = vsm_point_lights[point_light_idx].face_cameras[info.cube_face].proj;
        let mip_view = vsm_point_lights[point_light_idx].face_cameras[info.cube_face].view;

        pixel_footprint.center += world_normal * 0.007;
        let view_space_world_pos = mul(mip_view, float4(pixel_footprint.center, 1.0));
        let view_space_offset_world_pos = view_space_world_pos + filter_view_space_offset;
        let clip_filter_offset_world = mul(mip_proj, view_space_offset_world_pos);

        let clip_uv = ((clip_filter_offset_world.xy / clip_filter_offset_world.w) + 1.0) / 2.0;

        if(all(greaterThanEqual(clip_uv, 0.0)) && all(lessThan(clip_uv, 1.0)))
        {
            const int2 texel_coords = int2(clip_uv * (VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION / (1 << info.mip_level)));
            const uint point_page_array_index = get_vsm_point_page_array_idx(info.cube_face, point_light_idx);
            const uint vsm_page_entry = vsm_point_spot_page_table[info.mip_level].get()[int3(texel_coords, point_page_array_index)];
            info.page_uvs = clip_uv;
            info.page_texel_coords = texel_coords;

            if(get_is_allocated(vsm_page_entry))
            {
                sum += vsm_point_shadow_test(
                    vsm_memory_block,
                    vsm_point_lights,
                    info, 
                    vsm_page_entry, 
                    pixel_footprint.center, 
                    point_light_idx,
                    point_norm_dot);
            }
        }
        else
        {
            sum += 1.0f;
        }
    }
    return sum / PCF_NUM_SAMPLES;
}

float get_vsm_point_shadow_coarse(
    RenderGlobalData* globals, 
    VSMGlobals* vsm_globals,
    Texture2D<float> vsm_memory_block, 
    daxa::RWTexture2DArrayId<daxa_u32>* vsm_point_spot_page_table,
    VSMPointLight* vsm_point_lights, 
    float3 world_normal, 
    float3 world_position,
    int point_light_idx,
    float point_norm_dot)
{
    const float3 point_ws = vsm_point_lights[point_light_idx].light.position;
    const float3 point_to_frag_norm = normalize(world_position - point_ws);

    const int face_idx = cube_face_from_dir(point_to_frag_norm);
    if(face_idx == -1) 
    {
        return float(1.0f);
    }
    PointMipInfo info;
    info.mip_level = 6;
    info.cube_face = face_idx;
    info.page_texel_coords = uint2(0,0);

    let mip_proj = vsm_point_lights[point_light_idx].face_cameras[face_idx].proj;
    let mip_view = vsm_point_lights[point_light_idx].face_cameras[face_idx].view;

    world_position += world_normal * 0.07;
    let view_space_world_pos = mul(mip_view, float4(world_position, 1.0));
    let view_space_offset_world_pos = view_space_world_pos;
    let clip_filter_offset_world = mul(mip_proj, view_space_offset_world_pos);

    info.page_uvs = ((clip_filter_offset_world.xy / clip_filter_offset_world.w) + 1.0) / 2.0;

    if(all(greaterThanEqual(info.page_uvs, 0.0)) && all(lessThan(info.page_uvs, 1.0)))
    {
        const uint point_page_array_index = get_vsm_point_page_array_idx(face_idx, point_light_idx);
        const uint vsm_page_entry = vsm_point_spot_page_table[info.mip_level].get()[int3(info.page_texel_coords, point_page_array_index)];
        if(get_is_allocated(vsm_page_entry))
        {
            return vsm_point_shadow_test(
                vsm_memory_block,
                vsm_point_lights,
                info, 
                vsm_page_entry, 
                world_position, 
                point_light_idx,
                point_norm_dot);
        }
    }
    return 0.0f;
}












///
/// SPOT LIGHTS
///

float vsm_spot_shadow_test(
    RenderGlobalData* globals,
    VSMGlobals* vsm_globals,
    Texture2D<float> vsm_memory_block, 
    VSMSpotLight* vsm_spot_lights,
    SpotMipInfo info, 
    uint vsm_page_entry, 
    float3 world_position, 
    int light_index)
{
        let memory_page_coords = get_meta_coords_from_vsm_entry(vsm_page_entry);

        const int2 physical_texel_coords = int2(info.page_uvs * (VSM_POINT_SPOT_TEXTURE_RESOLUTION / (1 << int(info.mip_level))));
        const int2 in_page_texel_coords = int2(_mod(physical_texel_coords, float(VSM_PAGE_SIZE)));

        const uint2 in_memory_offset = memory_page_coords * VSM_PAGE_SIZE;
        const uint2 memory_texel_coord = in_memory_offset + in_page_texel_coords;


        let shadow_view = vsm_spot_lights[light_index].camera.view;
        let shadow_proj = vsm_spot_lights[light_index].camera.proj;
        const float4 world_position_in_shadow_vs = mul(shadow_view, float4(world_position, 1.0f));
        const float shadow_vs_offset = 0.1;
        const float4 offset_world_position_in_shadow_vs = float4(world_position_in_shadow_vs.xy, world_position_in_shadow_vs.z + shadow_vs_offset, 1.0f);
        const float4 world_position_in_shadow_cs = mul(shadow_proj, offset_world_position_in_shadow_vs);

        const float depth = world_position_in_shadow_cs.z / world_position_in_shadow_cs.w;
        const float vsm_depth = vsm_memory_block.Load(int3(memory_texel_coord, 0)).r;

        const bool is_in_shadow = depth < vsm_depth;
        return is_in_shadow ? 0.0f : 1.0f;
}

float get_vsm_spot_shadow(
    RenderGlobalData* globals,
    VSMGlobals* vsm_globals,
    Texture2D<float> vsm_memory_block, 
    daxa::RWTexture2DArrayId<daxa_u32>* vsm_point_spot_page_table,
    VSMSpotLight* vsm_spot_lights,
    float2 screen_uv, 
    float3 world_normal, 
    int spot_light_idx, 
    ScreenSpacePixelWorldFootprint pixel_footprint)
{
    SpotMipInfo info = project_into_spot_light(spot_light_idx, pixel_footprint, globals, vsm_spot_lights, vsm_globals);
    if(info.page_texel_coords.x == -1) 
    {
        return float(1.0f);
    }
    info.mip_level = clamp(info.mip_level, 0, 6);

    const float filter_radius = 0.05;
    float sum = 0.0;

    rand_seed(asuint(screen_uv.x + screen_uv.y * 13136.1235f) * globals.frame_index);

    for(int sample = 0; sample < PCF_NUM_SAMPLES; sample++)
    {
        //int final_sample = poisson
        float theta = (rand()) * 2 * PI;
        float r = sqrt(rand());
        let filter_rot_offset =  float2(cos(theta), sin(theta)) * r;

        let level = 0;
        let filter_view_space_offset = float4(filter_rot_offset * filter_radius, 0.0, 0.0);

        let mip_proj = vsm_spot_lights[spot_light_idx].camera.proj;
        let mip_view = vsm_spot_lights[spot_light_idx].camera.view;

        pixel_footprint.center += world_normal * 0.001;
        let view_space_world_pos = mul(mip_view, float4(pixel_footprint.center, 1.0));
        let view_space_offset_world_pos = view_space_world_pos + filter_view_space_offset;
        let clip_filter_offset_world = mul(mip_proj, view_space_offset_world_pos);

        let clip_uv = ((clip_filter_offset_world.xy / clip_filter_offset_world.w) + 1.0) / 2.0;

        if(all(greaterThanEqual(clip_uv, 0.0)) && all(lessThan(clip_uv, 1.0)))
        {
            const int2 texel_coords = int2(clip_uv * (VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION / (1 << info.mip_level)));
            const uint spot_page_array_index = VSM_SPOT_LIGHT_OFFSET + spot_light_idx;
            const uint vsm_page_entry = vsm_point_spot_page_table[info.mip_level].get()[int3(texel_coords, spot_page_array_index)];
            info.page_uvs = clip_uv;
            info.page_texel_coords = texel_coords;

            if(get_is_allocated(vsm_page_entry))
            {
                sum += vsm_spot_shadow_test(
                    globals,
                    vsm_globals,
                    vsm_memory_block,
                    vsm_spot_lights,
                    info, 
                    vsm_page_entry, 
                    pixel_footprint.center, 
                    spot_light_idx);
            }
        }
        else
        {
            sum += 1.0f;
        }
    }
    return sum / PCF_NUM_SAMPLES;
}


float get_vsm_spot_shadow_coarse(
    RenderGlobalData* globals,
    VSMGlobals* vsm_globals,
    Texture2D<float> vsm_memory_block, 
    daxa::RWTexture2DArrayId<daxa_u32>* vsm_point_spot_page_table,
    VSMSpotLight* vsm_spot_lights,
    float3 world_normal, 
    float3 world_position,
    int spot_light_idx)
{
    SpotMipInfo info;
    info.mip_level = 6;

    //int final_sample = poisson
    float theta = (rand()) * 2 * PI;
    float r = sqrt(rand());
    let filter_rot_offset =  float2(cos(theta), sin(theta)) * r;

    let mip_proj = vsm_spot_lights[spot_light_idx].camera.proj;
    let mip_view = vsm_spot_lights[spot_light_idx].camera.view;

    world_position += world_normal * 0.001;
    let view_space_world_pos = mul(mip_view, float4(world_position, 1.0));
    let view_space_offset_world_pos = view_space_world_pos;
    let clip_filter_offset_world = mul(mip_proj, view_space_offset_world_pos);

    let clip_uv = ((clip_filter_offset_world.xy / clip_filter_offset_world.w) + 1.0) / 2.0;

    if(all(greaterThanEqual(clip_uv, 0.0)) && all(lessThan(clip_uv, 1.0)))
    {
        const int2 texel_coords = int2(clip_uv * (VSM_POINT_SPOT_PAGE_TABLE_RESOLUTION / (1 << info.mip_level)));
        const uint spot_page_array_index = VSM_SPOT_LIGHT_OFFSET + spot_light_idx;
        const uint vsm_page_entry = vsm_point_spot_page_table[info.mip_level].get()[int3(texel_coords, spot_page_array_index)];
        info.page_uvs = clip_uv;
        info.page_texel_coords = texel_coords;

        if(get_is_allocated(vsm_page_entry))
        {
            return vsm_spot_shadow_test(
                globals,
                vsm_globals,
                vsm_memory_block,
                vsm_spot_lights,
                info, 
                vsm_page_entry, 
                world_position, 
                spot_light_idx);
        }
    }
    return 1.0f;
}