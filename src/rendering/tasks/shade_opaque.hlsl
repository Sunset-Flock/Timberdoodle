#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include "shade_opaque.inl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/sky_util.glsl"
#include "shader_lib/vsm_sampling.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/volumetric.hlsl"
#include "shader_lib/shading.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/lights.hlsl"
#include "../path_trace/kajiya/math_const.hlsl"

static int debug_mark_light_influence_counter = 0;

[[vk::push_constant]] ShadeOpaquePush push_opaque;

#define RT 0

#define AT deref(push_opaque.attachments).attachments

float compute_exposure(float average_luminance) 
{
    const float exposure_bias = AT.globals->postprocess_settings.exposure_bias;
    const float calibration = AT.globals->postprocess_settings.calibration;
    const float sensor_sensitivity = AT.globals->postprocess_settings.sensor_sensitivity;
    const float ev100 = log2(average_luminance * sensor_sensitivity * exposure_bias / calibration);
	const float exposure = 1.0 / (1.2 * exp2(ev100));
	return exposure;
}

// ndc going in needs to be in range [-1, 1]
float3 get_view_direction(float2 ndc_xy)
{
    const float3 camera_position = AT.globals->view_camera.position;
    const float4 unprojected_pos = mul(AT.globals->view_camera.inv_view_proj, float4(ndc_xy, 1.0, 1.0));
    const float3 world_direction = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera_position);
    return world_direction;
}

float map(float value, float min1, float max1, float min2, float max2)
{
    // Convert the current value to a percentage
    // 0% - min1, 100% - max1
    float perc = (value - min1) / (max1 - min1);

    // Do the same operation backwards with min2 and max2
    float value = perc * (max2 - min2) + min2;
    return value;
}

float3 get_vsm_spot_debug_page_color(ScreenSpacePixelWorldFootprint pixel_footprint)
{
    const uint spot_light_index = max(AT.globals.vsm_settings.force_spot_light_idx, 0);
    
    SpotMipInfo info = project_into_spot_light(spot_light_index, pixel_footprint, AT.globals, AT.vsm_spot_lights, AT.vsm_globals);

    let light = AT.vsm_spot_lights[spot_light_index].light;

    const float3 position_to_light = normalize(light.position - pixel_footprint.center);
    // the scale and offset computations can be done CPU-side
    float cos_outer = cos(light.outer_cone_angle - 0.02);
    float spot_scale = 1.0 / max(cos(light.inner_cone_angle) - cos_outer, 1e-4);
    float spot_offset = -cos_outer * spot_scale;
    float cd = dot(-position_to_light, light.direction);

    float angle_attenuation = clamp(cd * spot_scale + spot_offset, 0.0, 1.0);
    angle_attenuation = angle_attenuation * angle_attenuation;

    if(info.mip_level == -1 || angle_attenuation == 0.0f) 
    {
        return float3(0.6f);
    }

    float3 color = hsv2rgb(float3(0.5f, (6.0f - float(info.mip_level)) / 6.0f, 1.0f));
    const uint spot_page_array_index = spot_light_index + VSM_SPOT_LIGHT_OFFSET;
    const uint vsm_page_entry = AT.vsm_point_spot_page_table[info.mip_level].get()[int3(info.page_texel_coords.xy, spot_page_array_index)];

    const int2 physical_texel_coords = info.page_uvs * (VSM_POINT_SPOT_TEXTURE_RESOLUTION / (1 << int(info.mip_level)));
    const int2 in_page_texel_coords = int2(_mod(physical_texel_coords, float(VSM_PAGE_SIZE)));

    bool texel_near_border = any(greaterThan(in_page_texel_coords, int2(VSM_PAGE_SIZE - 2))) ||
                             any(lessThan(in_page_texel_coords, int2(2))) ||
                             any(lessThan(info.page_uvs, 0.02)) ||
                             any(greaterThan(info.page_uvs, 0.98));


    if(!get_is_visited_marked(vsm_page_entry)) 
    {
        color = float3(0.05, 0.05, 0.05);
    }

    if(texel_near_border)
    {
        color = float3(0.01, 0.01, 0.01);
    }

    return color;
}

float3 get_vsm_point_debug_page_color(ScreenSpacePixelWorldFootprint pixel_footprint)
{
    const uint point_light_index = max(AT.globals.vsm_settings.force_point_light_idx, 0);
    
    PointMipInfo info = project_into_point_light(point_light_index, pixel_footprint, AT.globals, AT.vsm_point_lights, AT.vsm_globals);
    if(info.mip_level == -1) 
    {
        return float3(0.6f);
    }

    float3 color = hsv2rgb(float3(6 - float(info.cube_face) / 6.0f, float(6 - int(info.mip_level)) / 6.0f, 1.0));
    const uint point_page_array_index = get_vsm_point_page_array_idx(info.cube_face, point_light_index);
    const uint vsm_page_entry = AT.vsm_point_spot_page_table[info.mip_level].get()[int3(info.page_texel_coords.xy, point_page_array_index)];

    const int2 physical_texel_coords = info.page_uvs * (VSM_POINT_SPOT_TEXTURE_RESOLUTION / (1 << int(info.mip_level)));
    const int2 in_page_texel_coords = int2(_mod(physical_texel_coords, float(VSM_PAGE_SIZE)));

    bool texel_near_border = any(greaterThan(in_page_texel_coords, int2(VSM_PAGE_SIZE - 1))) ||
                             any(lessThan(in_page_texel_coords, int2(1)));

    if(!get_is_visited_marked(vsm_page_entry)) 
    {
        color = float3(0.05, 0.05, 0.05);
    }

    if(texel_near_border)
    {
        color = float3(0.01, 0.01, 0.01);
    }

    return color;
}

float3 get_vsm_debug_page_color(ScreenSpacePixelWorldFootprint pixel_footprint)
{
    float3 color = float3(1.0, 1.0, 1.0);

    const bool level_forced = AT.globals->vsm_settings.force_clip_level != 0;
    const int force_clip_level = level_forced ? AT.globals->vsm_settings.forced_clip_level : -1;

    ClipInfo clip_info;
    clip_info = clip_info_from_uvs(ClipFromUVsInfo(
        force_clip_level,
        AT.globals->main_camera.position,   
        pixel_footprint,
        AT.vsm_clip_projections,
        AT.vsm_globals,
        AT.globals
    ));
    if(clip_info.clip_level >= VSM_CLIP_LEVELS) { return color; }

    const daxa_i32vec3 vsm_page_pix_coords = daxa_i32vec3(daxa_i32vec2(floor(clip_info.clip_depth_uv * VSM_DIRECTIONAL_PAGE_TABLE_RESOLUTION)), clip_info.clip_level);
    const uint is_dynamic_invalidated = unwrap_vsm_page_from_mask(vsm_page_pix_coords, AT.vsm_wrapped_pages);
    const int3 vsm_page_texel_coords = vsm_clip_info_to_wrapped_coords(clip_info, AT.vsm_clip_projections);
    const uint page_entry = Texture2DArray<uint>::get(AT.vsm_page_table).Load(int4(vsm_page_texel_coords, 0)).r;
    color.rgb = hsv2rgb(float3(pow(float(vsm_page_texel_coords.z) / float(VSM_CLIP_LEVELS - 1), 0.5), 1.0, 1.0));

    if(get_is_visited_marked(page_entry) || get_is_dirty(page_entry) || get_is_allocated(page_entry))
    {
        const int2 physical_page_coords = get_meta_coords_from_vsm_entry(page_entry);
        const int2 physical_texel_coords = virtual_uv_to_physical_texel(clip_info.clip_depth_uv, physical_page_coords);
        uint overdraw_amount = 0;
        if (AT.globals->settings.debug_draw_mode == DEBUG_DRAW_MODE_VSM_OVERDRAW)
        {
            overdraw_amount = RWTexture2D<uint>::get(AT.vsm_overdraw_debug)[physical_texel_coords].x;
        }
        const int2 in_page_texel_coords = int2(_mod(physical_texel_coords, float(VSM_PAGE_SIZE)));
        bool texel_near_border = any(greaterThan(in_page_texel_coords, int2(VSM_PAGE_SIZE - 2))) ||
                                 any(lessThan(in_page_texel_coords, int2(2)));
        if(texel_near_border)
        {
            color = float3(0.01, 0.01, 0.01);
        }
        else if(is_dynamic_invalidated != 0)
        {
            color.rgb = float3(1.0, 0.0, 1.0);
        }
        if (AT.globals->settings.debug_draw_mode == DEBUG_DRAW_MODE_VSM_OVERDRAW)
        {
            const float3 overdraw_color = 3.0 * TurboColormap(float(overdraw_amount) / 25.0);
            color.rgb = overdraw_color;
        }
    } else {
        if(get_is_dirty(page_entry)) {color = float3(0.0, 0.0, 1.0);}
    }
    return color;
}

float vsm_shadow_test(ClipInfo clip_info, uint page_entry, float3 world_position, float3 page_camera_position, float sun_norm_dot, float2 screen_uv, float3 normal)
{
    const int2 physical_page_coords = get_meta_coords_from_vsm_entry(page_entry);
    const int2 physical_texel_coords = virtual_uv_to_physical_texel(clip_info.clip_depth_uv, physical_page_coords);


    const float vsm_sample = Texture2D<float>::get(AT.vsm_memory_block).Load(int3(physical_texel_coords, 0)).r;

    const float4x4 vsm_shadow_view = deref_i(AT.vsm_clip_projections, clip_info.clip_level).camera.view;

    float4x4 vsm_shifted_shadow_view = vsm_shadow_view;
    vsm_shifted_shadow_view[0][3] = page_camera_position[0]; 
    vsm_shifted_shadow_view[1][3] = page_camera_position[1]; 
    vsm_shifted_shadow_view[2][3] = page_camera_position[2]; 

    const float4x4 vsm_shadow_proj = deref_i(AT.vsm_clip_projections, clip_info.clip_level).camera.proj;
    const float4x4 vsm_inv_shadow_proj = deref_i(AT.vsm_clip_projections, clip_info.clip_level).camera.inv_proj;

#if 0
    const float2 center_texel_uv = float2(int2(clip_info.clip_depth_uv * VSM_DIRECTIONAL_TEXTURE_RESOLUTION) + 0.5f) / VSM_DIRECTIONAL_TEXTURE_RESOLUTION;
    // const float2 center_texel_uv = float2(clip_info.clip_depth_uv * VSM_DIRECTIONAL_TEXTURE_RESOLUTION)) / VSM_DIRECTIONAL_TEXTURE_RESOLUTION;
    const float3 center_texel_ndc = float3((center_texel_uv) * 2.0f - 1.0f, 0.0f);
    const float3 ray_direction = mul(vsm_inv_shadow_proj, float4(0.0f, 0.0f, -1.0f, 1.0f)).xyz;
    const float3 ray_origin = mul(vsm_inv_shadow_proj, float4(center_texel_ndc, 1.0f)).xyz;
    const float3 plane_normal = mul(vsm_shifted_shadow_view, float4(normal, 1.0f)).xyz;
    const float3 plane_origin = mul(vsm_shifted_shadow_view, float4(world_position, 1.0f)).xyz;

    const float vsm_unbiased_projected_view_depth = ray_plane_intersection(ray_direction, ray_origin, plane_normal, plane_origin).z;
    const float vsm_unbiased_projected_depth = mul(vsm_shadow_proj, float4(center_texel_uv, vsm_unbiased_projected_view_depth, 1.0f)).z;
    const float lf = 2000.0f;
    const float ln = 0.01f;
    const float K = 0.0000001f + 0.0000002f * pow(max(0.0, clip_info.clip_level - 2), 2.0);
    const float scale_factor = min(1.0f / sun_norm_dot, 10);
    const float bias = pow(lf - (vsm_unbiased_projected_depth * (lf - ln)), 2.0f) / (lf * ln * (lf - ln));
    const float vsm_projected_depth = vsm_unbiased_projected_depth - (bias * K * scale_factor);
#else
    const float sqrt2 = 1.41421356f;
    const float quantize = 2.0f / (1 << 23);
    const float b = sqrt2 * AT.vsm_globals.clip_0_texel_world_size * pow(2.0f, clip_info.clip_level + 1) / 2.0f;
    const float bias = quantize + b * length(cross(normal, AT.globals.sky_settings.sun_direction)) / sun_norm_dot;

    const float3 view_projected_world_pos = (mul(vsm_shifted_shadow_view, daxa_f32vec4(world_position, 1.0))).xyz;

    const float view_space_offset = bias;//0.04;// / abs(sun_norm_dot);//0.004 * pow(2.0, clip_info.clip_level);// / max(abs(sun_norm_dot), 0.05);
    const float3 offset_view_pos = float3(view_projected_world_pos.xy, view_projected_world_pos.z + view_space_offset);

    const float4 vsm_projected_world = mul(vsm_shadow_proj, float4(offset_view_pos, 1.0));
    const float vsm_projected_depth = (vsm_projected_world.z / vsm_projected_world.w);
#endif

    const bool is_in_shadow = vsm_sample < vsm_projected_depth;
    return is_in_shadow ? 0.0 : 1.0;
}

float get_dir_shadow_rt(float3 position, float3 normal, float3 primary_ray, float2 uv)
{
    rand_seed(asuint(uv.x + uv.y * 13136.1235f) * AT.globals.frame_index);
    float offset_scale = 0.0;
    float sum = 0.0;

    for(int i = 0; i < PCF_NUM_SAMPLES; ++i)
    {
        float3 offset = ((float3(rand(), rand(), rand()) * 2.0f) - 1.0f) * offset_scale;
        //float3 light_position = position + AT.globals.sky_settings.sun_direction * 100000.0f;
        //float3 offset_light_position = light_position + offset;

        float t_max = 100000.0f;//length(offset_light_position - position);
        float3 direction = normalize(AT.globals.sky_settings.sun_direction);
        // float3 start = rt_calc_ray_start(position, normal, primary_ray);
        float t = rayquery_shadow_path(AT.tlas.get(), position, direction, t_max, AT.globals, AT.mesh_instances.instances);
        sum += float(t != t_max);
    }
    return 1.0f - (sum / PCF_NUM_SAMPLES);
}

float get_vsm_shadow(float2 screen_uv, float sun_norm_dot, ScreenSpacePixelWorldFootprint pixel_footprint, float3 normal, float3 tangent, float3 bitangent)
{
    const bool level_forced = AT.globals->vsm_settings.force_clip_level != 0;
    const int force_clip_level = level_forced ? AT.globals->vsm_settings.forced_clip_level : -1;

    ClipInfo clip_info;
    let base_clip_info = ClipFromUVsInfo(
        force_clip_level,
        AT.globals.main_camera.position,
        pixel_footprint,   
        AT.vsm_clip_projections,
        AT.vsm_globals,
        AT.globals
    );
    clip_info = clip_info_from_uvs(base_clip_info);
    if(clip_info.clip_level >= VSM_CLIP_LEVELS) { return 1.0; }

    const float filter_radius = 0.05;
    float sum = 0.0;

    rand_seed(asuint(screen_uv.x + screen_uv.y * 13136.1235f) * AT.globals.frame_index);

    for(int sample = 0; sample < PCF_NUM_SAMPLES; sample++)
    {
        float theta = (rand()) * 2 * PI;
        float r = sqrt(rand());
        let filter_rot_offset =  float2(cos(theta), sin(theta)) * r;

        let level = 0;
        let filter_view_space_offset = float4(filter_rot_offset * filter_radius * pow(1.0, clip_info.clip_level), 0.0, 0.0);

        let clip_proj = AT.vsm_clip_projections[clip_info.clip_level].camera.proj;
        let clip_view = AT.vsm_clip_projections[clip_info.clip_level].camera.view;

        let offset_world_center = pixel_footprint.center + filter_view_space_offset.x * tangent + filter_view_space_offset.y * bitangent;
        let view_space_world_pos = mul(clip_view, float4(offset_world_center, 1.0));
        let proj_filter_offset_world = mul(clip_proj, view_space_world_pos);

        let clip_uv = ((proj_filter_offset_world.xy / proj_filter_offset_world.w) + 1.0) / 2.0;
        let offset_info = ClipInfo(clip_info.clip_level, clip_uv, 0.0f);

        if(all(greaterThanEqual(offset_info.clip_depth_uv, 0.0)) && all(lessThan(offset_info.clip_depth_uv, 1.0)))
        {
            let vsm_page_texel_coords = vsm_clip_info_to_wrapped_coords(offset_info, AT.vsm_clip_projections);
            let page_entry = Texture2DArray<uint>::get(AT.vsm_page_table).Load(int4(vsm_page_texel_coords, 0)).r;
            const float3 page_camera_pos = Texture2DArray<float3>::get(AT.vsm_page_view_pos_row).Load(int4(vsm_page_texel_coords, 0));

            if(get_is_allocated(page_entry))
            {
                sum += vsm_shadow_test(offset_info, page_entry, offset_world_center, page_camera_pos, sun_norm_dot, screen_uv, normal);
            }
        }
    }
    return sum / PCF_NUM_SAMPLES;
}

float get_rt_point_shadow(GPUPointLight light, float3 position, float3 normal, float3 primary_ray, float2 uv)
{
    rand_seed(asuint(uv.x + uv.y * 13136.1235f) * AT.globals.frame_index);
    float offset_scale = 0.000;
    float sum = 0.0;
    for(int i = 0; i < PCF_NUM_SAMPLES; ++i)
    {
        float3 offset = ((float3(rand(), rand(), rand()) * 2.0f) - 1.0f) * offset_scale;
        float3 offset_light_position = light.position + offset;

        float t_max = length(offset_light_position - position) - 0.05;
        float3 direction = normalize(offset_light_position - position);
        float3 start = position;//rt_calc_ray_start(position, normal, primary_ray);
        float t = rayquery_shadow_path(AT.tlas.get(), start, direction, t_max, AT.globals, AT.mesh_instances.instances);
        sum += bool(t < t_max);
    }
    return 1.0f - float(sum / PCF_NUM_SAMPLES);
}

float3 point_lights_contribution(
    float2 screen_uv,
    float3 shading_normal,
    float3 world_normal,
    GPUPointLight * lights,
    uint light_count,
    ScreenSpacePixelWorldFootprint pixel_footprint,
    bool skip_shadows,
    float3 view_dir)
{
    float3 total_contribution = float3(0.0);
    let light_settings = AT.globals.light_settings;
#if LIGHTS_ENABLE_MASK_ITERATION
    let mask_volume = AT.light_mask_volume.get();
    uint4 light_mask = lights_get_mask(light_settings, pixel_footprint.center, mask_volume);
    light_mask = light_mask & light_settings.point_light_mask;
    light_mask = WaveActiveBitOr(light_mask);
    while (any(light_mask != uint4(0)))
    {
        uint light_index = lights_iterate_mask(light_settings, light_mask);
#else
    for(int light_index = 0; light_index < light_count; light_index++)
    {
#endif
        GPUPointLight light = lights[light_index];
        const float3 position_to_light = normalize(light.position - pixel_footprint.center);
        const float diffuse = max(dot(shading_normal, position_to_light), 0.0);
        const float point_norm_dot = dot(world_normal, position_to_light);

        const float to_light_dist = length(light.position - pixel_footprint.center);

        float attenuation = lights_attenuate_point(to_light_dist, light.cutoff);
        float shadowing = 1.0f;
        if(attenuation > 0.0f && !skip_shadows) {
#if RT
            shadowing = get_rt_point_shadow(light, pixel_footprint.center, world_normal, view_dir, screen_uv);
#else
            shadowing = get_vsm_point_shadow(
                AT.globals,
                AT.vsm_globals,
                Texture2D<float>::get(AT.vsm_memory_block),
                &(AT.vsm_point_spot_page_table[0]),
                AT.vsm_point_lights,
                screen_uv, 
                world_normal, 
                light_index,
                pixel_footprint,
                point_norm_dot);
#endif
                
            if (light_settings.debug_mark_influence && 
                light_settings.debug_draw_point_influence && 
                (light_settings.selected_debug_point_light == -1 || 
                 light_settings.selected_debug_point_light == light_index) &&
                (shadowing > 0.0f || !light_settings.debug_mark_influence_shadowed))
            {
                debug_mark_light_influence_counter += 1;
            }
        }

        total_contribution += light.color * diffuse * attenuation * light.intensity * shadowing;
    }
    return total_contribution;
}

float get_rt_spot_shadow(GPUSpotLight light, float3 position, float3 normal, float3 primary_ray, float2 uv)
{
    rand_seed(asuint(uv.x + uv.y * 13136.1235f) * AT.globals.frame_index);
    float offset_scale = 0.000;
    float sum = 0.0;
    for(int i = 0; i < PCF_NUM_SAMPLES; ++i)
    {
        float3 offset = ((float3(rand(), rand(), rand()) * 2.0f) - 1.0f) * offset_scale;
        float3 offset_light_position = light.position + offset;

        float t_max = length(offset_light_position - position) - 0.05;
        float3 direction = normalize(offset_light_position - position);
        float3 start = rt_calc_ray_start(position, normal, primary_ray);
        float t = rayquery_shadow_path(AT.tlas.get(), start, direction, t_max, AT.globals, AT.mesh_instances.instances);
        sum += t < t_max;
    }
    return 1.0f - float(sum / PCF_NUM_SAMPLES);
}

float3 spot_lights_contribution(
    float2 screen_uv,
    float3 shading_normal,
    float3 world_normal,
    GPUSpotLight * lights,
    uint light_count,
    ScreenSpacePixelWorldFootprint pixel_footprint, 
    bool skip_shadows,
    float3 view_dir)
{
    float3 total_contribution = float3(0.0);
#if LIGHTS_ENABLE_MASK_ITERATION
    let mask_volume = AT.light_mask_volume.get();
    let light_settings = AT.globals.light_settings;
    uint4 light_mask = lights_get_mask(light_settings, pixel_footprint.center, mask_volume);
    light_mask = light_mask & light_settings.spot_light_mask;
    light_mask = WaveActiveBitOr(light_mask);
    while (any(light_mask != uint4(0)))
    {
        uint light_index = lights_iterate_mask(light_settings, light_mask) - light_settings.first_spot_light_instance;
#else
    for(int light_index = 0; light_index < light_count; light_index++)
    {
#endif
        GPUSpotLight light = lights[light_index];
        const float3 position_to_light = normalize(light.position - pixel_footprint.center);
        const float diffuse = max(dot(shading_normal, position_to_light), 0.0);
        const float to_light_dist = length(light.position - pixel_footprint.center);
        const float attenuation = lights_attenuate_spot(position_to_light, to_light_dist, light);
        float shadowing = 1.0f;
        if(attenuation > 0.0f && !skip_shadows) {
#if RT
            shadowing = get_rt_spot_shadow(light, pixel_footprint.center, world_normal, view_dir, screen_uv);
#else
            shadowing = get_vsm_spot_shadow(
                AT.globals,
                AT.vsm_globals,
                Texture2D<float>::get(AT.vsm_memory_block),
                &(AT.vsm_point_spot_page_table[0]),
                AT.vsm_spot_lights,
                screen_uv, 
                world_normal, 
                light_index, 
                pixel_footprint);
#endif
                
            if (light_settings.debug_mark_influence && 
                light_settings.debug_draw_spot_influence && 
                (light_settings.selected_debug_spot_light == -1 || 
                 light_settings.selected_debug_spot_light == light_index) &&
                (shadowing > 0.0f || !light_settings.debug_mark_influence_shadowed))
            {
                debug_mark_light_influence_counter += 1;
            }
        }

        total_contribution += light.color * diffuse * attenuation * light.intensity * shadowing;
    }
    return total_contribution;
}

struct VsmLightVisibilityTester : LightVisibilityTesterI
{
    RaytracingAccelerationStructure tlas;
    RenderGlobalData* globals;
    ScreenSpacePixelWorldFootprint footprint;
    float2 screen_uv;
    float sun_light(MaterialPointData material_point, float3 incoming_ray)
    {
        const float3 sun_direction = AT.globals->sky_settings.sun_direction;
        const float sun_norm_dot = clamp(dot(material_point.normal, sun_direction), 0.0001, 1.0);
        // TODO: fix
        float3 tangent = float3(0.0, 0.0, 1.0);
        float shadow = AT.globals->vsm_settings.enable != 0 ? get_vsm_shadow(screen_uv, sun_norm_dot, footprint, material_point.normal, tangent, tangent) : 1.0f;
        const float final_shadow = sun_norm_dot * shadow.x;

        return final_shadow;
    }
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        return 0.0f;
    }
    float spot_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        return 1.0f;
    }
}

groupshared uint gs_sorter_array[1024]; // Shared memory for sorting
void BitonicSort(uint in_group_thread)
{
    uint index = in_group_thread;

    // Bitonic sorting phase
    for (uint k = 2; k <= 1024; k *= 2)
    {
        for (uint j = k / 2; j > 0; j /= 2)
        {
            uint ixj = index ^ j;
            if (ixj > index)
            {
                if ((index & k) == 0)
                {
                    if (gs_sorter_array[index] > gs_sorter_array[ixj])
                    {
                        uint temp = gs_sorter_array[index];
                        gs_sorter_array[index] = gs_sorter_array[ixj];
                        gs_sorter_array[ixj] = temp;
                    }
                }
                else
                {
                    if (gs_sorter_array[index] < gs_sorter_array[ixj])
                    {
                        uint temp = gs_sorter_array[index];
                        gs_sorter_array[index] = gs_sorter_array[ixj];
                        gs_sorter_array[ixj] = temp;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
}

[numthreads(SHADE_OPAQUE_WG_X, SHADE_OPAQUE_WG_Y, 1)]
[shader("compute")]
void entry_main_cs(
    uint3 svdtid : SV_DispatchThreadID,
    uint2 in_group_id : SV_GroupThreadID,
)
{
    let push = push_opaque;

    if (svdtid.x == 0 && svdtid.y == 0 && false)
    {
        push.attachments.attachments.globals.readback.first_pass_meshlet_count_post_cull = push.attachments.attachments.instantiated_meshlets.pass_counts[0];
        push.attachments.attachments.globals.readback.second_pass_meshlet_count_post_cull = push.attachments.attachments.instantiated_meshlets.pass_counts[1];

        ShaderDebugRectangleDraw rect;
        rect.color = float3(2, 1, 0);
        rect.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        rect.center = float3(0, 0, -2.5f);
        rect.span = float2(0.5f, 0.5f);
        debug_draw_rectangle(AT.globals.debug, rect);
        
        ShaderDebugCircleDraw circle;
        circle.color = float3(2,1,0);
        circle.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        circle.position = float3(0, 0, -2.5f);
        circle.radius = 0.5f;
        debug_draw_circle(AT.globals.debug, circle);
        
        ShaderDebugAABBDraw aabb;
        aabb.color = float3(2,1,0);
        aabb.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        aabb.position = float3(0, 0, -2.5f);
        aabb.size = (0.5f).xxx;
        debug_draw_aabb(AT.globals.debug, aabb);
        
        ShaderDebugBoxDraw box;
        box.color = float3(2,1,0);
        box.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        box.vertices[0] = float3(0,0,0) - 0.5f - float3(0,0,2.5f);
        box.vertices[1] = float3(0,0,1) - 0.5f - float3(0,0,2.5f);
        box.vertices[2] = float3(0,1,0) - 0.5f - float3(0,0,2.5f);
        box.vertices[3] = float3(0,1,1) - 0.5f - float3(0,0,2.5f);
        box.vertices[4] = float3(1,0,0) - 0.5f - float3(0,0,2.5f);
        box.vertices[5] = float3(1,0,1) - 0.5f - float3(0,0,2.5f);
        box.vertices[6] = float3(1,1,0) - 0.5f - float3(0,0,2.5f);
        box.vertices[7] = float3(1,1,1) - 0.5f - float3(0,0,2.5f);
        debug_draw_box(AT.globals.debug, box);

        ShaderDebugLineDraw line;
        line.color = float3(2,1,0);
        line.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        line.start = float3(0,0,0) - 0.5f - float3(0,0,2.5f);
        line.end = float3(1,1,1) - 0.5f - float3(0,0,2.5f);
        debug_draw_line(AT.globals.debug, line);

        ShaderDebugConeDraw cone;
        cone.color = float3(2,1,0);
        cone.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        cone.position = float3(0,0,-2.5f);
        cone.direction = normalize(float3(1,1,-1));
        cone.size = 1.0f;
        cone.angle = 0.8f;
        debug_draw_cone(AT.globals.debug, cone);

        ShaderDebugSphereDraw sphere;
        sphere.color = float3(2,1,0);
        sphere.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        sphere.position = float3(0,0,-2.5f);
        sphere.radius = 2.0f;
        debug_draw_sphere(AT.globals.debug, sphere);
    }

    int2 index = svdtid.xy;

    // Sort threads
    uint sort_key = 0;
    if (false) {
        uint triangle_id = INVALID_TRIANGLE_ID;
        if(all(lessThan(index, AT.globals->settings.render_target_size)))
        {
            triangle_id = AT.vis_image.get()[index].x;
        }
        bool triangle_id_valid = triangle_id != INVALID_TRIANGLE_ID;
    
        sort_key = (triangle_id << 10u) | (in_group_id.y << 5u) | (in_group_id.x);
        gs_sorter_array[in_group_id.x + in_group_id.y * 32] = sort_key;
        GroupMemoryBarrierWithGroupSync();
        BitonicSort(in_group_id.x + in_group_id.y * 32);
        GroupMemoryBarrierWithGroupSync();
        uint resorted_payload = gs_sorter_array[in_group_id.x + in_group_id.y * 32];
        uint2 resorted_in_group_id = uint2(resorted_payload & 0x1F, (resorted_payload >> 5u) & 0x1F);
    
        index = index - in_group_id + resorted_in_group_id;
    }

    let clk_start = clockARB();

    uint triangle_id = INVALID_TRIANGLE_ID;
    if(all(lessThan(index, AT.globals->settings.render_target_size)))
    {
        triangle_id = AT.vis_image.get()[index].x;
    }
    bool triangle_id_valid = triangle_id != INVALID_TRIANGLE_ID;

    const float2 screen_uv = (float2(svdtid.xy) + 0.5f) * AT.globals->settings.render_target_size_inv;

    const bool is_pixel_under_cursor = all(index == int2(floor(AT.globals->cursor_uv * AT.globals->settings.render_target_size)));
    if (is_pixel_under_cursor)
    {
        debug_pixel = true;
    }

    float3 atmo_position = get_atmo_position(AT.globals);

    float4 output_value = float4(0);
    float4 debug_value = float4(0);


    float4x4 view_proj = AT.globals->view_camera.view_proj;
    float3 camera_position = AT.globals->view_camera.position;
    CameraInfo camera = AT.globals->view_camera;

    float nonlinear_depth = AT.depth.get()[index];

    let primary_ray = normalize(pixel_index_to_world_space(camera, index, nonlinear_depth) - camera.position);

    float ambient_occlusion = 1.0f;
    if(triangle_id_valid)
    {
        daxa_BufferPtr(MeshletInstancesBufferHead) instantiated_meshlets = AT.instantiated_meshlets;
        daxa_BufferPtr(GPUMesh) meshes = AT.meshes;
        daxa_BufferPtr(daxa_f32mat4x3) combined_transforms = AT.combined_transforms;
        VisbufferTriangleGeometry visbuf_tri = visgeo_triangle_data(
            triangle_id,
            float2(index),
            push.size,
            push.inv_size,
            view_proj,
            instantiated_meshlets,
            meshes,
            combined_transforms
        );        
        TriangleGeometry tri_geo = visbuf_tri.tri_geo;
        TriangleGeometryPoint tri_point = visbuf_tri.tri_geo_point;
        float depth = visbuf_tri.depth;
        uint meshlet_triangle_index = visbuf_tri.meshlet_triangle_index;
        uint meshlet_instance_index = visbuf_tri.meshlet_instance_index;
        uint meshlet_index = visbuf_tri.meshlet_index;
        tri_point.face_normal = flip_face_normal_to_incoming(tri_point.face_normal, primary_ray);
        tri_point.world_normal = flip_normal_on_face_normal(tri_point.world_normal, tri_point.face_normal);

        {
            bool mark = false;
            switch (AT.globals.selected_mark_mode)
            {
                case MARK_SELECTED_MODE_ENTITY:
                {
                    mark = tri_geo.entity_index == AT.globals.selected_entity_index;
                    break;
                }
                case MARK_SELECTED_MODE_MESH:
                {
                    mark = 
                        tri_geo.entity_index == AT.globals.selected_entity_index &&
                        tri_geo.mesh_index == AT.globals.selected_mesh_index;
                    break;
                }
                case MARK_SELECTED_MODE_MESHLET:
                {
                    mark = 
                        tri_geo.entity_index == AT.globals.selected_entity_index &&
                        tri_geo.mesh_index == AT.globals.selected_mesh_index &&
                        visbuf_tri.meshlet_index == AT.globals.selected_meshlet_in_mesh_index;
                    break;
                }
                case MARK_SELECTED_MODE_TRIANGLE:
                {
                    mark = 
                        tri_geo.entity_index == AT.globals.selected_entity_index &&
                        tri_geo.mesh_index == AT.globals.selected_mesh_index &&
                        visbuf_tri.meshlet_index == AT.globals.selected_meshlet_in_mesh_index && 
                        visbuf_tri.meshlet_triangle_index == AT.globals.selected_triangle_in_meshlet_index;
                    break;
                }
            }
            AT.selected_mark_image.get()[index] = mark ? 1.0f : 0.0f;
        }

        if (is_pixel_under_cursor)
        {
            AT.globals.readback.hovered_entity = tri_geo.entity_index;
            AT.globals.readback.hovered_mesh_in_meshgroup = tri_geo.in_mesh_group_index;
            AT.globals.readback.hovered_mesh = tri_geo.mesh_index;
            AT.globals.readback.hovered_meshlet_in_mesh = visbuf_tri.meshlet_index;
            AT.globals.readback.hovered_triangle_in_meshlet = visbuf_tri.meshlet_triangle_index;
        }


        GPUMaterial material = GPU_MATERIAL_FALLBACK;
        if(tri_geo.material_index != INVALID_MANIFEST_INDEX)
        {
            material = AT.material_manifest[tri_geo.material_index];
        }

        MaterialPointData material_point = evaluate_material<SHADING_QUALITY_HIGH>(
            AT.globals,
            tri_geo,
            tri_point
        );
        if (AT.globals.settings.debug_material_quality == SHADING_QUALITY_LOW)
        {
            material_point = evaluate_material<SHADING_QUALITY_LOW>(
                AT.globals,
                tri_geo,
                tri_point
            );
        }
        material_point.normal = flip_normal_on_face_normal(material_point.normal, material_point.face_normal);
        material_point.geometry_normal = flip_normal_on_face_normal(material_point.normal, material_point.face_normal);
        let mapped_normal = material_point.normal;
        let albedo = material_point.albedo;
        

        // TODO(msakmary) refactor into a separate function.
        // ====================================================================================================================================
        bool skip_shadows = false;

        const float2 uv_offset = 0.5f * AT.globals->settings.render_target_size_inv.xy;
        float2 real_screen_space_uv = (float2(svdtid.xy) + float2(0.5f)) * AT.globals.settings.render_target_size_inv;
        float real_depth = depth;

        if (AT.globals.settings.draw_from_observer == 1u) 
        {
            const float4 main_cam_proj_world = mul(AT.globals->main_camera.view_proj, float4(tri_point.world_position, 1.0));
            const float2 ndc = main_cam_proj_world.xy / main_cam_proj_world.w;

            if(main_cam_proj_world.w < 0.0 || abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0)
            {
                skip_shadows = true;
            }
            real_screen_space_uv = (ndc + float2(1.0)) / float2(2.0);
            real_depth = main_cam_proj_world.z / main_cam_proj_world.w;
        }

        // Reprojecting screen space into world
        const float2 bottom_right = real_screen_space_uv + float2(uv_offset.x, uv_offset.y);
        const float3 bottom_right_ws = world_space_from_uv( bottom_right, real_depth, AT.globals.main_camera.inv_view_proj);

        const float2 bottom_left = real_screen_space_uv + float2(-uv_offset.x, uv_offset.y);
        const float3 bottom_left_ws = world_space_from_uv( bottom_left, real_depth, AT.globals.main_camera.inv_view_proj);

        const float2 top_right = real_screen_space_uv + float2(uv_offset.x, -uv_offset.y);
        const float3 top_right_ws = world_space_from_uv( top_right, real_depth, AT.globals.main_camera.inv_view_proj);

        const float2 top_left = real_screen_space_uv + float2(-uv_offset.x, -uv_offset.y);
        const float3 top_left_ws = world_space_from_uv( top_left, real_depth, AT.globals.main_camera.inv_view_proj);

        ScreenSpacePixelWorldFootprint ws_pixel_footprint;
        ws_pixel_footprint.center = tri_point.world_position;
        ws_pixel_footprint.bottom_right = ray_plane_intersection(normalize(bottom_right_ws - AT.globals.main_camera.position), AT.globals.main_camera.position, tri_point.face_normal, ws_pixel_footprint.center);
        ws_pixel_footprint.bottom_left = ray_plane_intersection(normalize(bottom_left_ws - AT.globals.main_camera.position), AT.globals.main_camera.position, tri_point.face_normal, ws_pixel_footprint.center);
        ws_pixel_footprint.top_right = ray_plane_intersection(normalize(top_right_ws - AT.globals.main_camera.position), AT.globals.main_camera.position, tri_point.face_normal, ws_pixel_footprint.center);
        ws_pixel_footprint.top_left = ray_plane_intersection(normalize(top_left_ws - AT.globals.main_camera.position), AT.globals.main_camera.position, tri_point.face_normal, ws_pixel_footprint.center);
        // ================================================================================================================

        const float3 sun_direction = AT.globals->sky_settings.sun_direction;
        const float sun_norm_dot = clamp(dot(tri_point.world_normal, sun_direction), 0.0001, 1.0);
        float shadow = 1.0f;
        if((AT.globals->vsm_settings.enable != 0 && !skip_shadows))
        {
#if RT
            shadow = get_dir_shadow_rt(tri_point.world_position, tri_point.world_normal, primary_ray, screen_uv);
#else 
            shadow = get_vsm_shadow(screen_uv, sun_norm_dot, ws_pixel_footprint, tri_point.face_normal, tri_point.world_tangent, tri_point.world_bitangent);
#endif
        }

        if (AT.globals->vsm_settings.shadow_everything == 1)
        {
            shadow = 0.0f;
        }
        const float final_shadow = sun_norm_dot * shadow.x;

        // Point lights and spot lights
        float3 point_lights_direct = point_lights_contribution(screen_uv, mapped_normal, tri_point.world_normal, AT.point_lights, AT.globals.vsm_settings.point_light_count, ws_pixel_footprint, skip_shadows, primary_ray);
        float3 spot_lights_direct = spot_lights_contribution(screen_uv, mapped_normal, tri_point.world_normal, AT.spot_lights, AT.globals.vsm_settings.spot_light_count, ws_pixel_footprint, skip_shadows, primary_ray);

        const float3 directional_light_direct = final_shadow * get_sun_direct_lighting(
            AT.globals, AT.transmittance, AT.sky,
            sun_direction, atmo_position);

        float3 indirect_lighting = {};        
        if (AT.globals.pgi_settings.enabled)
        {
            if (AT.globals.settings.draw_from_observer == 1)
            {
                float3 pgi_irradiance = pgi_sample_irradiance(
                    AT.globals,
                    &AT.globals.pgi_settings,
                    tri_point.world_position,
                    tri_point.world_normal,
                    material_point.normal,
                    camera.position,
                    AT.pgi_irradiance.get(),
                    AT.pgi_visibility.get(),
                    AT.pgi_info.get(),
                    AT.pgi_requests.get(),
                    PGI_PROBE_REQUEST_MODE_NONE
                );
                indirect_lighting = pgi_irradiance;
            }
            else
            {
                float3 pgi_irradiance = push.attachments.attachments.pgi_screen_irrdiance.get()[index].rgb;
                indirect_lighting = pgi_irradiance;
            }
        }
        else
        {
            const float4 compressed_indirect_lighting = TextureCube<float4>::get(AT.sky_ibl).SampleLevel(SamplerState::get(AT.globals->samplers.linear_clamp), mapped_normal, 0);
            indirect_lighting = compressed_indirect_lighting.rgb * 3.14 * 2 * compressed_indirect_lighting.a;
        }

        ambient_occlusion = 1.0f;
        const bool ao_enabled = (AT.globals.ao_settings.mode != AMBIENT_OCCLUSION_MODE_NONE) && !AT.ao_image.id.is_empty();
        if (ao_enabled)
        {
            ambient_occlusion = lerp(AT.ao_image.get().Load(index).r, 1.0f, 0.1f);
        }

        const bool rtgi_enabled = AT.globals.rtgi_settings.enabled;
        if (rtgi_enabled)
        {
            indirect_lighting = AT.rtgi_per_pixel_diffuse.get()[int3(index,0)].rgb * rcp(M_FRAC_1_PI);
            // ambient_occlusion = 1.0f;
        }


        const float3 lighting = (directional_light_direct + point_lights_direct + spot_lights_direct) + (indirect_lighting.rgb * ambient_occlusion);
        let shaded_color = albedo.rgb * M_FRAC_1_PI * lighting + material.emissive_color;


        float3 dummy_color = float3(1,0,1);
        uint mesh_group_index = AT.mesh_instances.instances[tri_geo.mesh_instance_index].mesh_group_index;
        switch(AT.globals->settings.debug_draw_mode)
        {
            case DEBUG_DRAW_MODE_OVERDRAW:
            {
                if (AT.overdraw_image.value != 0)
                {
                    let value = Texture2D<uint>::get(AT.overdraw_image)[index].x;
                    let scaled_value = float(value) * AT.globals->settings.debug_visualization_scale;
                    let color = TurboColormap(scaled_value);
                    output_value.rgb = color;
                }
                break;
            }
            case DEBUG_DRAW_MODE_TRIANGLE_CONNECTIVITY:
            {
                uint3 indices = tri_geo.vertex_indices;
                float3 color_x = hsv2rgb(float3(IdFloatScramble(indices.x), 1, 1));
                float3 color_y = hsv2rgb(float3(IdFloatScramble(indices.y), 1, 1));
                float3 color_z = hsv2rgb(float3(IdFloatScramble(indices.z), 1, 1));
                output_value.rgb = (tri_geo.barycentrics.x * color_x + tri_geo.barycentrics.y * color_y + tri_geo.barycentrics.z * color_z) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_TRIANGLE_ID:
            {
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(meshlet_triangle_index + meshlet_index + tri_geo.mesh_index + mesh_group_index + tri_geo.entity_index), 1, 1)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_MESHLET_ID:
            {
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(meshlet_index + tri_geo.mesh_index + mesh_group_index + tri_geo.entity_index), 1, IdFloatScramble(meshlet_triangle_index) * 0.2f + 0.8f)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_MESH_ID:
            {
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(tri_geo.mesh_index + mesh_group_index + tri_geo.entity_index), 1, 1)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_MESH_GROUP_ID:
            {
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(mesh_group_index + tri_geo.entity_index), 1, 1)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_ENTITY_ID:
            {
                output_value.rgb = hsv2rgb(float3(IdFloatScramble(tri_geo.entity_index), 1, 1)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_MESH_LOD:
            {
                uint lod = tri_geo.mesh_index % MAX_MESHES_PER_LOD_GROUP;
                output_value.rgb = TurboColormap(2 * float(lod) / float(MAX_MESHES_PER_LOD_GROUP)) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_VSM_OVERDRAW: 
            {
                let vsm_debug_color = get_vsm_debug_page_color(ws_pixel_footprint);
                output_value.rgb = vsm_debug_color * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_VSM_CLIP_LEVEL: 
            {
                if (AT.globals->vsm_settings.enable != 0)
                {
                    const float3 PERCEIVED_LUMINANCE_WEIGHTS = float3(0.2127, 0.7152, 0.0722);
                    let vsm_debug_color = get_vsm_debug_page_color(ws_pixel_footprint) * ambient_occlusion;
                    let debug_albedo = dot(lighting, PERCEIVED_LUMINANCE_WEIGHTS) * vsm_debug_color;
                    output_value.rgb = debug_albedo;
                }
                break;
            }
            case DEBUG_DRAW_MODE_VSM_SPOT_LEVEL:
            {
                const float3 PERCEIVED_LUMINANCE_WEIGHTS = float3(0.2127, 0.7152, 0.0722);
                let vsm_debug_color = get_vsm_spot_debug_page_color(ws_pixel_footprint) * ambient_occlusion;
                let debug_albedo = dot(lighting, PERCEIVED_LUMINANCE_WEIGHTS) * vsm_debug_color.rgb;
                output_value.rgb = debug_albedo;
                break;
            }
            case DEBUG_DRAW_MODE_VSM_POINT_LEVEL:
            {
                const float3 PERCEIVED_LUMINANCE_WEIGHTS = float3(0.2127, 0.7152, 0.0722);
                let vsm_debug_color = get_vsm_point_debug_page_color(ws_pixel_footprint) * ambient_occlusion;
                let debug_albedo = dot(lighting, PERCEIVED_LUMINANCE_WEIGHTS) * vsm_debug_color.rgb;
                output_value.rgb = debug_albedo;
                break;
            }
            case DEBUG_DRAW_MODE_DEPTH:
            {
                float depth = depth;
                let color = unband_z_color(index.x, index.y, linearise_depth(AT.globals.main_camera.near_plane, depth));
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_ALBEDO:
            {
                output_value.rgb = albedo;
                break;
            }
            case DEBUG_DRAW_MODE_FACE_NORMAL:
            {
                let color = tri_point.face_normal * 0.5 + 0.5f;
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_SMOOTH_NORMAL:
            {
                let color = tri_point.world_normal * 0.5 + 0.5f;
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_MAPPED_NORMAL:
            {
                let color = mapped_normal * 0.5 + 0.5f;
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_FACE_TANGENT:
            {
                let color = tri_point.world_tangent * 0.5 + 0.5f;
                output_value.rgb = color;
                break;
            }
            case DEBUG_DRAW_MODE_SMOOTH_TANGENT:
            {
                output_value.rgb = float3(frac(tri_point.uv), 0);
                break;
            }
            case DEBUG_DRAW_MODE_DIRECT_DIFFUSE:
            {
                output_value.rgb = (directional_light_direct + point_lights_direct + spot_lights_direct) * M_FRAC_1_PI;
                break;
            }
            case DEBUG_DRAW_MODE_INDIRECT_DIFFUSE:
            {
                output_value.rgb = indirect_lighting;
                break;
            }
            case DEBUG_DRAW_MODE_INDIRECT_DIFFUSE_AO:
            {
                output_value.rgb = indirect_lighting * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_AO:
            {
                output_value.rgb = ambient_occlusion.xxx;
                break;
            }
            case DEBUG_DRAW_MODE_ALL_DIFFUSE:
            {
                output_value.rgb = (directional_light_direct + point_lights_direct + spot_lights_direct) * M_FRAC_1_PI + indirect_lighting * ambient_occlusion + material.emissive_color;
                break;
            }
            case DEBUG_DRAW_MODE_UV:
            {
                output_value.rgb = float3(tri_point.uv, 1.0f) * ambient_occlusion;
                break;
            }
            case DEBUG_DRAW_MODE_PGI_CASCADE_SMOOTH:
            case DEBUG_DRAW_MODE_PGI_CASCADE_ABSOLUTE:
            {
                uint pgi_absolute_cascade = 0;
                bool pgi_is_center_8 = false;
                bool pgi_checker = false;
                float pgi_color_mul = 1.0f;
                PGISettings * pgi = &AT.globals.pgi_settings;
                for (int cascade = 0; cascade < pgi.cascade_count; ++cascade)
                {
                    let in_cascade = pgi_is_pos_in_cascade(pgi, material_point.position, cascade);
                    if (in_cascade)
                    {
                        pgi_absolute_cascade = cascade;
                        float3 grid_coord = pgi_grid_coord_of_position(pgi, material_point.position, cascade);
                        int4 base_probe = int4(floor(grid_coord), cascade);
                        pgi_is_center_8 = any(base_probe.xyz >= pgi.probe_count/2-1 && base_probe.xyz < pgi.probe_count/2);
                        let stable_index = pgi_probe_to_stable_index(pgi, base_probe);
                        pgi_checker = ((uint(stable_index.x + int(~0u >> 2)) & 0x1) != 0) ^ ((uint(stable_index.y + int(~0u >> 2)) & 0x1) != 0) ^ ((uint(stable_index.z + int(~0u >> 2)) & 0x1) != 0);
                        if (pgi_is_center_8)
                        {
                            pgi_color_mul *= 0.25f;
                        }
                        if (pgi_checker)
                        {
                            pgi_color_mul *= 0.75f;
                        }

                        break;
                    }
                }
                float smooth_cascade = pgi_select_cascade_smooth_spherical(pgi, material_point.position - AT.globals.main_camera.position);
                switch(AT.globals->settings.debug_draw_mode)
                {
                    case DEBUG_DRAW_MODE_PGI_CASCADE_SMOOTH:
                    {
                        float3 cascade_color = TurboColormap(float(smooth_cascade) * rcp(12)) * pgi_color_mul;
                        output_value.rgb = lerp(shaded_color * cascade_color, cascade_color, 0.4f);
                        break;
                    }
                    case DEBUG_DRAW_MODE_PGI_CASCADE_ABSOLUTE:
                    {
                        float3 cascade_color = TurboColormap(float(pgi_absolute_cascade) * rcp(12)) * pgi_color_mul;
                        output_value.rgb = lerp(shaded_color * cascade_color, cascade_color, 0.4f);
                        break;
                    }
                }
                break;
            }      
            case DEBUG_DRAW_MODE_PGI_LOW_QUALITY_SAMPLING:
            {
                float3 pgi_nearest_irradiance = pgi_sample_irradiance_nearest(
                    AT.globals,
                    &AT.globals.pgi_settings,
                    tri_point.world_position,
                    tri_point.world_normal,
                    material_point.normal,
                    camera.position,
                    AT.pgi_irradiance.get(),
                    AT.pgi_visibility.get(),
                    AT.pgi_info.get(),
                    AT.pgi_requests.get(),
                    PGI_PROBE_REQUEST_MODE_NONE
                );
                output_value.rgb = pgi_nearest_irradiance;
                break;
            }   
            case DEBUG_DRAW_MODE_LIGHT_MASK_VOLUME:
            {
                let mask_volume = AT.light_mask_volume.get();
                let light_settings = AT.globals.light_settings;
                uint4 light_mask = lights_get_mask(light_settings, material_point.position, mask_volume);
                int3 index = lights_get_mask_volume_cell(light_settings, material_point.position) / 4;
                //light_mask &= light_settings.spot_light_mask;
                
                bool checker = false;//((uint(index.x + int(~0u >> 2)) & 0x1) != 0) ^ ((uint(index.y + int(~0u >> 2)) & 0x1) != 0) ^ ((uint(index.z + int(~0u >> 2)) & 0x1) != 0);
                let light_count4 = countbits(light_mask);
                let light_count = light_count4.x + light_count4.y + light_count4.z + light_count4.w;
                if (lights_in_mask_volume(light_settings, lights_get_mask_volume_cell(light_settings, material_point.position)))
                {
                    output_value.rgb = TurboColormap(float(light_count) * rcp(123.0f)) * ambient_occlusion;
                }
                else
                {
                    output_value.rgb = ambient_occlusion.xxx;
                }
                output_value.rgb *= checker ? 0.6f : 1.0f;
                break;
            }  
            case DEBUG_DRAW_MODE_RTGI_DEBUG_PRIMARY_TRACE:
            {
                output_value.rgb = AT.rtgi_debug_primary_trace.get()[index/2].rgb;
                break;
            }
            case DEBUG_DRAW_MODE_NONE:
            default:
            output_value.rgb = shaded_color;
            break;
        }
        
        if (debug_mark_light_influence_counter && AT.globals.light_settings.debug_mark_influence)
        {
            float3 color = TurboColormap(float(debug_mark_light_influence_counter) * rcp(31.0f));
            output_value.rgb = lerp(color, output_value.rgb * color, 0.75f);
        }
    }
    else 
    {
        if (is_pixel_under_cursor)
        {
            AT.globals.readback.hovered_entity = ~0u;
        }

        const float2 ndc_xy = screen_uv * 2.0 - 1.0;
        const float3 view_direction = get_view_direction(ndc_xy);
        const float3 atmosphere_direct_illuminnace = get_atmosphere_illuminance_along_ray(
            AT.globals->sky_settings,
            AT.transmittance,
            AT.sky,
            AT.globals->samplers.linear_clamp,
            view_direction,
            atmo_position
        );
        const float3 sun_direct_illuminance = get_sun_direct_lighting(
            AT.globals, AT.transmittance, AT.sky,
            view_direction, atmo_position);
        const float3 total_direct_illuminance = sun_direct_illuminance + atmosphere_direct_illuminnace;
        output_value.rgb = total_direct_illuminance;
        debug_value.xyz = atmosphere_direct_illuminnace;
    }
        
    uint clocks = 0;
    switch(push.attachments.attachments.globals.settings.debug_draw_mode)
    {
        case DEBUG_DRAW_MODE_SHADE_OPAQUE_CLOCKS:
        {
            let clk_end = clockARB();
            clocks = uint(clk_end) - uint(clk_start);
            break;
        }
        case DEBUG_DRAW_MODE_PGI_EVAL_CLOCKS:
        case DEBUG_DRAW_MODE_RTAO_TRACE_CLOCKS:
        case DEBUG_DRAW_MODE_RTGI_TRACE_DIFFUSE_CLOCKS:
        {
            clocks = push.attachments.attachments.clocks_image.get()[index];
            break;
        }
    }
    if (clocks != 0)
    {
        output_value.rgb = TurboColormap(clocks * 0.0001f * push.attachments.attachments.globals.settings.debug_visualization_scale) * lerp(ambient_occlusion, 1.0f, 0.5f);
    }

    const float exposure = deref(AT.exposure);
    float3 exposed_color = output_value.rgb * exposure;
    
    AT.color_image.get()[index] = exposed_color;
}