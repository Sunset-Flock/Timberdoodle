#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "shade_opaque.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/sky_util.glsl"
#include "shader_lib/vsm_util.glsl"


DAXA_DECL_PUSH_CONSTANT(ShadeOpaquePush, push)
#define AT_FROM_PUSH deref(push.attachments).attachments

float compute_exposure(float average_luminance) 
{
    const float exposure_bias = deref(AT_FROM_PUSH.globals).postprocess_settings.exposure_bias;
    const float calibration = deref(AT_FROM_PUSH.globals).postprocess_settings.calibration;
    const float sensor_sensitivity = deref(AT_FROM_PUSH.globals).postprocess_settings.exposure_bias;
    const float ev100 = log2(average_luminance * sensor_sensitivity * exposure_bias / calibration);
	const float exposure = 1.0 / (1.2 * exp2(ev100));
	return exposure;
}

struct AtmosphereLightingInfo
{
    // illuminance from atmosphere along normal vector
    vec3 atmosphere_normal_illuminance;
    // illuminance from atmosphere along view vector
    vec3 atmosphere_direct_illuminance;
    // direct sun illuminance
    vec3 sun_direct_illuminance;
};

vec3 get_sun_direct_lighting(daxa_BufferPtr(SkySettings) settings, vec3 view_direction, vec3 world_position)
{
    const float bottom_atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        vec3(0.0, 0.0, length(world_position)),
        view_direction,
        vec3(0.0),
        deref(settings).atmosphere_bottom
    );
    bool view_ray_intersects_ground = bottom_atmosphere_intersection_distance >= 0.0;
    const vec3 direct_sun_illuminance = view_ray_intersects_ground ? 
        vec3(0.0) : 
        get_sun_illuminance(
            settings,
            AT_FROM_PUSH.transmittance,
            deref(AT_FROM_PUSH.globals).samplers.linear_clamp,
            view_direction,
            length(world_position),
            dot(deref(settings).sun_direction, normalize(world_position))
        );
    return direct_sun_illuminance;
}

// ndc going in needs to be in range [-1, 1]
vec3 get_view_direction(vec2 ndc_xy)
{
    daxa_BufferPtr(SkySettings) settings = deref(AT_FROM_PUSH.globals).sky_settings_ptr;
    vec3 world_direction; 
    if(deref(AT_FROM_PUSH.globals).settings.draw_from_observer == 1)
    {
        const vec3 camera_position = deref(AT_FROM_PUSH.globals).observer_camera.position;
        const vec4 unprojected_pos = deref(AT_FROM_PUSH.globals).observer_camera.inv_view_proj * vec4(ndc_xy, 1.0, 1.0);
        world_direction = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera_position);
    }
    else 
    {
        const vec3 camera_position = deref(AT_FROM_PUSH.globals).camera.position;
        const vec4 unprojected_pos = deref(AT_FROM_PUSH.globals).camera.inv_view_proj * vec4(ndc_xy, 1.0, 1.0);
        world_direction = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera_position);
    }
    return world_direction;
}

vec3 hsv2rgb(vec3 c) {
    vec4 k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

vec3 get_vsm_debug_page_color(vec2 uv, float depth)
{
    vec3 color = vec3(1.0, 1.0, 1.0);

    const mat4x4 inv_projection_view = deref(AT_FROM_PUSH.globals).camera.inv_view_proj;
    const int force_clip_level = -1;//deref(_globals).force_view_clip_level ? deref(_globals).vsm_debug_clip_level : -1;

    ClipInfo clip_info;
    // When we are using debug camera and no clip level is manually forced we need to
    // search through all the clip levels linearly to see if one level contains VSM entry
    // for this tile - This is a DEBUG ONLY thing the perf will probably suffer
    // if(deref(_globals).use_debug_camera && !deref(_globals).force_view_clip_level)
    // {
    //     for(daxa_i32 clip_level = 0; clip_level < VSM_CLIP_LEVELS; clip_level++)
    //     {
    //         clip_info = clip_info_from_uvs(ClipFromUVsInfo(
    //             uv,
    //             pc.offscreen_resolution,
    //             depth,
    //             inv_projection_view,
    //             camera_offset,
    //             clip_level
    //         ));
    //         const daxa_i32vec3 vsm_page_texel_coords = vsm_clip_info_to_wrapped_coords(clip_info);
    //         const daxa_u32 page_entry = texelFetch(daxa_utexture2DArray(_vsm_page_table), vsm_page_texel_coords, 0).r;

    //         if(get_is_visited_marked(page_entry)) {break;}
    //     }
    // }
    // else 
    // {
        uvec2 render_target_size = deref(AT_FROM_PUSH.globals).settings.render_target_size;
        daxa_BufferPtr(VSMClipProjection) vsm_clip_projections = AT_FROM_PUSH.vsm_clip_projections;
        daxa_BufferPtr(VSMGlobals) vsm_globals = AT_FROM_PUSH.vsm_globals;
        clip_info = clip_info_from_uvs(ClipFromUVsInfo(
            uv,
            render_target_size,
            depth,
            inv_projection_view,
            force_clip_level,
            vsm_clip_projections,
            vsm_globals
        ));
    // }
    if(clip_info.clip_level >= VSM_CLIP_LEVELS) { return color; }

    const ivec3 vsm_page_texel_coords = vsm_clip_info_to_wrapped_coords(clip_info, vsm_clip_projections);
    const uint page_entry = texelFetch(daxa_utexture2DArray(AT_FROM_PUSH.vsm_page_table), vsm_page_texel_coords, 0).r;

    if(get_is_allocated(page_entry))
    {
        const ivec2 physical_page_coords = get_meta_coords_from_vsm_entry(page_entry);
        const ivec2 physical_texel_coords = virtual_uv_to_physical_texel(clip_info.sun_depth_uv, physical_page_coords);
        const ivec2 in_page_texel_coords = ivec2(mod(physical_texel_coords, float(VSM_PAGE_SIZE)));
        bool texel_near_border = any(greaterThan(in_page_texel_coords, ivec2(126))) ||
                                 any(lessThan(in_page_texel_coords, ivec2(2)));
        if(texel_near_border)
        {
            color = vec3(0.001, 0.001, 0.001);
        } else {
            // color = clip_to_color[int(mod(clip_info.clip_level, float(NUM_CLIP_VIZ_COLORS)))];
            color = hsv2rgb(vec3(float(clip_info.clip_level) / float(VSM_CLIP_LEVELS), 1.0, 0.5));
            if(get_is_visited_marked(page_entry)) {color = vec3(1.0, 1.0, 0.0);}
        }
    } else {
        color = vec3(1.0, 0.0, 0.0);
        if(get_is_dirty(page_entry)) {color = vec3(0.0, 0.0, 1.0);}
    }
    return color;
}

float mip_map_level(vec2 ddx, vec2 ddy)
{
    float delta_max_sqr = max(dot(ddx, ddx), dot(ddy, ddy));
    return 0.5 * log2(delta_max_sqr);
}

layout(local_size_x = SHADE_OPAQUE_WG_X, local_size_y = SHADE_OPAQUE_WG_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    const vec2 screen_uv = vec2(gl_GlobalInvocationID.xy) * deref(AT_FROM_PUSH.globals).settings.render_target_size_inv;
    if ( all(equal(index, ivec2(0,0))) )
    {
        deref(deref(AT_FROM_PUSH.globals).debug).gpu_output.debug_ivec4.x = int(deref(AT_FROM_PUSH.instantiated_meshlets).first_count);
        deref(deref(AT_FROM_PUSH.globals).debug).gpu_output.debug_ivec4.y = int(deref(AT_FROM_PUSH.instantiated_meshlets).second_count);
    }
    const uint triangle_id = imageLoad(daxa_uimage2D(AT_FROM_PUSH.vis_image), index).x;
    vec4 output_value = vec4(0,0,0,0);
    vec4 debug_value = vec4(0, 0, 0, 0);
    if (triangle_id != INVALID_TRIANGLE_ID)
    {
        mat4x4 view_proj;
        vec3 camera_position;
        if (deref(AT_FROM_PUSH.globals).settings.draw_from_observer == 1)
        {
            view_proj = deref(AT_FROM_PUSH.globals).observer_camera.view_proj;
            camera_position = deref(AT_FROM_PUSH.globals).observer_camera.position;
        } 
        else
        {
            view_proj = deref(AT_FROM_PUSH.globals).camera.view_proj;
            camera_position = deref(AT_FROM_PUSH.globals).camera.position;
        }
        daxa_BufferPtr(MeshletInstancesBufferHead) instantiated_meshlets = AT_FROM_PUSH.instantiated_meshlets;
        daxa_BufferPtr(GPUMesh) meshes = AT_FROM_PUSH.meshes;
        daxa_BufferPtr(daxa_f32mat4x3) combined_transforms = AT_FROM_PUSH.combined_transforms;
        VisbufferTriangleData tri_data = visgeo_triangle_data(
            triangle_id, 
            vec2(index), 
            push.size,
            push.inv_size,
            view_proj, 
            instantiated_meshlets,
            meshes,
            combined_transforms);
        vec3 normal = tri_data.world_normal;

        // vec4 debug_tex_value = texelFetch(daxa_texture2D(AT_FROM_PUSH.debug_image), index, 0);

        GPUMaterial material;
        material.diffuse_texture_id.value = 0;
        material.normal_texture_id.value = 0;
        material.roughnes_metalness_id.value = 0;
        material.alpha_discard_enabled = false;
        material.normal_compressed_bc5_rg = false;
        if (tri_data.meshlet_instance.material_index != INVALID_MANIFEST_INDEX)
        {
            material = deref(AT_FROM_PUSH.material_manifest[tri_data.meshlet_instance.material_index]);
        }

        ivec2 diffuse_size = textureSize(daxa_texture2D(material.diffuse_texture_id), 0);
        const float manually_calc_mip = mip_map_level(tri_data.uv_ddx * diffuse_size, tri_data.uv_ddy * diffuse_size);

        vec3 albedo = (0.5f).xxx;
        if(material.diffuse_texture_id.value != 0)
        {
            albedo = textureGrad(daxa_sampler2D(material.diffuse_texture_id, deref(AT_FROM_PUSH.globals).samplers.linear_repeat_ani), tri_data.uv, tri_data.uv_ddx, tri_data.uv_ddy).rgb;
        }

        if(material.normal_texture_id.value != 0)
        {
            vec3 normal_map_value = vec3(0,0,0);
            if (material.normal_compressed_bc5_rg)
            {
                const vec2 raw = textureGrad(daxa_sampler2D(material.normal_texture_id, deref(AT_FROM_PUSH.globals).samplers.normals), tri_data.uv, tri_data.uv_ddx, tri_data.uv_ddy).rg;
                const vec2 rescaled_normal_rg = raw * 2.0f - 1.0f;
                const float normal_b = sqrt(clamp(1.0f - dot(rescaled_normal_rg.rg, rescaled_normal_rg.rg ), 0.0, 1.0));
                normal_map_value = vec3(rescaled_normal_rg, normal_b);
            }
            else
            {
                const vec3 raw = textureGrad(daxa_sampler2D(material.normal_texture_id, deref(AT_FROM_PUSH.globals).samplers.normals), tri_data.uv, tri_data.uv_ddx, tri_data.uv_ddy).rgb;
                normal_map_value = raw * 2.0f - 1.0f;
            }
            mat3 tbn = mat3(-tri_data.world_tangent, -cross(tri_data.world_tangent, tri_data.world_normal), tri_data.world_normal);
            normal = tbn * normal_map_value;
            debug_value = vec4(tri_data.world_tangent * 0.5 + 0.5, 1.0);
        }
        
        output_value = debug_value;
#if 0
        const vec3 light_position = vec3(-5,-5,15);
        const vec3 light_power = vec3(1,1,1) * 100;
        const float light_distance = length(tri_data.world_position - light_position);
        const vec3 to_light_dir = normalize(light_position - tri_data.world_position);
        vec3 color = (albedo.rgb * light_power) * (max(0.0, dot(to_light_dir, normal)) * 1/(light_distance*light_distance));
        output_value.rgb = color;
#else
        const vec3 sun_direction = deref(AT_FROM_PUSH.globals).sky_settings.sun_direction;
        const float sun_norm_dot = clamp(dot(normal, sun_direction), 0.0, 1.0);
        // This will be multiplied with shadows once added
        const float shadow = sun_norm_dot;
        daxa_BufferPtr(SkySettings) settings = deref(AT_FROM_PUSH.globals).sky_settings_ptr;
        // Because the atmosphere is using km as it's default units and we want one unit in world
        // space to be one meter we need to scale the position by a factor to get from meters -> kilometers
        const vec3 atmo_camera_position = deref(AT_FROM_PUSH.globals).camera.position * M_TO_KM_SCALE;
        vec3 world_camera_position = atmo_camera_position + vec3(0.0, 0.0, deref(settings).atmosphere_bottom + BASE_HEIGHT_OFFSET);

        const vec3 direct_lighting = shadow * get_sun_direct_lighting(settings, sun_direction, world_camera_position);
        const vec4 norm_indirect_ligting = texture( daxa_samplerCube( AT_FROM_PUSH.sky_ibl, deref(AT_FROM_PUSH.globals).samplers.linear_clamp), normal).rgba;
        const vec3 indirect_ligting = norm_indirect_ligting.rgb * norm_indirect_ligting.a;
        const vec3 lighting = direct_lighting + indirect_ligting;

        const uint visualize_clip_levels = deref(AT_FROM_PUSH.globals).vsm_settings.visualize_clip_levels;
        const vec3 vsm_debug_color = visualize_clip_levels == 0 ? vec3(1.0f) : get_vsm_debug_page_color(screen_uv, tri_data.depth);

        output_value.rgb = albedo.rgb * lighting * vsm_debug_color;
#endif

#if 0
        output_value.rgb = hsv2rgb(vec3(floor(manually_calc_mip) * 0.1, 1, 0.5));
#endif
#if 0
        output_value.rgb = normal * 0.5f + 0.5f;
#endif

        float combined_indices = tri_data.meshlet_instance.meshlet_index + tri_data.meshlet_instance.mesh_index * 100 + tri_data.meshlet_instance.entity_index * 1000;
        //output_value = vec4(vec3(cos(combined_indices), cos(1 + 2.53252343422 * combined_indices), cos(2 + 3.3111223232 * combined_indices)) * 0.5f + 0.5f, 1);
    } else {
        // scale uvs to be in the range [0, 1]
        const vec2 ndc_xy = (screen_uv * 2.0) - 1.0;
        const vec3 view_direction = get_view_direction(ndc_xy);

        daxa_BufferPtr(SkySettings) settings = deref(AT_FROM_PUSH.globals).sky_settings_ptr;
        // Because the atmosphere is using km as it's default units and we want one unit in world
        // space to be one meter we need to scale the position by a factor to get from meters -> kilometers
        const vec3 camera_position = deref(AT_FROM_PUSH.globals).settings.draw_from_observer == 1 ? 
            deref(AT_FROM_PUSH.globals).observer_camera.position * M_TO_KM_SCALE :
            deref(AT_FROM_PUSH.globals).camera.position * M_TO_KM_SCALE;

        vec3 world_camera_position = camera_position + vec3(0.0, 0.0, deref(settings).atmosphere_bottom + BASE_HEIGHT_OFFSET);

        const vec3 atmosphere_direct_illuminance = get_atmosphere_illuminance_along_ray(
            settings,
            AT_FROM_PUSH.transmittance,
            AT_FROM_PUSH.sky,
            deref(AT_FROM_PUSH.globals).samplers.linear_clamp,
            view_direction,
            world_camera_position
        );
        const vec3 total_direct_illuminance = get_sun_direct_lighting(settings, view_direction, world_camera_position) + atmosphere_direct_illuminance;
        // const vec4 total_direct_illuminance = texture( daxa_samplerCube( AT_FROM_PUSH.sky_ibl, deref(AT_FROM_PUSH.globals).samplers.linear_clamp), view_direction).rgba;
        // output_value = vec4(total_direct_illuminance.rgb * total_direct_illuminance.a, 1.0);
        output_value = vec4(total_direct_illuminance, 1.0);
    }

    const float exposure = compute_exposure(deref(AT_FROM_PUSH.luminance_average));
    const vec3 exposed_color = output_value.rgb * exposure;

    uvec2 detector_window_index;
    daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info = deref(AT_FROM_PUSH.globals).debug;
    debug_write_lens(
        debug_info, 
        AT_FROM_PUSH.debug_lens_image, 
        index, 
        vec4(exposed_color, 1.0f));
    imageStore(daxa_image2D(AT_FROM_PUSH.color_image), index, vec4(exposed_color, output_value.a));
}