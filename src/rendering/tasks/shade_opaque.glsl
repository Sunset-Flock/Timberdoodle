#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "shade_opaque.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/sky_util.glsl"


DAXA_DECL_PUSH_CONSTANT(ShadeOpaquePush, push)

float compute_exposure(float average_luminance) 
{
    const float exposure_bias = deref(push.attachments.globals).postprocess_settings.exposure_bias;
    const float calibration = deref(push.attachments.globals).postprocess_settings.calibration;
    const float sensor_sensitivity = deref(push.attachments.globals).postprocess_settings.exposure_bias;
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
            push.attachments.transmittance,
            deref(push.attachments.globals).samplers.linear_clamp,
            view_direction,
            length(world_position),
            dot(deref(settings).sun_direction, normalize(world_position))
        );
    return direct_sun_illuminance;
}

// ndc going in needs to be in range [-1, 1]
vec3 get_view_direction(vec2 ndc_xy)
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attachments.globals).sky_settings_ptr;
    vec3 world_direction; 
    if(deref(push.attachments.globals).settings.draw_from_observer == 1)
    {
        const vec3 camera_position = deref(push.attachments.globals).observer_camera.position;
        const vec4 unprojected_pos = deref(push.attachments.globals).observer_camera.inv_view_proj * vec4(ndc_xy, 1.0, 1.0);
        world_direction = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera_position);
    }
    else 
    {
        const vec3 camera_position = deref(push.attachments.globals).camera.position;
        const vec4 unprojected_pos = deref(push.attachments.globals).camera.inv_view_proj * vec4(ndc_xy, 1.0, 1.0);
        world_direction = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera_position);
    }
    return world_direction;
}

float mip_map_level(vec2 ddx, vec2 ddy)
{
    float delta_max_sqr = max(dot(ddx, ddx), dot(ddy, ddy));
    return 0.5 * log2(delta_max_sqr);
}

vec3 hsv2rgb(vec3 c) {
    vec4 k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
}

layout(local_size_x = SHADE_OPAQUE_WG_X, local_size_y = SHADE_OPAQUE_WG_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    if ( all(equal(index, ivec2(0,0))) )
    {
        deref(deref(push.attachments.globals).debug).gpu_output.debug_ivec4.x = int(deref(push.attachments.instantiated_meshlets).first_count);
        deref(deref(push.attachments.globals).debug).gpu_output.debug_ivec4.y = int(deref(push.attachments.instantiated_meshlets).second_count);
    }
    const uint triangle_id = imageLoad(daxa_uimage2D(push.attachments.vis_image), index).x;
    vec4 output_value = vec4(0,0,0,0);
    vec4 debug_value = vec4(0, 0, 0, 0);
    if (triangle_id != INVALID_TRIANGLE_ID)
    {
        mat4x4 view_proj;
        vec3 camera_position;
        if (deref(push.attachments.globals).settings.draw_from_observer == 1)
        {
            view_proj = deref(push.attachments.globals).observer_camera.view_proj;
            camera_position = deref(push.attachments.globals).observer_camera.position;
        } 
        else
        {
            view_proj = deref(push.attachments.globals).camera.view_proj;
            camera_position = deref(push.attachments.globals).camera.position;
        }
        VisbufferTriangleData tri_data = visgeo_triangle_data(
            triangle_id, 
            vec2(index), 
            push.size,
            push.inv_size,
            view_proj, 
            push.attachments.instantiated_meshlets,
            push.attachments.meshes,
            push.attachments.combined_transforms);
        vec3 normal = tri_data.world_normal;

        // vec4 debug_tex_value = texelFetch(daxa_texture2D(push.attachments.debug_image), index, 0);

        GPUMaterial material = deref(push.attachments.material_manifest[tri_data.meshlet_instance.material_index]);

        ivec2 diffuse_size = textureSize(daxa_texture2D(material.diffuse_texture_id), 0);
        const float manually_calc_mip = mip_map_level(tri_data.uv_ddx * diffuse_size, tri_data.uv_ddy * diffuse_size);

        vec3 albedo = (0.5f).xxx;
        if(material.diffuse_texture_id.value != 0)
        {
            albedo = textureGrad(daxa_sampler2D(material.diffuse_texture_id, deref(push.attachments.globals).samplers.linear_repeat_ani), tri_data.uv, tri_data.uv_ddx, tri_data.uv_ddy).rgb;
        }

        if(material.normal_texture_id.value != 0)
        {
            vec3 normal_map_value = vec3(0,0,0);
            if (material.normal_compressed_bc5_rg)
            {
                const vec2 raw = texture(daxa_sampler2D(material.normal_texture_id, deref(push.attachments.globals).samplers.linear_repeat_ani), tri_data.uv).rg;
                const vec2 rescaled_normal_rg = raw * 2.0f - 1.0f;
                const float normal_b = sqrt(clamp(1.0f - dot(rescaled_normal_rg.rg, rescaled_normal_rg.rg ), 0.0, 1.0));
                normal_map_value = vec3(rescaled_normal_rg, normal_b);
            }
            else
            {
                const vec3 raw = texture(daxa_sampler2D(material.normal_texture_id, deref(push.attachments.globals).samplers.linear_repeat_ani), tri_data.uv).rgb;
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
        const vec3 sun_direction = deref(push.attachments.globals).sky_settings.sun_direction;
        const float sun_norm_dot = clamp(dot(normal, sun_direction), 0.0, 1.0);
        // This will be multiplied with shadows once added
        const float shadow = sun_norm_dot;
        daxa_BufferPtr(SkySettings) settings = deref(push.attachments.globals).sky_settings_ptr;
        // Because the atmosphere is using km as it's default units and we want one unit in world
        // space to be one meter we need to scale the position by a factor to get from meters -> kilometers
        const vec3 atmo_camera_position = deref(push.attachments.globals).camera.position * M_TO_KM_SCALE;
        vec3 world_camera_position = atmo_camera_position + vec3(0.0, 0.0, deref(settings).atmosphere_bottom + BASE_HEIGHT_OFFSET);

        const vec3 direct_lighting = shadow * get_sun_direct_lighting(settings, sun_direction, world_camera_position);
        const vec4 norm_indirect_ligting = texture( daxa_samplerCube( push.attachments.sky_ibl, deref(push.attachments.globals).samplers.linear_clamp), normal).rgba;
        const vec3 indirect_ligting = norm_indirect_ligting.rgb * norm_indirect_ligting.a;
        const vec3 lighting = direct_lighting + indirect_ligting;
        output_value.rgb = albedo.rgb * lighting;
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
        const vec2 uv = vec2(gl_GlobalInvocationID.xy) * deref(push.attachments.globals).settings.render_target_size_inv;
        const vec2 ndc_xy = (uv * 2.0) - 1.0;
        const vec3 view_direction = get_view_direction(ndc_xy);

        daxa_BufferPtr(SkySettings) settings = deref(push.attachments.globals).sky_settings_ptr;
        // Because the atmosphere is using km as it's default units and we want one unit in world
        // space to be one meter we need to scale the position by a factor to get from meters -> kilometers
        const vec3 camera_position = deref(push.attachments.globals).settings.draw_from_observer == 1 ? 
            deref(push.attachments.globals).observer_camera.position * M_TO_KM_SCALE :
            deref(push.attachments.globals).camera.position * M_TO_KM_SCALE;

        vec3 world_camera_position = camera_position + vec3(0.0, 0.0, deref(settings).atmosphere_bottom + BASE_HEIGHT_OFFSET);

        const vec3 atmosphere_direct_illuminance = get_atmosphere_illuminance_along_ray(
            settings,
            push.attachments.transmittance,
            push.attachments.sky,
            deref(push.attachments.globals).samplers.linear_clamp,
            view_direction,
            world_camera_position
        );
        const vec3 total_direct_illuminance = get_sun_direct_lighting(settings, view_direction, world_camera_position) + atmosphere_direct_illuminance;
        // const vec4 total_direct_illuminance = texture( daxa_samplerCube( push.attachments.sky_ibl, deref(push.attachments.globals).samplers.linear_clamp), view_direction).rgba;
        // output_value = vec4(total_direct_illuminance.rgb * total_direct_illuminance.a, 1.0);
        output_value = vec4(total_direct_illuminance, 1.0);
    }

    const float exposure = compute_exposure(deref(push.attachments.luminance_average));
    const vec3 exposed_color = output_value.rgb * exposure;
    uvec2 detector_window_index;
    debug_write_lens(
        deref(push.attachments.globals).debug, 
        push.attachments.debug_lens_image, 
        index, 
        vec4(debug_value));
    imageStore(daxa_image2D(push.attachments.color_image), index, vec4(exposed_color, output_value.a));
}