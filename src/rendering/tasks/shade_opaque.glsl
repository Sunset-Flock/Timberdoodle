#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "shade_opaque.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/sky_util.glsl"

const vec4 sun_color = vec4(255.0, 240.0, 233.0, 255.0)/255.0; // 5800K

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

vec3 get_sun_illuminance(vec3 view_direction, float height, float zenith_cos_angle)
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attachments.globals).sky_settings_ptr;

    const float sun_solid_angle = 0.25 * PI / 180.0;
    const float min_sun_cos_theta = cos(sun_solid_angle);

    const vec3 sun_direction = deref(settings).sun_direction;
    const float cos_theta = dot(view_direction, sun_direction);
    if(cos_theta >= min_sun_cos_theta) 
    {
        TransmittanceParams transmittance_lut_params = TransmittanceParams(height, zenith_cos_angle);
        vec2 transmittance_texture_uv = transmittance_lut_to_uv(
            transmittance_lut_params,
            deref(settings).atmosphere_bottom,
            deref(settings).atmosphere_top
        );
        
        vec3 transmittance_to_sun = texture( 
            daxa_sampler2D( push.attachments.transmittance, deref(push.attachments.globals).samplers.linear_clamp),
            transmittance_texture_uv
        ).rgb;

        return transmittance_to_sun * sun_color.rgb * deref(settings).sun_brightness;
    }
    return vec3(0.0);
}

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
mat3 build_orthonormal_basis(vec3 n) {
    vec3 b1;
    vec3 b2;

    if (n.z < 0.0) {
        const float a = 1.0 / (1.0 - n.z);
        const float b = n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = vec3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        const float a = 1.0 / (1.0 + n.z);
        const float b = -n.x * n.y * a;
        b1 = vec3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }

    return mat3(b1, b2, n);
}

vec3 get_atmosphere_illuminance_along_ray(vec3 ray, vec3 world_camera_position, vec3 sun_direction, out bool intersects_ground)
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attachments.globals).sky_settings_ptr;
    const vec3 world_up = normalize(world_camera_position);

    const float view_zenith_angle = acos(dot(ray, world_up));
    const float light_view_angle = acos(clamp(dot(
        normalize(vec3(sun_direction.xy, 0.0)),
        normalize(vec3(ray.xy, 0.0))
        ),-1.0, 1.0)
    );

    const float bottom_atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_camera_position,
        ray,
        vec3(0.0),
        deref(settings).atmosphere_bottom
    );

    const float top_atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_camera_position,
        ray,
        vec3(0.0, 0.0, 0.0),
        deref(settings).atmosphere_bottom
    );

    intersects_ground = bottom_atmosphere_intersection_distance >= 0.0;
    const bool intersects_sky = top_atmosphere_intersection_distance >= 0.0;
    const float camera_height = length(world_camera_position);

    vec2 sky_uv = skyview_lut_params_to_uv(
        intersects_ground,
        SkyviewParams(view_zenith_angle, light_view_angle),
        deref(settings).atmosphere_bottom,
        deref(settings).atmosphere_top,
        vec2(deref(settings).sky_dimensions),
        camera_height
    );

    const vec4 unitless_atmosphere_illuminance_mult = texture(daxa_sampler2D(push.attachments.sky, deref(push.attachments.globals).samplers.linear_clamp) , sky_uv).rgba;
    const vec3 unitless_atmosphere_illuminance = unitless_atmosphere_illuminance_mult.rgb * unitless_atmosphere_illuminance_mult.a;
    const vec3 sun_color_weighed_atmosphere_illuminance = sun_color.rgb * unitless_atmosphere_illuminance;
    const vec3 atmosphere_scattering_illuminance = sun_color_weighed_atmosphere_illuminance * deref(settings).sun_brightness;

    TransmittanceParams transmittance_lut_params = TransmittanceParams(camera_height, dot(ray, world_up));
    vec2 transmittance_texture_uv = transmittance_lut_to_uv(
        transmittance_lut_params,
        deref(settings).atmosphere_bottom,
        deref(settings).atmosphere_top
    );

    vec3 atmosphere_transmittance = texture(
        daxa_sampler2D(push.attachments.transmittance, deref(push.attachments.globals).samplers.linear_clamp),
        transmittance_texture_uv
    ).rgb;

    if (!intersects_sky) { atmosphere_transmittance = vec3(1); }
    const mat3 sun_basis = build_orthonormal_basis(normalize(sun_direction));
    const vec3 stars_color = atmosphere_transmittance * get_star_radiance(ray * sun_basis) * float(!intersects_ground);

    return atmosphere_scattering_illuminance + stars_color;
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

AtmosphereLightingInfo get_atmosphere_lighting(vec3 view_direction, vec3 normal)
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attachments.globals).sky_settings_ptr;
    // Because the atmosphere is using km as it's default units and we want one unit in world
    // space to be one meter we need to scale the position by a factor to get from meters -> kilometers
    const vec3 camera_position = deref(push.attachments.globals).camera.position * M_TO_KM_SCALE;
    const vec3 world_camera_position = camera_position + vec3(0.0, 0.0, deref(settings).atmosphere_bottom + BASE_HEIGHT_OFFSET);

    const vec3 sun_direction = deref(settings).sun_direction;

    bool normal_ray_intersects_ground;
    bool view_ray_intersects_ground;
    const vec3 atmosphere_normal_illuminance = get_atmosphere_illuminance_along_ray(
        normal,
        world_camera_position,
        sun_direction,
        normal_ray_intersects_ground
    );
    const vec3 atmosphere_view_illuminance = get_atmosphere_illuminance_along_ray(
        view_direction,
        world_camera_position,
        sun_direction,
        view_ray_intersects_ground
    );

    const vec3 direct_sun_illuminance = view_ray_intersects_ground ? 
        vec3(0.0) : 
        get_sun_illuminance(
            view_direction,
            length(world_camera_position),
            dot(sun_direction, normalize(world_camera_position))
        );

    return AtmosphereLightingInfo(
        atmosphere_normal_illuminance,
        atmosphere_view_illuminance,
        direct_sun_illuminance
    );
}

// ndc going in needs to be in range [-1, 1]
vec3 get_view_direction(vec2 ndc_xy)
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attachments.globals).sky_settings_ptr;
    const vec3 camera_position = deref(push.attachments.globals).camera.position;//* M_TO_KM_SCALE;

    // Get the direction of ray contecting camera origin and current fragment on the near plane 
    // in world coordinate system
    const vec4 unprojected_pos = deref(push.attachments.globals).camera.inv_view_proj * vec4(ndc_xy, 1.0, 1.0);
    const vec3 world_direction = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera_position);

    return world_direction;
}

layout(local_size_x = SHADE_OPAQUE_WG_X, local_size_y = SHADE_OPAQUE_WG_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    const uint triangle_id = imageLoad(daxa_uimage2D(push.attachments.vis_image), index).x;
    vec4 output_value = vec4(0,0,0,0);
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

        vec4 debug_value = vec4(normal * 0.5f + 0.5f, 1);

        GPUMaterial material = deref(push.attachments.material_manifest[tri_data.meshlet_instance.material_index]);
        vec3 albedo = (0.5f).xxx;
        if(material.diffuse_texture_id.value != 0)
        {
            albedo = texture(daxa_sampler2D(material.diffuse_texture_id, deref(push.attachments.globals).samplers.linear_repeat), tri_data.uv).rgb;
        }

        if(material.normal_texture_id.value != 0)
        {
            vec3 normal_map = texture(daxa_sampler2D(material.normal_texture_id, deref(push.attachments.globals).samplers.linear_repeat), tri_data.uv).rgb;
            normal_map = normal_map * 2.0f - 1.0f;
            mat3 tbn = mat3(tri_data.world_tangent, cross(tri_data.world_tangent, tri_data.world_normal), tri_data.world_normal);
            normal = tbn * normal_map;
            debug_value = vec4(normal * 0.5f + 0.5f, 1);   
        }
        
        output_value = debug_value;
#if 1
        const vec3 light_position = vec3(-5,-5,15);
        const vec3 light_power = vec3(1,1,1) * 100000;
        const float light_distance = length(tri_data.world_position - light_position);
        const vec3 to_light_dir = normalize(light_position - tri_data.world_position);
        vec3 color = (albedo.rgb * light_power) * (max(0.0, dot(to_light_dir, normal)) * 1/(light_distance*light_distance));
        output_value.rgb = color;
#endif

        uvec2 detector_window_index;
        debug_write_detector_image(
            deref(push.attachments.globals).debug, 
            push.attachments.detector_image, 
            index, 
            debug_value);
        if (debug_in_detector_window(deref(push.attachments.globals).debug, index, detector_window_index))
        {
            output_value = debug_value;
        }
    } else {
        // scale uvs to be in the range [0, 1]
        const vec2 uv = vec2(gl_GlobalInvocationID.xy) * deref(push.attachments.globals).settings.render_target_size_inv;
        const vec2 ndc_xy = (uv * 2.0) - 1.0;
        const vec3 view_direction = get_view_direction(ndc_xy);

        AtmosphereLightingInfo atmosphere_lighting = get_atmosphere_lighting(view_direction, vec3(0.0, 0.0, 1.0));
        const vec3 total_direct_illuminance = 
            (atmosphere_lighting.atmosphere_direct_illuminance + atmosphere_lighting.sun_direct_illuminance);
        output_value = vec4(total_direct_illuminance, 1.0);
    }

    const float exposure = compute_exposure(deref(push.attachments.luminance_average));
    const vec3 exposed_color = output_value.rgb * exposure;
    imageStore(daxa_image2D(push.attachments.color_image), index, vec4(exposed_color, output_value.a));
}