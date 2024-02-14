#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "shade_opaque.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"

DAXA_DECL_PUSH_CONSTANT(ShadeOpaquePush, push)
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
        VisbufferTriangleUv tri_uv = visgeo_interpolated_uv( tri_data, push.attachments.meshes );
        vec3 geometry_normal = visgeo_interpolated_normal( tri_data, push.attachments.meshes );
        mat3x3 tbn = visgeo_tbn(tri_data, tri_uv, geometry_normal);

        // vec4 debug_value = vec4(uv,0,1);
        vec4 debug_value = vec4(geometry_normal * 0.5f + 0.5f, 1);

        vec4 color = debug_value;
        GPUMaterial material = deref(push.attachments.material_manifest[tri_data.meshlet_instance.material_index]);
        uvec2 window_index;
        if(material.diffuse_texture_id.value != 0)
        {
            color = texture(daxa_sampler2D(material.diffuse_texture_id, deref(push.attachments.globals).samplers.linear_repeat), tri_uv.uv);
        }

        vec3 normal = geometry_normal;
        if(material.normal_texture_id.value != 0)
        {
            normal = texture(daxa_sampler2D(material.normal_texture_id, deref(push.attachments.globals).samplers.linear_repeat), tri_uv.uv).rgb;
            normal = tbn * normal;
            color = vec4(normal * 0.5f + 0.5f, 1);
        }

#if 0
        const vec3 light_position = vec3(-5,-5,15);
        const vec3 light_power = vec3(1,1,1) * 100;
        const float light_distance = length(tri_data.world_position - light_position);
        const vec3 to_light_dir = normalize(light_position - tri_data.world_position);
        color.rgb = (color.rgb * light_power) * (max(0, dot(to_light_dir, normal)) * 1/(light_distance*light_distance));
#endif
        
        output_value = vec4(color.rgb,1);//vec4(uv, 0, 1);

        debug_write_detector_image(
            deref(push.attachments.globals).debug, 
            push.attachments.detector_image, 
            index, 
            debug_value);
        if (debug_in_detector_window(deref(push.attachments.globals).debug, index, window_index))
        {
            output_value = debug_value;
        }
    }

    imageStore(daxa_image2D(push.attachments.color_image), index, output_value);
}