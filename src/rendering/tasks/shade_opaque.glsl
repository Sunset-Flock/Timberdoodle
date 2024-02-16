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

        vec4 debug_value = vec4(tri_data.world_normal * 0.5f + 0.5f, 1);

        GPUMaterial material = deref(push.attachments.material_manifest[tri_data.meshlet_instance.material_index]);
        if(material.diffuse_texture_id.value != 0)
        {
            // color = texture(daxa_sampler2D(material.diffuse_texture_id, deref(push.attachments.globals).samplers.linear_repeat), tri_uv.uv);
        }

        if(material.normal_texture_id.value != 0)
        {
            vec3 normal_map = texture(daxa_sampler2D(material.normal_texture_id, deref(push.attachments.globals).samplers.linear_repeat), tri_data.uv).rgb;
        }
        
        output_value = debug_value;

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
    }

    imageStore(daxa_image2D(push.attachments.color_image), index, output_value);
}