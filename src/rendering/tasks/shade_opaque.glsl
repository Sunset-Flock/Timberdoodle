#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "shade_opaque.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"

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
        if (push.observer_pass == 1)
        {
            view_proj = deref(push.attachments.globals).observer_camera.view_proj;
        } 
        else
        {
            view_proj = deref(push.attachments.globals).camera.view_proj;
        }
        VisbufferTriangleData tri_data = get_visbuffer_triangle_data(
            triangle_id, 
            vec2(index), 
            push.size,
            push.inv_size,
            view_proj, 
            push.attachments.instantiated_meshlets,
            push.attachments.meshes,
            push.attachments.combined_transforms);
        vec2 uvs = get_interpolated_uv(
            tri_data,
            push.attachments.meshes
        );

        uvec2 main_cam_texel_pos = world_space_to_texel_index(deref(push.attachments.globals).camera, tri_data.world_position);

        if (push.observer_pass == 0)
        {
            debug_write_detector_image(
                deref(push.attachments.globals).debug, 
                push.attachments.detector_image, 
                index, 
                vec4(uvs,0,1));
        }

        vec4 color;
        GPUMaterial material = deref(push.attachments.material_manifest[tri_data.meshlet_instance.material_index]);
        uvec2 window_index;
        if(material.diffuse_texture_id.value != 0 && !debug_in_detector_window(deref(push.attachments.globals).debug, main_cam_texel_pos, window_index))
        {
            color = texture(daxa_sampler2D(material.diffuse_texture_id, deref(push.attachments.globals).samplers.linear_repeat), uvs);
        } else {
            color = vec4(uvs.x, uvs.y,0, 1);
        }
        
        output_value = vec4(color.rgb,1);//vec4(uvs, 0, 1);
    }

    imageStore(daxa_image2D(push.attachments.color_image), index, output_value);
}