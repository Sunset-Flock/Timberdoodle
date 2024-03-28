#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "decode_visbuffer_test.inl"

#include "shader_lib/visbuffer.glsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/sky_util.glsl"


DAXA_DECL_PUSH_CONSTANT(DecodeVisbufferTestPush, push)

layout(local_size_x = DECODE_VISBUFFER_TEST_X, local_size_y = DECODE_VISBUFFER_TEST_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    const uint triangle_id = imageLoad(daxa_uimage2D(push.attachments.vis_image), index).x;
    vec4 output_value = vec4(0,0,0,0);
    vec4 debug_value = vec4(0, 0, 0, 0);
    if (triangle_id != INVALID_TRIANGLE_ID)
    {
        #if 1
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

        if (length(normal) > 2)
        {
            imageStore(daxa_image2D(push.attachments.debug_image), index, vec4(normal, 1.0f));
        }
        #else
        if (triangle_id == 0)
        {
            imageStore(daxa_image2D(push.attachments.debug_image), index, vec4(triangle_id.xxx, 1.0f));
        }
        #endif
    }
}