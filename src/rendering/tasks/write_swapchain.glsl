#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "write_swapchain.inl"

#include "shader_lib/visbuffer.glsl"

DAXA_DECL_PUSH_CONSTANT(WriteSwapchainPush, push)
layout(local_size_x = WRITE_SWAPCHAIN_WG_X, local_size_y = WRITE_SWAPCHAIN_WG_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    float checker = ((((index.x / (4)) & 1) == 0) ^^ (((index.y / (4)) & 1) == 0)) ? 1.0f : 0.9f;
    const float checker2 = ((((index.x / (4*8)) & 1) == 0) ^^ (((index.y / (4*4)) & 1) == 0)) ? 1.0f : 0.9f;
    checker *= checker2;
    const uint triangle_id = imageLoad(daxa_uimage2D(vis_image), index).x;
    vec4 output_value = vec4(0,0,0,0);
    if (triangle_id != INVALID_TRIANGLE_ID)
    {
        uint instantiated_meshlet_index;
        uint triangle_index;
        decode_triangle_id(triangle_id, instantiated_meshlet_index, triangle_index);
        MeshletInstance inst_meshlet = unpack_meshlet_instance(deref(u_instantiated_meshlets).meshlets[instantiated_meshlet_index]);
        
        float f = float(inst_meshlet.entity_index * 10 + inst_meshlet.meshlet_index) * 0.93213213232;
        vec3 debug_color = vec3(cos(f), cos(f+2), cos(f+4));
        debug_color = debug_color * 0.5 + 0.5;
        //debug_color *= fract(inst_meshlet.entity_index * 0.0111);

        vec4 result = vec4(0,0,0,1);

        output_value = vec4(debug_color * checker,1);
    }
    else
    {
        output_value = vec4(vec3(1,1,1) * checker,1);
    }
    // vec4 debug_value = imageLoad(daxa_image2D(u_debug_image), index);
    vec4 debug_value = vec4(0,0,0,0);
    output_value = vec4(mix(output_value.xyz, debug_value.xyz, debug_value.a), output_value.a);
    imageStore(daxa_image2D(swapchain), index, output_value);
}