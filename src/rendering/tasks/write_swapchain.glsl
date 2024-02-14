#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "write_swapchain.inl"

#include "shader_lib/visbuffer.glsl"

vec3 index_to_color(uint index)
{
    return vec3(cos(index), cos(index * 2 + 1), cos(index * 3 + 2)) * 0.5 + 0.5;
}

DAXA_DECL_PUSH_CONSTANT(WriteSwapchainPush, push)
layout(local_size_x = WRITE_SWAPCHAIN_WG_X, local_size_y = WRITE_SWAPCHAIN_WG_Y) in;
void main()
{
    const ivec2 index = ivec2(gl_GlobalInvocationID.xy);
    if (index.x >= push.size.x || index.y >= push.size.y)
    {
        return;
    }
    //float checker = ((((index.x / (4)) & 1) == 0) ^^ (((index.y / (4)) & 1) == 0)) ? 1.0f : 0.9f;
    //const float checker2 = ((((index.x / (4*8)) & 1) == 0) ^^ (((index.y / (4*4)) & 1) == 0)) ? 1.0f : 0.9f;
    //checker *= checker2;
    //const uint triangle_id = imageLoad(daxa_uimage2D(push.uses.vis_image), index).x;
    //vec4 output_value = vec4(0,0,0,1);
    //vec4 color = vec4(0, 0, 0, 1);
    //if (triangle_id != INVALID_TRIANGLE_ID)
    //{
    //    uint instantiated_meshlet_index;
    //    uint triangle_index;
    //    decode_triangle_id(triangle_id, instantiated_meshlet_index, triangle_index);
    //    MeshletInstance inst_meshlet = unpack_meshlet_instance(deref(push.uses.instantiated_meshlets).meshlets[instantiated_meshlet_index]);
    //    vec2 uvs = imageLoad(daxa_image2D(push.uses.debug_image), index).rg;
    //    GPUMaterial material = deref(push.uses.material_manifest[inst_meshlet.material_index]);
    //    if(material.diffuse_texture_id.value != 0)
    //    {
    //        color = texture(daxa_sampler2D(material.diffuse_texture_id, deref(push.uses.globals).samplers.linear_repeat), uvs);
    //    } else {
    //        color = imageLoad(daxa_image2D(push.uses.debug_image), index);
    //    }
    //    
    //    float f = float(inst_meshlet.entity_index * 10 + inst_meshlet.meshlet_index) * 0.93213213232;
    //    vec3 debug_color = vec3(cos(f), cos(f+2), cos(f+4));
    //    debug_color = debug_color * 0.5 + 0.5;
    //    //debug_color *= fract(inst_meshlet.entity_index * 0.0111);
    //    vec4 result = vec4(0,0,0,1);
    //    output_value = vec4(debug_color * checker,1);
    //    output_value = color;
    //    if (false) {
    //        VisbufferTriangleData tri_data = visgeo_triangle_data(
    //            triangle_id, 
    //            vec2(index), 
    //            vec2(push.width, push.height),
    //            push.inv_size,
    //            push.uses.globals, 
    //            push.uses.instantiated_meshlets,
    //            push.uses.meshes,
    //            push.uses.combined_transforms);
    //        vec2 uvs = visgeo_interpolated_uv(
    //            tri_data,
    //            push.uses.meshes
    //        );
    //        
    //        output_value = vec4(uvs, 0, 1);
    //    }
    //    
    //}
    //else
    //{
    //    output_value = vec4(vec3(0.05) * checker,1);
    //    color = output_value;
    //}
    //// vec4 debug_value = imageLoad(daxa_image2D(push.uses.debug_image), index);
    //// output_value = vec4(mix(output_value.xyz, debug_value.xyz, debug_value.a), output_value.a);
    vec4 color = imageLoad(daxa_image2D(push.attachments.color_image), index);
    imageStore(daxa_image2D(push.attachments.swapchain), index, color);
}