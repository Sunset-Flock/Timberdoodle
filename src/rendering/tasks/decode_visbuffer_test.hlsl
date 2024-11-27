#extension GL_EXT_debug_printf : enable

#include <daxa/daxa.inl>

#include "decode_visbuffer_test.inl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/sky_util.glsl"


[[vk::push_constant]] DecodeVisbufferTestPush push;

[numthreads(DECODE_VISBUFFER_TEST_X,DECODE_VISBUFFER_TEST_Y, 1)]
[[shader("compute")]]
void entry_decode_visbuffer(uint2 index : SV_DispatchThreadID)
{
    uint triangle_id = INVALID_TRIANGLE_ID;
    if (all(lessThan(index, push.size)))
    {
        const uint triangle_id = RWTexture2D<uint>::get(push.attachments.vis_image)[index];
    }
    float4 output_value = float4(0,0,0,0);
    float4 debug_value = float4(0, 0, 0, 0);
    if (triangle_id != INVALID_TRIANGLE_ID)
    {
        float4x4 view_proj;
        float3 camera_position;
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
            float2(index), 
            push.size,
            push.inv_size,
            view_proj, 
            push.attachments.instantiated_meshlets,
            push.attachments.meshes,
            push.attachments.combined_transforms);
        float3 normal = tri_data.world_normal;

        // Used to prevent it beeing optimized away.
        if (length(normal) > 2.0f)
        {
            RWTexture2D<float4>::get(push.attachments.debug_image)[index] = float4(1,0,0,1);
        }
    }
}