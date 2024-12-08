#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_TASK_HEAD_BEGIN(GenGbufferH)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_READ_ONLY, daxa::RWTexture2DId<daxa_u32>, vis_image)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_WRITE_ONLY, daxa::RWTexture2DId<daxa_u32>, geo_normal_image)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(daxa_f32mat4x3), combined_transforms)
DAXA_TH_BUFFER_PTR(COMPUTE_SHADER_READ, daxa_BufferPtr(MeshletInstancesBufferHead), instantiated_meshlets)
DAXA_DECL_TASK_HEAD_END

struct GenGbufferPush
{
    GenGbufferH::AttachmentShaderBlob attachments;
    daxa_f32vec2 size;
    daxa_f32vec2 inv_size;
};

#define GEN_GBUFFER_X 8
#define GEN_GBUFFER_Y 8

#if DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG
#include "../../shader_lib/visbuffer.hlsl"
#include "../../shader_lib/misc.hlsl"

[[vk::push_constant]] GenGbufferPush push;

[[shader("compute")]]
[numthreads(GEN_GBUFFER_X, GEN_GBUFFER_Y, 1)]
func entry_gen_gbuffer(uint2 dtid : SV_DispatchThreadID)
{
    uint triangle_id = INVALID_TRIANGLE_ID;
    if (all(lessThan(dtid, push.size)))
    {
        triangle_id = push.attachments.vis_image.get()[dtid];
    }

    uint packed_geometric_normal = 0u;
    if (triangle_id != INVALID_TRIANGLE_ID)
    {
        float4x4 view_proj;
        float3 camera_position;
        if(push.attachments.globals->settings.draw_from_observer == 1)
        {
            view_proj = push.attachments.globals->observer_camera.view_proj;
            camera_position = push.attachments.globals->observer_camera.position;
        }
        else 
        {
            view_proj = push.attachments.globals->camera.view_proj;
            camera_position = push.attachments.globals->camera.position;
        }

        MeshletInstancesBufferHead* instantiated_meshlets = push.attachments.instantiated_meshlets;
        GPUMesh* meshes = push.attachments.meshes;
        daxa_f32mat4x3* combined_transforms = push.attachments.combined_transforms;
        VisbufferTriangleGeometry visbuf_tri = visgeo_triangle_data(
            triangle_id,
            float2(dtid),
            push.size,
            push.inv_size,
            view_proj,
            instantiated_meshlets,
            meshes,
            combined_transforms
        );   
        TriangleGeometry tri_geo = visbuf_tri.tri_geo;
        TriangleGeometryPoint tri_point = visbuf_tri.tri_geo_point;
        float depth = visbuf_tri.depth;
        uint meshlet_triangle_index = visbuf_tri.meshlet_triangle_index;
        uint meshlet_instance_index = visbuf_tri.meshlet_instance_index;
        uint meshlet_index = visbuf_tri.meshlet_index;

        packed_geometric_normal = compress_normal_octahedral_32(tri_point.world_normal);

        //push.attachments.debug_image.get()[dtid].xyz = tri_data.world_normal * 0.5f + 0.5f;

        push.attachments.geo_normal_image.get()[dtid] = packed_geometric_normal;
    }
}

#endif