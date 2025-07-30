#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/geometry.inl"
#include "../../shader_shared/visbuffer.inl"
#include "../../shader_shared/scene.inl"

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(GenGbufferH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_IMAGE_TYPED(READ_WRITE_CONCURRENT, daxa::RWTexture2DId<daxa_f32vec4>, debug_image)
DAXA_TH_IMAGE_TYPED(READ, daxa::RWTexture2DId<daxa_u32>, vis_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_u32>, face_normal_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_u32>, detail_normal_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_u32>, half_res_face_normal_image)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32>, half_res_depth_image)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUMaterial), material_manifest)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(GPUMesh), meshes)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(daxa_f32mat4x3), combined_transforms)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshletInstancesBufferHead), instantiated_meshlets)
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
#include "../../shader_lib/misc.hlsl"
#include "../../shader_lib/visbuffer.hlsl"
#include "../../shader_lib/shading.hlsl"

[[vk::push_constant]] GenGbufferPush push;

groupshared uint gs_face_normals[GEN_GBUFFER_X][GEN_GBUFFER_Y];
groupshared float gs_depths[GEN_GBUFFER_X][GEN_GBUFFER_Y];

[[shader("compute")]]
[numthreads(GEN_GBUFFER_X, GEN_GBUFFER_Y, 1)]
func entry_gen_gbuffer(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID)
{
    uint triangle_id = INVALID_TRIANGLE_ID;
    if (all(lessThan(dtid, push.size)))
    {
        triangle_id = push.attachments.vis_image.get()[dtid];
    }

    if (triangle_id != INVALID_TRIANGLE_ID)
    {
        CameraInfo camera = push.attachments.globals->view_camera;

        MeshletInstancesBufferHead* instantiated_meshlets = push.attachments.instantiated_meshlets;
        GPUMesh* meshes = push.attachments.meshes;
        daxa_f32mat4x3* combined_transforms = push.attachments.combined_transforms;
        VisbufferTriangleGeometry visbuf_tri = visgeo_triangle_data(
            triangle_id,
            float2(dtid),
            push.size,
            push.inv_size,
            camera.view_proj,
            instantiated_meshlets,
            meshes,
            combined_transforms
        );     
        TriangleGeometry tri_geo = visbuf_tri.tri_geo;
        TriangleGeometryPoint tri_point = visbuf_tri.tri_geo_point;
        float3 primary_ray = normalize(tri_point.world_position - camera.position);
        float depth = visbuf_tri.depth;
        uint meshlet_triangle_index = visbuf_tri.meshlet_triangle_index;
        uint meshlet_instance_index = visbuf_tri.meshlet_instance_index;
        uint meshlet_index = visbuf_tri.meshlet_index;

        MaterialPointData material_point = evaluate_material<SHADING_QUALITY_HIGH>(
            push.attachments.globals,
            tri_geo,
            tri_point
        );
        if (push.attachments.globals.settings.debug_material_quality == SHADING_QUALITY_LOW)
        {
            material_point = evaluate_material<SHADING_QUALITY_LOW>(
                push.attachments.globals,
                tri_geo,
                tri_point
            );
        }

        material_point.face_normal = flip_normal_to_incoming(
            material_point.face_normal,
            material_point.face_normal,
            primary_ray
        );
        uint packed_face_normal = compress_normal_octahedral_32(material_point.face_normal);
        push.attachments.face_normal_image.get()[dtid] = packed_face_normal;

        float3 detail_normal = material_point.normal;
        detail_normal = flip_normal_to_incoming(
            material_point.face_normal,
            detail_normal,
            primary_ray
        );
        
        uint packed_mapped_normal = compress_normal_octahedral_32(detail_normal);
        push.attachments.detail_normal_image.get()[dtid] = packed_mapped_normal;

        gs_face_normals[gtid.x][gtid.y] = packed_face_normal;
        gs_depths[gtid.x][gtid.y] = depth;
    }
    else
    {
        gs_face_normals[gtid.x][gtid.y] = 0u;
        gs_depths[gtid.x][gtid.y] = 0.0f;
    }

    GroupMemoryBarrierWithGroupSync();

    if (all((dtid.xy & uint2(1,1)) == uint2(0,0)))
    {
        uint2 half_out_idx = dtid.xy / 2;

        float4 depths = {
            gs_depths[gtid.x + 0][gtid.y + 0],
            gs_depths[gtid.x + 1][gtid.y + 0],
            gs_depths[gtid.x + 0][gtid.y + 1],
            gs_depths[gtid.x + 1][gtid.y + 1]
        };
        uint4 normals = {
            gs_face_normals[gtid.x + 0][gtid.y + 0],
            gs_face_normals[gtid.x + 1][gtid.y + 0],
            gs_face_normals[gtid.x + 0][gtid.y + 1],
            gs_face_normals[gtid.x + 1][gtid.y + 1]
        };

        float closest_depth = 0.0f;
        uint closest_face_normal = 0u;
        for (uint i = 0; i < 4; ++i)
        {
            if (depths[i] > closest_depth)
            {
                closest_depth = depths[i];
                closest_face_normal = normals[i];
            }
        }

        push.attachments.half_res_depth_image.get()[half_out_idx] = closest_depth;
        push.attachments.half_res_face_normal_image.get()[half_out_idx] = closest_face_normal;
    }
}

#endif