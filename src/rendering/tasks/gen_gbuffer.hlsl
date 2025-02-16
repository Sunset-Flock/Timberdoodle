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
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_WRITE_ONLY, daxa::RWTexture2DId<daxa_u32>, face_normal_image)
DAXA_TH_IMAGE_TYPED(COMPUTE_SHADER_STORAGE_WRITE_ONLY, daxa::RWTexture2DId<daxa_u32>, detail_normal_image)
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

    uint packed_face_normal = 0u;
    if (triangle_id != INVALID_TRIANGLE_ID)
    {
        CameraInfo* camera = Ptr<CameraInfo>(0);
        CameraInfo* camera_prev = Ptr<CameraInfo>(0);
        if(push.attachments.globals->settings.draw_from_observer == 1)
        {
            camera = &push.attachments.globals->observer_camera;
        }
        else 
        {
            camera = &push.attachments.globals->camera;
        }

        MeshletInstancesBufferHead* instantiated_meshlets = push.attachments.instantiated_meshlets;
        GPUMesh* meshes = push.attachments.meshes;
        daxa_f32mat4x3* combined_transforms = push.attachments.combined_transforms;
        VisbufferTriangleGeometry visbuf_tri = visgeo_triangle_data(
            triangle_id,
            float2(dtid),
            push.size,
            push.inv_size,
            camera->view_proj,
            instantiated_meshlets,
            meshes,
            combined_transforms
        );     
        TriangleGeometry tri_geo = visbuf_tri.tri_geo;
        TriangleGeometryPoint tri_point = visbuf_tri.tri_geo_point;
        float3 primary_ray = normalize(tri_point.world_position - camera->position);
        float depth = visbuf_tri.depth;
        uint meshlet_triangle_index = visbuf_tri.meshlet_triangle_index;
        uint meshlet_instance_index = visbuf_tri.meshlet_instance_index;
        uint meshlet_index = visbuf_tri.meshlet_index;

        packed_face_normal = compress_normal_octahedral_32(tri_point.face_normal);
        push.attachments.face_normal_image.get()[dtid] = packed_face_normal;

        float3 mapped_normal = tri_point.world_normal;
        GPUMaterial material = GPU_MATERIAL_FALLBACK;
        if(tri_geo.material_index != INVALID_MANIFEST_INDEX)
        {
            material = push.attachments.material_manifest[tri_geo.material_index];
        }

        float3 albedo = float3(material.base_color);
        if(material.diffuse_texture_id.value != 0)
        {
            albedo = Texture2D<float4>::get(material.diffuse_texture_id).SampleGrad(
                // SamplerState::get(AT.globals->samplers.nearest_repeat_ani),
                SamplerState::get(push.attachments.globals->samplers.linear_repeat_ani),
                tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy
            ).rgb;
        }
        
        mapped_normal = flip_normal_to_incoming(tri_point.face_normal, mapped_normal, primary_ray);
        tri_point.world_normal = flip_normal_to_incoming(tri_point.face_normal, tri_point.world_normal, primary_ray);

        if(material.normal_texture_id.value != 0)
        {
            float3 normal_map_value = float3(0);
            if(material.normal_compressed_bc5_rg)
            {
                const float2 raw = Texture2D<float4>::get(material.normal_texture_id).SampleGrad(
                    SamplerState::get(push.attachments.globals->samplers.normals),
                    tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy
                ).rg;
                const float2 rescaled_normal_rg = raw * 2.0f - 1.0f;
                const float normal_b = sqrt(clamp(1.0f - dot(rescaled_normal_rg, rescaled_normal_rg), 0.0, 1.0));
                normal_map_value = float3(rescaled_normal_rg, normal_b);
            }
            else
            {
                const float3 raw = Texture2D<float4>::get(material.normal_texture_id).SampleGrad(
                    SamplerState::get(push.attachments.globals->samplers.normals),
                    tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy
                ).rgb;
                normal_map_value = raw * 2.0f - 1.0f;
            }
            const float3x3 tbn = transpose(float3x3(tri_point.world_tangent, cross(tri_point.world_tangent, tri_point.world_normal), tri_point.world_normal));
            mapped_normal = mul(tbn, normal_map_value);
        }

        uint packed_mapped_normal = compress_normal_octahedral_32(mapped_normal);
        push.attachments.detail_normal_image.get()[dtid] = packed_mapped_normal;
    }
}

#endif