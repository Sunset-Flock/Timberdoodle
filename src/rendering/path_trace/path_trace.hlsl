#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "path_trace.inl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/misc.hlsl"

#define PI 3.1415926535897932384626433832795

float rand_normal_dist() {
    float theta = 2.0 * PI * rand();
    float rho = sqrt(-2.0 * log(rand()));
    return rho * cos(theta);
}

float3 rand_dir() {
    return normalize(float3(
        rand_normal_dist(),
        rand_normal_dist(),
        rand_normal_dist()));
}

float3 rand_hemi_dir(float3 nrm) {
    float3 result = rand_dir();
    return result * sign(dot(nrm, result));
}

struct RayPayload
{
    bool hit;
}

[[vk::push_constant]] ReferencePathTracePush ref_pt_push;

float compute_exposure(float average_luminance)
{
    const float exposure_bias = ref_pt_push.attach.globals->postprocess_settings.exposure_bias;
    const float calibration = ref_pt_push.attach.globals->postprocess_settings.calibration;
    const float sensor_sensitivity = ref_pt_push.attach.globals->postprocess_settings.sensor_sensitivity;
    const float ev100 = log2(average_luminance * sensor_sensitivity * exposure_bias / calibration);
	const float exposure = 1.0 / (1.2 * exp2(ev100));
	return exposure;
}

[shader("raygeneration")]
void ray_gen()
{
    let push = ref_pt_push;
    const int2 index = DispatchRaysIndex().xy;
    const float2 screen_uv = index * push.attach.globals->settings.render_target_size_inv;

    uint triangle_id = push.attach.vis_image.get()[index].x;
    bool triangle_id_valid = triangle_id != INVALID_TRIANGLE_ID;

    const float exposure = compute_exposure(deref(push.attach.luminance_average));

    float3 color = float3(0.1, 0.1, 0.1);

    if (!triangle_id_valid)
    {
        // draw sky
        color = float3(1, 0, 1);
        push.attach.pt_image.get()[index.xy] = float4(color * exposure, 1);
        return;
    }

    daxa_BufferPtr(MeshletInstancesBufferHead) instantiated_meshlets = push.attach.instantiated_meshlets;
    daxa_BufferPtr(GPUMesh) meshes = push.attach.meshes;
    daxa_BufferPtr(daxa_f32mat4x3) combined_transforms = push.attach.combined_transforms;
    VisbufferTriangleGeometry visbuf_tri = visgeo_triangle_data(
        triangle_id,
        float2(index),
        push.attach.globals.settings.render_target_size,
        1.0 / push.attach.globals.settings.render_target_size,
        push.attach.globals.camera.view_proj,
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

    // if (is_center_pixel)
    // {
    //     AT.globals.readback.hovered_entity = tri_geo.entity_index;
    // }

    // world_space_depth = length(tri_point.world_position - camera_position);

    float3 normal = tri_point.world_normal;
    GPUMaterial material = GPU_MATERIAL_FALLBACK;
    if(tri_geo.material_index != INVALID_MANIFEST_INDEX)
    {
        material = push.attach.material_manifest[tri_geo.material_index];
    }

    float3 albedo = float3(material.base_color);
    if(material.diffuse_texture_id.value != 0)
    {
        albedo = Texture2D<float4>::get(material.diffuse_texture_id).SampleGrad(
            // SamplerState::get(AT.globals->samplers.nearest_repeat_ani),
            SamplerState::get(push.attach.globals->samplers.linear_repeat_ani),
            tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy
        ).rgb;
    }

    color = albedo;

    push.attach.pt_image.get()[index.xy] = float4(color * exposure, 1);
}

[shader("anyhit")]
void any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = ref_pt_push;
    const float3 hit_location = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const uint primitive_index = PrimitiveIndex();
    GPUMesh *mesh = {};
    
    const uint mesh_instance_index = InstanceID();
    MeshInstance* mesh_instance = push.attach.mesh_instances.instances + mesh_instance_index;
    mesh = push.attach.meshes + mesh_instance->mesh_index;

    const int primitive_indices[3] = {
        mesh.primitive_indices[3 * primitive_index],
        mesh.primitive_indices[3 * primitive_index + 1],
        mesh.primitive_indices[3 * primitive_index + 2],
    };

    const float2 uvs[3] = {
        mesh.vertex_uvs[primitive_indices[0]],
        mesh.vertex_uvs[primitive_indices[1]],
        mesh.vertex_uvs[primitive_indices[2]],
    };
    const float2 interp_uv = uvs[0] + attr.barycentrics.x * (uvs[1] - uvs[0]) + attr.barycentrics.y* (uvs[2] - uvs[0]);
    const GPUMaterial *material = &push.attach.material_manifest[mesh.material_index];
    
    if (mesh.material_index == INVALID_MANIFEST_INDEX)
    {
        return;
    }

    const bool has_opacity_texture = !material.opacity_texture_id.is_empty();
    const bool alpha_discard_enabled = material.alpha_discard_enabled;
    if(has_opacity_texture && alpha_discard_enabled)
    {
        let opacity_tex = Texture2D<float>::get(material.opacity_texture_id);
        let opacity = opacity_tex.SampleLevel(SamplerState::get(push.attach.globals->samplers.linear_repeat), interp_uv, 2).r;
        if(opacity < 0.5) {
            IgnoreHit();
        }
    }
}

[shader("closesthit")]
void closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.hit = true;
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    payload.hit = false;
}
