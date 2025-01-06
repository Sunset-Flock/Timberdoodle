#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "path_trace.inl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/sky_util.glsl"
#include "shader_lib/shading.hlsl"
#include "hash.hlsl"

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
    float t;
    float2 barycentrics;
    uint primitive_index;
    uint geometry_index;
    uint instance_id;
}
struct ShadowRayPayload
{
    bool is_shadowed;
    
    static ShadowRayPayload new_hit() {
        ShadowRayPayload res;
        res.is_shadowed = true;
        return res;
    }
}

[[vk::push_constant]] ReferencePathTracePush ref_pt_push;
#define AT deref(ref_pt_push.attachments).attachments

float compute_exposure(float average_luminance)
{
    const float exposure_bias = AT.globals->postprocess_settings.exposure_bias;
    const float calibration = AT.globals->postprocess_settings.calibration;
    const float sensor_sensitivity = AT.globals->postprocess_settings.sensor_sensitivity;
    const float ev100 = log2(average_luminance * sensor_sensitivity * exposure_bias / calibration);
    const float exposure = 1.0 / (1.2 * exp2(ev100));
    return exposure;
}

float3 get_view_direction(float2 ndc_xy)
{
    float3 world_direction; 
    const float3 camera_position = AT.globals->camera.position;
    const float4 unprojected_pos = mul(AT.globals->camera.inv_view_proj, float4(ndc_xy, 1.0, 1.0));
    world_direction = normalize((unprojected_pos.xyz / unprojected_pos.w) - camera_position);
    return world_direction;
}

static float3 atmo_position;

float3 sample_sky(float3 dir)
{
    const float3 atmosphere_direct_illuminnace = get_atmosphere_illuminance_along_ray(
        AT.globals->sky_settings_ptr,
        AT.transmittance,
        AT.sky,
        AT.globals->samplers.linear_clamp,
        dir,
        atmo_position
    );
    const float3 sun_direct_illuminance = get_sun_direct_lighting(
        AT.globals, AT.transmittance, AT.sky,
        dir, atmo_position);
    const float3 total_direct_illuminance = sun_direct_illuminance + atmosphere_direct_illuminnace;
    return total_direct_illuminance;
}

bool rt_is_shadowed(RaytracingAccelerationStructure acceleration_structure, RayDesc ray) {
    ShadowRayPayload shadow_payload = ShadowRayPayload::new_hit();
    TraceRay(
        acceleration_structure,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xff, 0, 0, 1, ray, shadow_payload
    );
    return shadow_payload.is_shadowed;
}

float3 eval_fresnel_schlick(float3 f0, float3 f90, float cos_theta) {
    return lerp(f0, f90, pow(max(0.0, 1.0 - cos_theta), 5));
}


[shader("raygeneration")]
void ray_gen()
{
    const int2 index = DispatchRaysIndex().xy;
    const float2 screen_uv = index * AT.globals->settings.render_target_size_inv;

    rand_seed(index.x + index.y * AT.globals.settings.render_target_size.x + AT.globals.frame_index * AT.globals.settings.render_target_size.x * AT.globals.settings.render_target_size.y);

    uint triangle_id = AT.vis_image.get()[index].x;
    bool triangle_id_valid = triangle_id != INVALID_TRIANGLE_ID;

    const float exposure = compute_exposure(deref(AT.luminance_average));

    atmo_position = get_atmo_position(AT.globals);
    const float2 ndc_xy = screen_uv * 2.0 - 1.0;
    const float3 view_direction = get_view_direction(ndc_xy);

    if (!triangle_id_valid)
    {
        // draw sky
        AT.pt_image.get()[index.xy] = float4(sample_sky(view_direction) * exposure, 1);
        return;
    }

    daxa_BufferPtr(MeshletInstancesBufferHead) instantiated_meshlets = AT.instantiated_meshlets;
    daxa_BufferPtr(GPUMesh) meshes = AT.meshes;
    daxa_BufferPtr(daxa_f32mat4x3) combined_transforms = AT.combined_transforms;
    VisbufferTriangleGeometry visbuf_tri = visgeo_triangle_data(
        triangle_id,
        float2(index),
        AT.globals.settings.render_target_size,
        1.0 / AT.globals.settings.render_target_size,
        AT.globals.camera.view_proj,
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

    float3 normal = tri_point.world_normal;
    GPUMaterial material = GPU_MATERIAL_FALLBACK;
    if(tri_geo.material_index != INVALID_MANIFEST_INDEX)
    {
        material = AT.material_manifest[tri_geo.material_index];
    }

    float3 albedo = float3(material.base_color);
    if(material.diffuse_texture_id.value != 0)
    {
        albedo = Texture2D<float4>::get(material.diffuse_texture_id).SampleGrad(
            SamplerState::get(AT.globals->samplers.linear_repeat_ani),
            tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy
        ).rgb;
    }

    if(material.normal_texture_id.value != 0)
    {
        float3 normal_map_value = float3(0);
        if(material.normal_compressed_bc5_rg)
        {
            const float2 raw = Texture2D<float4>::get(material.normal_texture_id).SampleGrad(
                SamplerState::get(AT.globals->samplers.normals),
                tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy
            ).rg;
            const float2 rescaled_normal_rg = raw * 2.0f - 1.0f;
            const float normal_b = sqrt(clamp(1.0f - dot(rescaled_normal_rg, rescaled_normal_rg), 0.0, 1.0));
            normal_map_value = float3(rescaled_normal_rg, normal_b);
        }
        else
        {
            const float3 raw = Texture2D<float4>::get(material.normal_texture_id).SampleGrad(
                SamplerState::get(AT.globals->samplers.normals),
                tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy
            ).rgb;
            normal_map_value = raw * 2.0f - 1.0f;
        }
        const float3x3 tbn = transpose(float3x3(tri_point.world_tangent, cross(tri_point.world_tangent, tri_point.world_normal), tri_point.world_normal));
        normal = mul(tbn, normal_map_value);
    }

    RayDesc shadow_ray = {};
    shadow_ray.Direction = AT.globals.sky_settings.sun_direction;
    shadow_ray.Origin = tri_point.world_position;
    shadow_ray.TMax = 1000.0f;
    shadow_ray.TMin = 0.001f;

    const float3 SUN_COL = get_sun_direct_lighting(
            AT.globals, AT.transmittance, AT.sky,
            AT.globals.sky_settings.sun_direction, atmo_position);
    const bool is_shadowed = rt_is_shadowed(AT.tlas.get(), shadow_ray);

    float3 throughput = albedo;
    float3 total_radiance = throughput * select(is_shadowed, 0.0, SUN_COL) * max(dot(normal, shadow_ray.Direction), 0);
    total_radiance += albedo * material.emissive_color;

    const uint MAX_EYE_PATH_LENGTH = 3;
    RayDesc outgoing_ray = {};
    outgoing_ray.Direction = rand_hemi_dir(normal);
    outgoing_ray.Origin = tri_point.world_position;
    outgoing_ray.TMax = 1000.0f;
    outgoing_ray.TMin = 0.001f;

    // throughput *= max(dot(outgoing_ray.Direction, normal), 0);

    for (uint path_length = 0; path_length < MAX_EYE_PATH_LENGTH; ++path_length) {
        RayPayload payload;
        TraceRay(AT.tlas.get(), {}, ~0, 0, 0, 0, outgoing_ray, payload);

        if (payload.hit)
        {
            TriangleGeometry tri_geo = rt_get_triangle_geo(
                payload.barycentrics,
                payload.instance_id,
                payload.geometry_index,
                payload.primitive_index,
                AT.globals.scene.meshes,
                AT.globals.scene.entity_to_meshgroup,
                AT.globals.scene.mesh_groups,
                AT.mesh_instances.instances
            );
            TriangleGeometryPoint tri_point = rt_get_triangle_geo_point(
                tri_geo,
                AT.globals.scene.meshes,
                AT.globals.scene.entity_to_meshgroup,
                AT.globals.scene.mesh_groups,
                AT.globals.scene.entity_combined_transforms
            );
            MaterialPointData material_point = evaluate_material(
                AT.globals,
                tri_geo,
                tri_point
            );
            if (dot(tri_point.world_normal, outgoing_ray.Direction) > 0)
                tri_point.world_normal *= -1;
            
            shadow_ray.Origin = tri_point.world_position;
            const bool is_shadowed = rt_is_shadowed(AT.tlas.get(), shadow_ray);
            total_radiance += material_point.albedo * material_point.emissive * throughput;
            total_radiance += material_point.albedo * throughput * select(is_shadowed, 0.0, SUN_COL) * max(dot(tri_point.world_normal, shadow_ray.Direction), 0);
            throughput *= material_point.albedo;

            outgoing_ray.Origin = tri_point.world_position;
            outgoing_ray.Direction = rand_hemi_dir(tri_point.world_normal);
            // throughput *= max(dot(outgoing_ray.Direction, tri_point.world_normal), 0);
        }
        else
        {
            total_radiance += throughput * sample_sky(outgoing_ray.Direction);
            break;
        }

    }

    float4 prev_radiance = AT.history_image.get()[index.xy];

    const float sample_count = 1;
    float tsc = sample_count + prev_radiance.w;
    // tsc = min(tsc, 20);
    float lrp = sample_count / max(1.0, tsc);
    total_radiance /= max(1.0, sample_count);
    total_radiance = lerp(prev_radiance.rgb, total_radiance, lrp);

    AT.history_image.get()[index.xy] = float4(total_radiance, tsc);
    AT.pt_image.get()[index.xy] = float4(total_radiance * exposure, 1);
}

[shader("anyhit")]
void any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    const float3 hit_location = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    const uint primitive_index = PrimitiveIndex() * 3;
    GPUMesh *mesh = {};
    const uint mesh_instance_index = InstanceID();
    MeshInstance* mesh_instance = AT.mesh_instances.instances + mesh_instance_index;
    mesh = AT.globals.scene.meshes + mesh_instance->mesh_index;

    const int primitive_indices[3] = {
        mesh.primitive_indices[primitive_index],
        mesh.primitive_indices[primitive_index + 1],
        mesh.primitive_indices[primitive_index + 2],
    };

    const float2 uvs[3] = {
        mesh.vertex_uvs[primitive_indices[0]],
        mesh.vertex_uvs[primitive_indices[1]],
        mesh.vertex_uvs[primitive_indices[2]],
    };
    const float2 interp_uv = uvs[0] + attr.barycentrics.x * (uvs[1] - uvs[0]) + attr.barycentrics.y * (uvs[2] - uvs[0]);
    const GPUMaterial *material = &AT.globals.scene.materials[mesh.material_index];
    
    if (mesh.material_index == INVALID_MANIFEST_INDEX)
    {
        return;
    }

    float alpha = 1.0;
    if (material.opacity_texture_id.value != 0 && material.alpha_discard_enabled)
    {
        // TODO: WHAT THE FUCK IS THIS BUG? WHY ARE WE SAMPLING diffuse_texture_id IN THIS BRANCH??
        alpha = Texture2D<float4>::get(material.diffuse_texture_id)
            .Sample( SamplerState::get(AT.globals->samplers.linear_repeat), interp_uv).a; 
    }
    else if (material.diffuse_texture_id.value != 0 && material.alpha_discard_enabled)
    {
        alpha = Texture2D<float4>::get(material.diffuse_texture_id)
            .Sample( SamplerState::get(AT.globals->samplers.linear_repeat), interp_uv).a; 
    }
    if(alpha < 0.5) {
        IgnoreHit();
    }
}

[shader("closesthit")]
void closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    payload.hit = true;
    payload.t = RayTCurrent();
    payload.barycentrics = attr.barycentrics;
    payload.primitive_index = PrimitiveIndex();
    payload.instance_id = InstanceID();
    payload.geometry_index = GeometryIndex();
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    payload.hit = false;
}

[shader("miss")]
void shadow_miss(inout ShadowRayPayload payload)
{
    payload.is_shadowed = false;
}
