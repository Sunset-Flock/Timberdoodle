#pragma once

#define DAXA_RAY_TRACING 1
#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "pgi_update.inl"

// Not in pgi_update.inl because this shader includes a lot shading specific inl files, such as vsm.
DAXA_DECL_RAY_TRACING_TASK_HEAD_BEGIN(PGITraceProbeLightingH)
DAXA_TH_BUFFER_PTR(READ_WRITE_CONCURRENT, daxa_RWBufferPtr(RenderGlobalData), globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(PGIIndirections), probe_indirections)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_u32vec4>, light_mask_volume)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_f32vec4>, probe_color)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_f32vec2>, probe_visibility)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DArrayIndex<daxa_f32vec4>, probe_info)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayIndex<daxa_u32>, probe_requests)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, sky_transmittance)
DAXA_TH_IMAGE_ID(SAMPLED, REGULAR_2D, sky)
DAXA_TH_TLAS_PTR(READ, tlas)
DAXA_TH_IMAGE_TYPED(READ_WRITE, daxa::RWTexture2DArrayIndex<daxa_f32vec4>, trace_result)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(MeshInstancesBufferHead), mesh_instances)
// VSM:
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMGlobals), vsm_globals)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMPointLight), vsm_point_lights)
DAXA_TH_BUFFER_PTR(READ, daxa_BufferPtr(VSMSpotLight), vsm_spot_lights)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, vsm_memory_block)
DAXA_TH_IMAGE_TYPED_MIP_ARRAY(READ, daxa::RWTexture2DArrayId<daxa_u32>, vsm_point_spot_page_table, 8)
DAXA_DECL_TASK_HEAD_END

struct PGITraceProbeLightingPush
{
    daxa_BufferPtr(PGITraceProbeLightingH::AttachmentShaderBlob) attach;
};

#if DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG

#include "../../shader_lib/pgi.hlsl"
#include "../../shader_lib/misc.hlsl"
#include "../../shader_lib/debug.glsl"
#include "../../shader_lib/shading.hlsl"
#include "../../shader_lib/raytracing.hlsl"
#include "../../shader_lib/vsm_sampling.hlsl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/misc.hlsl"

[[vk::push_constant]] PGITraceProbeLightingPush pgi_trace_probe_lighting_push;

#define PI 3.1415926535897932384626433832795

float rand_normal_dist() {
    float theta = 2.0 * PI * rand();
    float rho = sqrt(-2.0 * log(rand()));
    return rho * cos(theta);
}

struct RayPayload
{
    bool hit;
    float4 color_depth;
    int4 probe_index;
    float probe_validity;
}

func trace_shadow_ray(RaytracingAccelerationStructure tlas, float3 position, float3 light_position, float3 flat_normal, float3 incoming_ray) -> bool
{
    float3 start = rt_calc_ray_start(position, flat_normal, incoming_ray);

    RayDesc ray = {};
    ray.Direction = normalize(light_position - position);
    ray.Origin = start;
    ray.TMax = length(light_position - position) * 1.01f; 
    ray.TMin = 0.0f;

    RayPayload payload;
    payload.hit = true; // Only runs miss shader. Miss shader writes false.
    TraceRay(tlas, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, ~0, 0, 0, 0, ray, payload);

    bool path_occluded = payload.hit;
    return !path_occluded;
}

struct PGILightVisibilityTester : LightVisibilityTesterI
{
    RaytracingAccelerationStructure tlas;
    RenderGlobalData* globals;
    float sun_light(MaterialPointData material_point, float3 incoming_ray)
    {
        let sky = globals->sky_settings;
        let light_visible = trace_shadow_ray(tlas, material_point.position, material_point.position + sky.sun_direction * 100000000.0f, material_point.face_normal, incoming_ray);
        return light_visible ? 1.0f : 0.0f;
    }
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        let push = pgi_trace_probe_lighting_push;

        GPUPointLight point_light = globals.scene.point_lights[light_index];
        float3 to_light = point_light.position - material_point.position;
        float3 to_light_dir = normalize(to_light);

        //return 1.0f;
        let RAYTRACED_POINT_SHADOWS = false;
        if (RAYTRACED_POINT_SHADOWS)
        {        
            let light_visible = trace_shadow_ray(tlas, material_point.position, point_light.position, material_point.face_normal, incoming_ray);
            return light_visible ? 1.0f : 0.0f;
        }
        else
        {
            const float point_norm_dot = dot(material_point.position, to_light_dir);
            return get_vsm_point_shadow_coarse(
                push.attach.globals,
                push.attach.vsm_globals,
                push.attach.vsm_memory_block.get(),
                &(push.attach.vsm_point_spot_page_table[0]),
                push.attach.vsm_point_lights,
                material_point.normal, 
                material_point.position,
                light_index,
                point_norm_dot);
        }
    }
    float spot_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        let push = pgi_trace_probe_lighting_push;

        GPUSpotLight spot_light = globals.scene.spot_lights[light_index];

        //return 1.0f;
        let RAYTRACED_POINT_SHADOWS = true;
        if (RAYTRACED_POINT_SHADOWS)
        {        
            let light_visible = trace_shadow_ray(tlas, material_point.position, spot_light.position, material_point.face_normal, incoming_ray);
            return light_visible ? 1.0f : 0.0f;
        }
        else
        {
            return get_vsm_spot_shadow_coarse(
                push.attach.globals,
                push.attach.vsm_globals,
                push.attach.vsm_memory_block.get(),
                &(push.attach.vsm_point_spot_page_table[0]),
                push.attach.vsm_spot_lights,
                material_point.normal, 
                material_point.position,
                light_index);
        }
    }
}

[shader("raygeneration")]
void entry_ray_gen()
{
    let push = pgi_trace_probe_lighting_push;
    PGISettings* settings = &push.attach.globals.pgi_settings;
    PGISettings reg_settings = *settings;
    int3 dtid = DispatchRaysIndex().xyz;

    uint indirect_index = {};
    int4 probe_index = {};
    int2 probe_texel = {};
    {
        indirect_index = dtid.x / (reg_settings.probe_trace_resolution * reg_settings.probe_trace_resolution);
        uint local_index = (dtid.x - indirect_index * (reg_settings.probe_trace_resolution * reg_settings.probe_trace_resolution));
        probe_texel.y = local_index / reg_settings.probe_trace_resolution;
        probe_texel.x = local_index - reg_settings.probe_trace_resolution * probe_texel.y;

        uint indirect_package = ((uint*)(push.attach.probe_indirections + 1))[indirect_index];
        probe_index = pgi_unpack_indirect_probe(indirect_package);
    }
    
    PGICascade reg_cascade = settings->cascades[probe_index.w];

    uint frame_index = push.attach.globals.frame_index;

    // Seed is the same for all threads processing a probe.
    // This is important to be able to efficiently reconstruct data between tracing and probe texel updates.
    float2 in_texel_offset = pgi_probe_trace_noise(probe_index, frame_index);

    PGIProbeInfo probe_info = PGIProbeInfo::load(reg_settings, reg_cascade, push.attach.probe_info.get(), probe_index);

    float3 probe_position = pgi_probe_index_to_worldspace(reg_settings, reg_cascade, probe_info, probe_index);

    uint3 probe_texture_base_index = uint3(pgi_indirect_index_to_trace_tex_offset(reg_settings, indirect_index), 0);
    uint3 probe_texture_index = probe_texture_base_index + uint3(probe_texel, 0);
    uint3 trace_result_texture_index = probe_texture_index;

    float2 probe_uv = float2(float2(probe_texel) + in_texel_offset) * rcp(reg_settings.probe_trace_resolution);

    float3 probe_normal = pgi_probe_uv_to_probe_normal(probe_uv);

    // Randomly Offset Origin tangentially to the trace direction
    // Reduces artifacts caused by specific probe positioning.
    float3 sample_origin = probe_position;
    float3 sample_direction = probe_normal;

    RayDesc ray = {};
    ray.Direction = sample_direction;
    ray.Origin = sample_origin;
    ray.TMax = 100000.0f;
    ray.TMin = 0.0f;

    RayPayload payload;
    payload.color_depth = float4(0,0,0,0);
    payload.probe_index = probe_index;
    payload.probe_validity = probe_info.validity;
    payload.hit = true;

    TraceRay(RaytracingAccelerationStructure::get(push.attach.tlas), 0u, ~0, 0, 0, 0, ray, payload);

    RWTexture2DArray<float4> trace_result_tex = push.attach.trace_result.get();

    trace_result_tex[trace_result_texture_index] = payload.color_depth;
}

[shader("anyhit")]
void entry_any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = pgi_trace_probe_lighting_push;
    if (!rt_is_alpha_hit(
        push.attach.globals,
        push.attach.mesh_instances,
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.materials,
        attr.barycentrics))
    {
        IgnoreHit();
    }
}

[shader("closesthit")]
void entry_closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = pgi_trace_probe_lighting_push;

    payload.hit = true;

    PGISettings* pgi_settings = &push.attach.globals.pgi_settings;
    PGISettings reg_settings = *pgi_settings;
    PGICascade reg_cascade = pgi_settings->cascades[payload.probe_index.w];
    float3 hit_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

    uint request_mode = pgi_get_probe_request_mode(
        push.attach.globals,
        reg_settings,
        reg_cascade,
        push.attach.probe_requests.get_formatted(),
        payload.probe_index);
    request_mode += 1; // direct(0) becomes indirect(1), indirect(1) becomes none(2) 
    
    const float indirect_ao_range = reg_cascade.max_visibility_distance * 0.15f;
    const float pgi_enabled = push.attach.globals.pgi_settings.enabled ? 1.0f : 0.0f;
    const float ambient_occlusion = (1.0f - max(0.0f,(indirect_ao_range - RayTCurrent()))/indirect_ao_range);

    MeshInstance* mi = push.attach.mesh_instances.instances;
    TriangleGeometry tri_geo = rt_get_triangle_geo(
        attr.barycentrics,
        InstanceID(),
        GeometryIndex(),
        PrimitiveIndex(),
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.entity_to_meshgroup,
        push.attach.globals.scene.mesh_groups,
        mi
    );
    TriangleGeometryPoint tri_point = rt_get_triangle_geo_point(
        tri_geo,
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.entity_to_meshgroup,
        push.attach.globals.scene.mesh_groups,
        push.attach.globals.scene.entity_combined_transforms
    );

    const float relative_t = RayTCurrent() * rcp(length(reg_cascade.probe_spacing));
    if (relative_t < 1.5f)
    {
        MaterialPointData material_point = evaluate_material<SHADING_QUALITY_LOW>(
            push.attach.globals,
            tri_geo,
            tri_point
        );
        bool double_sided_or_blend = ((material_point.material_flags & MATERIAL_FLAG_DOUBLE_SIDED) != MATERIAL_FLAG_NONE);
        bool backface = dot(WorldRayDirection(), tri_point.face_normal) > 0.01f && !double_sided_or_blend;
        payload.color_depth.rgb = float3(0,0,0);

        // ~10% speedup
        if (payload.probe_validity < 0.01f)
        {
            payload.color_depth = float4(0,0,0, RayTCurrent() * (backface ? -1 : 1));
            return;
        }

        if (!backface)
        {

            PGILightVisibilityTester light_vis_tester = PGILightVisibilityTester(RaytracingAccelerationStructure::get(push.attach.tlas), push.attach.globals);
            payload.color_depth.rgb = shade_material<SHADING_QUALITY_LOW>(
                push.attach.globals, 
                push.attach.sky_transmittance,
                push.attach.sky,
                material_point, 
                push.attach.globals.view_camera.position,
                WorldRayDirection(), 
                light_vis_tester, 
                push.attach.light_mask_volume.get(),
                push.attach.probe_color.get(), 
                push.attach.probe_visibility.get(), 
                push.attach.probe_info.get(),
                push.attach.probe_requests.get_formatted(),
                request_mode,
                ambient_occlusion
            ).rgb;

            payload.color_depth.a = RayTCurrent();
        }

        if (backface)
        {
            payload.color_depth.a = -RayTCurrent();
        }

        // ShaderDebugLineDraw line;
        // line.color = float3(2,1,0);
        // line.coord_space = DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE;
        // line.start = WorldRayDirection() * RayTCurrent() + WorldRayOrigin();
        // line.end = WorldRayDirection() * RayTCurrent() + WorldRayOrigin() + tri_point.face_normal * 0.1f;
        // debug_draw_line(push.attach.globals.debug, line);
    }
    else // Far hits are not shaded fully, instead resample projected probe radiance 
    {
        // TODO: Early out and shade these coherently in a followup pass.

        const float3 sample_offset_direction = tri_point.face_normal;
        const float3 sample_direction = float3(0,0,0); // ignored with probe_relative_sample_dir
        const float3 sample_position = RayTCurrent() * WorldRayDirection() + WorldRayOrigin();

        PGISampleInfo info = PGISampleInfoNearestSurfaceRadiance();
        
        payload.color_depth.rgb = pgi_sample_probe_volume(
            push.attach.globals, &push.attach.globals.pgi_settings, info,
            sample_position, push.attach.globals.view_camera.position, tri_point.face_normal, tri_point.face_normal,
            push.attach.probe_color.get(),
            push.attach.probe_visibility.get(),
            push.attach.probe_info.get(),
            push.attach.probe_requests.get()
        ) * ambient_occlusion;
        payload.color_depth.a = RayTCurrent();
    }

    
}

[shader("miss")]
void entry_miss(inout RayPayload payload)
{
    let push = pgi_trace_probe_lighting_push;

    payload.hit = false;

    payload.color_depth.rgb = shade_sky(push.attach.globals, push.attach.sky_transmittance, push.attach.sky, WorldRayDirection());
    payload.color_depth.a = 1000.0f;
}

#endif