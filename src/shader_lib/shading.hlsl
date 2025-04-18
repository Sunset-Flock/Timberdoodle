#pragma once

#include <daxa/daxa.inl>

#include "shader_lib/debug.glsl"
#include "shader_lib/transform.glsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/sky_util.glsl"
#include "shader_lib/vsm_util.glsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/volumetric.hlsl"
#include "shader_lib/geometry.hlsl"
#include "shader_lib/pgi.hlsl"

// DO NOT INCLUDE VSM SHADING NOR RAY TRACED SHADING HEADERS HERE!

static const uint SHADING_QUALITY_NONE = 0;
static const uint SHADING_QUALITY_LOW = 1;
static const uint SHADING_QUALITY_HIGH = 2;
typedef uint ShadingQuality;

func evaluate_material<ShadingQuality SHADING_QUALITY>(RenderGlobalData* globals, TriangleGeometry tri_geo, TriangleGeometryPoint tri_point) -> MaterialPointData
{
    GPUMaterial material = {};
    if (tri_geo.material_index == INVALID_MANIFEST_INDEX)
    {
        material = GPU_MATERIAL_FALLBACK;
    }
    else
    {
        material = globals.scene.materials[tri_geo.material_index];
    }

    MaterialPointData ret = {};
    ret.position = tri_point.world_position;

    ret.emissive = material.emissive_color;
    ret.alpha = 1.0f;
    ret.albedo = material.base_color;
    if (!material.diffuse_texture_id.is_empty())
    {
        float4 diffuse_fetch = float4(0,0,0,0);
        if (SHADING_QUALITY > SHADING_QUALITY_LOW)
        {
            diffuse_fetch = Texture2D<float4>::get(material.diffuse_texture_id).SampleGrad(globals.samplers.linear_repeat_ani.get(), tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy);
        }
        else
        {
            diffuse_fetch = Texture2D<float4>::get(material.diffuse_texture_id).SampleLevel(globals.samplers.linear_repeat_ani.get(), float2(0,0), 16.0f);
        }
        ret.albedo *= diffuse_fetch.rgb;
        ret.alpha = diffuse_fetch.a;
    }

    if (SHADING_QUALITY > SHADING_QUALITY_LOW)
    {
        ret.geometry_normal = tri_point.world_normal;
        ret.face_normal = tri_point.face_normal;
        ret.normal = tri_point.world_normal;
        if(material.normal_texture_id.value != 0)
        {
            float3 normal_map_value = float3(0);
            if(material.normal_compressed_bc5_rg)
            {
                const float2 raw = Texture2D<float4>::get(material.normal_texture_id).SampleGrad(
                    globals->samplers.normals.get(),
                    tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy
                ).rg;
                const float2 rescaled_normal_rg = raw * 2.0f - 1.0f;
                const float normal_b = sqrt(clamp(1.0f - dot(rescaled_normal_rg, rescaled_normal_rg), 0.0, 1.0));
                normal_map_value = float3(rescaled_normal_rg, normal_b);
            }
            else
            {
                const float3 raw = Texture2D<float4>::get(material.normal_texture_id).SampleGrad(
                    globals->samplers.normals.get(),
                    tri_point.uv, tri_point.uv_ddx, tri_point.uv_ddy
                ).rgb;
                normal_map_value = raw * 2.0f - 1.0f;
            }
            if (dot(normal_map_value, -1) < 0.9999)
            {
                const float3x3 tbn = transpose(float3x3(tri_point.world_tangent, tri_point.world_bitangent, tri_point.world_normal));
                ret.normal = mul(tbn, normal_map_value);
            }
        }
    }
    else
    {
        ret.geometry_normal = tri_point.face_normal;
        ret.face_normal = tri_point.face_normal;
        ret.normal = tri_point.face_normal;
    }

    if (material.alpha_discard_enabled)
    {
        ret.material_flags = ret.material_flags | MATERIAL_FLAG_ALPHA_DISCARD | MATERIAL_FLAG_DOUBLE_SIDED;
    }
    if (material.blend_enabled)
    {
        ret.material_flags = ret.material_flags | MATERIAL_FLAG_BLEND | MATERIAL_FLAG_DOUBLE_SIDED;
    }
    if (material.double_sided_enabled)
    {
        ret.material_flags = ret.material_flags | MATERIAL_FLAG_DOUBLE_SIDED;
    }

    return ret;
}

float3 get_sun_direct_lighting(
    RenderGlobalData* globals, 
    daxa_ImageViewId sky_transmittance,
    daxa_ImageViewId sky,
    float3 view_direction, 
    float3 atmo_position)
{
    const float bottom_atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        float3(0.0, 0.0, length(atmo_position)),
        view_direction,
        float3(0.0),
        globals.sky_settings.atmosphere_bottom
    );
    
    bool view_ray_intersects_ground = bottom_atmosphere_intersection_distance >= 0.0;
    const float3 direct_sun_illuminance = view_ray_intersects_ground ? 
        float3(0.0) : 
        get_sun_illuminance(
            &globals.sky_settings,
            sky_transmittance,
            globals->samplers.linear_clamp,
            view_direction,
            length(atmo_position),
            dot(globals.sky_settings.sun_direction, normalize(atmo_position))
        );

    return direct_sun_illuminance;
}

func shade_material<ShadingQuality SHADING_QUALITY, LIGHT_VIS_TESTER_T : LightVisibilityTesterI>(
    RenderGlobalData* globals, 
    daxa_ImageViewId sky_transmittance,
    daxa_ImageViewId sky,
    MaterialPointData material_point, 
    float3 origin,
    float3 incoming_ray,
    LIGHT_VIS_TESTER_T light_visibility,
    Texture2DArray<float4> probe_irradiance,
    Texture2DArray<float2> probe_visibility,
    Texture2DArray<float4> probe_infos,
    RWTexture2DArray<uint> probe_requests,
    uint pgi_request_mode,
) -> float4
{
    // TODO: material_point.normal is busted only in ray tracing for some reason
    // Flipping normal if backface (in case of double sided triangles)
    material_point.normal = flip_normal_to_incoming(material_point.face_normal, material_point.normal, incoming_ray);
    material_point.geometry_normal = flip_normal_to_incoming(material_point.face_normal, material_point.geometry_normal, incoming_ray);

    float3 diffuse_light = float3(0,0,0);

    float3 atmo_position = get_atmo_position(globals);

    // Sun Shading
    {
        float sun_visibility = max(0.0f, dot(material_point.geometry_normal, globals.sky_settings.sun_direction));

        if (sun_visibility > 0.0f)
        {
            sun_visibility *= light_visibility.sun_light(material_point, incoming_ray);
        }

        if (sun_visibility > 0.0f)
        {
            float3 sun_light = get_sun_direct_lighting(
                globals,
                sky_transmittance,
                sky,
                globals.sky_settings.sun_direction,
                atmo_position
            );
            diffuse_light += sun_light * sun_visibility;
        }
    }

    // Indirect Diffuse
    {
        float3 indirect_diffuse = float3(0,0,0);
        if (SHADING_QUALITY == SHADING_QUALITY_LOW)
        {
            indirect_diffuse = pgi_sample_irradiance_nearest(
                globals, 
                &globals.pgi_settings, 
                material_point.position, 
                material_point.geometry_normal, 
                material_point.geometry_normal, 
                incoming_ray, probe_irradiance, 
                probe_visibility, 
                probe_infos, 
                probe_requests, 
                pgi_request_mode);
        }
        if (SHADING_QUALITY == SHADING_QUALITY_HIGH)
        {
            indirect_diffuse = pgi_sample_irradiance(
                globals, 
                &globals.pgi_settings, 
                material_point.position, 
                material_point.geometry_normal, 
                material_point.geometry_normal, 
                incoming_ray, probe_irradiance, 
                probe_visibility, 
                probe_infos, 
                probe_requests, 
                pgi_request_mode);
        }
        diffuse_light += indirect_diffuse;
    }

    // Emissive
    {
        diffuse_light += material_point.emissive;
    }

    return float4(material_point.albedo * diffuse_light, material_point.alpha);
}

static float3 DEBUG_atmosphere_direct_illuminnace;
func shade_sky(
    RenderGlobalData* globals, 
    daxa_ImageViewId sky_transmittance,
    daxa_ImageViewId sky,
    float3 incoming_ray) -> float3
{
    float3 atmo_position = get_atmo_position(globals);

    const float3 atmosphere_direct_illuminnace = get_atmosphere_illuminance_along_ray(
        globals->sky_settings,
        sky_transmittance,
        sky,
        globals->samplers.linear_clamp,
        incoming_ray,
        atmo_position
    );
    return atmosphere_direct_illuminnace;
}