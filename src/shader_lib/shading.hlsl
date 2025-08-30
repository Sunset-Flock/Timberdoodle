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
#include "shader_lib/lights.hlsl"
#include "../rendering/path_trace/kajiya/math_const.hlsl"

// DO NOT INCLUDE VSM SHADING NOR RAY TRACED SHADING HEADERS HERE!

static const uint SHADING_QUALITY_NONE = 0;
static const uint SHADING_QUALITY_LOW = 1;
static const uint SHADING_QUALITY_HIGH = 2;
static const uint SHADING_QUALITY_HIGH_STOCHASTIC = 3;
static const uint SHADING_QUALITY_ONLY_DIRECT = 4;
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
            diffuse_fetch = Texture2D<float4>::get(material.diffuse_texture_id).SampleLevel(globals.samplers.linear_repeat_ani.get(), tri_point.uv, 8.0f);
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
    float3 origin, // Camera Position or Ray Origin
    float3 incoming_ray,
    LIGHT_VIS_TESTER_T light_visibility,
    Texture2DArray<uint4> light_mask_volume,
    Texture2DArray<float4> probe_irradiance,
    Texture2DArray<float2> probe_visibility,
    Texture2DArray<float4> probe_infos,
    RWTexture2DArray<uint> probe_requests,
    uint pgi_request_mode,
    float ambient_occlusion = 1.0f
) -> float4
{    
    material_point.face_normal = flip_face_normal_to_incoming(material_point.face_normal, incoming_ray);
    material_point.geometry_normal = flip_normal_on_face_normal(material_point.geometry_normal, material_point.face_normal);
    material_point.normal = flip_normal_on_face_normal(material_point.normal, material_point.face_normal);

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

    // Local Lights
    {
        let light_settings = globals.light_settings;
        uint4 light_mask = lights_get_mask(light_settings, material_point.position, light_mask_volume);
        // Point Lights
        {
            uint4 point_light_mask = light_mask & light_settings.point_light_mask;
            //point_light_mask = WaveActiveBitOr(point_light_mask); // Way faster to be divergent in rt
    #if LIGHTS_ENABLE_MASK_ITERATION
            while (any(point_light_mask != uint4(0)))
            {
                uint point_light_idx = lights_iterate_mask(light_settings, point_light_mask);
    #else
            for(int point_light_idx = 0; point_light_idx < light_settings.point_light_count; ++point_light_idx)
            {
    #endif
                GPUPointLight light = globals.scene.point_lights[point_light_idx];
                const float3 position_to_light = normalize(light.position - material_point.position);
                
                const float to_light_dist = length(light.position - material_point.position);

                float attenuation = lights_attenuate_point(to_light_dist, light.cutoff);
                if (attenuation > 0.0f)
                {
                    let light_visibility = light_visibility.point_light(material_point, incoming_ray, point_light_idx);
                    let diffuse = max(dot(material_point.normal, position_to_light), 0.0);
                    diffuse_light += light.color * diffuse * attenuation * light.intensity * light_visibility;
                }
            }
        }

        // Spot Lights
        {
            float3 total_contribution = float3(0.0);
            light_mask = light_mask & light_settings.spot_light_mask;
            light_mask = WaveActiveBitOr(light_mask);
    #if LIGHTS_ENABLE_MASK_ITERATION
            while (any(light_mask != uint4(0)))
            {
                uint spot_light_idx = lights_iterate_mask(light_settings, light_mask) - light_settings.first_spot_light_instance;
    #else
            for(int spot_light_idx = 0; spot_light_idx < light_settings.spot_light_count; ++spot_light_idx)
            {
    #endif
                GPUSpotLight light = globals.scene.spot_lights[spot_light_idx];
                const float3 position_to_light = normalize(light.position - material_point.position);
                const float to_light_dist = length(light.position - material_point.position);
                const float attenuation = lights_attenuate_spot(position_to_light, to_light_dist, light);
                if (attenuation > 0.0f)
                {
                    let light_visibility = light_visibility.spot_light(material_point, incoming_ray, spot_light_idx);
                    let diffuse = max(dot(material_point.normal, position_to_light), 0.0);
                    diffuse_light += light.color * diffuse * attenuation * light.intensity * light_visibility;
                }
            }
        }
    }

    // Indirect Diffuse
    if (SHADING_QUALITY < SHADING_QUALITY_ONLY_DIRECT)
    {
        PGISampleInfo pgi_sample_info = PGISampleInfo();
        pgi_sample_info.cascade_mode = SHADING_QUALITY == SHADING_QUALITY_HIGH_STOCHASTIC ? PGI_CASCADE_MODE_STOCHASTIC_BLEND : PGI_CASCADE_MODE_BLEND;
        pgi_sample_info.probe_blend_nearest = SHADING_QUALITY == SHADING_QUALITY_LOW;
        pgi_sample_info.request_mode = pgi_request_mode;

        const float3 indirect_diffuse = pgi_sample_probe_volume(
            globals, &globals.pgi_settings, pgi_sample_info,
            material_point.position, origin, material_point.normal, material_point.face_normal, 
            probe_irradiance, probe_visibility, probe_infos, probe_requests
        );
        diffuse_light += indirect_diffuse * ambient_occlusion;
    }

    return float4(material_point.albedo * M_FRAC_1_PI * diffuse_light + material_point.emissive, material_point.alpha);
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