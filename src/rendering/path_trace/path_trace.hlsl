#pragma once

#include "path_trace.inl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/sky_util.glsl"
#include "shader_lib/shading.hlsl"

#include "kajiya/hash.hlsl"
#include "kajiya/rt.hlsl"
#include "kajiya/brdf.hlsl"
#include "kajiya/brdf_lut.hlsl"
#include "kajiya/layered_brdf.hlsl"

// Does not include the segment used to connect to the sun
static const uint MAX_EYE_PATH_LENGTH = 16;

static const uint RUSSIAN_ROULETTE_START_PATH_LENGTH = 3;
static const float MAX_RAY_LENGTH = FLT_MAX;
//static const float MAX_RAY_LENGTH = 5.0;

// Rough-smooth-rough specular paths are a major source of fireflies.
// Enabling this option will bias roughness of path vertices following
// reflections off rough interfaces.
static const bool FIREFLY_SUPPRESSION = true;
static const bool FURNACE_TEST = !true;
static const bool FURNACE_TEST_EXCLUDE_DIFFUSE = !true;
static const bool USE_PIXEL_FILTER = true;
static const bool INDIRECT_ONLY = !true;
static const bool GREY_ALBEDO_FIRST_BOUNCE = !true;
static const bool BLACK_ALBEDO_FIRST_BOUNCE = !true;
static const bool ONLY_SPECULAR_FIRST_BOUNCE = !true;
static const bool USE_SOFT_SHADOWS = true;
static const bool SHOW_ALBEDO = !true;

static const bool USE_LIGHTS = true;
static const bool USE_EMISSIVE = true;
static const bool RESET_ACCUMULATION = !true;
static const bool ROLLING_ACCUMULATION = !true;
static const bool TRACE_PRIMARY = !true;

static const float DEFAULT_ROUGHNESS = 1.0;
static const float DEFAULT_METALNESS = 0.0;

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

float pixel_cone_spread_angle_from_image_height(float image_height) {
    return atan(2.0 * AT.globals.camera.inv_proj._11 / image_height);
}

RayCone pixel_ray_cone_from_image_height(float image_height) {
    RayCone res;
    res.width = 0.0;
    res.spread_angle = pixel_cone_spread_angle_from_image_height(image_height);
    return res;
}

// Assume this is constant no matter where we sample. This is wrong but faster
static float3 atmo_position;
float3 sample_environment_light(float3 dir) {
    // return 0.5.xxx;

    if (FURNACE_TEST) {
        return 0.5.xxx;
    }

    const float3 atmosphere_direct_illuminnace = get_atmosphere_illuminance_along_ray(
        AT.globals->sky_settings,
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

#define SUN_DIRECTION (AT.globals.sky_settings.sun_direction)

#define SUN_COLOR get_sun_direct_lighting(\
    AT.globals, AT.transmittance, AT.sky,\
    SUN_DIRECTION, atmo_position)

float3 uniform_sample_cone(float2 urand, float cos_theta_max) {
    float cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    float sin_theta = sqrt(saturate(1.0 - cos_theta * cos_theta));
    float phi = urand.y * (PI * 2.0);
    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}
float3 sample_sun_direction(float2 urand, bool soft) {
    if (soft) {
        static const float sun_angular_radius_cos = cos(0.5f * PI / 360.0);
        if (sun_angular_radius_cos < 1.0) {
            const float3x3 basis = build_orthonormal_basis(normalize(SUN_DIRECTION));
            return mul(basis, uniform_sample_cone(urand, sun_angular_radius_cos));
        }
    }

    return SUN_DIRECTION;
}

// Approximate Gaussian remap
// https://www.shadertoy.com/view/MlVSzw
float inv_error_function(float x, float truncation) {
    static const float ALPHA = 0.14;
    static const float INV_ALPHA = 1.0 / ALPHA;
    static const float K = 2.0 / (PI * ALPHA);

    float y = log(max(truncation, 1.0 - x*x));
    float z = K + 0.5 * y;
    return sqrt(max(0.0, sqrt(z*z - y * INV_ALPHA) - z)) * sign(x);
}

float remap_unorm_to_gaussian(float x, float truncation) {
    return inv_error_function(x * 2.0 - 1.0, truncation);
}


[shader("raygeneration")]
void ray_gen()
{
    const uint2 px = DispatchRaysIndex().xy;
    let output_tex = AT.history_image.get();
    let acceleration_structure = AT.tlas.get();
    
    const float exposure = compute_exposure(deref(AT.luminance_average));
    atmo_position = get_atmo_position(AT.globals);
    const float2 screen_uv = (float2(px) + 0.5) * AT.globals->settings.render_target_size_inv;
    const float2 ndc_xy = screen_uv * 2.0 - 1.0;
    const float3 view_direction = get_view_direction(ndc_xy);

    float4 prev;
    if (ROLLING_ACCUMULATION) {
        prev = float4(output_tex[px].rgb, 8);
    } else {
        prev = select(RESET_ACCUMULATION, 0, output_tex[px]);
    }

    if (prev.w < 1000)
    {
        float4 radiance_sample_count_packed = 0.0;
        uint rng = hash_combine2(hash_combine2(px.x, hash1(px.y)), AT.globals.frame_index);
        static const uint sample_count = 1;
        
        for (uint sample_idx = 0; sample_idx < sample_count; ++sample_idx) {
            RayDesc outgoing_ray;
            outgoing_ray.TMin = 0.0;
            outgoing_ray.TMax = FLT_MAX;

            float3 throughput = 1.0.xxx;
            float3 total_radiance = 0.0.xxx;

            float roughness_bias = 0.0;

            RayCone ray_cone = pixel_ray_cone_from_image_height(
                DispatchRaysDimensions().y
            );

            // Bias for texture sharpness
            ray_cone.spread_angle *= 0.3;

            [loop]
            for (uint path_length = 0; path_length < MAX_EYE_PATH_LENGTH; ++path_length) {
                if (path_length == 1) {
                    outgoing_ray.TMax = MAX_RAY_LENGTH;
                }

                GbufferPathVertex primary_hit;

                if (path_length == 0)
                {
                    outgoing_ray.Origin = AT.globals.camera.position;
                    outgoing_ray.Direction = view_direction;
                }

                if (!TRACE_PRIMARY && path_length == 0)
                {
                    uint triangle_id = AT.vis_image.get()[px].x;
                    bool triangle_id_valid = triangle_id != INVALID_TRIANGLE_ID;
                    if (triangle_id_valid) {
                        daxa_BufferPtr(MeshletInstancesBufferHead) instantiated_meshlets = AT.meshlet_instances;
                        daxa_BufferPtr(GPUMesh) meshes = AT.globals.scene.meshes;
                        daxa_BufferPtr(daxa_f32mat4x3) combined_transforms = AT.globals.scene.entity_combined_transforms;
                        VisbufferTriangleGeometry visbuf_tri = visgeo_triangle_data(
                            triangle_id,
                            float2(px),
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
                            material = AT.globals.scene.materials[tri_geo.material_index];
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
                            if (dot(normal_map_value, -1) < 0.9999)
                            {
                                const float3x3 tbn = transpose(float3x3(tri_point.world_tangent, tri_point.world_bitangent, tri_point.world_normal));
                                normal = mul(tbn, normal_map_value);
                            }
                        }

                        GbufferData gbuffer = GbufferData::create_zero();
                        gbuffer.albedo = albedo;
                        gbuffer.normal = normal;
                        gbuffer.roughness = DEFAULT_ROUGHNESS;
                        gbuffer.metalness = DEFAULT_METALNESS;
                        gbuffer.emissive = albedo * material.emissive_color;

                        primary_hit.gbuffer_packed = gbuffer.pack();
                        primary_hit.is_hit = true;
                        primary_hit.ray_t = length(tri_point.world_position - AT.globals.camera.position);
                        primary_hit.position = tri_point.world_position;
                    } else {
                        primary_hit.is_hit = false;
                        primary_hit.ray_t = FLT_MAX;
                    }
                }
                else
                {
                    primary_hit = GbufferRaytrace::with_ray(outgoing_ray)
                        .with_cone(ray_cone)
                        .with_cull_back_faces(false)
                        .with_path_length(path_length)
                        .trace(acceleration_structure);
                }

                if (primary_hit.is_hit) {
                    // TODO
                    const float surface_spread_angle = 0.0;
                    ray_cone = ray_cone.propagate(surface_spread_angle, primary_hit.ray_t);

                    const float3 to_light_norm = sample_sun_direction(
                        float2(uint_to_u01_float(hash1_mut(rng)), uint_to_u01_float(hash1_mut(rng))),
                        true
                    );

                    GbufferData gbuffer = primary_hit.gbuffer_packed.unpack();
                    
                    const bool is_shadowed =
                        (INDIRECT_ONLY && path_length == 0) ||
                        rt_is_shadowed(
                            acceleration_structure,
                            new_ray(
                                rt_calc_ray_start(primary_hit.position, gbuffer.normal, outgoing_ray.Direction),
                                to_light_norm,
                                0,
                                FLT_MAX
                        ));

                    if (SHOW_ALBEDO) {
                        output_tex[px] = float4(gbuffer.albedo, 1);
                        return;
                    }

                    if (dot(gbuffer.normal, outgoing_ray.Direction) >= 0.0) {
                        if (0 == path_length) {
                            // Flip the normal for primary hits so we don't see blackness
                            gbuffer.normal = -gbuffer.normal;
                        } else {
                            break;
                        }
                    }

                    if (FURNACE_TEST && !FURNACE_TEST_EXCLUDE_DIFFUSE) {
                        gbuffer.albedo = 1;
                    }

                    //gbuffer.albedo = float3(0.966653, 0.802156, 0.323968); // Au from Mitsuba
                    //gbuffer.albedo = 0;
                    //gbuffer.metalness = 1.0;
                    //gbuffer.roughness = 0.5;//lerp(gbuffer.roughness, 1.0, 0.8);

                    if (INDIRECT_ONLY && path_length == 0) {
                        gbuffer.albedo = 1.0;
                        gbuffer.metalness = 0.0;
                    }

                    // For reflection comparison against RTR
                    /*if (path_length == 0) {
                        gbuffer.albedo = 0;
                    }*/

                    if (ONLY_SPECULAR_FIRST_BOUNCE && path_length == 0) {
                        gbuffer.albedo = 1.0;
                        gbuffer.metalness = 1.0;
                        //gbuffer.roughness = 0.01;
                    }

                    if (GREY_ALBEDO_FIRST_BOUNCE && path_length == 0) {
                        gbuffer.albedo = 0.5;
                    }
                    
                    if (BLACK_ALBEDO_FIRST_BOUNCE && path_length == 0) {
                        gbuffer.albedo = 0.0;
                    }

                    //gbuffer.roughness = lerp(gbuffer.roughness, 0.0, 0.8);
                    //gbuffer.metalness = 1.0;
                    //gbuffer.albedo = max(gbuffer.albedo, 1e-3);
                    //gbuffer.roughness = 0.07;
                    //gbuffer.roughness = clamp((int(primary_hit.position.x * 0.2) % 5) / 5.0, 1e-4, 1.0);

                    const float3x3 tangent_to_world = build_orthonormal_basis(gbuffer.normal);
                    const float3 wi = mul(to_light_norm, tangent_to_world);

                    float3 wo = mul(-outgoing_ray.Direction, tangent_to_world);

                    // Hack for shading normals facing away from the outgoing ray's direction:
                    // We flip the outgoing ray along the shading normal, so that the reflection's curvature
                    // continues, albeit at a lower rate.
                    if (wo.z < 0.0) {
                        wo.z *= -0.25;
                        wo = normalize(wo);
                    }

                    LayeredBrdf brdf = LayeredBrdf::from_gbuffer_ndotv(gbuffer, wo.z, AT.brdf_lut.get(), AT.globals.samplers.linear_clamp.get());

                    if (FIREFLY_SUPPRESSION) {
                        brdf.specular_brdf.roughness = lerp(brdf.specular_brdf.roughness, 1.0, roughness_bias);
                    }

                    if (FURNACE_TEST && FURNACE_TEST_EXCLUDE_DIFFUSE) {
                        brdf.diffuse_brdf.albedo = 0.0.xxx;
                    }

                    if (!FURNACE_TEST && !(ONLY_SPECULAR_FIRST_BOUNCE && path_length == 0)) {
                        const float3 brdf_value = brdf.evaluate_directional_light(wo, wi);
                        const float3 light_radiance = select(is_shadowed, 0.0, SUN_COLOR);
                        total_radiance += throughput * brdf_value * light_radiance * max(0.0, wi.z);

                        if (USE_EMISSIVE) {
                            total_radiance += gbuffer.emissive * throughput;
                        }
                        
                        // NOTE(grundlett): Not yet implemented
                        #if 0
                        if (USE_LIGHTS && frame_constants.triangle_light_count > 0/* && path_length > 0*/) {   // rtr comp
                            const float light_selection_pmf = 1.0 / frame_constants.triangle_light_count;
                            const uint light_idx = hash1_mut(rng) % frame_constants.triangle_light_count;
                            //const float light_selection_pmf = 1;
                            //for (uint light_idx = 0; light_idx < frame_constants.triangle_light_count; light_idx += 1)
                            {
                                const float2 urand = float2(
                                    uint_to_u01_float(hash1_mut(rng)),
                                    uint_to_u01_float(hash1_mut(rng))
                                );

                                TriangleLight triangle_light = TriangleLight::from_packed(triangle_lights_dyn[light_idx]);
                                LightSampleResultArea light_sample = sample_triangle_light(triangle_light.as_triangle(), urand);
                                const float3 shadow_ray_origin = primary_hit.position;
                                const float3 to_light_ws = light_sample.pos - primary_hit.position;
                                const float dist_to_light2 = dot(to_light_ws, to_light_ws);
                                const float3 to_light_norm_ws = to_light_ws * rsqrt(dist_to_light2);

                                const float to_psa_metric =
                                    max(0.0, dot(to_light_norm_ws, gbuffer.normal))
                                    * max(0.0, dot(to_light_norm_ws, -light_sample.normal))
                                    / dist_to_light2;

                                if (to_psa_metric > 0.0) {
                                    float3 wi = mul(to_light_norm_ws, tangent_to_world);

                                    const bool is_shadowed =
                                        rt_is_shadowed(
                                            acceleration_structure,
                                            new_ray(
                                                shadow_ray_origin,
                                                to_light_norm_ws,
                                                1e-3,
                                                sqrt(dist_to_light2) - 2e-3
                                        ));

                                    total_radiance +=
                                        select(is_shadowed, 0,
                                            throughput * triangle_light.radiance() * brdf.evaluate(wo, wi) / light_sample.pdf.value * to_psa_metric / light_selection_pmf);
                                }
                            }
                        }
                        #endif
                    }

                    float3 urand;
                    BrdfSample brdf_sample = BrdfSample::invalid();

                    #if 0
                    if (path_length == 0) {
                        const uint noise_offset = frame_constants.frame_index;

                        urand = bindless_textures[BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0][
                            (px + int2(noise_offset * 59, noise_offset * 37)) & 255
                        ].xyz * 255.0 / 256.0 + 0.5 / 256.0;

                        urand.x += uint_to_u01_float(hash1(frame_constants.frame_index));
                        urand.y += uint_to_u01_float(hash1(frame_constants.frame_index + 103770841));
                        urand.z += uint_to_u01_float(hash1(frame_constants.frame_index + 828315679));

                        urand = frac(urand);
                    } else
                    #endif
                    {
                        urand = float3(
                            uint_to_u01_float(hash1_mut(rng)),
                            uint_to_u01_float(hash1_mut(rng)),
                            uint_to_u01_float(hash1_mut(rng)));
                    }

                    brdf_sample = brdf.sample(wo, urand);

                    if (brdf_sample.is_valid()) {
                        if (FIREFLY_SUPPRESSION) {
                            roughness_bias = lerp(roughness_bias, 1.0, 0.5 * brdf_sample.approx_roughness);
                        }

                        outgoing_ray.Origin = rt_calc_ray_start(primary_hit.position, gbuffer.normal, outgoing_ray.Direction);
                        outgoing_ray.Direction = mul(tangent_to_world, brdf_sample.wi);
                        outgoing_ray.TMin = 0;
                        throughput *= brdf_sample.value_over_pdf;
                    } else {
                         break;
                    }

                    if (FURNACE_TEST) {
                        // Short-circuit the path tracing
                        total_radiance += throughput * sample_environment_light(outgoing_ray.Direction);
                        break;
                    }

                    // Russian roulette
                    if (path_length >= RUSSIAN_ROULETTE_START_PATH_LENGTH) {
                        const float rr_coin = uint_to_u01_float(hash1_mut(rng));
                        const float continue_p = max(gbuffer.albedo.r, max(gbuffer.albedo.g, gbuffer.albedo.b));
                        if (rr_coin > continue_p) {
                            break;
                        } else {
                            throughput /= continue_p;
                        }
                    }
                } else {
                    total_radiance += throughput * sample_environment_light(outgoing_ray.Direction);
                    break;
                }
            }

            if (all(total_radiance >= 0.0)) {
                radiance_sample_count_packed += float4(total_radiance, 1.0);
            }
        }

        float4 cur = radiance_sample_count_packed;

        float tsc = cur.w + prev.w;
        float lrp = cur.w / max(1.0, tsc);
        cur.rgb /= max(1.0, cur.w);

        float3 result = max(0.0.xxx, lerp(prev.rgb, cur.rgb, lrp));

        output_tex[px] = float4(result, max(1, tsc));

        AT.pt_image.get()[px] = float4(result * exposure, 1);
    }
}

[shader("anyhit")]
void any_hit(inout GbufferRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = ref_pt_push;
    if (!rt_is_alpha_hit(
        push.attachments.attachments.globals,
        push.attachments.attachments.mesh_instances,
        push.attachments.attachments.globals.scene.meshes,
        push.attachments.attachments.globals.scene.materials,
        attr.barycentrics))
    {
        IgnoreHit();
    }
}

[shader("closesthit")]
void closest_hit(inout GbufferRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let primitive_index = PrimitiveIndex();
    let instance_id = InstanceID();
    let geometry_index = GeometryIndex();

    TriangleGeometry tri_geo = rt_get_triangle_geo(
        attr.barycentrics,
        instance_id,
        geometry_index,
        primitive_index,
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
    if (dot(tri_point.world_normal, WorldRayDirection()) > 0)
        tri_point.world_normal *= -1;

    GbufferData gbuffer = GbufferData::create_zero();
    gbuffer.albedo = material_point.albedo;
    gbuffer.normal = tri_point.world_normal;
    gbuffer.roughness = DEFAULT_ROUGHNESS;
    gbuffer.metalness = DEFAULT_METALNESS;
    gbuffer.emissive = material_point.albedo * material_point.emissive;

    payload.gbuffer_packed = gbuffer.pack();
    payload.t = RayTCurrent();
}

[shader("miss")]
void miss(inout GbufferRayPayload payload) {}

[shader("miss")]
void shadow_miss(inout ShadowRayPayload payload) { payload.is_shadowed = false; }
