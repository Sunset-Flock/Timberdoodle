#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtao.inl"

#include "shader_lib/visbuffer.hlsl"
#include "shader_lib/misc.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/pgi.hlsl"
#include "shader_lib/shading.hlsl"
#include "shader_lib/vsm_sampling.hlsl"

#define PI 3.1415926535897932384626433832795

[[vk::push_constant]] RayTraceAmbientOcclusionPush rt_ao_push;

float3 rand_dir() {
    return normalize(float3(
        rand() * 2.0f - 1.0f,
        rand() * 2.0f - 1.0f,
        rand() * 2.0f - 1.0f));
}

float3 rand_hemi_dir(float3 nrm) {
    float3 result = rand_dir();
    return result * sign(dot(nrm, result));
}

float2 concentric_sample_disc()
{
    float r = sqrt(rand());
    float theta = rand() * 2 * PI;
    return float2(cos(theta), sin(theta)) * r;
}

float3 cosine_sample_hemi()
{
    float2 d = concentric_sample_disc();
    float z = sqrt(max(0.0f, 1.0f - d.x * d.x - d.y * d.y));
    return float3(d.x, d.y, z);
}

struct RayPayload
{
    bool miss;
    // RTAO:
    float power;
    // RTGI:
    float3 color;
    float3 normal;
    bool skip_sky_shading_on_miss;
}

[shader("raygeneration")]
void ray_gen()
{
    let clk_start = clockARB();
    let push = rt_ao_push;
    const int2 index = DispatchRaysIndex().xy;
    const float2 screen_uv = index * push.attach.globals->settings.render_target_size_inv;

    const float depth = push.attach.view_cam_depth.get()[index];
    float4 output_value = float4(0);
    float4 debug_value = float4(0);

    if(depth != 0.0f)
    {
        CameraInfo camera = push.attach.globals.main_camera;
        const float3 world_position = pixel_index_to_world_space(camera, index, depth);
        const float3 detail_normal = uncompress_normal_octahedral_32(push.attach.view_cam_detail_normals.get()[index].r);
        const float3 primary_ray = normalize(world_position - push.attach.globals.main_camera.position);
        const float3 corrected_face_normal = flip_normal_to_incoming(detail_normal, detail_normal, primary_ray);
        const float3 sample_pos = rt_calc_ray_start(world_position, corrected_face_normal, primary_ray);
        const float3 world_tangent = normalize(cross(corrected_face_normal, float3(0,0,1) + 0.0001));
        const float3x3 tbn = transpose(float3x3(world_tangent, cross(world_tangent, corrected_face_normal), corrected_face_normal));
            
        const uint RAY_COUNT = push.attach.globals.ppd_settings.sample_count;
        const uint thread_seed = (index.x * push.attach.globals->settings.render_target_size.y + index.y) * push.attach.globals.frame_index;
        rand_seed(RAY_COUNT * thread_seed);


        RaytracingAccelerationStructure tlas = daxa::acceleration_structures[push.attach.tlas.index()];
        RayPayload payload = {};

        if (push.attach.globals.ppd_settings.debug_primary_trace != 0)
        {
            RayDesc ray = {};
            ray.Origin = camera.position;
            ray.TMax = 1000000000.0f;
            ray.TMin = 0.0f;
            ray.Direction = primary_ray;

            payload.miss = false; // need to set to false as we skip the closest hit shader
            payload.color = float3(0,0,0);
            payload.normal = detail_normal;
            TraceRay(tlas, 0, ~0, 0, 0, 0, ray, payload);

            float4 value = float4(payload.color, 1.0f);
            push.attach.ppd_raw_image.get()[index.xy] = value;
        }
        else 
        {
            if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_RTAO)
            {
                RayDesc ray = {};
                ray.Origin = sample_pos;
                ray.TMax = push.attach.globals.ppd_settings.ao_range;
                ray.TMin = 0.0f;

                float ao_factor = 0.0f;
                for (uint ray_i = 0; ray_i < RAY_COUNT; ++ray_i)
                {
                    const float3 hemi_sample = cosine_sample_hemi();
                    const float3 sample_dir = mul(tbn, hemi_sample);
                    ray.Direction = sample_dir;
                    payload.miss = false; // need to set to false as we skip the closest hit shader
                    payload.power = 1.0f;
                    TraceRay(tlas, 0, ~0, 0, 0, 0, ray, payload);
                    ao_factor += payload.miss ? 0 : payload.power;
                }

                let ao_value = 1.0f - ao_factor * rcp(RAY_COUNT);
                push.attach.ppd_raw_image.get()[index.xy] = float4(ao_value,0,0,0);
            }
            else // RTGI
            {
                RayDesc ray = {};
                ray.Origin = sample_pos;
                if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_FULL_RTGI)
                {
                    ray.TMax = 1000000000.0f;
                }
                else if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_SHORT_RANGE_RTGI)
                {
                    ray.TMax = push.attach.globals.ppd_settings.short_range_rtgi_range;
                }
                ray.TMin = 0.0f;

                float3 acc_light = 0.0f;
                int valid_rays = 0;
                for (uint ray_i = 0; ray_i < RAY_COUNT; ++ray_i)
                {
                    const float3 hemi_sample = cosine_sample_hemi();
                    const float3 sample_dir = mul(tbn, hemi_sample);
                    ray.Direction = sample_dir;
                    payload.miss = false; // need to set to false as we skip the closest hit shader
                    payload.color = float3(0,0,0);
                    payload.normal = detail_normal;
                    TraceRay(tlas, 0, ~0, 0, 0, 0, ray, payload);

                    acc_light += payload.color;
                }

                float4 value = float4(acc_light * rcp(RAY_COUNT), 1.0f);
                push.attach.ppd_raw_image.get()[index.xy] = value;
            }
        }
    }
    let clk_end = clockARB();
    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTAO_TRACE_CLOCKS)
    {
        push.attach.clocks_image.get()[index] = uint(clk_end - clk_start);
    }
}

[shader("anyhit")]
void any_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = rt_ao_push;

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

func trace_shadow_ray(RaytracingAccelerationStructure tlas, float3 position, float3 light_position, float3 flat_normal, float3 incoming_ray) -> bool
{
    float3 start = rt_calc_ray_start(position, flat_normal, incoming_ray);

    RayDesc ray = {};
    ray.Direction = normalize(light_position - position);
    ray.Origin = start;
    ray.TMax = length(light_position - position) * 1.01f;
    ray.TMin = 0.0f;

    RayPayload payload = {};
    payload.miss = false; // Only runs miss shader. Miss shader writes false.
    payload.skip_sky_shading_on_miss = true;
    TraceRay(tlas, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, ~0, 0, 0, 0, ray, payload);

    return payload.miss;
}

struct PGILightVisibilityTester : LightVisibilityTesterI
{
    RaytracingAccelerationStructure tlas;
    RenderGlobalData* globals;
    float sun_light(MaterialPointData material_point, float3 incoming_ray)
    {
        let sky = globals->sky_settings;
        let light_visible = trace_shadow_ray(tlas, material_point.position, material_point.position + sky.sun_direction * 1000000, material_point.face_normal, incoming_ray);
        return light_visible ? 1.0f : 0.0f;
    }
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        let push = rt_ao_push;

        GPUPointLight point_light = globals.scene.point_lights[light_index];
        float3 to_light = point_light.position - material_point.position;
        float3 to_light_dir = normalize(to_light);

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
        let push = rt_ao_push;

        GPUSpotLight spot_light = globals.scene.spot_lights[light_index];

        let RAYTRACED_POINT_SHADOWS = false;
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

[shader("closesthit")]
void closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = rt_ao_push;

    payload.miss = false;

    let primitive_index = PrimitiveIndex();
    let instance_id = InstanceID();
    let geometry_index = GeometryIndex();
    
    TriangleGeometry tri_geo = rt_get_triangle_geo(
        attr.barycentrics,
        instance_id,
        geometry_index,
        primitive_index,
        push.attach.globals.scene.meshes,
        push.attach.globals.scene.entity_to_meshgroup,
        push.attach.globals.scene.mesh_groups,
        push.attach.mesh_instances.instances
    );

    if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_RTAO)
    {
        if (tri_geo.material_index != INVALID_MANIFEST_INDEX)
        {
            const GPUMaterial material = push.attach.globals.scene.materials[tri_geo.material_index];

            bool emissive = any(material.emissive_color > 0.0f);
            payload.miss = emissive;
        }
        

        if (!payload.miss)
        {
            const float3 hit_location = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
            const uint primitive_index = PrimitiveIndex();
            
            const uint mesh_instance_index = InstanceID();
            MeshInstance* mesh_instance = push.attach.mesh_instances.instances + mesh_instance_index;
            GPUMesh *mesh = push.attach.globals.scene.meshes + mesh_instance->mesh_index;

            
            float3 color = GPU_MATERIAL_FALLBACK.base_color;
            if (mesh.material_index == INVALID_MANIFEST_INDEX)
            {
                const GPUMaterial *material = push.attach.globals.scene.materials + mesh.material_index;
                
                float3 color = material.base_color;

                if (!material.diffuse_texture_id.is_empty())
                {
                    let albedo_tex = Texture2D<float3>::get(material.diffuse_texture_id);
                    let albedo = albedo_tex.SampleLevel(SamplerState::get(push.attach.globals->samplers.linear_repeat), float2(0,0), 16).rgb;
                    color = color * albedo;
                }
            }

            let luma = clamp((color.r + color.g + color.b) / 3.0f, 0.01f, 0.8f);

            payload.power = 1.0f - square(square(square(square(luma))));
            payload.power *= sqrt((push.attach.globals.ppd_settings.ao_range - RayTCurrent())/push.attach.globals.ppd_settings.ao_range);
        }
    }
    else // RTGI
    {
        float3 hit_position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        PGISettings* pgi_settings = &push.attach.globals.pgi_settings;

        uint request_mode = PGI_PROBE_REQUEST_MODE_INDIRECT;
        {
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
            MaterialPointData material_point = evaluate_material<SHADING_QUALITY_LOW>(
                push.attach.globals,
                tri_geo,
                tri_point
            );
            bool double_sided_or_blend = ((material_point.material_flags & MATERIAL_FLAG_DOUBLE_SIDED) != MATERIAL_FLAG_NONE);
            PGILightVisibilityTester light_vis_tester = PGILightVisibilityTester(push.attach.tlas.get(), push.attach.globals);
            payload.color = shade_material<SHADING_QUALITY_HIGH>(
                push.attach.globals, 
                push.attach.sky_transmittance,
                push.attach.sky,
                material_point, 
                WorldRayOrigin(),
                WorldRayDirection(), 
                light_vis_tester, 
                push.attach.light_mask_volume.get(),
                push.attach.pgi_irradiance.get(), 
                push.attach.pgi_visibility.get(), 
                push.attach.pgi_info.get(),
                push.attach.pgi_requests.get_formatted(),
                request_mode
            ).rgb * (2.0f * 3.141f);
        }
    }
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    let push = rt_ao_push;
    payload.miss = true;

    if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_FULL_RTGI)
    {
        if (!payload.skip_sky_shading_on_miss)
        {
            payload.color = shade_sky(push.attach.globals, push.attach.sky_transmittance, push.attach.sky, WorldRayDirection());
        }
    }
    else if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_SHORT_RANGE_RTGI)
    {
        let push = rt_ao_push;
        payload.color = pgi_sample_irradiance(
            push.attach.globals, 
            &push.attach.globals.pgi_settings, 
            WorldRayOrigin(), // Little counter intuitive here but we want the origin as the shading point and the cam position as origin
            payload.normal, 
            payload.normal, 
            push.attach.globals.main_camera.position,
            normalize(WorldRayOrigin() - push.attach.globals.main_camera.position), 
            push.attach.pgi_irradiance.get(), 
            push.attach.pgi_visibility.get(), 
            push.attach.pgi_info.get(), 
            push.attach.pgi_requests.get_formatted(), 
            PGI_PROBE_REQUEST_MODE_DIRECT);
    }
}

static const uint kRadius = 1;
static const uint kWidth = 1 + 2 * kRadius;
static const float kernel1D[kWidth] = {0.27901, 0.44198, 0.27901};
static const float kernel[kWidth][kWidth] = {
  {kernel1D[0] * kernel1D[0], kernel1D[0] * kernel1D[1], kernel1D[0] * kernel1D[2]},
  {kernel1D[1] * kernel1D[0], kernel1D[1] * kernel1D[1], kernel1D[1] * kernel1D[2]},
  {kernel1D[2] * kernel1D[0], kernel1D[2] * kernel1D[1], kernel1D[2] * kernel1D[2]},
};

float DepthWeight(float linear_depth_a, float linear_depth_b, float3 normalCur, float3 viewDir, float4x4 proj, float phi)
{  
    float angleFactor = max(0.25, -dot(normalCur, viewDir));

    float diff = abs(linear_depth_a - linear_depth_b);
    return exp(-diff * angleFactor / phi);
}

float NormalWeight(float3 normalPrev, float3 normalCur, float phi)
{
    return max(0.01f, square(dot(normalPrev, normalCur)));
    float3 dd = normalPrev - normalCur;
    float d = dot(dd, dd);
    return exp(-d / phi);
}

#define DENOISER_TAP_WIDTH 1
[[vk::push_constant]] RTAODenoiserPush denoiser_push;
[shader("compute")]
[numthreads(RTAO_DENOISER_X, RTAO_DENOISER_Y, 1)]
void entry_rtao_denoiser(int2 index : SV_DispatchThreadID)
{
    let push = denoiser_push;

    if (any(greaterThan(index, push.size)))
    {
        return;
    }

    let VALIDITY_FRAMES = 8;

    CameraInfo camera = push.attach.globals->main_camera;

    float local_depth = push.attach.depth.get()[index];
    float local_depth_linear = linearise_depth(local_depth, camera.near_plane);
    float3 local_normal = uncompress_normal_octahedral_32(push.attach.normals.get()[index]);
    float3 ws_pos = pixel_index_to_world_space(camera, index, local_depth);
    float3 ray = normalize(ws_pos - camera.position);

    // Blur new samples.
    float3 raw_ppd = float3(0,0,0);
    float raw_ppd_weight_acc = 0.0f;
    {
        for (int col = 0; col < kWidth; col++)
        {
            for (int row = 0; row < kWidth; row++)
            {
                int2 offset = int2(row - kRadius, col - kRadius);
                int2 pos = int2(index) + offset;
                
                if (any(pos >= int2(push.size)) || any(pos < int2(0,0)))
                {
                    continue;
                }

                float kernelWeight = kernel[row][col];

                float4 raw = push.attach.ppd_raw.get()[pos];
                float3 irrad = raw.rgb;
                float3 normal = uncompress_normal_octahedral_32(push.attach.normals.get()[pos]);
                float depth = push.attach.depth.get()[pos];

                float depth_linear = linearise_depth(depth, camera.near_plane);

                float normalWeight = NormalWeight(normal, local_normal, 0.1f /*TODO*/);
                float depthWeight = DepthWeight(depth_linear, local_depth_linear, local_normal, ray, camera.proj, 0.1f /*TODO*/);
                
                float weight = depthWeight * normalWeight * kernelWeight;
                raw_ppd += pow(irrad, 0.2f) * weight;
                raw_ppd_weight_acc += weight;
            }
        }
    }
    raw_ppd = pow(raw_ppd * rcp(raw_ppd_weight_acc), 5.0f);
    raw_ppd = clamp(raw_ppd, float3(0,0,0), (100000000000.0f).xxx);

    float non_linear_depth = push.attach.depth.get()[index];
    float4 prev_frame_ndc = {};
    {
        float4x4 view_proj = camera.view_proj;
        float4x4 view_proj_prev = push.attach.globals->main_camera_prev_frame.view_proj;
        float3 camera_position = camera.position;
        float3 pixel_world_position = pixel_index_to_world_space(camera, index, non_linear_depth);

        prev_frame_ndc = mul(view_proj_prev, float4(pixel_world_position, 1.0f));
        prev_frame_ndc.xyz /= prev_frame_ndc.w;
    }

    float3 accepted_history_ao = {};
    float acceptance = 0.0f;
    float validity = 0.0f;
    {
        float2 pixel_prev_uv = (prev_frame_ndc.xy * 0.5f + 0.5f);        
        float2 pixel_prev_index = pixel_prev_uv * float2(camera.screen_size);
        float2 base_texel = floor(pixel_prev_index - 0.5f); 
        float2 interpolants = clamp((pixel_prev_index - (base_texel + 0.5f)), float2(0,0), float2(1,1));
        float2 gather_uv = (base_texel + 0.5f) / float2(camera.screen_size);
        
        float4 history_ao_r = push.attach.ppd_history.get().GatherRed(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_ao_g = push.attach.ppd_history.get().GatherGreen(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_ao_b = push.attach.ppd_history.get().GatherBlue(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_validity = push.attach.ppd_history.get().GatherAlpha(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_depth = push.attach.depth_history.get().GatherRed(push.attach.globals.samplers.linear_clamp.get(), gather_uv).wzxy;
        uint4 history_normal = push.attach.normal_history.get().GatherRed(push.attach.globals.samplers.linear_clamp.get(), gather_uv).wzxy;
        let hist_normal_tex = push.attach.depth_history.get();
        float4 sample_linear_weights = float4(
            (1.0f - interpolants.x) * (1.0f - interpolants.y),
            interpolants.x * (1.0f - interpolants.y),
            (1.0f - interpolants.x) * interpolants.y,
            interpolants.x * interpolants.y
        );

        // push.attach.debug_image.get()[index] = float4(uncompress_normal_octahedral_32(push.attach.normal_history.get()[index]), 1.0f);

        float linear_prev_depth = linearise_depth(prev_frame_ndc.z, camera.near_plane);
        float3 interpolated_ao = 0.0f;
        float interpolated_ao_weight_sum = 0.0f;
        float interpolated_validity_sum = 0.0f;
        for (int i = 0; i < 4; ++i)
        {
            float depth = history_depth[i];
            float linear_depth = linearise_depth(depth, camera.near_plane);
            float depth_diff = abs(linear_prev_depth - linear_depth);
            float acceptable_diff = 0.05f;
            const float depth_weight = DepthWeight(linear_depth, linear_prev_depth, local_normal, ray, camera.proj, 0.1);
            float3 normal = uncompress_normal_octahedral_32(history_normal[i]);
            float normalWeight = NormalWeight(normal, local_normal, 0.01f /*TODO*/);
            float validity_weight = square(history_validity[i]);

            if (depth_weight > 0.1f)
            {
                float weight = sample_linear_weights[i] * depth_weight * normalWeight * square(history_validity[i] * rcp(VALIDITY_FRAMES));
                interpolated_ao_weight_sum += weight;
                interpolated_ao += pow(float3(history_ao_r[i], history_ao_g[i], history_ao_b[i]), 0.2f) * weight; // pow used to blend in a more perceptual color space relative to perception of brightness
                interpolated_validity_sum += history_validity[i] * weight;
            }
        }
        if (interpolated_ao_weight_sum > 0.01f)
        {
            validity = interpolated_validity_sum * rcp(interpolated_ao_weight_sum);
            acceptance = 1.0f;
            accepted_history_ao = pow(interpolated_ao * rcp(interpolated_ao_weight_sum), 5.0f);
        }
        if (any(base_texel < 0.0f || (base_texel + 1.0f >= float2(camera.screen_size))))
        {
            validity = 0;
            acceptance = 0.0f;
        }
        //accepted_history_ao = push.attach.history.get().SampleLevel(push.attach.globals.samplers.linear_clamp.get(), pixel_prev_uv, 0.0f).x;
        //push.attach.debug_image.get()[index].xy = interpolated_ao;
    }

    validity = min(validity, VALIDITY_FRAMES);

    acceptance = min(validity * rcp(VALIDITY_FRAMES), push.attach.globals.ppd_settings.denoiser_accumulation_max_epsi);
    float exp_average = lerp(1.0f, 0.1f, acceptance);

    let new_ao = lerp(raw_ppd, accepted_history_ao, acceptance);
    float new_depth = push.attach.depth.get()[index];

    let new_validity = 1.0f + validity;
    push.attach.ppd_image.get()[index] = float4(new_ao, new_validity);
}