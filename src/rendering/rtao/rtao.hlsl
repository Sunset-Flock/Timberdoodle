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

#define PI 3.1415926535897932384626433832795
#define RTAO_RANGE 1.5f
#define RTAO_RANGE_FALLOFF 1

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

        if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_RTAO)
        {
            RayDesc ray = {};
            ray.Origin = sample_pos;
            ray.TMax = RTAO_RANGE;
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
            if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_RTGI)
            {
                ray.TMax = 1000000000.0f;
            }
            else if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_RTGI_HYBRID)
            {
                ray.TMax = 4.0f;
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
    let clk_end = clockARB();
    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTAO_TRACE_CLOCKS)
    {
        push.attach.debug_image.get()[index] = float4((clk_end - clk_start),0,0,0);
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

struct PGILightVisibilityTester : LightVisibilityTesterI
{
    RaytracingAccelerationStructure tlas;
    RenderGlobalData* globals;
    float sun_light(MaterialPointData material_point, float3 incoming_ray)
    {
        let sky = globals->sky_settings;

        float t_max = 10000.0f;
        float3 start = rt_calc_ray_start(material_point.position, material_point.geometry_normal, incoming_ray);
        float3 dir = sky.sun_direction;
        
        
        RayDesc ray = {};
        ray.Direction = dir;
        ray.Origin = start;
        ray.TMax = t_max;
        ray.TMin = 0.0f;

        RayPayload payload;
        payload.miss = false; // Only runs miss shader. Miss shader writes false.
        TraceRay(tlas, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, ~0, 0, 0, 0, ray, payload);

        bool path_occluded = !payload.miss;

        return path_occluded ? 0.0f : 1.0f;
    }
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index)
    {
        float3 start = rt_calc_ray_start(material_point.position, material_point.geometry_normal, incoming_ray);
        GPUPointLight point_light = globals.scene.point_lights[light_index];
        float3 to_light = point_light.position - material_point.position;
        float3 to_light_dir = normalize(to_light);
        #if 0
        
        RayDesc ray = {};
        ray.Direction = normalize(to_light);
        ray.Origin = start;
        ray.TMax = length(to_light) * 1.01f;
        ray.TMin = 0.0f;

        RayPayload payload;
        payload.hit = true; // Only runs miss shader. Miss shader writes false.
        TraceRay(tlas, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_CULL_BACK_FACING_TRIANGLES, ~0, 0, 0, 0, ray, payload);

        bool path_occluded = payload.hit;
        #endif

        
        // const int face_idx = cube_face_from_dir(to_light_dir);
        // const float4x4 point_view_projection = point_light.face_cameras[face_idx].view_proj;
        // float4 light_projected_pos = mul(point_view_projection, float4(material_point.position, 1.0f));

        // let constant_cascade = 4;

        // uint prev_page_state_point;
        // const uint point_page_array_index = get_vsm_point_page_array_idx(face_idx, light_idx);
        // InterlockedOr(
        //     AT.vsm_point_spot_page_table[constant_cascade].get_formatted()[uint3(vsm_point_page_coords.xy, point_page_array_index)], // [mip].get()[x, y, array_layer]
        //     uint(requests_allocation_mask() | visited_marked_mask()),
        //     prev_page_state_point
        // );
        // if(!get_requests_allocation(prev_page_state_point) && !get_is_allocated(prev_page_state_point))
        // {
        //     uint allocation_index;
        //     InterlockedAdd(AT.vsm_allocation_requests->counter, 1u, allocation_index);
        //     if(allocation_index < MAX_VSM_ALLOC_REQUESTS)
        //     {
        //         AT.vsm_allocation_requests.requests[allocation_index] = AllocationRequest(int3(vsm_point_page_coords.xy, point_page_array_index), 0u, constant_cascade);
        //     }
        // }]
        // else 
        // {
        //     if(get_is_allocated(prev_page_state_point))
        //     {
        //          if (!get_is_visited_marked(prev_page_state_point))
        //          {
        //               InterlockedOr(
        //                   AT.vsm_meta_memory_table.get_formatted()[get_meta_coords_from_vsm_entry(prev_page_state_point)],
        //                   meta_memory_visited_mask()
        //               );
        //         }
        //         
        //         SAMPLE HERE
        //     }
        // }

        return 1.0f;//path_occluded ? 0.0f : 1.0f;
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
            if (mesh.material_index == INVALID_MANIFEST_INDEX)
            {
                return;
            }
            const GPUMaterial *material = push.attach.globals.scene.materials + mesh.material_index;

            let luma_constant = (material.base_color.r + material.base_color.g + material.base_color.b) / 3.0f;

            payload.power = 1.0f - square(luma_constant);

            if (material.diffuse_texture_id.is_empty())
            {
                return;
            }

            let albedo_tex = Texture2D<float3>::get(material.diffuse_texture_id);
            let albedo = albedo_tex.SampleLevel(SamplerState::get(push.attach.globals->samplers.linear_repeat), float2(0,0), 16).rgb;

            let luma = (albedo.r + albedo.g + albedo.b) / 3.0f;

            payload.power = 1.0f - square(square(luma))
            #if RTAO_RANGE_FALLOFF
            * sqrt((RTAO_RANGE - RayTCurrent())/RTAO_RANGE)
            #endif
            ;
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
            material_point.position = hit_position;
            bool double_sided_or_blend = ((material_point.material_flags & MATERIAL_FLAG_DOUBLE_SIDED) != MATERIAL_FLAG_NONE);
            PGILightVisibilityTester light_vis_tester = PGILightVisibilityTester( push.attach.tlas.get(), push.attach.globals);
            payload.color = shade_material<SHADING_QUALITY_HIGH>(
                push.attach.globals, 
                push.attach.sky_transmittance,
                push.attach.sky,
                material_point, 
                WorldRayOrigin(),
                WorldRayDirection(), 
                light_vis_tester, 
                push.attach.light_mask_volume.get(),
                push.attach.pgi_radiance.get(), 
                push.attach.pgi_visibility.get(), 
                push.attach.pgi_info.get(),
                push.attach.pgi_requests.get_formatted(),
                request_mode
            ).rgb;
        }
    }
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    let push = rt_ao_push;
    payload.miss = true;

    if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_RTGI)
    {
        payload.color = shade_sky(push.attach.globals, push.attach.sky_transmittance, push.attach.sky, WorldRayDirection());
    }
    else if (push.attach.globals.ppd_settings.mode == PER_PIXEL_DIFFUSE_MODE_RTGI_HYBRID)
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
            push.attach.pgi_radiance.get(), 
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
  // float d = max(0.05, dot(normalCur, normalPrev));
  // return d * d;
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

    CameraInfo camera = push.attach.globals->main_camera;

    float local_depth = push.attach.depth.get()[index];
    float local_depth_linear = linearise_depth(local_depth, camera.near_plane);
    float3 local_normal = uncompress_normal_octahedral_32(push.attach.normals.get()[index]);
    float3 ws_pos = pixel_index_to_world_space(camera, index, local_depth);
    float3 ray = normalize(ws_pos - camera.position);

    // Blur new samples.
    float3 raw_ppd = push.attach.ppd_raw.get()[index].rgb;
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

                float normalWeight = NormalWeight(normal, local_normal, 0.01f /*TODO*/);
                float depthWeight = DepthWeight(depth_linear, local_depth_linear, local_normal, ray, camera.proj, 0.1f /*TODO*/);
                
                float weight = depthWeight * normalWeight;
                raw_ppd += irrad * weight * kernelWeight;
                raw_ppd_weight_acc += weight * kernelWeight;
            }
        }
    }
    raw_ppd *= rcp(raw_ppd_weight_acc + 0.1f);
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
        //history_ao = max(history_ao, 0.0f);
        float4 sample_linear_weights = float4(
            (1.0f - interpolants.x) * (1.0f - interpolants.y),
            interpolants.x * (1.0f - interpolants.y),
            (1.0f - interpolants.x) * interpolants.y,
            interpolants.x * interpolants.y
        );

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
            float normalWeight = 1.0f;//NormalWeight(normal, local_normal, 0.01f /*TODO*/);
            float validity_weight = history_validity[i];

            if (depth_weight > 0.1)
            {
                float weight = sample_linear_weights[i] * depth_weight * normalWeight * history_validity[i];
                interpolated_ao_weight_sum += weight;
                interpolated_ao += float3(history_ao_r[i], history_ao_g[i], history_ao_b[i]) * weight;
                interpolated_validity_sum += history_validity[i] * weight;
            }
        }
        if (interpolated_ao_weight_sum > 0.01f)
        {
            validity = interpolated_validity_sum * rcp(interpolated_ao_weight_sum);
            acceptance = 1.0f;
            accepted_history_ao = interpolated_ao * rcp(interpolated_ao_weight_sum);
        }
        if (any(base_texel < 0.0f || (base_texel + 1.0f >= float2(camera.screen_size))))
        {
            acceptance = 0.0f;
        }
        //accepted_history_ao = push.attach.history.get().SampleLevel(push.attach.globals.samplers.linear_clamp.get(), pixel_prev_uv, 0.0f).x;
        //push.attach.debug_image.get()[index].xy = interpolated_ao;
    }

    let validity_frames = 16;

    validity = min(validity, validity_frames);

    acceptance = min(validity * rcp(validity_frames), 0.95f);
    float exp_average = lerp(1.0f, 0.1f, acceptance);

    let new_ao = lerp(raw_ppd, accepted_history_ao, acceptance);
    float new_depth = push.attach.depth.get()[index];

    let new_validity = 1.0f + validity;
    push.attach.ppd_image.get()[index] = float4(new_ao, new_validity);
}