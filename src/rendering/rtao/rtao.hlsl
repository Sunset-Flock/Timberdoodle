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

#define PI 3.1415926535897932384626433832795
#define RTAO_RANGE 2.0f
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
    float power;
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
        CameraInfo camera = push.attach.globals.camera;
        const float3 world_position = pixel_index_to_world_space(camera, index, depth);
        const float3 detail_normal = uncompress_normal_octahedral_32(push.attach.view_cam_detail_normals.get()[index].r);
        const float3 primary_ray = normalize(world_position - push.attach.globals.camera.position);
        const float3 corrected_face_normal = flip_normal_to_incoming(detail_normal, detail_normal, primary_ray);
        const float3 sample_pos = rt_calc_ray_start(world_position, corrected_face_normal, primary_ray);
        const float3 world_tangent = normalize(cross(corrected_face_normal, float3(0,0,1) + 0.0001));
        const float3x3 tbn = transpose(float3x3(world_tangent, cross(world_tangent, corrected_face_normal), corrected_face_normal));
            
        const uint AO_RAY_COUNT = push.attach.globals.settings.ao_samples;
        const uint thread_seed = (index.x * push.attach.globals->settings.render_target_size.y + index.y) * push.attach.globals.frame_index;
        rand_seed(AO_RAY_COUNT * thread_seed);

        RayDesc ray = {};
        ray.Origin = sample_pos;
        ray.TMax = RTAO_RANGE;
        ray.TMin = 0.0f;

        RaytracingAccelerationStructure tlas = daxa::acceleration_structures[push.attach.tlas.index()];
        RayPayload payload = {};
        float ao_factor = 0.0f;
        for (uint ray_i = 0; ray_i < AO_RAY_COUNT; ++ray_i)
        {
            const float3 hemi_sample = cosine_sample_hemi();
            const float3 sample_dir = mul(tbn, hemi_sample);
            ray.Direction = sample_dir;
            payload.miss = false; // need to set to false as we skip the closest hit shader
            payload.power = 1.0f;
            TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, ~0, 0, 0, 0, ray, payload);
            ao_factor += payload.miss ? 0 : payload.power;
        }

        let ao_value = 1.0f - ao_factor * rcp(AO_RAY_COUNT);
        push.attach.ao_image.get()[index.xy] = ao_value;
        //push.attach.debug_image.get()[index.xy] = float4(ao_value.xxx, 1.0f);
    }
    let clk_end = clockARB();
    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_RTAO_TRACE_CLOCKS)
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

        if ((mesh.vertex_uvs == Ptr<float2>(0)) || material.diffuse_texture_id.is_empty())
        {
            return;
        }

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

        let albedo_tex = Texture2D<float3>::get(material.diffuse_texture_id);
        let albedo = albedo_tex.SampleLevel(SamplerState::get(push.attach.globals->samplers.linear_repeat), interp_uv, 5).rgb;

        let luma = (albedo.r + albedo.g + albedo.b) / 3.0f;

        payload.power = lerp(1.0 - square(luma * luma_constant), 0.2f, 0.2f) 
        #if RTAO_RANGE_FALLOFF
        * sqrt((RTAO_RANGE - RayTCurrent())/RTAO_RANGE)
        #endif
        ;
    }
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    payload.miss = true;
}

float DepthWeight(float linear_depth_a, float linear_depth_b, float3 normalCur, float3 viewDir, float4x4 proj, float phi)
{  
  float angleFactor = max(0.25, -dot(normalCur, viewDir));

  float diff = abs(linear_depth_a - linear_depth_b);
  return exp(-diff * angleFactor / phi);
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

    CameraInfo camera = push.attach.globals->camera;


    // Blur new samples.
    float blurred_new_ao = push.attach.src.get()[index];
    if (false) {
        // Reads and box blurrs a 6x6 area.
        // Uses 3x3 taps. Each tap is doing a 2x2 linear interpolation.
        // All samples are averaged for the box blur.
        const int2 start_index = index - int2(2 * (DENOISER_TAP_WIDTH/2), 2 * (DENOISER_TAP_WIDTH/2));
        const float2 start_uv = float2(start_index) * push.inv_size;
        const float2 tap_uv_increment = 2.0f * push.inv_size;
        for (uint i = 0; i < DENOISER_TAP_WIDTH * DENOISER_TAP_WIDTH; ++i)
        {
            let x = i % DENOISER_TAP_WIDTH;
            let y = i / DENOISER_TAP_WIDTH;

            let v = push.attach.src.get().SampleLevel(push.attach.globals.samplers.linear_clamp.get(), start_uv + tap_uv_increment * float2(x,y), 0.0f).r;
            blurred_new_ao += v * (1.0f / (DENOISER_TAP_WIDTH * DENOISER_TAP_WIDTH));
        }
    }

    float non_linear_depth = push.attach.depth.get()[index];
    float3 ray = normalize(pixel_index_to_world_space(camera, index, non_linear_depth) - camera.position);
    float4 prev_frame_ndc = {};
    {
        float4x4 view_proj = camera.view_proj;
        float4x4 view_proj_prev = push.attach.globals->camera_prev_frame.view_proj;
        float3 camera_position = camera.position;
        float3 pixel_world_position = pixel_index_to_world_space(camera, index, non_linear_depth);

        prev_frame_ndc = mul(view_proj_prev, float4(pixel_world_position, 1.0f));
        prev_frame_ndc.xyz /= prev_frame_ndc.w;
    }

    float accepted_history_ao = {};
    float acceptance = 0.0f;
    {
        float2 pixel_prev_uv = (prev_frame_ndc.xy * 0.5f + 0.5f);
        float2 pixel_prev_index = pixel_prev_uv * float2(camera.screen_size);
        float2 base_texel = max(float2(0,0), floor(pixel_prev_index - 0.5f)); 
        float2 interpolants = clamp((pixel_prev_index - (base_texel + 0.5f)), float2(0,0), float2(1,1));
        float2 gather_uv = (base_texel + 1.0f) / float2(camera.screen_size);
        
        float4 history_ao = push.attach.history.get().GatherRed(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_depth = push.attach.history.get().GatherGreen(push.attach.globals.samplers.linear_clamp.get(), gather_uv).wzxy;
        //history_ao = max(history_ao, 0.0f);
        float4 sample_linear_weights = float4(
            (1.0f - interpolants.x) * (1.0f - interpolants.y),
            interpolants.x * (1.0f - interpolants.y),
            (1.0f - interpolants.x) * interpolants.y,
            interpolants.x * interpolants.y
        );

        float linear_prev_depth = linearise_depth(prev_frame_ndc.z, camera.near_plane);
        float3 face_normal = uncompress_normal_octahedral_32(push.attach.face_normals.get()[index]);
        float interpolated_ao = 0.0f;
        float interpolated_ao_weight_sum = 0.0f;
        for (int i = 0; i < 4; ++i)
        {
            float depth = history_depth[i];
            float linear_depth = linearise_depth(depth, camera.near_plane);
            float depth_diff = abs(linear_prev_depth - linear_depth);
            float acceptable_diff = 0.05f;
            const float depth_weight = DepthWeight(linear_depth, linear_prev_depth, face_normal, ray, camera.proj, 0.1);

            if (depth_weight > 0.1)
            {
                interpolated_ao_weight_sum += sample_linear_weights[i] * depth_weight;
                interpolated_ao += history_ao[i] * sample_linear_weights[i] * depth_weight;
            }
        }
        if (interpolated_ao_weight_sum > 0.01f)
        {
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

    float exp_average = lerp(1.0f, 0.02f, acceptance);

    float new_ao = lerp(accepted_history_ao, blurred_new_ao, exp_average);
    float new_depth = push.attach.depth.get()[index];

    push.attach.dst.get()[index] = float2(new_ao, new_depth);
}