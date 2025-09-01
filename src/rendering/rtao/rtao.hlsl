#pragma once

#define DAXA_RAY_TRACING 1

#include <daxa/daxa.inl>

#include "rtao.inl"

#include "shader_lib/misc.hlsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/raytracing.hlsl"
#include "shader_lib/depth_util.glsl"
#include "shader_lib/transform.hlsl"

#define PI 3.1415926535897932384626433832795

[[vk::push_constant]] RayTraceAmbientOcclusionPush rt_ao_push;

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
        CameraInfo camera = push.attach.globals.view_camera;
        const float3 world_position = pixel_index_to_world_space(camera, index, depth);
        float3 detail_normal = uncompress_normal_octahedral_32(push.attach.view_cam_detail_normals.get()[index].r);
        const float3 primary_ray = normalize(world_position - push.attach.globals.view_camera.position);
        
        const float3 sample_pos = rt_calc_ray_start(world_position, detail_normal, primary_ray);
        const float3 world_tangent = normalize(cross(detail_normal, float3(0,0,1) + 0.0001));
        const float3x3 tbn = transpose(float3x3(world_tangent, cross(world_tangent, detail_normal), detail_normal));
            
        const uint RAY_COUNT = push.attach.globals.ao_settings.sample_count;
        const uint thread_seed = (index.x * push.attach.globals->settings.render_target_size.y + index.y) * push.attach.globals.frame_index;
        rand_seed(RAY_COUNT * thread_seed);

        RaytracingAccelerationStructure tlas = daxa::acceleration_structures[push.attach.tlas.index()];
        RayPayload payload = {};

        if (push.attach.globals.ao_settings.mode == AMBIENT_OCCLUSION_MODE_RTAO)
        {
            RayDesc ray = {};
            ray.Origin = sample_pos;
            ray.TMax = push.attach.globals.ao_settings.ao_range;
            ray.TMin = 0.0f;

            float ao_factor = 0.0f;
            for (uint ray_i = 0; ray_i < RAY_COUNT; ++ray_i)
            {
                const float3 hemi_sample = rand_cosine_sample_hemi();
                const float3 sample_dir = mul(tbn, hemi_sample);
                ray.Direction = sample_dir;
                payload.miss = false; // need to set to false as we skip the closest hit shader
                payload.power = 1.0f;
                TraceRay(tlas, 0, ~0, 0, 0, 0, ray, payload);
                ao_factor += payload.miss ? 0 : payload.power;
            }

            let ao_value = 1.0f - ao_factor * rcp(RAY_COUNT);
            push.attach.rtao_raw_image.get()[index.xy] = float4(ao_value,0,0,0);
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

[shader("closesthit")]
void closest_hit(inout RayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    let push = rt_ao_push;

    payload.miss = false;
    {
        if (!payload.miss)
        {
            payload.power *= sqrt((push.attach.globals.ao_settings.ao_range - RayTCurrent())/push.attach.globals.ao_settings.ao_range);
        }
    }
}

[shader("miss")]
void miss(inout RayPayload payload)
{
    let push = rt_ao_push;
    payload.miss = true;
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
    float3 raw_rtao = float3(0,0,0);
    float raw_rtao_weight_acc = 0.0f;
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

                float4 raw = push.attach.rtao_raw.get()[pos];
                float3 irrad = raw.rgb;
                float3 normal = uncompress_normal_octahedral_32(push.attach.normals.get()[pos]);
                float depth = push.attach.depth.get()[pos];

                float depth_linear = linearise_depth(depth, camera.near_plane);

                float normalWeight = NormalWeight(normal, local_normal, 0.1f /*TODO*/);
                float depthWeight = DepthWeight(depth_linear, local_depth_linear, local_normal, ray, camera.proj, 0.1f /*TODO*/);
                
                float weight = depthWeight * normalWeight * kernelWeight;
                raw_rtao += pow(irrad, 0.2f) * weight;
                raw_rtao_weight_acc += weight;
            }
        }
    }
    raw_rtao = pow(raw_rtao * rcp(raw_rtao_weight_acc), 5.0f);
    raw_rtao = clamp(raw_rtao, float3(0,0,0), (100000000000.0f).xxx);

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
        
        float4 history_ao_r = push.attach.rtao_history.get().GatherRed(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_ao_g = push.attach.rtao_history.get().GatherGreen(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_ao_b = push.attach.rtao_history.get().GatherBlue(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_validity = push.attach.rtao_history.get().GatherAlpha(push.attach.globals.samplers.nearest_clamp.get(), gather_uv).wzxy;
        float4 history_depth = push.attach.depth_history.get().GatherRed(push.attach.globals.samplers.linear_clamp.get(), gather_uv).wzxy;
        uint4 history_normal = push.attach.normal_history.get().GatherRed(push.attach.globals.samplers.linear_clamp.get(), gather_uv).wzxy;
        let hist_normal_tex = push.attach.depth_history.get();
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
    }

    validity = min(validity, VALIDITY_FRAMES);

    acceptance = min(validity * rcp(VALIDITY_FRAMES), push.attach.globals.ao_settings.denoiser_accumulation_max_epsi);
    float exp_average = lerp(1.0f, 0.1f, acceptance);

    let new_ao = lerp(raw_rtao, accepted_history_ao, acceptance);
    float new_depth = push.attach.depth.get()[index];

    let new_validity = 1.0f + validity;
    push.attach.rtao_image.get()[index] = float4(new_ao, new_validity);
}