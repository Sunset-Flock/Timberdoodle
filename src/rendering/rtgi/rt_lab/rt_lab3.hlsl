#pragma once

#include "rt_lab3.inl"

#include "rt_lab_trace.hlsl" // rt_lab_trace_radiance_ex (hit shading of the lab rays) + RTGI_SHADE_CONTEXT + rt_lab_shared.hlsl
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtLab3TracePush rt_lab3_trace_push;
[[vk::push_constant]] RtLab3TemporalPush rt_lab3_temporal_push;
[[vk::push_constant]] RtLab3SpatialPush rt_lab3_spatial_push;
[[vk::push_constant]] RtLab3ShadePush rt_lab3_shade_push;
[[vk::push_constant]] RtLab3FilterPush rt_lab3_filter_push;

// == Reservoir =========================================================================================================
// One light sample (x_s, n_s, L_o) with its unbiased contribution weight W and confidence M, for the target function
// p_hat(x) = luminance(L_o) cos(receiver normal, direction to x_s). Estimate of the cosine weighted mean radiance:
// L_o cos W / pi (a fresh cosine ray has W = pi / cos: estimate L_o).

struct RtLab3Reservoir
{
    float3 position;
    uint normal;
    uint age; // frames since the sample was traced
    float ray_history; // rays shot at this pixel over time (not the confidence M), the history of the ray redistribution
    float3 radiance;
    float weight;
    float count;
    bool unvalidated; // came from a neighbor by spatial reuse, visibility from this pixel not tested yet (the shade pass does it)
};

func rt_lab3_empty_reservoir() -> RtLab3Reservoir
{
    RtLab3Reservoir r;
    r.position = float3(0.0f, 0.0f, 0.0f);
    r.normal = 0u;
    r.age = 0u;
    r.ray_history = 0.0f;
    r.radiance = float3(0.0f, 0.0f, 0.0f);
    r.weight = 0.0f;
    r.count = 0.0f;
    r.unvalidated = false;
    return r;
}

// 16 byte layout, see rt_lab3.inl.
// Shared exponent colour (RGB9E5).
func rt_lab3_pack_rgb9e5(float3 rgb) -> uint
{
    rgb = clamp(rgb, 0.0f, 65408.0f);
    const float max_channel = max(rgb.x, max(rgb.y, rgb.z));
    if (max_channel <= 0.0f)
    {
        return 0u;
    }
    int shared_exponent = max(-16, int(floor(log2(max_channel)))) + 16;
    float denominator = exp2(float(shared_exponent - 24));
    if (int(max_channel / denominator + 0.5f) == 512)
    {
        shared_exponent += 1;
        denominator *= 2.0f;
    }
    const uint3 m = uint3(rgb / denominator + 0.5f);
    return m.x | (m.y << 9u) | (m.z << 18u) | (uint(shared_exponent) << 27u);
}

func rt_lab3_unpack_rgb9e5(uint v) -> float3
{
    const float scale = exp2(float(int(v >> 27u)) - 24.0f);
    return float3(float(v & 511u), float((v >> 9u) & 511u), float((v >> 18u) & 511u)) * scale;
}

// Confidence M and ray history: 8 bit, sqrt coded (max 1016, 2.5 % steps around 100).
func rt_lab3_encode_count(float x) -> uint
{
    return min(255u, uint(sqrt(max(x, 0.0f)) * 8.0f + 0.5f));
}

func rt_lab3_decode_count(uint b) -> float
{
    const float v = float(b) * 0.125f;
    return v * v;
}

// owner_position = world position of the pixel the reservoir belongs to.
func rt_lab3_unpack(uint4 v, float3 owner_position) -> RtLab3Reservoir
{
    const uint direction_x = ((v.y & 0xFFFu) << 4u) | 8u;
    const uint direction_y = (((v.y >> 12u) & 0xFFFu) << 4u) | 8u;
    RtLab3Reservoir r;
    r.position = owner_position + uncompress_normal_octahedral_32(direction_x | (direction_y << 16u)) * f16tof32(v.z & 0xFFFFu);
    r.weight = f16tof32(v.z >> 16u);
    r.count = rt_lab3_decode_count(v.y >> 24u);
    r.radiance = rt_lab3_unpack_rgb9e5(v.x) * RTGI_RADIANCE_SCALE;
    r.normal = (((v.w & 0xFFu) << 8u) | 0x80u) | ((((v.w >> 8u) & 0xFFu) << 8u | 0x80u) << 16u);
    r.age = (v.w >> 16u) & 0x7Fu;
    r.unvalidated = ((v.w >> 23u) & 1u) != 0u;
    r.ray_history = rt_lab3_decode_count(v.w >> 24u);
    return r;
}

func rt_lab3_load(Texture2DArray<uint4> reservoirs, int2 pixel, uint slot, float3 owner_position) -> RtLab3Reservoir
{
    return rt_lab3_unpack(reservoirs[int3(pixel, int(slot))], owner_position);
}

func rt_lab3_store(RWTexture2DArray<uint4> reservoirs, uint2 pixel, uint slot, RtLab3Reservoir r, float3 owner_position)
{
    // A reservoir that is not finite is dropped (an empty one has weight 0 and count 0).
    const bool valid = isfinite(r.weight) && all(isfinite(r.position)) && all(isfinite(r.radiance)) && isfinite(r.count);
    if (!valid)
    {
        r = rt_lab3_empty_reservoir();
        r.position = owner_position;
    }
    const float3 to_sample = r.position - owner_position;
    const float distance = length(to_sample);
    const uint direction = compress_normal_octahedral_32(distance > 1e-6f ? to_sample / distance : float3(0.0f, 0.0f, 1.0f));
    const uint direction_x = min(((direction & 0xFFFFu) + 8u) >> 4u, 4095u);
    const uint direction_y = min(((direction >> 16u) + 8u) >> 4u, 4095u);
    const uint normal_x = min(((r.normal & 0xFFFFu) + 128u) >> 8u, 255u);
    const uint normal_y = min(((r.normal >> 16u) + 128u) >> 8u, 255u);
    reservoirs[uint3(pixel, slot)] = uint4(
        rt_lab3_pack_rgb9e5(r.radiance * rcp(RTGI_RADIANCE_SCALE)),
        direction_x | (direction_y << 12u) | (rt_lab3_encode_count(r.count) << 24u),
        f32tof16(min(distance, 65504.0f)) | (f32tof16(clamp(r.weight, 0.0f, 65504.0f)) << 16u),
        normal_x | (normal_y << 8u) | (min(r.age, 127u) << 16u) | (r.unvalidated ? (1u << 23u) : 0u) | (rt_lab3_encode_count(isfinite(r.ray_history) ? r.ray_history : 0.0f) << 24u));
}

// The seed of a pass: per frame, pixel and pass (own stream per use).
func rt_lab3_seed(RenderGlobalData* globals, uint2 pixel, uint stream) -> uint
{
    return uint(globals->trunk_flt_frame_index) * 257u + pixel.x * 9629u + pixel.y * 10069u + stream * 1000003u + 0x2545F491u;
}

// Shadow ray from the receiver to the sample point: true when nothing blocks it (alpha tested).
func rt_lab3_visible(RenderGlobalData* globals, MeshInstancesBufferHead* mesh_instances, daxa_u64 tlas, float3 receiver, float3 receiver_normal, float bias, float3 target) -> bool
{
    const float3 origin = receiver + receiver_normal * bias;
    const float3 to_target = target - origin;
    const float distance_to_target = length(to_target);
    if (distance_to_target < 1e-4f)
    {
        return true;
    }
    RayDesc ray = {};
    ray.Origin = origin;
    ray.Direction = to_target / distance_to_target;
    ray.TMin = 0.0f;
    ray.TMax = distance_to_target * 0.98f; // stop short of the surface the sample sits on
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(RaytracingAccelerationStructure::get(tlas), 0, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            rt_is_alpha_hit(
                globals,
                mesh_instances,
                globals.scene.meshes,
                globals.scene.materials,
                q.CandidateTriangleBarycentrics(),
                q.CandidatePrimitiveIndex(), q.CandidateInstanceID(), ray.Origin, ray.Direction, q.CandidateTriangleRayT()))
        {
            q.CommitNonOpaqueTriangleHit();
        }
    }
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

// Reconnects the sample of `r` (found from `source_position`) to a receiver: its target function value there and the Jacobian of the shift
// (solid angle at the receiver / solid angle at the source: cos(phi_receiver) d_source^2 / (cos(phi_source) d_receiver^2), phi = angle between
// the sample's normal and the direction to the visible point). False when the sample cannot be reused (below the horizon, facing away, too close,
// Jacobian outside the allowed range).
func rt_lab3_shift(RtLab3Reservoir r, float3 source_position, float3 receiver_position, float3 receiver_normal, RtLab3Settings s, out float p_hat, out float jacobian) -> bool
{
    p_hat = 0.0f;
    jacobian = 1.0f;
    const float3 to_sample = r.position - receiver_position;
    const float distance_receiver = length(to_sample);
    if (distance_receiver < max(s.min_reconnection_distance, 1e-4f))
    {
        return false;
    }
    const float3 direction = to_sample / distance_receiver;
    const float cos_receiver = dot(receiver_normal, direction);
    if (cos_receiver <= 1e-4f)
    {
        return false;
    }
    const float3 sample_normal = uncompress_normal_octahedral_32(r.normal);
    const float3 from_source = r.position - source_position;
    const float distance_source = max(length(from_source), 1e-4f);
    const float cos_phi_receiver = dot(sample_normal, -direction);
    const float cos_phi_source = dot(sample_normal, -(from_source / distance_source));
    if (cos_phi_receiver <= 1e-4f || cos_phi_source <= 1e-4f)
    {
        return false;
    }
    jacobian = (cos_phi_receiver * distance_source * distance_source) / (cos_phi_source * distance_receiver * distance_receiver);
    const float limit = max(s.jacobian_max, 1.0f);
    if (!(jacobian > 1.0f / limit && jacobian < limit))
    {
        return false;
    }
    p_hat = rt_lab_luminance(r.radiance) * cos_receiver;
    return p_hat > 0.0f;
}

// Running RIS state of a pixel (temporal pass).
struct RtLab3Merge
{
    RtLab3Reservoir selected;
    float selected_p_hat;
    float weight_sum;
    float count_sum;
    float ray_history; // reprojected ray history of the reused reservoirs
};

// Merges one reprojected reservoir (found from source_position, confidence scaled by m_scale) into the pixel's RIS.
func rt_lab3_merge_temporal(inout RtLab3Merge m, RtLab3Reservoir previous, float3 source_position, float m_scale, float ray_history_cap, float3 position, float3 normal, float pixel_width_ws,
    RtLab3Settings s, RenderGlobalData* globals, MeshInstancesBufferHead* mesh_instances, daxa_u64 tlas)
{
    // Sample age cap: the reused sample is one frame older afterwards, it may not get older than the cap (its M is dropped with it).
    // The limit is drawn per reuse (cap x (1 - jitter x random)): a sample dies somewhere in [(1 - jitter) cap, cap], not all on the same frame.
    if (s.temporal_max_age > 0 && float(previous.age + 1u) > float(s.temporal_max_age) * (1.0f - saturate(s.temporal_max_age_jitter) * rand()))
    {
        return;
    }
    previous.count = min(previous.count, max(s.temporal_max_m, 0.0f)) * m_scale;
    if (!(previous.count > 0.0f) || !(previous.weight > 0.0f) || !isfinite(previous.weight))
    {
        return;
    }
    float p_hat;
    float jacobian;
    bool usable = rt_lab3_shift(previous, source_position, position, normal, s, p_hat, jacobian);
    if (usable && s.temporal_visibility != 0)
    {
        usable = rt_lab3_visible(globals, mesh_instances, tlas, position, normal, pixel_width_ws, previous.position);
    }
    if (!usable)
    {
        return;
    }
    const float reuse_weight = p_hat * jacobian * previous.weight * previous.count;
    m.count_sum += previous.count;
    if (reuse_weight > 0.0f && isfinite(reuse_weight))
    {
        m.weight_sum += reuse_weight;
        if (rand() * m.weight_sum < reuse_weight)
        {
            m.selected.position = previous.position;
            m.selected.normal = previous.normal;
            m.selected.age = previous.age + 1u;
            m.selected.radiance = previous.radiance;
            m.selected_p_hat = p_hat;
        }
    }
}

// Does the previous frame's pixel show the receiver's surface (inside the image, geometry, normal + plane test)? Its world position in last frame's camera.
func rt_lab3_temporal_tap(Texture2D<float> depth_history, Texture2D<uint> normal_history, CameraInfo prev_camera, int2 pixel, int2 size, float3 position, float3 normal,
    float rcp_pixel_width_ws, RtLab3Settings s, out float3 tap_position) -> bool
{
    tap_position = float3(0.0f, 0.0f, 0.0f);
    if (any(pixel < 0) || any(pixel >= size))
    {
        return false;
    }
    const float prev_depth = depth_history[pixel];
    if (prev_depth <= 0.0f)
    {
        return false;
    }
    const float3 prev_normal = uncompress_normal_octahedral_32(normal_history[pixel]);
    tap_position = linear_depth_to_world_space(prev_camera, rt_lab_ndc(uint2(pixel), rcp(float2(size))), prev_depth);
    return dot(normal, prev_normal) >= s.temporal_normal_threshold &&
           rt_lab_plane_weight(position, normal, tap_position, prev_normal, rcp_pixel_width_ws, s.temporal_plane_tolerance) > 0.0f;
}

// The footprint of a pixel in last frame: the 2x2 pixels around the exact (fractional) reprojected position that show the receiver's surface, bilinear weights.
struct RtLab3Footprint
{
    int2 tap_pixel[4];
    float3 tap_position[4];
    float tap_weight[4];
    float weight_total; // 0 = no history (temporal reuse off, reset, nothing found)
    float partial;      // footprint overhangs an edge (valid weight < 1): less confidence
};

func rt_lab3_temporal_footprint(RtLab3TemporalPush push, RtLab3Settings s, float3 position, float3 normal, float rcp_pixel_width_ws) -> RtLab3Footprint
{
    RtLab3Footprint fp;
    fp.weight_total = 0.0f;
    fp.partial = 1.0f;
    [ForceUnroll] for (uint t = 0u; t < 4u; ++t)
    {
        fp.tap_pixel[t] = int2(0, 0);
        fp.tap_position[t] = float3(0.0f, 0.0f, 0.0f);
        fp.tap_weight[t] = 0.0f;
    }
    if (s.temporal_enabled == 0 || s.reset != 0)
    {
        return fp;
    }
    const float2 size = float2(push.size);
    const CameraInfo prev_camera = push.attach.globals.view_camera_prev_frame;
    const float4 clip = mul(prev_camera.view_proj, float4(position, 1.0f));
    if (clip.w <= 0.0f)
    {
        return fp;
    }
    const float2 prev_uv = (clip.xy / clip.w) * 0.5f + 0.5f;
    const float2 prev_continuous = prev_uv * size - 0.5f;
    const int2 prev_base = int2(floor(prev_continuous));
    const float2 prev_frac = prev_continuous - floor(prev_continuous);
    [ForceUnroll] for (uint t = 0u; t < 4u; ++t)
    {
        fp.tap_pixel[t] = prev_base + int2(t & 1u, t >> 1u);
        if (rt_lab3_temporal_tap(push.attach.lab_depth_history.get(), push.attach.lab_face_normal_history.get(), prev_camera, fp.tap_pixel[t], int2(push.size),
                position, normal, rcp_pixel_width_ws, s, fp.tap_position[t]))
        {
            const float bilinear = ((t & 1u) != 0u ? prev_frac.x : 1.0f - prev_frac.x) * ((t >> 1u) != 0u ? prev_frac.y : 1.0f - prev_frac.y);
            fp.tap_weight[t] = bilinear;
            fp.weight_total += bilinear;
        }
    }
    if (fp.weight_total > 1e-5f)
    {
        fp.partial = lerp(1.0f, saturate(fp.weight_total), saturate(s.temporal_partial_scale));
    }
    return fp;
}

// Temporal reuse of one slot: coherent stochastic bilinear: ONE valid tap of the footprint picked by its bilinear weight with ONE random number per frame for the
// whole image (golden ratio sequence, shifted per slot). Neighbors pick the same tap, so a smooth shift moves the image by whole pixels and copies every old
// reservoir about once. Independent random numbers per pixel duplicate some reservoirs and drop others every frame, and the duplicates grow into clumps.
// The slot reads the slot (slot % slots of the history pixel) of the tap.
func rt_lab3_temporal_merge_slot(RtLab3TemporalPush push, RtLab3Footprint fp, RtLab3Settings s, uint slot, float3 position, float3 normal, float pixel_width_ws, float ray_history_cap,
    inout RtLab3Merge merge)
{
    if (!(fp.weight_total > 1e-5f))
    {
        return;
    }
    const float random = frac(float(uint(push.attach.globals.trunk_flt_frame_index)) * 0.61803398875f + 0.5f + float(slot) * 0.38196601125f);
    float pick = random * fp.weight_total;
    uint chosen = 0u;
    bool found = false;
    [ForceUnroll] for (uint t = 0u; t < 4u; ++t)
    {
        if (!found && fp.tap_weight[t] > 0.0f)
        {
            chosen = t;
            pick -= fp.tap_weight[t];
            found = pick <= 0.0f;
        }
    }
    const uint previous_slots = push.attach.slot_count_history.get()[fp.tap_pixel[chosen]];
    if (previous_slots == 0u)
    {
        return;
    }
    RtLab3Reservoir previous = rt_lab3_load(push.attach.history.get(), fp.tap_pixel[chosen], slot % previous_slots, fp.tap_position[chosen]);
    // The ray history (rays shot at this surface over time, slot 0) is carried whenever the history pixel shows the same surface, whether or not the
    // reservoir sample itself can be reused (age cap, empty reservoir, rejected shift): else those pixels look like disocclusions and ask for the maximum
    // rays all the time, and the real disocclusions get no more than the rest.
    if (slot == 0u)
    {
        merge.ray_history += min(previous.ray_history, ray_history_cap) * fp.partial;
    }
    rt_lab3_merge_temporal(merge, previous, fp.tap_position[chosen], fp.partial, ray_history_cap, position, normal, pixel_width_ws, s,
        push.attach.globals, push.attach.mesh_instances, push.attach.tlas);
}

// Temporal reuse of slot 0, shared with ReTIR (rt_lab_retir.hlsl: one reservoir per pixel).
func rt_lab3_temporal_reuse(RtLab3TemporalPush push, RtLab3Settings s, float3 position, float3 normal, float pixel_width_ws, float rcp_pixel_width_ws, float ray_history_cap, inout RtLab3Merge merge)
{
    const RtLab3Footprint fp = rt_lab3_temporal_footprint(push, s, position, normal, rcp_pixel_width_ws);
    rt_lab3_temporal_merge_slot(push, fp, s, 0u, position, normal, pixel_width_ws, ray_history_cap, merge);
}

// == Trace =============================================================================================================

[shader("compute")]
[numthreads(RT_LAB_TRACE_THREADS, 1, 1)]
func entry_rt_lab3_trace(uint ray_index : SV_DispatchThreadID)
{
    let push = rt_lab3_trace_push;
    let settings = push.attach.globals.rt_lab_settings;
    if (ray_index >= min(push.attach.ray_counters->ray_count, rt_lab_ray_budget(push.size, settings, push.ray_capacity)))
    {
        return;
    }
    const uint packed_pixel = push.attach.ray_list[ray_index];
    const uint2 dtid = uint2(packed_pixel & 0xFFFFu, packed_pixel >> 16u);
    const uint pixel_ray = ray_index - push.attach.ray_alloc.get()[dtid].x;
    const float depth = push.attach.lab_depth.get()[dtid];
    const float3 face_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);

    const uint ray_seed = rt_lab_ray_seed(settings, push.attach.globals.trunk_flt_frame_index, dtid, pixel_ray);
    rand_seed(ray_seed);
    const float3 tangent = normalize(cross(face_normal, float3(0, 0, 1) + 0.0001f));
    const float3x3 tbn = transpose(float3x3(tangent, cross(tangent, face_normal), face_normal));
    const float3 direction = uncompress_normal_octahedral_32(compress_normal_octahedral_32(mul(tbn, rand_cosine_sample_hemi())));

    float3 world_pos;
    float t_min;
    RayDesc ray = {};
    ray.Origin    = rt_lab_ray_launch(push.attach.globals.view_camera, dtid, push.size, depth, face_normal, world_pos, t_min);
    ray.Direction = direction;
    ray.TMin      = t_min;
    ray.TMax      = RT_LAB_RAY_T_MAX;

    let ctx = RTGI_SHADE_CONTEXT(push.attach);
    float hit_t;
    float3 hit_normal;
    const float3 color = rt_lab_trace_radiance_ex(ctx, settings, ray, rt_lab_hit_seed(ray_seed), true, hit_t, hit_normal);

    RtLab3Ray result;
    result.radiance = color * RTGI_RADIANCE_SCALE;
    result.hit_normal = compress_normal_octahedral_32(hit_normal);
    result.position = ray.Origin + direction * min(hit_t, RT_LAB3_SKY_DISTANCE);
    result.direction = compress_normal_octahedral_32(direction);
    push.attach.rays[ray_index] = result;
}

// == Temporal (initial candidates + temporal reuse) =====================================================================
// The pixel's rays are the candidates of a RIS (weight p_hat / p_source = pi luminance, the cosine cancels), the reprojected reservoir of last
// frame is merged in (weight p_hat_receiver(sample) J W M). Output = next frame's history.

[shader("compute")]
[numthreads(RT_LAB3_PASS_X, RT_LAB3_PASS_Y, 1)]
func entry_rt_lab3_temporal(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab3_temporal_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const RtLab3Settings s = *push.settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        push.attach.out_slot_count.get()[dtid] = 0u;
        return;
    }
    const float2 size = float2(push.size);
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(size)), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    const float pixel_width_ws = rt_lab_pixel_width_ws(camera, size.y, depth);
    const float rcp_pixel_width_ws = rcp(pixel_width_ws);

    // One reservoir per ray (at least one: a pixel without a ray keeps its history).
    const uint2 alloc = push.attach.ray_alloc.get()[dtid];
    const uint slots = clamp(alloc.y, 1u, RT_LAB3_MAX_SLOTS);
    push.attach.out_slot_count.get()[dtid] = slots;
    // Ray history: the rays shot over time at this pixel, capped at the target of the ray redistribution (so a converged pixel asks for no extras).
    // Kept in slot 0 only.
    const float ray_history_cap = max(push.attach.globals.rt_lab_settings.fast_convergence_samples, 1.0f);
    const RtLab3Footprint footprint = rt_lab3_temporal_footprint(push, s, position, normal, rcp_pixel_width_ws);

    for (uint slot = 0u; slot < slots; ++slot)
    {
        rand_seed(rt_lab3_seed(push.attach.globals, dtid, 1u + slot * 64u));

        // Candidates: the rays of the slot (ray k belongs to slot k % RT_LAB3_MAX_SLOTS).
        RtLab3Reservoir selected = rt_lab3_empty_reservoir();
        float selected_p_hat = 0.0f;
        float weight_sum = 0.0f;
        float count_sum = 0.0f;
        for (uint k = slot; k < alloc.y; k += RT_LAB3_MAX_SLOTS)
        {
            const RtLab3Ray ray = push.attach.rays[alloc.x + k];
            count_sum += 1.0f;
            if (any(!isfinite(ray.radiance)) || any(!isfinite(ray.position)))
            {
                continue;
            }
            const float luminance = rt_lab_luminance(ray.radiance);
            const float candidate_weight = PI * luminance; // p_hat / p_source = luminance cos / (cos / pi)
            if (!(candidate_weight > 0.0f))
            {
                continue;
            }
            weight_sum += candidate_weight;
            if (rand() * weight_sum < candidate_weight)
            {
                selected.position = ray.position;
                selected.normal = ray.hit_normal;
                selected.radiance = ray.radiance;
                selected_p_hat = luminance * max(dot(normal, uncompress_normal_octahedral_32(ray.direction)), 0.0f);
            }
        }

        RtLab3Merge merge;
        merge.selected = selected;
        merge.selected_p_hat = selected_p_hat;
        merge.weight_sum = weight_sum;
        merge.count_sum = count_sum;
        merge.ray_history = 0.0f;
        rt_lab3_temporal_merge_slot(push, footprint, s, slot, position, normal, pixel_width_ws, ray_history_cap, merge);
        selected = merge.selected;
        selected_p_hat = merge.selected_p_hat;
        weight_sum = merge.weight_sum;
        count_sum = merge.count_sum;

        selected.count = count_sum;
        selected.ray_history = slot == 0u ? min(merge.ray_history + float(alloc.y), ray_history_cap) : 0.0f;
        selected.weight = (selected_p_hat > 0.0f && count_sum > 0.0f) ? weight_sum / (count_sum * selected_p_hat) : 0.0f;
        selected.unvalidated = false;
        rt_lab3_store(push.attach.out_reservoir.get(), dtid, slot, selected, position);
    }
}

// == Spatial reuse =====================================================================================================

[shader("compute")]
[numthreads(RT_LAB3_PASS_X, RT_LAB3_PASS_Y, 1)]
func entry_rt_lab3_spatial(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab3_spatial_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const RtLab3Settings s = *push.settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const float depth = push.attach.lab_depth.get()[dtid];
    if (depth == 0.0f)
    {
        return;
    }
    const float2 size = float2(push.size);
    const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(size)), depth);
    const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
    const float pixel_width_ws = rt_lab_pixel_width_ws(camera, size.y, depth);
    const float rcp_pixel_width_ws = rcp(pixel_width_ws);
    const int taps = clamp(s.spatial_taps, 0, RT_LAB3_MAX_SPATIAL_TAPS);
    const uint slots = push.attach.slot_count.get()[dtid];

    // Every slot (= every ray of the pixel) runs its own reuse with its own taps.
    for (uint slot = 0u; slot < slots; ++slot)
    {
        rand_seed(rt_lab3_seed(push.attach.globals, dtid, 2u + push.pass_index + slot * 64u));

        // The slot's own reservoir (shift = identity).
        RtLab3Reservoir own = rt_lab3_load(push.attach.in_reservoir.get(), int2(dtid), slot, position);
        RtLab3Reservoir selected = own;
        float selected_p_hat = 0.0f;
        float weight_sum = 0.0f;
        float count_sum = own.count;
        {
            // No reconnection limits for the own sample (identity shift): a close hit would otherwise be dropped.
            const float3 to_own = own.position - position;
            const float own_p_hat = rt_lab_luminance(own.radiance) * max(dot(normal, to_own / max(length(to_own), 1e-4f)), 0.0f);
            if (own.count > 0.0f && own.weight > 0.0f && own_p_hat > 0.0f)
            {
                weight_sum = own_p_hat * own.weight * own.count;
                selected_p_hat = own_p_hat;
            }
        }

        for (int i = 0; i < taps; ++i)
        {
            // Uniform disc sample around the pixel.
            const float radius = sqrt(rand()) * max(s.spatial_radius, 1.0f);
            const float angle = rand() * 2.0f * PI;
            const int2 tap = int2(dtid) + int2(round(float2(cos(angle), sin(angle)) * radius));
            if (all(tap == int2(dtid)) || any(tap < 0) || any(tap >= int2(push.size)))
            {
                continue;
            }
            const float tap_depth = push.attach.lab_depth.get()[tap];
            if (tap_depth == 0.0f)
            {
                continue;
            }
            const float3 tap_normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[tap]);
            const float3 tap_position = linear_depth_to_world_space(camera, rt_lab_ndc(uint2(tap), rcp(size)), tap_depth);
            // Surface test: the normal test, and either the plane test or the plain world space distance (in pixel widths of the receiving pixel).
            const bool on_surface = s.spatial_surface_test == RT_LAB3_SURFACE_DISTANCE
                ? length(tap_position - position) * rcp_pixel_width_ws <= s.spatial_distance_factor
                : rt_lab_plane_weight(position, normal, tap_position, tap_normal, rcp_pixel_width_ws, s.spatial_plane_tolerance) > 0.0f;
            if (dot(normal, tap_normal) < s.spatial_normal_threshold || !on_surface)
            {
                continue;
            }
            // The neighbor's slot of the same index (slots of the neighbor wrap around).
            const uint tap_slots = push.attach.slot_count.get()[tap];
            if (tap_slots == 0u)
            {
                continue;
            }
            // Cheap rejection on the packed words before the full unpack.
            const uint4 neighbor_packed = push.attach.in_reservoir.get()[int3(tap, int(slot % tap_slots))];
            const float neighbor_count = min(rt_lab3_decode_count(neighbor_packed.y >> 24u), max(s.spatial_max_m, 0.0f));
            if (!(f16tof32(neighbor_packed.z >> 16u) > 0.0f) || !(neighbor_count > 0.0f))
            {
                continue;
            }
            RtLab3Reservoir neighbor = rt_lab3_unpack(neighbor_packed, tap_position);
            neighbor.count = neighbor_count;
            float p_hat;
            float jacobian;
            const bool usable = rt_lab3_shift(neighbor, tap_position, position, normal, s, p_hat, jacobian);
            if (!usable)
            {
                continue;
            }
            const float reuse_weight = p_hat * jacobian * neighbor.weight * neighbor.count;
            count_sum += neighbor.count;
            if (reuse_weight > 0.0f && isfinite(reuse_weight))
            {
                weight_sum += reuse_weight;
                if (rand() * weight_sum < reuse_weight)
                {
                    selected.position = neighbor.position;
                    selected.normal = neighbor.normal;
                    selected.age = neighbor.age;
                    selected.radiance = neighbor.radiance;
                    selected_p_hat = p_hat;
                    selected.unvalidated = true;
                }
            }
        }

        selected.count = count_sum;
        selected.weight = (selected_p_hat > 0.0f && count_sum > 0.0f) ? weight_sum / (count_sum * selected_p_hat) : 0.0f;
        rt_lab3_store(push.attach.out_reservoir.get(), dtid, slot, selected, position);
    }
}

// == Shade =============================================================================================================

[shader("compute")]
[numthreads(RT_LAB3_PASS_X, RT_LAB3_PASS_Y, 1)]
func entry_rt_lab3_shade(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab3_shade_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const RtLab3Settings s = *push.settings;
    const CameraInfo camera = push.attach.globals.view_camera;
    const float depth = push.attach.lab_depth.get()[dtid];
    float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (depth > 0.0f)
    {
        const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(float2(push.size))), depth);
        const float3 normal = uncompress_normal_octahedral_32(push.attach.lab_face_normal.get()[dtid]);
        const uint slots = max(push.attach.slot_count.get()[dtid], 1u);
        float3 estimate_sum = float3(0.0f, 0.0f, 0.0f);
        float ray_history = 0.0f;
        for (uint slot = 0u; slot < slots; ++slot)
        {
            const RtLab3Reservoir r = rt_lab3_load(push.attach.in_reservoir.get(), int2(dtid), slot, position);
            float3 estimate = float3(0.0f, 0.0f, 0.0f);
            // A sample from a neighbor: ONE visibility ray from this pixel (a blocked sample would leak light through thin geometry).
            const bool blocked = r.unvalidated && s.spatial_visibility != 0 && r.weight > 0.0f && r.count > 0.0f
                && !rt_lab3_visible(push.attach.globals, push.attach.mesh_instances, push.attach.tlas, position, normal,
                    rt_lab_pixel_width_ws(camera, float(push.size.y), depth), r.position);
            if (r.weight > 0.0f && r.count > 0.0f && !blocked)
            {
                const float3 direction = normalize(r.position - position);
                estimate = r.radiance * (max(dot(normal, direction), 0.0f) * r.weight / PI);
            }
            if (push.samples_bound != 0u && slot == 0u)
            {
                // The mean of the estimates after each of this frame's rays; a pixel without a ray this frame keeps the estimate of its reservoir.
                const float4 samples = push.attach.samples.get()[dtid];
                if (samples.a > 0.0f)
                {
                    estimate = samples.rgb;
                }
            }
            // The estimate of every slot (ray) goes to the ray filter.
            push.attach.slot_estimates.get()[int3(dtid, int(slot))] = float4(estimate * rcp(RTGI_RADIANCE_SCALE), 0.0f);
            estimate_sum += estimate;
            if (slot == 0u)
            {
                ray_history = r.ray_history;
            }
        }
        result = float4(estimate_sum / float(slots), ray_history); // .a = the pixel's accumulated rays shot (the ray redistribution's history)
    }
    push.attach.shaded.get()[dtid] = result;

    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RT_LAB3_SAMPLE_COUNT)
    {
        // Rays shot at the pixel over time, saturating at the target of the ray redistribution.
        const uint resolution_scale = max(push.attach.globals.settings.render_target_size.x / push.size.x, 1u);
        const float3 color = depth > 0.0f ? Heatmap(result.a / max(push.attach.globals.rt_lab_settings.fast_convergence_samples, 1.0f)) : float3(0.0f, 0.0f, 0.0f);
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
            float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend), resolution_scale);
    }
}

// == Ray filter (after the reuse) ======================================================================================
// Clamps the final estimate of a pixel against the estimates of the neighbor pixels (pixel distance only, the pixel itself excluded), hue
// preserving: one ratio for all channels, from the worst channel. Same ceilings as the old lab's ray filter.

// The quad trick of the RTGI prefilter. The neighborhood statistics are per 2x2 QUAD of pixels: the group loads its quads + a halo of quads once (every texel read
// and logged once), each quad stores the SUM of its valid pixels' values and their count, ONE thread per quad builds the ceiling from the quads around it (the
// quad itself excluded, so a firefly never raises its own ceiling) and the 4 pixels of the quad share it. The radius of the ray filter counts quads here:
// radius 2 = 5x5 quads = 10x10 pixels for 24 taps (instead of 24 taps over 5x5 pixels).
#define RT_LAB3_FILTER_HALO RT_LAB_RAY_FILTER_MAX_RADIUS // in quads
#define RT_LAB3_FILTER_QUADS (RT_LAB3_PASS_X / 2)        // quads per group and axis
#define RT_LAB3_FILTER_TILE (RT_LAB3_FILTER_QUADS + 2 * RT_LAB3_FILTER_HALO)
groupshared float4 gs_rt_lab3_filter[RT_LAB3_FILTER_TILE * RT_LAB3_FILTER_TILE];       // per quad: .rgb sum of the valid pixels' values (log radiance in perceptual mode), .a their count
groupshared float4 gs_rt_lab3_ceiling[RT_LAB3_FILTER_QUADS * RT_LAB3_FILTER_QUADS];     // per quad of the group: .rgb ceiling, .a 1 = valid

[shader("compute")]
[numthreads(RT_LAB3_PASS_X, RT_LAB3_PASS_Y, 1)]
func entry_rt_lab3_filter(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID, uint gindex : SV_GroupIndex)
{
    let push = rt_lab3_filter_push;
    // The preset's ray filter (Trace Settings), applied after the reuse: on the per pixel estimate instead of per ray.
    const RtLabSettings filter = push.attach.globals.rt_lab_settings;
    const bool filter_on = filter.ray_filter_mode != RT_LAB_RAY_FILTER_OFF;
    const bool perceptual = filter.ray_filter_mode == RT_LAB_RAY_FILTER_PERCEPTUAL;
    // RTGI perceptual floor (calc_perceptual_radiance_floor): keeps log() of dark values bounded.
    const float log_floor = RTGI_RADIANCE_SCALE * 1e-3f / max(push.attach.globals.exposure, 1e-12f);
    const int radius = clamp(filter.ray_filter_radius, 0, RT_LAB_RAY_FILTER_MAX_RADIUS);
    if (filter_on)
    {
        // Every thread takes part in the loads (also the ones past the image edge), the barriers are group uniform.
        const int2 tile_quad_origin = int2(gid) * RT_LAB3_FILTER_QUADS - RT_LAB3_FILTER_HALO;
        for (uint i = gindex; i < RT_LAB3_FILTER_TILE * RT_LAB3_FILTER_TILE; i += RT_LAB3_PASS_X * RT_LAB3_PASS_Y)
        {
            const int2 quad = tile_quad_origin + int2(i % RT_LAB3_FILTER_TILE, i / RT_LAB3_FILTER_TILE);
            float4 entry = float4(0.0f, 0.0f, 0.0f, 0.0f);
            [unroll] for (int q = 0; q < 4; ++q)
            {
                const int2 texel = quad * 2 + int2(q & 1, q >> 1);
                if (all(texel >= 0) && all(texel < int2(push.size)) && push.attach.lab_depth.get()[texel] > 0.0f)
                {
                    const float3 radiance = push.attach.shaded.get()[texel].rgb;
                    entry += float4(perceptual ? log(max(radiance, log_floor)) : radiance, 1.0f);
                }
            }
            gs_rt_lab3_filter[i] = entry;
        }
        GroupMemoryBarrierWithGroupSync();
        // One thread per quad builds the ceiling of the quad from the quads around it (not the quad itself).
        if (((gtid.x | gtid.y) & 1u) == 0u)
        {
            const int2 quad_local = int2(gtid >> 1u);
            float3 sum = float3(0.0f, 0.0f, 0.0f);
            float count = 0.0f;
            for (int y = -radius; y <= radius; ++y)
            {
                for (int x = -radius; x <= radius; ++x)
                {
                    if (x == 0 && y == 0)
                    {
                        continue;
                    }
                    const float4 entry = gs_rt_lab3_filter[(quad_local.y + RT_LAB3_FILTER_HALO + y) * RT_LAB3_FILTER_TILE + quad_local.x + RT_LAB3_FILTER_HALO + x];
                    sum += entry.rgb;
                    count += entry.a;
                }
            }
            float4 quad_ceiling = float4(1e30f, 1e30f, 1e30f, 0.0f);
            if (count > 0.0f)
            {
                quad_ceiling = float4(perceptual
                    ? exp(sum / count + max(filter.ray_filter_tolerance, 0.0f))
                    : max(sum / count, log_floor) * max(filter.ray_filter_factor, 1.0f), 1.0f);
            }
            gs_rt_lab3_ceiling[quad_local.y * RT_LAB3_FILTER_QUADS + quad_local.x] = quad_ceiling;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (any(dtid >= push.size))
    {
        return;
    }
    const float4 value = push.attach.shaded.get()[dtid];
    float3 result = value.rgb;
    float3 ceiling = float3(1e30f, 1e30f, 1e30f); // the neighborhood ceiling, also applied to the fresh rays of the cell input
    if (filter_on && push.attach.lab_depth.get()[dtid] > 0.0f)
    {
        const int2 quad_local = int2(gtid >> 1u);
        ceiling = gs_rt_lab3_ceiling[quad_local.y * RT_LAB3_FILTER_QUADS + quad_local.x].rgb;
    }
    // The ray filter per ray: every slot's estimate is clamped against the ceiling (hue preserving), then they are averaged.
    const uint slots = push.attach.slot_count.get()[dtid];
    if (slots > 0u)
    {
        float3 estimate_sum = float3(0.0f, 0.0f, 0.0f);
        for (uint slot = 0u; slot < slots; ++slot)
        {
            float3 estimate = push.attach.slot_estimates.get()[int3(dtid, int(slot))].rgb * RTGI_RADIANCE_SCALE;
            if (any(!isfinite(estimate)))
            {
                continue;
            }
            const float3 over = estimate / max(ceiling, 1e-12f);
            estimate *= 1.0f / max(max(over.x, max(over.y, over.z)), 1.0f);
            estimate_sum += estimate;
        }
        result = estimate_sum / float(slots);
    }
    push.attach.filtered.get()[dtid] = float4(result, value.a);
    if (push.cell_input_bound != 0u)
    {
        float weight = 0.0f;
        float3 cell_color = result;
        if (push.attach.lab_depth.get()[dtid] > 0.0f)
        {
            if (push.cell_source == RT_LAB3_CELL_SOURCE_RAYS)
            {
                // The pixel's fresh rays only (independent samples), each clamped against the ceiling of the neighbors' estimates.
                const uint2 alloc = push.attach.ray_alloc.get()[dtid];
                float3 ray_sum = float3(0.0f, 0.0f, 0.0f);
                float ray_weight = 0.0f;
                for (uint k = 0u; k < alloc.y; ++k)
                {
                    float3 radiance = push.attach.rays[alloc.x + k].radiance;
                    if (any(!isfinite(radiance)))
                    {
                        continue;
                    }
                    const float3 over = radiance / max(ceiling, 1e-12f);
                    radiance *= 1.0f / max(max(over.x, max(over.y, over.z)), 1.0f);
                    ray_sum += radiance;
                    ray_weight += 1.0f;
                }
                weight = ray_weight;
                cell_color = ray_weight > 0.0f ? ray_sum / ray_weight : float3(0.0f, 0.0f, 0.0f);
            }
            else if (push.cell_weight_mode == RT_LAB3_CELL_WEIGHT_RAYS)
            {
                weight = float(push.attach.ray_alloc.get()[dtid].y);
            }
            else if (push.cell_weight_mode == RT_LAB3_CELL_WEIGHT_M)
            {
                weight = value.a;
            }
            else if (push.sample_weights != 0u)
            {
                // Temporal sample processing: the estimate stands for the pixel's fresh ray count of samples.
                weight = float(push.attach.ray_alloc.get()[dtid].y);
            }
            else
            {
                // One sample per ray (slot): the denoiser gets the number of independent estimates the mean stands for.
                weight = value.a > 0.0f ? float(slots) : 0.0f;
            }
        }
        push.attach.cell_input.get()[dtid] = float4(cell_color, weight);
    }
}
