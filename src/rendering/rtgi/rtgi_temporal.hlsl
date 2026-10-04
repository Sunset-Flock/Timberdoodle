#pragma once

#include "rtgi_temporal.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"
#include "shader_lib/brdf.hlsl"
#include "shader_lib/debug.glsl"

[[vk::push_constant]] RtgiTemporalReprojectPush rtgi_temporal_reproject_push;
[[vk::push_constant]] RtgiTemporalAccumulatePush rtgi_temporal_accumulate_push;

float apply_bilinear_custom_weights_soft_normalize( float s00, float s10, float s01, float s11, float4 w )
{
    float max_v = 0.0f;
    max_v = max(max_v, w.x > 0.1f ? s00 : 0.0f);
    max_v = max(max_v, w.y > 0.1f ? s10 : 0.0f);
    max_v = max(max_v, w.z > 0.1f ? s01 : 0.0f);
    max_v = max(max_v, w.w > 0.1f ? s11 : 0.0f);
    const float max_v_clamp_4 = min(max_v, 4.0f);
    const float v_acc = s00 * w.x + s10 * w.y + s01 * w.z + s11 * w.w;
    const float weight_sum = dot( w, 1.0f );
    // A larger exponent causes MORE normalization -> more streak artifacting.
    // A smaller exponent causes LESS normalization -> more temporal instability.
    const float SOFT_NORMALIZE_EXPONENT = 0.66f;
    const float soft_normalized_weight_sum = pow( weight_sum, SOFT_NORMALIZE_EXPONENT);
    return max(max_v_clamp_4, v_acc * rcp(soft_normalized_weight_sum));
    // return v_acc;
}

// The accumulation pass re-reads history by gathering the 2x2 block whose top-left texel is
// bilinear.origin. We store (origin + 1) because origin can be -1 at the screen edge (uv==0),
// and origin + 1 is always >= 0, so it fits an unsigned u16x2.
uint2 rtgi_reproject_corner(int2 origin)
{
    return uint2(origin + 1); // origin in [-1, size-1] -> [0, size]
}

// Parallax stretch penalty, [0,1]. A surface seen at a grazing angle covers very few pixels; when camera
// motion makes it much less grazing — e.g. a wall revealed by moving sideways, going from a 1-pixel strip
// covering 10 meters to a 10-pixel-wide wall — its thin previous-frame history is reprojected/stretched
// across all the new pixels, smearing that one accumulated strip along the whole surface.
//
// We detect this from the change in surface FORESHORTENING between the two camera positions: foreshortening
// f = |dot(view_dir, normal)| is ~0 at extreme grazing and grows as the surface faces us more. The screen
// footprint of a fixed surface patch scales with f, so f_cur / f_prev is how much wider (in pixels) the
// patch became this frame — i.e. the stretch factor. We only penalize growth (>1); the surface becoming
// MORE grazing is a harmless compression.
//
// Only the ULTRA-strong cases are penalized: there is a deadzone below a ~4x stretch (2 "stops") so the
// mild footprint changes of ordinary motion are left completely untouched; past that the penalty ramps up
// with strength. This uses only the camera translation (rotation leaves camera position, hence both
// foreshortenings, unchanged) and the current surface point/normal, so no extra fetches.
func calc_parallax_penalty(float3 world_pos, float3 normal_ws, float3 cam_pos, float3 cam_pos_prev, float strength) -> float
{
    const float3 view_cur  = normalize(world_pos - cam_pos);
    const float3 view_prev = normalize(world_pos - cam_pos_prev);
    const float graze_cur  = max(abs(dot(view_cur,  normal_ws)), 1e-3f); // foreshortening now
    const float graze_prev = max(abs(dot(view_prev, normal_ws)), 1e-3f); // foreshortening last frame
    const float stretch    = graze_cur / graze_prev;                      // >1 == thin strip stretched wide
    // No penalty until the footprint has grown ~2x (1 stop), then ramp with strength.
    const float STRETCH_DEADZONE_STOPS = 1.0f; // log2(2)
    return saturate((log2(max(stretch, 1.0f)) - STRETCH_DEADZONE_STOPS) * strength);
}

// === Temporal Reprojection =================================================================
// Determines where this pixel's history lives (prev-frame bilinear footprint) and how valid it is.
// Outputs only the addressing + weights + final sample count; it does NOT touch color/statistics
// history. This lets the accumulation pass — and future pre-trace consumers (reprojected radiance,
// per-pixel ray budget / redistribution) — read history cheaply.
groupshared uint gs_tile_total_desired; // sum of desired_total (1+extra) per non-sky pixel in tile
groupshared uint gs_tile_geo_count;    // number of non-sky pixels in tile
groupshared uint gs_tile_desired_specular; // specular part of gs_tile_total_desired (statistics)
groupshared uint gs_tile_specular_base;    // specular part of gs_tile_geo_count (statistics)

[shader("compute")]
[numthreads(RTGI_TEMPORAL_X,RTGI_TEMPORAL_Y,1)]
func entry_temporal_reproject(uint2 dtid : SV_DispatchThreadID, uint2 gtid : SV_GroupThreadID, uint2 gid : SV_GroupID)
{
    let push = rtgi_temporal_reproject_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;

    if (gtid.x == 0 && gtid.y == 0)
    {
        gs_tile_total_desired = 0u;
        gs_tile_geo_count = 0u;
        gs_tile_desired_specular = 0u;
        gs_tile_specular_base = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    // Per-thread ray-demand contribution to this tile. Out-of-bounds and sky pixels contribute 0 and
    // are excluded from the geometry count. ALL threads MUST reach the barrier further below, so this
    // uses structured control flow instead of early returns — early returns would deadlock the barrier
    // on any tile that mixes sky / geometry / out-of-bounds pixels (i.e. nearly every silhouette tile).
    uint thread_desired_total = 0u;
    uint thread_desired_specular = 0u;
    uint thread_geo_inc = 0u;

    const uint2 halfres_pixel_index = dtid;
    if (!any(dtid.xy >= push.size))
    {
    // Load and precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);

    const PixelData pixel = calc_pixel_data(dtid, inv_half_res_render_target_size, camera, push.attach.half_res_depth.get(), push.attach.half_res_normal.get());
    const float pixel_width_ws = rtgi_half_res_pixel_width_ws(inv_half_res_render_target_size, camera.near_plane, pixel.depth_vs);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);

    if (pixel.depth_vs == 0.0f)
    {
        // Sky: no valid history. A negative sample count is a sentinel that lets the accumulation
        // pass early-out on sky by reading only the sample count image (no separate depth fetch).
        // corner/weights are left unwritten since accumulation returns before reading them.
        push.attach.half_res_sample_count.get()[halfres_pixel_index] = rtgi_pack_sample_counts_sky();
        push.attach.specular_sample_count.get()[halfres_pixel_index] = 0.0f;
        push.attach.ray_impact.get()[halfres_pixel_index] = float2(1.0f, 1.0f);
    }
    else
    {

    // Load relevant global data
    CameraInfo* previous_camera = &push.attach.globals->view_camera_prev_frame;

    const float3 expected_world_position_prev_frame = pixel.position_ws;
    const float4 ndc_prev_frame_pre_div = mul(previous_camera.view_proj, float4(expected_world_position_prev_frame, 1.0f));
    const float3 ndc_prev_frame = ndc_prev_frame_pre_div.xyz / ndc_prev_frame_pre_div.w;
    const float2 uv_prev_frame = ndc_prev_frame.xy * 0.5f + 0.5f;

    // Load previous frame half res depth
    const Bilinear bilinear_filter_at_prev_pos = get_bilinear_filter( saturate( uv_prev_frame ), half_res_render_target_size );
    const float2 reproject_gather_uv = ( float2( bilinear_filter_at_prev_pos.origin ) + 1.0 ) * inv_half_res_render_target_size;
    SamplerState linear_clamp_s = push.attach.globals.samplers.linear_clamp.get();
    const float4 depth_reprojected4 = push.attach.half_res_depth_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
    const uint4 face_normals_packed_reprojected4 = push.attach.half_res_normal_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
    const uint4 samplecnt_packed4 = push.attach.half_res_sample_count_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
    const float4 samplecnt_reprojected4 = float4(
        rtgi_unpack_normal_count(samplecnt_packed4.x),
        rtgi_unpack_normal_count(samplecnt_packed4.y),
        rtgi_unpack_normal_count(samplecnt_packed4.z),
        rtgi_unpack_normal_count(samplecnt_packed4.w));

    // Calculate plane distance based occlusion and normal similarity
    float4 occlusion = float4(1.0f, 1.0f, 1.0f, 1.0f);
    float4 normal_similarity = float4(1.0f, 1.0f, 1.0f, 1.0f);
    {
        const float in_screen = all(uv_prev_frame > 0.0f && uv_prev_frame < 1.0f) ? 1.0f : 0.0f;
        const float3 other_face_normals[] = {
            uncompress_normal_octahedral_32(face_normals_packed_reprojected4.x),
            uncompress_normal_octahedral_32(face_normals_packed_reprojected4.y),
            uncompress_normal_octahedral_32(face_normals_packed_reprojected4.z),
            uncompress_normal_octahedral_32(face_normals_packed_reprojected4.w),
        };
        // Note: hard normal weights (dot(other, pixel.normal) > -0.3) cause too much
        // dis-occlusion on fine detailed geometry, so only the soft normal_similarity is used.
        normal_similarity = {
            calc_similar_normal_weight(other_face_normals[0], pixel.normal_ws),
            calc_similar_normal_weight(other_face_normals[1], pixel.normal_ws),
            calc_similar_normal_weight(other_face_normals[2], pixel.normal_ws),
            calc_similar_normal_weight(other_face_normals[3], pixel.normal_ws),
        };

        // high quality geometric weights
        float4 surface_weights = float4( 0.0f, 0.0f, 0.0f, 0.0f );
        {
            const float2 texel_ndc_prev_frame[4] = {
                float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(0,0)) * inv_half_res_render_target_size * 2.0f - 1.0f,
                float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(1,0)) * inv_half_res_render_target_size * 2.0f - 1.0f,
                float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(0,1)) * inv_half_res_render_target_size * 2.0f - 1.0f,
                float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(1,1)) * inv_half_res_render_target_size * 2.0f - 1.0f,
            };
            const float3 texel_ws_prev_frame[4] = {
                rtgi_half_res_depth_to_world_space(*previous_camera, texel_ndc_prev_frame[0], depth_reprojected4[0]),
                rtgi_half_res_depth_to_world_space(*previous_camera, texel_ndc_prev_frame[1], depth_reprojected4[1]),
                rtgi_half_res_depth_to_world_space(*previous_camera, texel_ndc_prev_frame[2], depth_reprojected4[2]),
                rtgi_half_res_depth_to_world_space(*previous_camera, texel_ndc_prev_frame[3], depth_reprojected4[3]),
            };
            surface_weights = {
                calc_similar_surface_weight_dist_limited(pixel_width_ws_rcp, expected_world_position_prev_frame, pixel.normal_ws, texel_ws_prev_frame[0], other_face_normals[0], 2),
                calc_similar_surface_weight_dist_limited(pixel_width_ws_rcp, expected_world_position_prev_frame, pixel.normal_ws, texel_ws_prev_frame[1], other_face_normals[1], 2),
                calc_similar_surface_weight_dist_limited(pixel_width_ws_rcp, expected_world_position_prev_frame, pixel.normal_ws, texel_ws_prev_frame[2], other_face_normals[2], 2),
                calc_similar_surface_weight_dist_limited(pixel_width_ws_rcp, expected_world_position_prev_frame, pixel.normal_ws, texel_ws_prev_frame[3], other_face_normals[3], 2),
            };
            surface_weights[0] *= depth_reprojected4[0] != 0.0f;
            surface_weights[1] *= depth_reprojected4[1] != 0.0f;
            surface_weights[2] *= depth_reprojected4[2] != 0.0f;
            surface_weights[3] *= depth_reprojected4[3] != 0.0f;
        }

        occlusion = surface_weights * in_screen;
    }

    const float4 sample_weights = get_bilinear_custom_weights( bilinear_filter_at_prev_pos, occlusion * normal_similarity );

    // For good quality reprojection we need multiple prev frame samples to properly avoid unwanted ghosting etc.
    // But for thin geometry its very hard or impossible to get 4 valid prev frame samples.
    // So we count the neighborhood pixels and scale the disocclusion threshold based on how easy it is to reproject.
    // So easy to reproject pixels have tight disocclusion, while thin things are allowed to have blurry ghosty reprojection.
    const float disocclusion_threshold = 0.025f;
    const float total_sample_weights = dot(1.0f, sample_weights);
    const bool disocclusion = total_sample_weights < disocclusion_threshold;

    // Calc new sample count
    // MUST NOT NORMALIZE SAMPLECOUNT
    // WHEN SAMPLECOUNT IS NORMALIZED, PARTIAL DISOCCLUSIONS WILL GET FULL SAMPLECOUNT FROM THE VALID SAMPLES
    // THIS CAUSES THE PARTIALLY DISOCCLUDED SAMPLES TO IMMEDIATELY TAKE ON A FULL SAMPLECOUNT
    // THEY GET STUCK IN THEIR FIRST FRAME HISTORY IMMEDIATELY
    //
    // Reasoning behind the soft normalized sample count:
    // * to be correct, one has to not normalize the sample counter
    // * this however causes sample counters to be perpetually low on thin moving things because the temporal pass runs at half resolution :(.
    // * simply normalizing the sample count is also very bad, it causes "streaking" and crawling color on slow moving disocclusions :(
    // * as a compromise a mix of both is used, on partial disocclusion, the samplecount is partially normalized
    //   * this still causes the "streaking" artifacts but MUCH less so :)
    //   * its good enough to very significantly increase temporal stability :)
    //   * the streaking it causes is nearly completely hidden by the post blur :)
    float reprojected_sample_count = apply_bilinear_custom_weights_soft_normalize( samplecnt_reprojected4.x, samplecnt_reprojected4.y, samplecnt_reprojected4.z, samplecnt_reprojected4.w, sample_weights );
    if (any(isnan(reprojected_sample_count)))
    {
        reprojected_sample_count = {};
    }
    // write_debug_image(push.attach.debug_image.get(), 0, dtid, float4(Heatmap(float(reprojected_sample_count) * rcp(64)), 2.0f), 2);

    // Carry the reprojected history count forward: the trace pass increments it by the number of rays
    // it shoots (and uses it to drive adaptive ray count). Disocclusion -> 0, so the trace pass starts
    // fresh (adding the rays it bursts that frame).
    float reprojected_history_count = disocclusion ? 0.0f : min(rtgi_history_count_cap(rtgi_settings.max_temporal_samples), reprojected_sample_count);

    // Parallax stretch: drop history where a grazing surface's thin previous-frame strip is being
    // stretched across many new pixels this frame (e.g. a wall revealed by lateral motion), so those
    // pixels re-converge with fresh samples instead of smearing. Only the ultra-strong (>~4x) cases bite.
    float parallax_penalty = 0.0f;
    if (rtgi_settings.temporal_parallax_penalty_strength > 0.0f)
    {
        parallax_penalty = calc_parallax_penalty(
            pixel.position_ws, pixel.normal_ws, camera.position, previous_camera.position,
            rtgi_settings.temporal_parallax_penalty_strength);
        // An active penalty first drops the history to at most the ray demand target (fast_convergence_samples),
        // THEN scales it: otherwise a long history above the target stays above it after the cut and the
        // stretched pixel requests no extra rays at all.
        if (parallax_penalty > 0.0f)
        {
            reprojected_history_count = min(reprojected_history_count, rtgi_settings.fast_convergence_samples) * (1.0f - parallax_penalty);
        }
    }

    // Write reprojection metadata consumed by the trace pass and entry_temporal_accumulate. The normal
    // field carries the (penalized) history count. The fast field is unused by the trace/allocate passes
    // (they only read the normal field), so we borrow it to hand the parallax penalty to the accumulate
    // pass — which reprojects the fast-history frame count and applies the SAME penalty to it — without
    // recomputing geometry there. Penalty [0,1] is scaled to the fast field's [0,15] range for precision
    // and divided back in accumulate; accumulate then overwrites this texel with the real fast count.
    push.attach.half_res_sample_count.get()[halfres_pixel_index] = rtgi_pack_sample_counts(reprojected_history_count, parallax_penalty * 15.0f);
    push.attach.reproject_corner.get()[halfres_pixel_index] = rtgi_reproject_corner(bilinear_filter_at_prev_pos.origin);
    push.attach.reproject_weights.get()[halfres_pixel_index] = sample_weights;

    // Specular history samples along the surface motion (the virtual motion is only known in accumulate;
    // the surface footprint is a good enough predictor of how much specular history survives).
    float reprojected_specular_count = 0.0f;
    if (!disocclusion)
    {
        const float4 sf4 = push.attach.specular_frames_history.get().GatherRed(linear_clamp_s, reproject_gather_uv).wzxy;
        reprojected_specular_count = apply_bilinear_custom_weights(sf4.x, sf4.y, sf4.z, sf4.w, sample_weights);
        if (isnan(reprojected_specular_count)) { reprojected_specular_count = 0.0f; }
        // Own strength (specular_temporal_parallax_penalty_strength); same stretch measure as the diffuse penalty.
        const float specular_parallax_penalty = rtgi_settings.specular_temporal_parallax_penalty_strength > 0.0f
            ? calc_parallax_penalty(pixel.position_ws, pixel.normal_ws, camera.position, previous_camera.position,
                                    rtgi_settings.specular_temporal_parallax_penalty_strength)
            : 0.0f;
        // Same as diffuse: clamp to the specular ray demand target first, then scale.
        if (specular_parallax_penalty > 0.0f)
        {
            const float specular_target = min(rtgi_settings.specular_fast_convergence_samples, rtgi_settings.specular_max_temporal_frames);
            reprojected_specular_count = min(reprojected_specular_count, specular_target) * (1.0f - specular_parallax_penalty);
        }
    }
    push.attach.specular_sample_count.get()[halfres_pixel_index] = reprojected_specular_count;

    // This geometry pixel wants 1 base ray + adaptive extras PER SIGNAL (shared with the allocate pass), the extras
    // split by the diffuse / specular share (rtgi_calc_ray_share). Incoming luma: last frame's accumulated history at the NEAREST texel
    // (deliberately crude: only "was geometry" is checked). Total disocclusions (incl. pixels revealed at the screen
    // edge) read NOTHING: the nearest texel there belongs to the occluder / another surface, so the brightness
    // split would run on wrong data. Without history the share falls back to the material reflectances.
    float2 reprojected_luma = float2(0.0f, 0.0f);
    if (!disocclusion)
    {
        const uint2 nearest = min(uint2(saturate(uv_prev_frame) * half_res_render_target_size), uint2(half_res_render_target_size) - 1u);
        if (push.attach.half_res_depth_history.get()[nearest] != 0.0f)
        {
            const float4 prev_spec = push.attach.half_res_specular_history.get()[nearest];
            reprojected_luma = float2(
                push.attach.half_res_diffuse_history.get()[nearest].w,
                dot(prev_spec.rgb, float3(0.25f, 0.5f, 0.25f)));
            reprojected_luma = select(isnan(reprojected_luma), float2(0.0f, 0.0f), max(reprojected_luma, 0.0f));
        }
    }
    const uint packed_normal_roughness = push.attach.half_res_normal_roughness.get()[halfres_pixel_index];
    const float NoV = saturate(dot(unpack_normal_roughness_normal(packed_normal_roughness), normalize(camera.position - pixel.position_ws)));
    RtgiRayImpact impact = rtgi_calc_ray_share(reprojected_luma, push.attach.half_res_albedo_metalness.get()[halfres_pixel_index],
        unpack_normal_roughness_roughness(packed_normal_roughness), NoV, push.attach.globals.inv_exposure, rtgi_settings.specular_enabled != 0, rtgi_settings.ray_share_slope);
    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_RAY_SHARE)
    {
        // Specular share of the pixel's extra-ray weight: 0 = all diffuse (cold), 0.5 = even, 1 = all specular (hot).
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, halfres_pixel_index,
            float4(Heatmap(0.5f * impact.specular), 1.0f + push.attach.globals.settings.debug_visualization_blend), 2);
    }
    // Material: full metal needs no diffuse extras, very rough specular needs no specular extras (base rays stay).
    const float2 material_factors = rtgi_calc_ray_material_factors(push.attach.half_res_albedo_metalness.get()[halfres_pixel_index],
        unpack_normal_roughness_roughness(packed_normal_roughness), NoV, rtgi_settings);
    impact.diffuse *= material_factors.x;
    impact.specular *= material_factors.y;
    if (push.attach.globals.settings.debug_draw_mode == DEBUG_DRAW_MODE_RTGI_RAY_MATERIAL)
    {
        // Red = diffuse material factor, green = specular material factor (yellow = both 1).
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, halfres_pixel_index,
            float4(material_factors.x, material_factors.y, 0.0f, 1.0f + push.attach.globals.settings.debug_visualization_blend), 2);
    }
    push.attach.ray_impact.get()[halfres_pixel_index] = float2(impact.diffuse, impact.specular);
    const RtgiRayDemand demand = rtgi_calc_ray_demand(reprojected_history_count, reprojected_specular_count, impact, rtgi_settings);
    thread_desired_total = demand.diffuse + demand.specular;
    thread_desired_specular = demand.specular;
    thread_geo_inc = rtgi_ray_demand_base(demand); // base rays (1 diffuse + 1 specular)
    } // end else (geometry pixel)
    } // end if (in bounds)

    // Accumulate per-tile ray demand into groupshared, then thread 0 flushes to the buffers.
    // ALL threads reach this barrier (sky / oob contributed 0 above), so it is uniform.
    InterlockedAdd(gs_tile_total_desired, thread_desired_total);
    InterlockedAdd(gs_tile_geo_count, thread_geo_inc);
    InterlockedAdd(gs_tile_desired_specular, thread_desired_specular);
    InterlockedAdd(gs_tile_specular_base, thread_desired_specular > 0u ? 1u : 0u);

    GroupMemoryBarrierWithGroupSync();

    if (gtid.x == 0 && gtid.y == 0)
    {
        // total_geo_rays = base rays (one per signal per geometry pixel), total_extra_rays = everything above.
        const uint tile_extra = gs_tile_total_desired - gs_tile_geo_count;
        if (tile_extra > 0u) InterlockedAdd(push.attach.ray_counters->total_extra_rays, tile_extra);
        if (gs_tile_geo_count > 0u) InterlockedAdd(push.attach.ray_counters->total_geo_rays, gs_tile_geo_count);
        if (gs_tile_total_desired > 0u)
        {
            // Statistics: the base request is only min_ray_budget per signal per pixel (the distribute pass
            // guarantees that much base coverage, more only from leftover budget), extras in full.
            const float min_budget = clamp(rtgi_settings.min_ray_budget, 0.0f, 1.0f);
            const uint specular_base = gs_tile_specular_base;
            const uint diffuse_base = gs_tile_geo_count - specular_base;
            const uint diffuse_desired = gs_tile_total_desired - gs_tile_desired_specular;
            InterlockedAdd(push.attach.ray_counters->requested_diffuse_rays, (diffuse_desired - diffuse_base) + uint(float(diffuse_base) * min_budget + 0.5f));
            InterlockedAdd(push.attach.ray_counters->requested_specular_rays, (gs_tile_desired_specular - specular_base) + uint(float(specular_base) * min_budget + 0.5f));
        }
    }
}

// == Ray demand: diffuse / specular share ======================================
// Per pixel, the extra rays are split between diffuse and specular by which of the two is VISIBLY brighter:
//   diffuse  = incoming diffuse luma  x luma(albedo * (1 - metalness))            (Lambert: avg radiance x albedo)
//   specular = incoming specular luma x luma(env BRDF(F0, roughness, NoV)),  F0 = lerp(0.04, albedo, metalness)
// Incoming luma = last frame's accumulated history at the NEAREST reprojected texel (deliberately crude). A signal
// without history (0: disocclusion / off screen / was sky) borrows the other signal's incoming luma, or both use
// the same when neither has any, so only the material decides.
// The two final radiances are compared in PERCEPTUAL (log) space (linear_to_perceptual, incl. its floor), in stops:
//   stops    = log2(specular / diffuse)
//   specular_share = 1 / (1 + 2^(-slope * stops)),   diffuse_share = 1 - specular_share
// slope = rtgi_settings.ray_share_slope (live). slope 2 == squared radiance ratio: 1 / (1 + (diffuse / specular)^2).
//   factor   = 2 * share   (equal radiance -> 1x each; the pixel's total extra-ray weight stays 2)
// So a pixel whose specular is 1 stop brighter than its diffuse gives specular 2/3 and diffuse 1/3 of the weight at
// slope 1, 4/5 and 1/5 at slope 2. The deficit spans 0..32 extras while the share only scales 0..2x, so a steep
// slope is needed for the radiance comparison to matter against the history deficit. The factors scale each signal's deficit-driven extras (rtgi_calc_ray_demand); base rays stay.
// Specular disabled -> diffuse factor 1. #define 0 = both 1.
#define RTGI_RAY_SHARE 1

func rtgi_calc_ray_share(float2 incoming_luma, float4 albedo_metalness, float roughness, float NoV, float inv_exposure, bool specular_enabled, float slope) -> RtgiRayImpact
{
    RtgiRayImpact impact;
    impact.diffuse = 1.0f;
    impact.specular = 1.0f;
    const float3 Y = float3(0.25f, 0.5f, 0.25f); // YCoCg luma, same as the stored histories
    const float metalness = saturate(albedo_metalness.a);
    const float3 albedo = albedo_metalness.rgb;
    const float diffuse_reflectance = dot(brdf_diffuse_color(albedo, metalness), Y);
    const float specular_reflectance = specular_enabled ? dot(brdf_env_approx(brdf_specular_f0(albedo, metalness), roughness, NoV), Y) : 0.0f;
    // Missing history: borrow the other signal's incoming radiance (or the same unit radiance for both).
    const float fallback = max(max(incoming_luma.x, incoming_luma.y), 1.0f);
    const float incoming_diffuse = incoming_luma.x > 0.0f ? incoming_luma.x : (incoming_luma.y > 0.0f ? incoming_luma.y : fallback);
    const float incoming_specular = incoming_luma.y > 0.0f ? incoming_luma.y : (incoming_luma.x > 0.0f ? incoming_luma.x : fallback);
    const bool any_history = incoming_luma.x > 0.0f || (specular_enabled && incoming_luma.y > 0.0f);
#if RTGI_RAY_SHARE
    if (specular_enabled)
    {
        // Raw (RTGI_RADIANCE_SCALE scaled) radiance, the space linear_to_perceptual expects; natural log -> stops.
        // Neither signal has history: compare the reflectances directly. The unit fallback radiance can sit below
        // the perceptual floor, which would clamp both sides equal and force an even split regardless of material.
        const float stops = any_history
            ? (linear_to_perceptual(incoming_specular * specular_reflectance, inv_exposure) -
               linear_to_perceptual(incoming_diffuse * diffuse_reflectance, inv_exposure)) * 1.44269504f
            : log2(max(specular_reflectance, 1e-4f) / max(diffuse_reflectance, 1e-4f));
        const float specular_share = rcp(1.0f + exp2(-max(slope, 0.0f) * stops));
        impact.specular = 2.0f * specular_share;
        impact.diffuse = 2.0f * (1.0f - specular_share);
    }
#endif
    return impact;
}

// == Ray demand: material factors ============================================
// Material only (no history needed), so they act at disocclusions too. They scale the extras; base rays stay.
// Diffuse (metalness): leaving diffuse out entirely changes the pixel by E = log2((D + S) / S) stops, with
// D = luma(albedo (1 - metal)) and S = luma(env BRDF). While E is above a just-noticeable difference the diffuse
// signal matters; factor = saturate(E / tolerance). For albedo >> 0.04, D / S ~ (1 - m) / m, so E ~ -log2(m):
// tolerance 0.1 stops -> 1 up to metalness ~0.93, 0.74 at 0.95, 0.29 at 0.98, 0 at 1, nearly albedo independent.
// Specular (roughness): 1 - smoothstep(start, end, roughness). Near roughness 1 the lobe is as wide as the
// cosine lobe, the specular signal is low frequency and dim (env BRDF), so extra rays buy almost nothing visible.
#define RTGI_RAY_MATERIAL 1

func rtgi_calc_ray_material_factors(float4 albedo_metalness, float roughness, float NoV, RtgiSettings settings) -> float2
{
    float2 factors = float2(1.0f, 1.0f);
#if RTGI_RAY_MATERIAL
    if (settings.specular_enabled != 0 && settings.ray_diffuse_metal_tolerance_stops > 0.0f)
    {
        const float3 Y = float3(0.25f, 0.5f, 0.25f);
        const float metalness = saturate(albedo_metalness.a);
        const float D = dot(brdf_diffuse_color(albedo_metalness.rgb, metalness), Y);
        const float S = max(dot(brdf_env_approx(brdf_specular_f0(albedo_metalness.rgb, metalness), roughness, NoV), Y), 1e-4f);
        const float stops = log2((D + S) / S);
        factors.x = saturate(stops / settings.ray_diffuse_metal_tolerance_stops);
    }
    if (settings.ray_specular_roughness_cutoff_start < settings.ray_specular_roughness_cutoff_end)
    {
        factors.y = 1.0f - smoothstep(settings.ray_specular_roughness_cutoff_start, settings.ray_specular_roughness_cutoff_end, roughness);
    }
#endif
    return factors;
}

// Draws the temporal-accumulate debug overlays. Called on every geometry pixel — including the
// no-ray early-out path — so the overlay has no holes where a pixel skipped tracing this frame.
void rtgi_temporal_accumulate_debug_draw(uint2 dtid, float sample_count, float blend, float ao_guide, float perceptual_radiance)
{
    let push = rtgi_temporal_accumulate_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    RWTexture2D<float4> dbg = push.attach.debug_image.get();
    let debug_mode = push.attach.globals.settings.debug_draw_mode;
    const float debug_alpha = 1.0f + push.attach.globals.settings.debug_visualization_blend;
    if (debug_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_HISTORY_LENGTH)
    {
        // Relative to the fast convergence target (ray demand target): 1 = no extra rays requested any more.
        const float t = saturate(sample_count * rcp(max(rtgi_settings.fast_convergence_samples, 1.0f)));
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(t), debug_alpha), 2);
    }
    else if (debug_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_TEMPORAL_REACTIVITY)
    {
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(blend), debug_alpha), 2);
    }
    else if (debug_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_AO_GUIDE_TEMPORAL)
    {
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(ao_guide), debug_alpha), 2);
    }
    else if (debug_mode == DEBUG_DRAW_MODE_RTGI_DIFFUSE_PERCEPTUAL_MEAN_TEMPORAL)
    {
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(perceptual_radiance_colormap(perceptual_radiance, push.attach.globals.exposure), debug_alpha + 1.0f), 2);
    }
}

// === Specular temporal accumulation =========================================================
// Port of NVIDIA NRD ReBLUR's specular reprojection (REBLUR_TemporalAccumulation.cs.hlsl), adapted to the
// half-res RTGI layout. Two candidate histories are fetched every frame:
//   surface motion (smb) : the shared corner + weights the diffuse history uses,
//   virtual motion (vmb) : the reflected image moves like a point BEHIND the reflector, at the (dominant
//                          direction scaled) hit distance along the view ray (NRD GetXvirtual, curvature 0).
// Each gets its own accumulated frame count, scaled by its footprint quality and capped by a confidence:
//   vmb confidence : parallax (virtual uv from this frame's vs last frame's hit distance must stay inside the
//                    lobe spread) x normal (previous normal at the virtual footprint inside the lobe angle),
//   smb confidence : how far the view direction to the surface point turned vs the lobe half-angle (stricter
//                    for low NoV, long histories and far hits; relaxed above roughness 0.8).
// The selector prefers the motion with the longer surviving history (NRD "virtualHistoryAmount") and falls
// back to surface motion while the camera does not move.
// Gloss + shading normal test: the current specular lobe and the history tap's lobe are both built from
// (detail normal, roughness) and compared by their total overlap (rtgi_spec_lobe_overlap), which is the weight.
// Applied to every history tap of BOTH motions; the virtual motion additionally runs NRD's "prev-prev" test one
// step further along the virtual motion. All tests fade in with the pixels traveled ("jitter friendly"), so a
// static view never rejects history over half-res representative flicker.
// Not ported (yet): curvature estimation, CatRom history fetch.
struct RtgiSpecularAccumulation
{
    float4 specular;
    float4 fast;          // .x fast brightness mean, .y fast relative variance, .z fast frame count
    float frames;
    float virtual_amount; // debug: 0 = surface motion history, 1 = virtual motion history
    float blend;          // debug: weight of this frame's sample (1 = no history)
};

static const float RTGI_SPEC_MAX_PERCENT_OF_LOBE_VOLUME = 0.75f;
static const float RTGI_SPEC_MIN_LOBE_ANGLE = 0.002f; // stand-in for NRD_NORMAL_ENCODING_ERROR

// NRD / MathLib ImportanceSampling::GetSpecularLobeTanHalfAngle. `roughness` is the perceptual (linear) one.
func rtgi_spec_lobe_tan_half_angle(float roughness, float percent_of_volume) -> float
{
    const float p = saturate(percent_of_volume);
    return saturate(roughness) * sqrt(p / (1.0f - p + 1e-6f));
}

func rtgi_linear_step(float a, float b, float x) -> float
{
    return saturate((x - a) / (b - a));
}

// == Lobe overlap ==
// A specular lobe is approximated in normal space by a von Mises-Fisher distribution around the shading normal,
// with the sharpness of the GGX normal distribution: kappa = 2 / alpha^2 (alpha = roughness^2). Two lobes are
// compared by their normalized overlap  W = <f0, f1> / sqrt(<f0, f0> <f1, f1>)  (1 = identical lobes, -> 0 as
// they separate in direction and/or width). For vMF lobes this is closed form:
//     W = sqrt(C(2 k0) C(2 k1)) / C(|k0 n0 + k1 n1|),   C(k) = k / (4 pi sinh k)
// Evaluated in log space (the 4 pi cancels) so mirror sharp lobes don't overflow sinh.
static const float RTGI_SPEC_LOBE_MAX_KAPPA = 1e4f;
// Lobes are compared as if they were at least this rough. Half-res detail normals of normal-mapped surfaces
// differ by a few degrees between neighboring texels; without a floor a glossy lobe rejects its own history
// whenever the reprojection slides by a sub-texel amount. 0.25 -> overlap 0.5 at ~4 deg (0.2 was ~2.7 deg).
static const float RTGI_SPEC_LOBE_COMPARE_MIN_ROUGHNESS = 0.125f;

// Lobe-based rejection only ramps in once the reprojection actually moved this many half-res pixels. Slow motion
// resamples the same neighborhood (sub-texel shifts), where lobe differences are texture detail, not disocclusion.
static const float RTGI_SPEC_REJECT_RAMP_START_PX = 0.5f;
static const float RTGI_SPEC_REJECT_RAMP_END_PX = 3.0f;
func rtgi_spec_reject_ramp(float pixels_traveled) -> float
{
    return smoothstep(RTGI_SPEC_REJECT_RAMP_START_PX, RTGI_SPEC_REJECT_RAMP_END_PX, pixels_traveled);
}

func rtgi_spec_lobe_kappa(float roughness) -> float
{
    roughness = max(roughness, RTGI_SPEC_LOBE_COMPARE_MIN_ROUGHNESS);
    const float alpha = max(roughness * roughness, BRDF_MIN_GGX_ALPHA);
    return min(2.0f / (alpha * alpha), RTGI_SPEC_LOBE_MAX_KAPPA);
}


func rtgi_spec_lobe_overlap(float3 n0, float roughness0, float3 n1, float roughness1) -> float
{
    const float k0 = rtgi_spec_lobe_kappa(roughness0);
    const float k1 = rtgi_spec_lobe_kappa(roughness1);
    const float k01 = length(k0 * n0 + k1 * n1);
    const float log_w = 0.5f * (rtgi_log_vmf_norm(2.0f * k0) + rtgi_log_vmf_norm(2.0f * k1)) - rtgi_log_vmf_norm(k01);
    return saturate(exp(log_w));
}

// == Catmull-Rom history (NRD ReBLUR "BicubicFilterNoCornersWithFallbackToBilinearFilterWithCustomWeights") ==
// 12-tap Catmull-Rom (4x4 footprint without the corners), evaluated with 5 bilinear fetches. Bilinear history
// fetches blur the history a little every frame under motion; Catmull-Rom keeps it sharp. Only allowed when all
// 12 taps are valid history (otherwise the caller falls back to the bilinear custom weights).
static const float RTGI_SPEC_CATROM_SHARPNESS = 0.5f; // 0.5 == Catmull-Rom

// Clamped to >= 0 against Catmull-Rom's negative lobes (radiance and hit distance are never negative).
func rtgi_spec_catrom_history(Texture2D<float4> tex, float2 sample_pos, float2 inv_size) -> float4
{
    let push = rtgi_temporal_accumulate_push;
    SamplerState s = push.attach.globals.samplers.linear_clamp.get();
    const float K = RTGI_SPEC_CATROM_SHARPNESS;
    const float2 center_pos = floor(sample_pos - 0.5f) + 0.5f;
    const float2 f = saturate(sample_pos - center_pos);
    const float2 w0 = f * (f * (-K * f + 2.0f * K) - K);
    const float2 w1 = f * (f * ((2.0f - K) * f - (3.0f - K))) + 1.0f;
    const float2 w2 = f * (f * (-(2.0f - K) * f + (3.0f - 2.0f * K)) + K);
    const float2 w3 = f * (f * (K * f - K));
    const float2 w12 = w1 + w2;
    const float2 tc = w2 / w12;
    const float4 w = float4(w12.x * w0.y, w0.x * w12.y, w12.x * w12.y, w3.x * w12.y);
    const float w4 = w12.x * w3.y;
    const float sum = dot(w, 1.0f) + w4;
    float4 color = tex.SampleLevel(s, (center_pos + float2(tc.x, -1.0f)) * inv_size, 0) * w.x;
    color += tex.SampleLevel(s, (center_pos + float2(-1.0f, tc.y)) * inv_size, 0) * w.y;
    color += tex.SampleLevel(s, (center_pos + tc) * inv_size, 0) * w.z;
    color += tex.SampleLevel(s, (center_pos + float2(2.0f, tc.y)) * inv_size, 0) * w.w;
    color += tex.SampleLevel(s, (center_pos + float2(tc.x, 2.0f)) * inv_size, 0) * w4;
    return sum < 1e-4f ? float4(0, 0, 0, 0) : max(color / sum, 0.0f);
}

// NRD smbAllowCatRom: all 12 non-corner taps of the 4x4 footprint around sample_pos must lie on the current
// surface plane (previous-frame depth, binary plane test at 1 pixel width).
func rtgi_spec_catrom_footprint_valid(float2 sample_pos, float2 inv_size, CameraInfo* previous_camera, float3 position_ws, float3 face_normal, float pixel_width_ws_rcp) -> bool
{
    let push = rtgi_temporal_accumulate_push;
    SamplerState s = push.attach.globals.samplers.linear_clamp.get();
    const float2 origin = floor(sample_pos - 0.5f) - 1.0f; // top-left texel of the 4x4 footprint
    uint valid = 0;
    [unroll]
    for (uint q = 0; q < 4; ++q)
    {
        const uint2 quad_offset = uint2(q & 1, q >> 1) * 2;
        const float2 quad_origin = origin + float2(quad_offset);
        const float4 depths = push.attach.half_res_depth_history.get().GatherRed(s, (quad_origin + 1.0f) * inv_size).wzxy;
        [unroll]
        for (uint i = 0; i < 4; ++i)
        {
            const uint2 t = quad_offset + uint2(i & 1, i >> 1);
            const bool corner = (t.x == 0 || t.x == 3) && (t.y == 0 || t.y == 3);
            if (corner) { continue; }
            const float2 tap_ndc = (quad_origin + float2(float(i & 1), float(i >> 1)) + 0.5f) * inv_size * 2.0f - 1.0f;
            const float3 tap_ws = rtgi_half_res_depth_to_world_space(*previous_camera, tap_ndc, depths[i]);
            const float plane_dist_px = abs(calc_plane_distance(position_ws, face_normal, tap_ws)) * pixel_width_ws_rcp;
            valid += (depths[i] != 0.0f && plane_dist_px < 1.0f) ? 1u : 0u;
        }
    }
    return valid == 12u;
}

func rtgi_gather_rgba_bilinear(Texture2D<float4> tex, SamplerState s, float2 gather_uv, float4 weights) -> float4
{
    const float4 r = tex.GatherRed(s, gather_uv).wzxy;
    const float4 g = tex.GatherGreen(s, gather_uv).wzxy;
    const float4 b = tex.GatherBlue(s, gather_uv).wzxy;
    const float4 a = tex.GatherAlpha(s, gather_uv).wzxy;
    return apply_bilinear_custom_weights(float4(r[0], g[0], b[0], a[0]), float4(r[1], g[1], b[1], a[1]), float4(r[2], g[2], b[2], a[2]), float4(r[3], g[3], b[3], a[3]), weights);
}

// Frames always use the bilinear custom weights (NRD reads accumSpeed bilinearly too); the color uses the
// Catmull-Rom fetch when use_catrom.
func rtgi_gather_specular_history(float2 gather_uv, float4 weights, bool use_catrom, float2 sample_pos, float2 inv_size, out float4 history, out float frames, out float4 fast)
{
    let push = rtgi_temporal_accumulate_push;
    SamplerState s = push.attach.globals.samplers.linear_clamp.get();
    // Fast history: always the bilinear custom weights (statistics, not image detail).
    fast = rtgi_gather_rgba_bilinear(push.attach.specular_fast_history_history.get(), s, gather_uv, weights);
    if (any(isnan(fast))) { fast = float4(0, 0, 0, 0); }
    const float4 f = push.attach.specular_frames_history.get().GatherRed(s, gather_uv).wzxy;
    frames = apply_bilinear_custom_weights(f[0], f[1], f[2], f[3], weights);
    history = use_catrom
        ? rtgi_spec_catrom_history(push.attach.half_res_specular_history.get(), sample_pos, inv_size)
        : rtgi_gather_rgba_bilinear(push.attach.half_res_specular_history.get(), s, gather_uv, weights);
    if (any(isnan(history)) || isnan(frames)) { history = float4(0, 0, 0, 0); frames = 0.0f; }
}

// NRD: accumSpeed *= lerp(footprintQuality, 1, 1 / (1 + accumSpeed)).
func rtgi_spec_apply_footprint_quality(float frames, float4 footprint_weights) -> float
{
    const float quality = sqrt(saturate(dot(footprint_weights, 1.0f)));
    return frames * lerp(quality, 1.0f, 1.0f / (1.0f + frames));
}

func rtgi_temporal_accumulate_specular(uint2 dtid, float rays_shot, bool surface_disocclusion, float2 surface_gather_uv, float4 surface_weights, float2 half_res_size) -> RtgiSpecularAccumulation
{
    let push = rtgi_temporal_accumulate_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    RtgiSpecularAccumulation ret;
    ret.specular = float4(0, 0, 0, 0);
    ret.fast = float4(0, 0, 0, 0);
    ret.frames = 0.0f;
    ret.virtual_amount = 0.0f;
    ret.blend = 1.0f;
    if (rtgi_settings.specular_enabled == 0)
    {
        return ret;
    }

    const bool pre_blurred_present = !push.attach.specular_pre_blurred.index.is_empty();
    const float4 new_spec = pre_blurred_present ? push.attach.specular_pre_blurred.get()[dtid] : push.attach.pre_filtered_specular_new.get()[dtid];
    const uint center_normal_roughness = push.attach.half_res_normal_roughness.get()[dtid];
    const float roughness = unpack_normal_roughness_roughness(center_normal_roughness);
    const float2 inv_size = rcp(half_res_size);

    const CameraInfo camera = push.attach.globals->view_camera;
    CameraInfo* previous_camera = &push.attach.globals->view_camera_prev_frame;
    const PixelData pixel = calc_pixel_data(dtid, inv_size, camera, push.attach.half_res_depth.get(), push.attach.half_res_face_normals.get());
    const float pixel_width_ws = rtgi_half_res_pixel_width_ws(inv_size, camera.near_plane, pixel.depth_vs);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);
    const float3 N_face = pixel.normal_ws; // geometry (plane) tests
    const float3 N = unpack_normal_roughness_normal(center_normal_roughness); // shading normal: lobe tests
    const float3 V = normalize(camera.position - pixel.position_ws); // towards the eye
    const float NoV = abs(dot(N, V));
    SamplerState linear_clamp = push.attach.globals.samplers.linear_clamp.get();

    // Hit distance for tracking: min over the 3x3 neighborhood, ignoring 0 (no-ray / failed samples).
    float hit_for_tracking = RTGI_SPECULAR_MAX_HIT_DISTANCE;
    [unroll]
    for (int y = -1; y <= 1; ++y)
    {
        [unroll]
        for (int x = -1; x <= 1; ++x)
        {
            const int2 p = clamp(int2(dtid) + int2(x, y), int2(0, 0), int2(half_res_size) - 1);
            const float h = push.attach.pre_filtered_specular_new.get()[p].a;
            hit_for_tracking = h > 0.0f ? min(hit_for_tracking, h) : hit_for_tracking;
        }
    }

    // == Surface motion ==
    const float4 smb_clip = mul(previous_camera.view_proj, float4(pixel.position_ws, 1.0f));
    const float2 smb_uv = (smb_clip.xy / smb_clip.w) * 0.5f + 0.5f;
    const float2 pixel_uv = (float2(dtid) + 0.5f) * inv_size;
    const float mv_length_in_pixels = length((smb_uv - pixel_uv) * half_res_size);
    const float slow_motion_factor = saturate(mv_length_in_pixels / 0.25f);

    const bool catrom_enabled = rtgi_settings.specular_catrom_history != 0;
    const float2 smb_sample_pos = smb_uv * half_res_size;
    float4 surface_history = float4(0, 0, 0, 0);
    float4 surface_fast = float4(0, 0, 0, 0);
    float smb_frames = 0.0f;
    bool smb_allow_catrom = false;
    bool surface_valid = !surface_disocclusion;
    if (surface_valid)
    {
        // Gloss + shading normal test of the surface footprint (on top of the diffuse geometry weights).
        const uint4  prev_normal_roughness = push.attach.half_res_normal_roughness_history.get().GatherRed(linear_clamp, surface_gather_uv).wzxy;
        const float  jitter_ramp = rtgi_spec_reject_ramp(mv_length_in_pixels);
        float4 lobe_weights;
        [unroll]
        for (uint i = 0; i < 4; ++i)
        {
            const float w = rtgi_spec_lobe_overlap(N, roughness, unpack_normal_roughness_normal(prev_normal_roughness[i]), unpack_normal_roughness_roughness(prev_normal_roughness[i]));
            lobe_weights[i] = lerp(1.0f, w, jitter_ramp);
        }
        const float4 tested_weights = surface_weights * lobe_weights;
        surface_valid = dot(tested_weights, 1.0f) > 0.025f;
        if (surface_valid)
        {
            // CatRom needs the full 12-tap footprint on this surface and the inner 2x2 lobes overlapping.
            smb_allow_catrom = catrom_enabled && all(lobe_weights > 0.5f) &&
                rtgi_spec_catrom_footprint_valid(smb_sample_pos, inv_size, previous_camera, pixel.position_ws, N_face, pixel_width_ws_rcp);
            rtgi_gather_specular_history(surface_gather_uv, tested_weights, smb_allow_catrom, smb_sample_pos, inv_size, surface_history, smb_frames, surface_fast);
            // Footprint quality from the GEOMETRIC footprint only (NRD: occlusion, not lobes). The lobe weights only
            // pick which taps contribute; folding them in here would shrink the history every moving frame.
            smb_frames = rtgi_spec_apply_footprint_quality(smb_frames, surface_weights);
        }
    }

    // == Virtual motion ==
    const float dominant_factor = rtgi_spec_dominant_factor(NoV, roughness);
    const float3 virtual_position = pixel.position_ws - V * hit_for_tracking * dominant_factor;
    const float4 vmb_clip = mul(previous_camera.view_proj, float4(virtual_position, 1.0f));
    const float2 vmb_uv = (vmb_clip.xy / vmb_clip.w) * 0.5f + 0.5f;
    const float vmb_pixels_traveled = length((vmb_uv - smb_uv) * half_res_size);

    bool virtual_valid = false;
    float4 virtual_history = float4(0, 0, 0, 0);
    float4 virtual_fast = float4(0, 0, 0, 0);
    float vmb_frames = 0.0f;
    float3 vmb_normal = N;
    float vmb_roughness = roughness;

    if (rtgi_settings.specular_virtual_reprojection != 0 && vmb_clip.w > 0.0f && all(vmb_uv > 0.0f) && all(vmb_uv < 1.0f))
    {
        const Bilinear bilinear = get_bilinear_filter(vmb_uv, half_res_size);
        const float2 gather_uv = (bilinear.origin + 1.0f) * inv_size;
        SamplerState s = push.attach.globals.samplers.linear_clamp.get();
        const float4 prev_depths = push.attach.half_res_depth_history.get().GatherRed(s, gather_uv).wzxy;
        const uint4 prev_normals_packed = push.attach.half_res_face_normals_history.get().GatherRed(s, gather_uv).wzxy;
        const uint4 prev_normal_roughness = push.attach.half_res_normal_roughness_history.get().GatherRed(s, gather_uv).wzxy;
        const float jitter_ramp = rtgi_spec_reject_ramp(vmb_pixels_traveled);
        float4 occlusion = float4(0, 0, 0, 0);
        float4 geometric_occlusion = float4(0, 0, 0, 0);
        float3 normal_acc = float3(0, 0, 0);
        float roughness_acc = 0.0f;
        const float4 bilinear_weights = get_bilinear_custom_weights(bilinear, float4(1, 1, 1, 1));
        [unroll]
        for (uint i = 0; i < 4; ++i)
        {
            const float2 tap_ndc = (bilinear.origin + float2(float(i & 1), float(i >> 1)) + 0.5f) * inv_size * 2.0f - 1.0f;
            const float3 tap_ws = rtgi_half_res_depth_to_world_space(*previous_camera, tap_ndc, prev_depths[i]);
            const float3 tap_n = uncompress_normal_octahedral_32(prev_normals_packed[i]);
            // Disocclusion: the reflected image must come from the same reflector (same plane, facing), weighted
            // by how much the tap's lobe (detail normal + gloss) overlaps the current one.
            const float plane = calc_similar_surface_weight(pixel_width_ws_rcp, pixel.position_ws, N_face, tap_ws, tap_n, 2.0f);
            const float facing = dot(tap_n, N_face) > 0.5f ? 1.0f : 0.0f;
            const float3 tap_detail_normal = unpack_normal_roughness_normal(prev_normal_roughness[i]);
            const float tap_roughness = unpack_normal_roughness_roughness(prev_normal_roughness[i]);
            const float lobe_overlap = lerp(1.0f, rtgi_spec_lobe_overlap(N, roughness, tap_detail_normal, tap_roughness), jitter_ramp);
            geometric_occlusion[i] = prev_depths[i] != 0.0f ? step(0.5f, plane) * facing : 0.0f;
            occlusion[i] = geometric_occlusion[i] * lobe_overlap;
            normal_acc += tap_detail_normal * bilinear_weights[i];
            roughness_acc += tap_roughness * bilinear_weights[i];
        }
        vmb_normal = length(normal_acc) > 1e-6f ? normalize(normal_acc) : N;
        vmb_roughness = roughness_acc;
        const float4 weights = get_bilinear_custom_weights(bilinear, occlusion);
        if (dot(weights, 1.0f) > 0.025f)
        {
            virtual_valid = true;
            // NRD vmbAllowCatRom: complete 2x2 virtual footprint, and surface motion allows CatRom too
            // (reduces over-sharpening in disoccluded areas).
            const bool vmb_allow_catrom = smb_allow_catrom && all(occlusion > 0.5f);
            rtgi_gather_specular_history(gather_uv, weights, vmb_allow_catrom, vmb_uv * half_res_size, inv_size, virtual_history, vmb_frames, virtual_fast);
            vmb_frames = rtgi_spec_apply_footprint_quality(vmb_frames, get_bilinear_custom_weights(bilinear, geometric_occlusion));
        }
    }

    if (!surface_valid && !virtual_valid)
    {
        // Disoccluded in both motions: restart from the new sample (fast history too, 0 fast frames like diffuse).
        ret.specular = new_spec;
        ret.fast = float4(dot(new_spec.rgb, float3(0.25f, 0.5f, 0.25f)), 0.0f, 0.0f, 0.0f);
        ret.frames = rays_shot;
        return ret;
    }

    // == Virtual motion confidence ==
    float virtual_confidence = 0.0f;
    if (virtual_valid)
    {
        const float percent_of_volume = RTGI_SPEC_MAX_PERCENT_OF_LOBE_VOLUME / (1.0f + vmb_frames);
        const float lobe_tan = max(rtgi_spec_lobe_tan_half_angle(roughness, percent_of_volume), RTGI_SPEC_MIN_LOBE_ANGLE);

        // Parallax: the virtual footprint computed with last frame's hit distance must land within the lobe spread
        // (in pixels) of the one computed with this frame's hit distance.
        const float hit_prev = virtual_history.a > 0.0f ? virtual_history.a : hit_for_tracking;
        const float3 virtual_position_prev = pixel.position_ws - V * hit_prev * dominant_factor;
        const float4 vmb_clip_prev = mul(previous_camera.view_proj, float4(virtual_position_prev, 1.0f));
        const float2 vmb_uv_prev = (vmb_clip_prev.xy / vmb_clip_prev.w) * 0.5f + 0.5f;
        const float virtual_distance = length(virtual_position - camera.position);
        const float pixel_radius_at_virtual = 0.5f * rtgi_half_res_pixel_width_ws(inv_size, camera.near_plane, virtual_distance);
        float r = min(hit_for_tracking, hit_prev) * lobe_tan / max(pixel_radius_at_virtual, 1e-6f);
        r *= 0.5f;                       // strengthen the test
        r = max(r, 0.1f * roughness);    // clean up dirt for high roughness
        const float d = length((vmb_uv_prev - vmb_uv) * half_res_size);
        const float parallax_weight = rtgi_linear_step(r, 0.0f, d);

        // Lobe: overlap of the current lobe with the (filtered) previous lobe at the virtual footprint ("jitter friendly").
        virtual_confidence = lerp(1.0f, rtgi_spec_lobe_overlap(N, roughness, vmb_normal, vmb_roughness), rtgi_spec_reject_ramp(vmb_pixels_traveled));

        // Prev-prev test (NRD): one step further along the virtual motion, the previous frame must still show an
        // overlapping lobe. Catches virtual motion that slides across a normal / material boundary.
        if (vmb_pixels_traveled > 0.0f)
        {
            const float step_between_taps = min(vmb_pixels_traveled * 2.0f, 2.0f) + vmb_pixels_traveled;
            const float2 vmb_dir = (vmb_uv - smb_uv) * rsqrt(max(dot(vmb_uv - smb_uv, vmb_uv - smb_uv), 1e-12f)) * inv_size;
            const float2 prev_prev_uv = vmb_uv + vmb_dir * step_between_taps;
            if (all(prev_prev_uv > 0.0f) && all(prev_prev_uv < 1.0f))
            {
                const uint2 p = uint2(prev_prev_uv * half_res_size);
                const uint pp_normal_roughness = push.attach.half_res_normal_roughness_history.get()[p];
                const float3 pp_normal = unpack_normal_roughness_normal(pp_normal_roughness);
                const float pp_roughness = unpack_normal_roughness_roughness(pp_normal_roughness);
                virtual_confidence = min(virtual_confidence, rtgi_spec_lobe_overlap(vmb_normal, vmb_roughness, pp_normal, pp_roughness));
            }
        }

        virtual_confidence *= parallax_weight;
    }

    // == Surface motion confidence ==
    float surface_confidence = 0.0f;
    if (surface_valid)
    {
        const float3 V_prev = normalize(previous_camera.position - pixel.position_ws);
        float a = atan2(length(cross(V, V_prev)), max(dot(V, V_prev), 1e-6f));
        a *= lerp(0.1f, 1.0f, slow_motion_factor);

        const float non_linear_accum_speed = 1.0f / (1.0f + smb_frames);
        const float h = lerp(surface_history.a, new_spec.a, non_linear_accum_speed);
        const float frustum_size = pixel_width_ws * min(half_res_size.x, half_res_size.y);
        float tana0 = rtgi_spec_lobe_tan_half_angle(roughness, RTGI_SPEC_MAX_PERCENT_OF_LOBE_VOLUME);
        tana0 *= lerp(NoV, 1.0f, roughness);              // stricter at grazing angles (very V dependent lobe)
        tana0 *= non_linear_accum_speed;                  // stricter for long histories
        tana0 /= saturate(h / frustum_size) + 1e-6f;      // relaxed in corners, where the reflection is near the surface
        const float a0 = max(atan(tana0), RTGI_SPEC_MIN_LOBE_ANGLE);
        surface_confidence = pow(rtgi_linear_step(a0, 0.0f, a), 4.0f);
        // Very rough specular regresses to surface motion.
        surface_confidence = lerp(surface_confidence, 1.0f, rtgi_linear_step(0.8f, 0.9f, roughness));
    }

    // == Frame limits ==
    const float max_frames = rtgi_settings.specular_max_temporal_frames;
    smb_frames = min(smb_frames, max_frames * surface_confidence);
    vmb_frames = min(vmb_frames, max_frames * virtual_confidence);

    // == Selector (NRD virtualHistoryAmount) ==
    float virtual_amount;
    if (!virtual_valid)      { virtual_amount = 0.0f; }
    else if (!surface_valid) { virtual_amount = 1.0f; }
    else
    {
        // 1 if vmb >= smb, pulled towards smb by how much shorter the virtual history is.
        virtual_amount = saturate(1.0f + (vmb_frames - smb_frames) / (1.0f + 0.5f * max(vmb_frames, smb_frames)));
        // No motion: the surface history is exact.
        virtual_amount *= slow_motion_factor;
    }

    ret.virtual_amount = virtual_amount;
    const float4 history = lerp(surface_history, virtual_history, virtual_amount);
    float history_frames = lerp(smb_frames, vmb_frames, virtual_amount);

    // Surface vs virtual disagreement: both reprojections are valid but fetched very different reflections.
    // Shorten the carried history in proportion to how many stops apart they are.
    if (surface_valid && virtual_valid && virtual_amount > 0.0f && rtgi_settings.specular_reprojection_disagreement_strength > 0.0f)
    {
        const float floor_value = calc_perceptual_radiance_floor(push.attach.globals.inv_exposure);
        const float surface_brightness = max(dot(surface_history.rgb, RTGI_CHANNEL_PERCEIVED_BRIGHTNESS), floor_value);
        const float virtual_brightness = max(dot(virtual_history.rgb, RTGI_CHANNEL_PERCEIVED_BRIGHTNESS), floor_value);
        const float disagreement_stops = abs(log2(surface_brightness / virtual_brightness));
        // Deadzone covers the noise between two converging histories; the cut ramps in with camera motion
        // (slow motion fetches nearly the same history twice, a mismatch there is noise, not a wrong reflection).
        const float DISAGREEMENT_DEADZONE_STOPS = 0.5f;
        const float cut = saturate((disagreement_stops - DISAGREEMENT_DEADZONE_STOPS) * 0.5f * rtgi_settings.specular_reprojection_disagreement_strength)
            * virtual_amount * rtgi_spec_reject_ramp(mv_length_in_pixels);
        history_frames *= 1.0f - cut;
    }

    // Fast history follows the same reprojection as the color (surface / virtual, blended by virtual_amount).
    const float4 fast_history = lerp(surface_fast, virtual_fast, virtual_amount);

    if (rays_shot <= 0.0f)
    {
        // No new sample: carry the fast history unchanged (no new fast observation, frame count not advanced).
        ret.specular = history;
        ret.fast = fast_history;
        ret.frames = history_frames;
        ret.blend = 0.0f;
        return ret;
    }

    // == Fast history (mirrors the diffuse fast history in entry_temporal_accumulate) ==
    // Short-window brightness mean + relative variance. Its only consumer is the blend confidence below: where the
    // slow history's brightness diverges from the fast mean (lighting / reflection changed), the slow history is
    // trusted less (anti-lag); fast relative variance (plain noise) earns some of that trust back.
    float blend_frames = history_frames;
    ret.fast = fast_history;
    if (rtgi_settings.specular_fast_history_enabled != 0)
    {
        const float3 Y = float3(0.25f, 0.5f, 0.25f);
        const float FAST_HISTORY_FRAMES = clamp(float(rtgi_settings.specular_fast_history_frames), 1.0f, 15.0f);
        const float fast_frames = min(fast_history.z + 1.0f, 15.0f);
        const float fast_blend = 1.0f / (1.0f + min(fast_frames, FAST_HISTORY_FRAMES));
        const float fast_mean_prev = fast_history.x;
        const float fast_var_prev = fast_history.y;
        float new_brightness = dot(new_spec.rgb, Y);
        // Temporal firefly filter on the fast stats only: clamp toward fast mean + N std devs once the window is full.
        if (fast_frames > FAST_HISTORY_FRAMES && rtgi_settings.specular_temporal_firefly_filter_enabled)
        {
            const float ceiling = fast_mean_prev * (1.0f + sqrt(fast_var_prev) * rtgi_settings.specular_temporal_firefly_std_dev_clamp);
            new_brightness = min(new_brightness, ceiling);
        }
        const float fast_mean = lerp(fast_mean_prev, new_brightness, fast_blend);
        const float new_relative_variance_point = min(square((new_brightness - fast_mean_prev) / max(fast_mean_prev, 1e-6f)), 4.0f);
        const float fast_var = lerp(fast_var_prev, new_relative_variance_point, fast_blend);
        ret.fast = float4(fast_mean, fast_var, fast_frames, 0.0f);

        const float adaptation_point = min(square(abs(fast_mean - new_brightness) / max(fast_mean, 1e-6f)), 4.0f);
        const float adaptation_variance = lerp(fast_var_prev, adaptation_point, fast_blend);
        const float variance_scaling = square(1.0f + sqrt(adaptation_variance) * rtgi_settings.specular_temporal_variance_fast_history_blend);
        const float slow_mean = dot(history.rgb, Y);
        const float slow_to_fast_ratio = max(slow_mean, fast_mean) / (min(slow_mean, fast_mean) + 1e-8f);
        const float mean_diff_scaling = square(1.0f / max(1.0f, slow_to_fast_ratio));
        if (history_frames > FAST_HISTORY_FRAMES)
        {
            blend_frames = min(history_frames, history_frames * variance_scaling * mean_diff_scaling);
        }
    }

    float blend = min(1.0f, rays_shot / (blend_frames + rays_shot));
    if (!rtgi_settings.specular_temporal_accumulation_enabled)
    {
        blend = 1.0f;
    }
    ret.specular = lerp(history, new_spec, blend);
    ret.blend = blend;
    // Counted in SAMPLES (rays), like the diffuse history: a multi-ray burst converges proportionally faster.
    ret.frames = history_frames + rays_shot;
    return ret;
}

// === Temporal Accumulation =================================================================
// Consumes the reprojection metadata to read color/statistics history and blend it with the new frame.
[shader("compute")]
[numthreads(RTGI_TEMPORAL_X,RTGI_TEMPORAL_Y,1)]
func entry_temporal_accumulate(uint2 dtid : SV_DispatchThreadID)
{
    let push = rtgi_temporal_accumulate_push;
    let rtgi_settings = push.attach.globals.rtgi_settings;
    if (any(dtid.xy >= push.size))
    {
        return;
    }

    // Load and precalculate constants
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);

    // Read the reprojected carry sample count (written by the reproject pass, read by the trace pass).
    // Only the sky sentinel (<0) survives as an early-out; disocclusion is detected from the weights.
    const uint packed_carry = push.attach.half_res_sample_count.get()[dtid.xy];
    const float reproj_carry_sample_count = rtgi_unpack_normal_count(packed_carry);
    if (reproj_carry_sample_count < 0.0f)
    {
        return; // sky sentinel
    }
    // Parallax stretch penalty the reproject pass stashed in the fast field ([0,15] -> [0,1]); applied to
    // the fast-history frame count below just like it was applied to the normal count in reproject.
    const float parallax_penalty = rtgi_unpack_fast_count(packed_carry) * (1.0f / 15.0f);
    // Increment the sample count by the rays the trace pass shot this frame (moved here from trace),
    // clamped to the history cap.
    const float rays_shot_virtual_samples = (float(push.attach.ray_count_image.get()[dtid.xy]));
    const float accumulated_sample_count = rtgi_accumulate_sample_count(reproj_carry_sample_count, rays_shot_virtual_samples, rtgi_settings.max_temporal_samples);
    const uint2 corner_plus_one = push.attach.reproject_corner.get()[dtid.xy];
    const float2 reproject_gather_uv = rtgi_reproject_gather_uv(corner_plus_one, inv_half_res_render_target_size);
    const float4 sample_weights = push.attach.reproject_weights.get()[dtid.xy];

    // Detect disocclusion from the reprojection footprint weights (identical test to the reproject
    // pass), since the sample count no longer drops to zero on disocclusion.
    const float disocclusion_threshold = 0.025f;
    const bool disocclusion = dot(sample_weights, 1.0f) < disocclusion_threshold;
    SamplerState linear_clamp_s = push.attach.globals.samplers.linear_clamp.get();

    // Reproject color & statistics history using the precomputed bilinear custom weights.
    float4 reprojected_diffuse = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float2 reprojected_diffuse2 = float2(0.0f, 0.0f);
    float reprojected_fast_temporal_mean = 0.0f;
    float reprojected_fast_temporal_variance = 0.0f;
    float reprojected_ao_guide = 0.0f;
    float reprojected_perceptual_radiance = 0.0f;
    float reprojected_fast_frames = 0.0f;
    {
        // Diffuse
        const float4 diffuse_r = push.attach.half_res_diffuse_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse_g = push.attach.half_res_diffuse_history.get().GatherGreen( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse_b = push.attach.half_res_diffuse_history.get().GatherBlue( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse_a = push.attach.half_res_diffuse_history.get().GatherAlpha( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse_samples[4] = {
            float4(diffuse_r[0], diffuse_g[0], diffuse_b[0], diffuse_a[0]),
            float4(diffuse_r[1], diffuse_g[1], diffuse_b[1], diffuse_a[1]),
            float4(diffuse_r[2], diffuse_g[2], diffuse_b[2], diffuse_a[2]),
            float4(diffuse_r[3], diffuse_g[3], diffuse_b[3], diffuse_a[3]),
        };
        reprojected_diffuse = apply_bilinear_custom_weights( diffuse_samples[0], diffuse_samples[1], diffuse_samples[2], diffuse_samples[3], sample_weights );
        if (any(isnan(reprojected_diffuse)))
        {
            reprojected_diffuse = {};
        }

        // Diffuse2
        const float4 diffuse2_r = push.attach.half_res_diffuse2_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 diffuse2_g = push.attach.half_res_diffuse2_history.get().GatherGreen( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float2 diffuse2_samples[4] = {
            float2(diffuse2_r[0], diffuse2_g[0]),
            float2(diffuse2_r[1], diffuse2_g[1]),
            float2(diffuse2_r[2], diffuse2_g[2]),
            float2(diffuse2_r[3], diffuse2_g[3]),
        };
        reprojected_diffuse2 = apply_bilinear_custom_weights( diffuse2_samples[0], diffuse2_samples[1], diffuse2_samples[2], diffuse2_samples[3], sample_weights );
        if (any(isnan(reprojected_diffuse2)))
        {
            reprojected_diffuse2 = {};
        }

        // Fast temporal history (f16x2): .x=fast_mean .y=fast_rel_var
        const float4 stat_r = push.attach.fast_temporal_history_history.get().GatherRed(  linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 stat_g = push.attach.fast_temporal_history_history.get().GatherGreen( linear_clamp_s, reproject_gather_uv ).wzxy;
        reprojected_fast_temporal_mean     = apply_bilinear_custom_weights( stat_r[0], stat_r[1], stat_r[2], stat_r[3], sample_weights );
        reprojected_fast_temporal_variance = apply_bilinear_custom_weights( stat_g[0], stat_g[1], stat_g[2], stat_g[3], sample_weights );
        if (isnan(reprojected_fast_temporal_mean))     reprojected_fast_temporal_mean = 0.0f;
        if (isnan(reprojected_fast_temporal_variance)) reprojected_fast_temporal_variance = 0.0f;

        const float4 fg4 = push.attach.half_res_ao_guide_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        reprojected_ao_guide = apply_bilinear_custom_weights( fg4.x, fg4.y, fg4.z, fg4.w, sample_weights );
        if (isnan(reprojected_ao_guide)) { reprojected_ao_guide = 0.0f; }

        // Temporal geometric mean history (log-space R16_SFLOAT)
        const float4 gm4 = push.attach.temporal_perceptual_radiance_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        reprojected_perceptual_radiance = apply_bilinear_custom_weights( gm4[0], gm4[1], gm4[2], gm4[3], sample_weights );
        if (isnan(reprojected_perceptual_radiance)) { reprojected_perceptual_radiance = 0.0f; }

        // Fast-history frame count: unpacked from the fast field of the previous frame's packed counter
        // texel. Reprojected like the rest so it follows the surface.
        const uint4 ffp4 = push.attach.half_res_sample_count_history.get().GatherRed( linear_clamp_s, reproject_gather_uv ).wzxy;
        const float4 ff4 = float4(
            rtgi_unpack_fast_count(ffp4.x),
            rtgi_unpack_fast_count(ffp4.y),
            rtgi_unpack_fast_count(ffp4.z),
            rtgi_unpack_fast_count(ffp4.w));
        reprojected_fast_frames = apply_bilinear_custom_weights( ff4[0], ff4[1], ff4[2], ff4[3], sample_weights );
        if (isnan(reprojected_fast_frames)) { reprojected_fast_frames = 0.0f; }
    }

    // Fast-history age in FRAMES (not ray samples): +1 per frame, reset to 0 on disocclusion. This is the
    // quantity the fast history should ramp on — a multi-ray disocclusion burst inflates the ray-sample
    // count by up to fast_convergence_samples in a single frame, which previously made the fast history hit full
    // confidence (and the firefly clamp) after one frame and lock the pixel onto that first, still-noisy
    // value. Counting frames makes the fast window ramp over FAST_HISTORY_FRAMES actual frames as intended.
    // Parallax stretch also drops the fast-history frame count (same penalty as the normal count), so a
    // stretched pixel's short fast window resets and the firefly clamp doesn't lock onto the smeared strip.
    const float accumulated_fast_frames = disocclusion ? 0.0f : (reprojected_fast_frames + 1.0f) * (1.0f - parallax_penalty);

    // Load new diffuse data
    const bool diffuse_pre_blurred_present = !push.attach.half_res_diffuse_pre_blurred.index.is_empty();

    float4 new_diffuse = diffuse_pre_blurred_present ? push.attach.half_res_diffuse_pre_blurred.get()[dtid.xy] : push.attach.pre_filtered_diffuse_new.get()[dtid.xy];
    float2 new_diffuse2 = diffuse_pre_blurred_present ? push.attach.half_res_diffuse2_pre_blurred.get()[dtid.xy] : push.attach.pre_filtered_diffuse2_new.get()[dtid.xy];
    float new_ao_guide = push.attach.ao_guide_new.get()[dtid.xy];

    // Determine accumulated fast history

    // Fast-history window length in frames. Capped at 15 to match the 6-bit fast counter storage range.
    const float FAST_HISTORY_FRAMES = clamp(float(rtgi_settings.temporal_fast_history_frames), 1.0f, 15.0f);
    // == Fast History ================
    // Ramp on the fast-history FRAME count, not the ray-sample count: one frame == one fast observation,
    // regardless of how many rays that frame bursted. This lets the fast window fill over
    // FAST_HISTORY_FRAMES frames instead of collapsing to full confidence on a single disocclusion burst.
    const float fast_blend_factor = (1.0f / (1.0f + min(accumulated_fast_frames, FAST_HISTORY_FRAMES)));
    float fast_mean_diff_scaling = 1.0f;
    float fast_variance_scaling = 1.0f;
    float accumulated_fast_mean = 0.0f;
    float accumulated_fast_relative_variance = 0.0f;
    float fast_std_dev_relative = 0.0f;
    if (rtgi_settings.temporal_fast_history_enabled)
    {
        // Temporal Fast History inspired by [DD2018: Tomasz Stachowiak - Stochastic all the things](https://www.youtube.com/watch?v=MyTOGHqyquU)

        // Fast History only stores brightness to save space.
        float new_fast_brightness = new_diffuse.w;
        // Temporal firefly filter — applied ONLY to the fast history (not the main color). Clamp a bright
        // outlier toward the reprojected fast mean + N std devs before it enters the fast mean/variance.
        // Skipped on disocclusion (no valid reprojected mean) — clamping there would pull it toward black.
        if (!disocclusion && accumulated_fast_frames > FAST_HISTORY_FRAMES && rtgi_settings.temporal_firefly_filter_enabled)
        {
            const float brightness_ratio = reprojected_fast_temporal_mean * (1.0f + sqrt(reprojected_fast_temporal_variance) * rtgi_settings.temporal_firefly_std_dev_clamp) / max(new_fast_brightness, 1e-8f);
            new_fast_brightness *= min(1.0f, brightness_ratio);
        }
        // On disocclusion there is no valid reprojected fast history (reprojected mean/variance ~0),
        // so reset the fast history to the new sample instead of lerping from garbage. Otherwise the
        // fast mean initializes near zero and the temporal firefly filter clamps the pixel toward black.
        accumulated_fast_mean = disocclusion ? new_fast_brightness : lerp(reprojected_fast_temporal_mean, new_fast_brightness, fast_blend_factor);

        // Relative variance EMA: point estimate uses OLD (reprojected) mean so the residual is
        // computed before the mean shifts — unbiased and dimensionless (fp16-safe at any radiance scale).
        // Clamped to 4.0 (2σ) to prevent fp16 overflow when the mean is uninitialized (zero).
        const float old_mean_safe = max(reprojected_fast_temporal_mean, 1e-6f);
        const float new_relative_variance_point = min(square((new_fast_brightness - reprojected_fast_temporal_mean) / old_mean_safe), 4.0f);
        accumulated_fast_relative_variance = disocclusion ? 0.0f : lerp(reprojected_fast_temporal_variance, new_relative_variance_point, fast_blend_factor);
        fast_std_dev_relative = sqrt(accumulated_fast_relative_variance);

        // Recompute the original "wrong" point estimate (uses post-update mean) to drive fast_variance_scaling,
        // preserving the original adaptation model exactly.
        const float wrong_point_estimate = min(square(abs(accumulated_fast_mean - new_fast_brightness) / max(accumulated_fast_mean, 1e-6f)), 4.0f);
        const float adaptation_relative_variance = lerp(reprojected_fast_temporal_variance, wrong_point_estimate, fast_blend_factor);
        fast_variance_scaling = square(1.0f + sqrt(adaptation_relative_variance) * rtgi_settings.temporal_variance_fast_history_blend);

        const float slow_history_mean = reprojected_diffuse.w;
        const float slow_to_fast_mean_ratio = max(slow_history_mean, accumulated_fast_mean) / (min(slow_history_mean, accumulated_fast_mean) + 0.00000001f);
        const float relevant_fast_to_slow_mean_ratio = max(1.0f, slow_to_fast_mean_ratio);
        fast_mean_diff_scaling = square(1.0f / relevant_fast_to_slow_mean_ratio);
    }

    // Debug tape (center pixel only): fast-history mean +/- its std dev, plus the slow-history
    // radiance. fast_std_dev_relative is relative to the mean, so absolute std = mean * relative.
    //   x (red)    = fast mean + std dev
    //   y (green)  = fast mean - std dev
    //   z (blue)   = fast mean
    //   w (yellow) = slow-history radiance (brightness)
    const uint2 tape_center_pixel = uint2(half_res_render_target_size) / 2;
    if (all(dtid == tape_center_pixel))
    {
        const float fast_std_dev_absolute = accumulated_fast_mean * reprojected_fast_temporal_variance;
        push.attach.globals.readback.debug_value = float4(
            reprojected_diffuse.w,
            accumulated_fast_mean,
            fast_std_dev_absolute,
        0);
    }

    const float specular_rays_shot = float(push.attach.specular_ray_count_image.get()[dtid.xy]);
    const RtgiSpecularAccumulation specular_accumulation = rtgi_temporal_accumulate_specular(
        dtid, specular_rays_shot, disocclusion, reproject_gather_uv, sample_weights, half_res_render_target_size);
    push.attach.half_res_specular_accumulated.get()[dtid.xy] = specular_accumulation.specular;
    {
        // Convergence statistics: history length relative to the max history, per signal (every geometry pixel
        // passes here, including the no-ray path below). One atomic triple per wave.
        const float diffuse_convergence = saturate(accumulated_sample_count / max(float(rtgi_settings.max_temporal_samples), 1.0f));
        const float specular_convergence = saturate(specular_accumulation.frames / max(rtgi_settings.specular_max_temporal_frames, 1.0f));
        const uint wave_diffuse = WaveActiveSum(uint(diffuse_convergence * RTGI_CONVERGENCE_SCALE + 0.5f));
        const uint wave_specular = WaveActiveSum(uint(specular_convergence * RTGI_CONVERGENCE_SCALE + 0.5f));
        const uint wave_pixels = WaveActiveCountBits(true);
        if (WaveIsFirstLane())
        {
            InterlockedAdd(push.attach.ray_counters->convergence_diffuse_sum, wave_diffuse);
            InterlockedAdd(push.attach.ray_counters->convergence_specular_sum, wave_specular);
            InterlockedAdd(push.attach.ray_counters->convergence_pixels, wave_pixels);
        }
        // Distribution: history relative to the RAY DEMAND target (fast convergence samples; specular capped by its
        // max frames like rtgi_calc_ray_demand). Buckets 0..N-2 split [0, target) evenly, the last bucket holds
        // everything at or above the target (no extra rays requested any more).
        // Per bucket a wave ballot, one atomic per non-empty bucket per wave.
        const float diffuse_target = max(rtgi_settings.fast_convergence_samples, 1.0f);
        const float specular_target = max(min(rtgi_settings.specular_fast_convergence_samples, rtgi_settings.specular_max_temporal_frames), 1.0f);
        const uint below_target_buckets = RTGI_CONVERGENCE_BUCKETS - 1u;
        const uint bucket_diffuse = min(uint(max(accumulated_sample_count, 0.0f) / diffuse_target * float(below_target_buckets)), below_target_buckets);
        const uint bucket_specular = min(uint(max(specular_accumulation.frames, 0.0f) / specular_target * float(below_target_buckets)), below_target_buckets);
        for (uint b = 0u; b < RTGI_CONVERGENCE_BUCKETS; ++b)
        {
            const uint count_diffuse = WaveActiveCountBits(bucket_diffuse == b);
            const uint count_specular = WaveActiveCountBits(bucket_specular == b);
            if (WaveIsFirstLane())
            {
                if (count_diffuse > 0u) { InterlockedAdd(push.attach.ray_counters->convergence_histogram_diffuse[b], count_diffuse); }
                if (count_specular > 0u) { InterlockedAdd(push.attach.ray_counters->convergence_histogram_specular[b], count_specular); }
            }
        }
    }
    push.attach.specular_frames_accumulated.get()[dtid.xy] = specular_accumulation.frames;
    push.attach.specular_fast_history_accumulated.get()[dtid.xy] = specular_accumulation.fast;
    {
        let debug_mode = push.attach.globals.settings.debug_draw_mode;
        const float debug_alpha = 1.0f + push.attach.globals.settings.debug_visualization_blend;
        if (debug_mode == DEBUG_DRAW_MODE_RTGI_SPECULAR_HISTORY_LENGTH)
        {
            // Relative to the specular fast convergence target (ray demand target, capped by the max frames).
            const float t = saturate(specular_accumulation.frames * rcp(max(min(rtgi_settings.specular_fast_convergence_samples, rtgi_settings.specular_max_temporal_frames), 1.0f)));
            write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(t), debug_alpha), 2);
        }
        else if (debug_mode == DEBUG_DRAW_MODE_RTGI_SPECULAR_TEMPORAL_REACTIVITY)
        {
            write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(specular_accumulation.blend), debug_alpha), 2);
        }
        else if (debug_mode == DEBUG_DRAW_MODE_RTGI_SPECULAR_HIT_DISTANCE_TEMPORAL)
        {
            const float hit = specular_accumulation.specular.a;
            write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(rtgi_hit_distance_debug_color(hit), debug_alpha), 2);
        }
        else if (debug_mode == DEBUG_DRAW_MODE_RTGI_SPECULAR_PERCEPTUAL_MEAN_TEMPORAL)
        {
            const float3 rgb = specular_accumulation.specular.rgb;
            const float perceptual = linear_to_perceptual(dot(rgb, float3(0.25f, 0.5f, 0.25f)), push.attach.globals.inv_exposure);
            write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(perceptual_radiance_colormap(perceptual, push.attach.globals.exposure), debug_alpha + 1.0f), 2);
        }
        else if (debug_mode == DEBUG_DRAW_MODE_RTGI_SPECULAR_VIRTUAL_AMOUNT)
        {
            write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(specular_accumulation.virtual_amount), debug_alpha), 2);
        }
    }

    // No-ray pixel (repacked dispatch only): this geometry pixel received no ray from the budget this
    // frame, so there is no new radiance to integrate. As long as we have valid history (not a
    // disocclusion), keep it 100% — write the reprojected history straight through and add nothing.
    // rays_shot_virtual_samples is authored 0 by the blend pass for such pixels; the classic per-pixel trace always
    // shoots >= 1 ray on geometry, so this branch never triggers there.
    if (rays_shot_virtual_samples == 0.0f && !disocclusion)
    {
        // No new sample integrated this frame, so the fast history mean is unchanged — carry the fast
        // frame count as-is (don't advance confidence for a frame that added no fast observation), but
        // still apply the parallax stretch penalty so a stretched no-ray pixel drops its smeared history.
        push.attach.half_res_sample_count.get()[dtid.xy] = rtgi_pack_sample_counts(accumulated_sample_count, reprojected_fast_frames * (1.0f - parallax_penalty)); // == reprojected carry, aged above max_temporal_samples (rays_shot_virtual_samples == 0)
        push.attach.half_res_diffuse_accumulated.get()[dtid.xy] = reprojected_diffuse;
        push.attach.half_res_diffuse2_accumulated.get()[dtid.xy] = reprojected_diffuse2;
        push.attach.fast_temporal_history_accumulated.get()[dtid] = float2(reprojected_fast_temporal_mean, reprojected_fast_temporal_variance);
        push.attach.half_res_ao_guide_accumulated.get()[dtid.xy] = reprojected_ao_guide;
        push.attach.temporal_perceptual_radiance_accumulated.get()[dtid.xy] = reprojected_perceptual_radiance;
        // Draw debug overlays here too — this path returns before the main draw below, and skipping it
        // is exactly what left holes on no-ray pixels in the temporal debug views. blend = 0 (no new
        // sample integrated this frame).
        rtgi_temporal_accumulate_debug_draw(dtid, accumulated_sample_count, 0.0f, reprojected_ao_guide, reprojected_perceptual_radiance);
        return;
    }

    const float max_sample_count = rtgi_settings.max_temporal_samples;
    // Accumulate Color
    float history_confidence = accumulated_sample_count;
    float fast_history_based_confidence = history_confidence * fast_variance_scaling * fast_mean_diff_scaling;
    if (accumulated_sample_count > FAST_HISTORY_FRAMES)
    {
        history_confidence = min(accumulated_sample_count * 1.0f, fast_history_based_confidence);
    }
    // Batch temporal integration: this frame contributed `rays_shot_virtual_samples` fresh samples for this pixel (the
    // blend pass already averaged them into new_diffuse), not just one. A running average that grows from
    // N to N+k prior samples weights the new batch mean by k/(N+k), NOT 1/(N+k). history_confidence plays
    // the role of the effective prior count (+1 regularization), so the correct batch weight is
    // rays_shot_virtual_samples/(1+history_confidence) — which collapses to the old 1/(1+history_confidence) at rays_shot_virtual_samples==1.
    // Without the rays_shot_virtual_samples factor, pixels that trace multiple rays (adaptive/redistributed budget) got the
    // SAME per-frame weight as single-ray pixels and never converged any faster. Clamp to 1 since the
    // variance scaling above can push history_confidence below rays_shot_virtual_samples.
    float blend = min(1.0f, float(rays_shot_virtual_samples) / (1.0f + history_confidence));
    float co_cg_blend = blend;
    if (!rtgi_settings.temporal_accumulation_enabled)
    {
        blend = 1.0f;
        co_cg_blend = 1.0f;
    }

    // Determine accumulated diffuse
    float4 accumulated_diffuse = disocclusion ? new_diffuse : lerp(reprojected_diffuse, new_diffuse, blend);
    float2 accumulated_diffuse2 = disocclusion ? new_diffuse2 : lerp(reprojected_diffuse2, new_diffuse2, co_cg_blend);

    // Guides ramp on the FAST history first (quick reaction while it fills), then switch to the normal
    // (slow) color blend once the fast-history window is full (accumulated_fast_frames >= FAST_HISTORY_FRAMES).
    const float guide_blend = accumulated_fast_frames < FAST_HISTORY_FRAMES ? fast_blend_factor : blend;

    // Determine accumulated ambient occlusion guide — same blend as color so it converges and stops boiling.
    // (Previously floored at 0.033, which kept injecting 3.3% fresh noisy guide every frame forever.)
    float accumulated_ao_guide = disocclusion ? new_ao_guide : lerp(reprojected_ao_guide, new_ao_guide, guide_blend);

    // write_debug_image(push.attach.debug_image.get(), 0, dtid, float4(Heatmap(accumulated_sample_count * rcp(64)), 2.0f), 2);

    // Temporal geometric mean: EMA of log(radiance) — same blend as color
    float new_perceptual_radiance = push.attach.perceptual_radiance_new.get()[dtid.xy];
    const bool invalid_new_perceptual_radiance = isinf(new_perceptual_radiance);
    if (invalid_new_perceptual_radiance)
    {
        new_perceptual_radiance = 0.0f;
    }
    const float accumulated_perceptual_radiance = disocclusion ? new_perceptual_radiance : lerp(reprojected_perceptual_radiance, new_perceptual_radiance, invalid_new_perceptual_radiance ? 0.0f : guide_blend);

    // Write Textures
    push.attach.half_res_sample_count.get()[dtid.xy] = rtgi_pack_sample_counts(accumulated_sample_count, accumulated_fast_frames); // carry + rays shot, for next frame's reproject
    push.attach.half_res_diffuse_accumulated.get()[dtid.xy] = accumulated_diffuse;
    push.attach.half_res_diffuse2_accumulated.get()[dtid.xy] = accumulated_diffuse2;
    push.attach.fast_temporal_history_accumulated.get()[dtid] = float2(accumulated_fast_mean, accumulated_fast_relative_variance);
    push.attach.half_res_ao_guide_accumulated.get()[dtid.xy] = accumulated_ao_guide;
    push.attach.temporal_perceptual_radiance_accumulated.get()[dtid.xy] = accumulated_perceptual_radiance;

    rtgi_temporal_accumulate_debug_draw(dtid, accumulated_sample_count, blend, accumulated_ao_guide, accumulated_perceptual_radiance);
}
