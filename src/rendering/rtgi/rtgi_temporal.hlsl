#pragma once

#include "rtgi_temporal.inl"

#include "shader_lib/transform.hlsl"
#include "shader_lib/misc.hlsl"
#include "rtgi_shared.hlsl"
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

float2 rtgi_reproject_gather_uv(uint2 corner_plus_one, float2 inv_half_res_render_target_size)
{
    return float2(corner_plus_one) * inv_half_res_render_target_size; // == (origin + 1) * inv_size
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
    }
    GroupMemoryBarrierWithGroupSync();

    // Per-thread ray-demand contribution to this tile. Out-of-bounds and sky pixels contribute 0 and
    // are excluded from the geometry count. ALL threads MUST reach the barrier further below, so this
    // uses structured control flow instead of early returns — early returns would deadlock the barrier
    // on any tile that mixes sky / geometry / out-of-bounds pixels (i.e. nearly every silhouette tile).
    uint thread_desired_total = 0u;
    uint thread_geo_inc = 0u;

    const uint2 halfres_pixel_index = dtid;
    if (!any(dtid.xy >= push.size))
    {
    // Load and precalculate constants
    const CameraInfo camera = push.attach.globals->view_camera;
    const float2 half_res_render_target_size = push.attach.globals.settings.render_target_size.xy >> 1;
    const float2 inv_half_res_render_target_size = rcp(half_res_render_target_size);

    const PixelData pixel = calc_pixel_data(dtid, inv_half_res_render_target_size, camera, push.attach.half_res_depth.get(), push.attach.half_res_normal.get());
    const float pixel_width_ws = calc_pixel_width_ws(inv_half_res_render_target_size, camera.near_plane, pixel.ndc.z);
    const float pixel_width_ws_rcp = rcp(pixel_width_ws);

    if (pixel.ndc.z == 0.0f)
    {
        // Sky: no valid history. A negative sample count is a sentinel that lets the accumulation
        // pass early-out on sky by reading only the sample count image (no separate depth fetch).
        // corner/weights are left unwritten since accumulation returns before reading them.
        push.attach.half_res_sample_count.get()[halfres_pixel_index] = rtgi_pack_sample_counts_sky();
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
            const float3 texel_ndc_prev_frame[4] = {
                float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(0,0)) * inv_half_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[0]),
                float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(1,0)) * inv_half_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[1]),
                float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(0,1)) * inv_half_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[2]),
                float3(float2(bilinear_filter_at_prev_pos.origin + 0.5f + float2(1,1)) * inv_half_res_render_target_size * 2.0f - 1.0f, depth_reprojected4[3]),
            };
            const float4 texel_ws_prev_frame_pre_div[4] = {
                mul(previous_camera.inv_view_proj, float4(texel_ndc_prev_frame[0], 1.0f)),
                mul(previous_camera.inv_view_proj, float4(texel_ndc_prev_frame[1], 1.0f)),
                mul(previous_camera.inv_view_proj, float4(texel_ndc_prev_frame[2], 1.0f)),
                mul(previous_camera.inv_view_proj, float4(texel_ndc_prev_frame[3], 1.0f)),
            };
            const float3 texel_ws_prev_frame[4] = {
                texel_ws_prev_frame_pre_div[0].xyz / texel_ws_prev_frame_pre_div[0].w,
                texel_ws_prev_frame_pre_div[1].xyz / texel_ws_prev_frame_pre_div[1].w,
                texel_ws_prev_frame_pre_div[2].xyz / texel_ws_prev_frame_pre_div[2].w,
                texel_ws_prev_frame_pre_div[3].xyz / texel_ws_prev_frame_pre_div[3].w,
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
    float reprojected_history_count = disocclusion ? 0.0f : min(rtgi_settings.max_temporal_samples, reprojected_sample_count);

    // Parallax stretch: drop history where a grazing surface's thin previous-frame strip is being
    // stretched across many new pixels this frame (e.g. a wall revealed by lateral motion), so those
    // pixels re-converge with fresh samples instead of smearing. Only the ultra-strong (>~4x) cases bite.
    float parallax_penalty = 0.0f;
    if (rtgi_settings.temporal_parallax_penalty_strength > 0.0f)
    {
        parallax_penalty = calc_parallax_penalty(
            pixel.position_ws, pixel.normal_ws, camera.position, previous_camera.position,
            rtgi_settings.temporal_parallax_penalty_strength);
        reprojected_history_count *= (1.0f - parallax_penalty);
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

    // This geometry pixel wants 1 base ray + its adaptive extras (shared with the allocate pass).
    thread_desired_total = calc_desired_ray_count(reprojected_history_count, rtgi_settings.fast_convergence_samples);
    thread_geo_inc = 1u;
    } // end else (geometry pixel)
    } // end if (in bounds)

    // Accumulate per-tile ray demand into groupshared, then thread 0 flushes to the buffers.
    // ALL threads reach this barrier (sky / oob contributed 0 above), so it is uniform.
    InterlockedAdd(gs_tile_total_desired, thread_desired_total);
    InterlockedAdd(gs_tile_geo_count, thread_geo_inc);

    GroupMemoryBarrierWithGroupSync();

    if (gtid.x == 0 && gtid.y == 0)
    {
        const uint tile_extra = gs_tile_total_desired - gs_tile_geo_count;
        if (tile_extra > 0u) InterlockedAdd(push.attach.ray_counters->total_extra_rays, tile_extra);
        if (gs_tile_geo_count > 0u) InterlockedAdd(push.attach.ray_counters->total_geo_rays, gs_tile_geo_count);
    }
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
    if (debug_mode == DEBUG_DRAW_MODE_RTGI_HISTORY_LENGTH)
    {
        const float t = sample_count * rcp(float(rtgi_settings.max_temporal_samples));
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(t), debug_alpha), 2);
    }
    else if (debug_mode == DEBUG_DRAW_MODE_RTGI_TEMPORAL_REACTIVITY)
    {
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(blend), debug_alpha), 2);
    }
    else if (debug_mode == DEBUG_DRAW_MODE_RTGI_AO_GUIDE_TEMPORAL)
    {
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(Heatmap(ao_guide), debug_alpha), 2);
    }
    else if (debug_mode == DEBUG_DRAW_MODE_RTGI_PERCEPTUAL_MEAN_TEMPORAL)
    {
        write_debug_image(dbg, push.attach.globals.settings.debug_visualization_tile, dtid, float4(perceptual_radiance_colormap(perceptual_radiance, push.attach.globals.exposure), debug_alpha + 1.0f), 2);
    }
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
    const float accumulated_sample_count = min(rtgi_settings.max_temporal_samples, reproj_carry_sample_count + rays_shot_virtual_samples);
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
        push.attach.half_res_sample_count.get()[dtid.xy] = rtgi_pack_sample_counts(accumulated_sample_count, reprojected_fast_frames * (1.0f - parallax_penalty)); // == reprojected carry (rays_shot_virtual_samples == 0)
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
