#pragma once

#include "rt_lab_gradient_oct.inl"

#include "rt_lab_shared.hlsl"
#include "shader_lib/debug.glsl"

// The exposed radiance floor of the log ratio and of the bins' variance: the gradient's one perceptual floor (rt_lab_shared.hlsl).
#define RT_LAB_GRADIENT_OCT_FLOOR RT_LAB_GRADIENT_PERCEPTUAL_FLOOR

// Directional pyramid of the temporal gradient re-trace samples, see rt_lab_gradient_oct.inl.

[[vk::push_constant]] RtLabGradientOctPush rt_lab_gradient_oct_push;

// Octahedral texel index of a direction.
func rt_lab_gradient_oct_texel(float3 direction) -> uint2
{
    return min(uint2(map_octahedral(direction) * float(RT_LAB_GRADIENT_OCT)), uint2(RT_LAB_GRADIENT_OCT - 1u, RT_LAB_GRADIENT_OCT - 1u));
}

func rt_lab_gradient_oct_texel_direction(uint2 texel) -> float3
{
    return unmap_octahedral((float2(texel) + 0.5f) / float(RT_LAB_GRADIENT_OCT));
}

// Solid angle of an octahedral texel relative to its neighbours: the map is area uniform on the octahedron
// |x| + |y| + |z| = 1, a surface element at p subtends dA cos(a) / |p|^2 which is proportional to (|dx| + |dy| + |dz|)^3.
func rt_lab_gradient_oct_solid_angle(float3 direction) -> float
{
    const float l1 = abs(direction.x) + abs(direction.y) + abs(direction.z);
    return l1 * l1 * l1;
}

// World space size of an area of this level at a position: the lab pixels it covers x the pixel width there x the tolerance scale.
func rt_lab_gradient_oct_footprint(CameraInfo camera, RtLabGradientOctPush push, float3 position) -> float
{
    const float distance = max(length(position - camera.position), 1e-3f);
    return max(float(push.area_pixels) * rt_lab_pixel_width_ws(camera, float(push.lab_height), distance) * push.distance_scale, 1e-4f);
}

// Gaussian of a world space distance.
func rt_lab_gradient_oct_distance_weight(float3 a, float3 b, float sigma) -> float
{
    const float3 difference = a - b;
    return exp(-0.5f * dot(difference, difference) / (sigma * sigma));
}

[shader("compute")]
[numthreads(RT_LAB_GRADIENT_OCT, RT_LAB_GRADIENT_OCT, 1)]
func entry_rt_lab_gradient_oct(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_gradient_oct_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const CameraInfo camera = push.attach.globals.view_camera;

    if (push.mode == RT_LAB_GRADIENT_OCT_MODE_POSITIONS_CELLS || push.mode == RT_LAB_GRADIENT_OCT_MODE_POSITIONS_AREAS)
    {
        // One thread per area: the smooth majority position of its children. Start at the weighted mean, then mean shift
        // iterations with a shrinking gaussian: the estimate slowly locks into the densest cluster and outliers lose their weight.
        float3 positions[16];
        float base_weights[16];
        float3 mean = float3(0.0f, 0.0f, 0.0f);
        float base_sum = 0.0f;
        [ForceUnroll]
        for (uint i = 0u; i < 16u; ++i)
        {
            const uint2 child = dtid * RT_LAB_GRADIENT_OCT_AREA + uint2(i % RT_LAB_GRADIENT_OCT_AREA, i / RT_LAB_GRADIENT_OCT_AREA);
            positions[i] = float3(0.0f, 0.0f, 0.0f);
            base_weights[i] = 0.0f;
            if (all(child < push.src_size))
            {
                const float4 value = push.attach.src_positions.get()[child];
                if (value.w > 0.0f)
                {
                    positions[i] = value.xyz;
                    base_weights[i] = value.w;
                    mean += value.xyz * value.w;
                    base_sum += value.w;
                }
            }
        }
        if (base_sum <= 0.0f)
        {
            push.attach.dst_positions.get()[dtid] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            return;
        }
        mean /= base_sum;
        const float sigma_start = rt_lab_gradient_oct_footprint(camera, push, mean);
        float weight_sum = base_sum;
        for (uint iteration = 0u; iteration < 3u; ++iteration)
        {
            const float sigma = sigma_start * (1.0f - 0.25f * float(iteration));
            float3 accumulated = float3(0.0f, 0.0f, 0.0f);
            float total = 0.0f;
            [ForceUnroll]
            for (uint i = 0u; i < 16u; ++i)
            {
                const float weight = base_weights[i] * (base_weights[i] > 0.0f ? rt_lab_gradient_oct_distance_weight(positions[i], mean, sigma) : 0.0f);
                accumulated += weight * positions[i];
                total += weight;
            }
            if (total > 1e-6f)
            {
                mean = accumulated / total;
                weight_sum = total;
            }
        }
        push.attach.dst_positions.get()[dtid] = float4(mean, weight_sum);
        return;
    }

    const uint2 area = dtid / uint2(RT_LAB_GRADIENT_OCT, RT_LAB_GRADIENT_OCT);
    const uint2 texel = dtid % uint2(RT_LAB_GRADIENT_OCT, RT_LAB_GRADIENT_OCT);

    if (push.mode == RT_LAB_GRADIENT_OCT_MODE_BIN_CELLS)
    {
        // This texel's share of the cell samples of the area, each weighted by the distance of the cell's surface point to the
        // area position (far off samples are not accepted).
        float4 sums = float4(0.0f, 0.0f, 0.0f, 0.0f);    // sum w old, sum w new, sum w, sum w^2
        float3 moments = float3(0.0f, 0.0f, 0.0f);       // sum w old^2, sum w old new, sum w new^2
        const float4 area_position = push.attach.area_positions.get()[area];
        if (area_position.w > 0.0f)
        {
            const float sigma = 0.5f * rt_lab_gradient_oct_footprint(camera, push, area_position.xyz);
            for (uint y = 0u; y < RT_LAB_GRADIENT_OCT_AREA; ++y)
            {
                for (uint x = 0u; x < RT_LAB_GRADIENT_OCT_AREA; ++x)
                {
                    const uint2 cell = area * RT_LAB_GRADIENT_OCT_AREA + uint2(x, y);
                    if (any(cell >= push.src_size))
                    {
                        continue;
                    }
                    const float4 sample = push.attach.src_cells.get()[cell];
                    const float4 cell_position = push.attach.src_positions.get()[cell];
                    if (sample.z <= 0.0f || cell_position.w <= 0.0f)
                    {
                        continue;
                    }
                    const float3 direction = unmap_octahedral(float2(sample.z - 1.0f, sample.w));
                    if (all(rt_lab_gradient_oct_texel(direction) == texel))
                    {
                        const float weight = rt_lab_gradient_oct_distance_weight(cell_position.xyz, area_position.xyz, sigma);
                        sums += float4(weight * sample.x, weight * sample.y, weight, weight * weight);
                        moments += weight * float3(sample.x * sample.x, sample.x * sample.y, sample.y * sample.y);
                    }
                }
            }
        }
        push.attach.dst_maps.get()[dtid] = sums;
        push.attach.dst_moments.get()[dtid] = float4(moments, 0.0f);
    }
    else if (push.mode == RT_LAB_GRADIENT_OCT_MODE_BIN_MAPS)
    {
        // The children's sums, each child weighted by the distance of its position to the area position.
        float4 sums = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 moments = float3(0.0f, 0.0f, 0.0f);
        const float4 area_position = push.attach.area_positions.get()[area];
        if (area_position.w > 0.0f)
        {
            const float sigma = 0.5f * rt_lab_gradient_oct_footprint(camera, push, area_position.xyz);
            for (uint y = 0u; y < RT_LAB_GRADIENT_OCT_AREA; ++y)
            {
                for (uint x = 0u; x < RT_LAB_GRADIENT_OCT_AREA; ++x)
                {
                    const uint2 child = area * RT_LAB_GRADIENT_OCT_AREA + uint2(x, y);
                    const uint2 child_texel = child * RT_LAB_GRADIENT_OCT + texel;
                    if (any(child_texel >= push.src_size))
                    {
                        continue;
                    }
                    const float4 child_position = push.attach.src_positions.get()[child];
                    if (child_position.w <= 0.0f)
                    {
                        continue;
                    }
                    const float weight = rt_lab_gradient_oct_distance_weight(child_position.xyz, area_position.xyz, sigma);
                    const float4 child_sums = push.attach.src_maps.get()[child_texel];
                    sums += float4(weight * child_sums.xyz, weight * weight * child_sums.w);
                    moments += weight * push.attach.src_moments.get()[child_texel].xyz;
                }
            }
        }
        push.attach.dst_maps.get()[dtid] = sums;
        push.attach.dst_moments.get()[dtid] = float4(moments, 0.0f);
    }
    else if (push.mode == RT_LAB_GRADIENT_OCT_MODE_DENOISE)
    {
        // Accumulates the sums of the same texel over the 5x5 neighbouring areas (the neighbours are different samples: the
        // weights add as w, the squared weights as w^2). A neighbour counts by a gaussian of the world space distance between
        // the two area positions (reach 2.5 area footprints: neighbours on the same surface weigh ~0.7 at the corners, probes on
        // another surface far away weigh 0). Off: copy.
        const float4 center_sums = push.attach.src_maps.get()[dtid];
        const float4 center_moments = push.attach.src_moments.get()[dtid];
        const float4 area_position = push.attach.area_positions.get()[area];
        if (push.denoise == 0u || area_position.w <= 0.0f)
        {
            push.attach.dst_maps.get()[dtid] = center_sums;
            push.attach.dst_moments.get()[dtid] = center_moments;
            return;
        }
        const int2 areas = int2(push.size / uint2(RT_LAB_GRADIENT_OCT, RT_LAB_GRADIENT_OCT));
        const float sigma = 2.5f * rt_lab_gradient_oct_footprint(camera, push, area_position.xyz);
        float4 sums = float4(0.0f, 0.0f, 0.0f, 0.0f);
        float3 moments = float3(0.0f, 0.0f, 0.0f);
        for (int dy = -RT_LAB_GRADIENT_OCT_DENOISE_RADIUS; dy <= RT_LAB_GRADIENT_OCT_DENOISE_RADIUS; ++dy)
        {
            for (int dx = -RT_LAB_GRADIENT_OCT_DENOISE_RADIUS; dx <= RT_LAB_GRADIENT_OCT_DENOISE_RADIUS; ++dx)
            {
                const int2 neighbor = int2(area) + int2(dx, dy);
                if (any(neighbor < 0) || any(neighbor >= areas))
                {
                    continue;
                }
                const float4 neighbor_position = push.attach.area_positions.get()[uint2(neighbor)];
                if (neighbor_position.w <= 0.0f)
                {
                    continue;
                }
                const float weight = rt_lab_gradient_oct_distance_weight(neighbor_position.xyz, area_position.xyz, sigma);
                const uint2 neighbor_texel = uint2(neighbor) * RT_LAB_GRADIENT_OCT + texel;
                const float4 neighbor_sums = push.attach.src_maps.get()[neighbor_texel];
                sums += float4(weight * neighbor_sums.xyz, weight * weight * neighbor_sums.w);
                moments += weight * push.attach.src_moments.get()[neighbor_texel].xyz;
            }
        }
        push.attach.dst_maps.get()[dtid] = sums;
        push.attach.dst_moments.get()[dtid] = float4(moments, 0.0f);
    }
    else
    {
        // Cosine convolution: the cosine weighted mean of the old and the new radiance over the texels that hold samples.
        // Both use the same texels, so the ratio is the change of the covered directions only. support = share of the
        // cosine lobe's weight that is covered by samples.
        const float3 center_direction = rt_lab_gradient_oct_texel_direction(texel);
        float old_sum = 0.0f;
        float new_sum = 0.0f;
        float covered_weight = 0.0f;
        float lobe_weight = 0.0f;
        float weight_total = 0.0f;
        float weight_squared_total = 0.0f;
        for (uint y = 0u; y < RT_LAB_GRADIENT_OCT; ++y)
        {
            for (uint x = 0u; x < RT_LAB_GRADIENT_OCT; ++x)
            {
                const uint2 bin = uint2(x, y);
                const float3 direction = rt_lab_gradient_oct_texel_direction(bin);
                const float weight = max(dot(center_direction, direction), 0.0f) * rt_lab_gradient_oct_solid_angle(direction);
                lobe_weight += weight;
                const float4 sums = push.attach.src_maps.get()[area * RT_LAB_GRADIENT_OCT + bin];
                if (sums.z > 0.0f && weight > 0.0f)
                {
                    old_sum += weight * sums.x / sums.z;
                    new_sum += weight * sums.y / sums.z;
                    covered_weight += weight;
                    weight_total += sums.z;
                    weight_squared_total += sums.w;
                }
            }
        }
        float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (covered_weight > 0.0f)
        {
            const float old_mean = old_sum / covered_weight;
            const float new_mean = new_sum / covered_weight;
            // Variance of ln(new / old): the ratio estimator r = new / old, the residual new - r old of every sample has mean 0 under the
            // fitted ratio, the mean square residual of a bin is (sum w (new - r old)^2) / sum w from the second moments, the variance of the
            // bin's weighted mean that times sum w^2 / (sum w)^2. The lobe mean is a sum of bin means with weights a_j, so var = sum a_j^2 v_j,
            // delta method: / new^2. One degree of freedom is spent on r: x n_eff / (n_eff - 1). Too few effective samples = no knowledge.
            const float ratio = new_mean / max(old_mean, RT_LAB_GRADIENT_OCT_FLOOR);
            float variance_sum = 0.0f;
            for (uint y = 0u; y < RT_LAB_GRADIENT_OCT; ++y)
            {
                for (uint x = 0u; x < RT_LAB_GRADIENT_OCT; ++x)
                {
                    const uint2 bin = uint2(x, y);
                    const float3 direction = rt_lab_gradient_oct_texel_direction(bin);
                    const float weight = max(dot(center_direction, direction), 0.0f) * rt_lab_gradient_oct_solid_angle(direction);
                    const float4 sums = push.attach.src_maps.get()[area * RT_LAB_GRADIENT_OCT + bin];
                    if (sums.z > 0.0f && weight > 0.0f)
                    {
                        const float3 moments = push.attach.src_moments.get()[area * RT_LAB_GRADIENT_OCT + bin].xyz;
                        const float mean_square = max(moments.z - 2.0f * ratio * moments.y + ratio * ratio * moments.x, 0.0f) / sums.z;
                        const float share = weight / covered_weight;
                        variance_sum += share * share * mean_square * sums.w / (sums.z * sums.z);
                    }
                }
            }
            const float effective_samples = weight_total * weight_total / max(weight_squared_total, 1e-12f);
            float variance = 1e6f;
            if (effective_samples > 2.0f)
            {
                const float new_floored = max(new_mean, RT_LAB_GRADIENT_OCT_FLOOR);
                variance = variance_sum * effective_samples / (effective_samples - 1.0f) / (new_floored * new_floored);
            }
            result = float4(old_mean, new_mean, covered_weight / lobe_weight, variance);
        }
        push.attach.dst_maps.get()[dtid] = result;
    }
}

// == Resolve + debug draw ===========================================================================================

[[vk::push_constant]] RtLabGradientOctResolvePush rt_lab_gradient_oct_resolve_push;

// Lighting change as a color: near black = unchanged, red = brighter, blue = darker, saturated at +-2 stops. Black = no samples.
func rt_lab_gradient_oct_color(float4 convolved) -> float3
{
    if (convolved.z <= 0.0f)
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    const float stops = clamp(log2(max(convolved.y, RT_LAB_GRADIENT_OCT_FLOOR) / max(convolved.x, RT_LAB_GRADIENT_OCT_FLOOR)) * 0.5f, -1.0f, 1.0f);
    const float3 grey = float3(0.03f, 0.03f, 0.03f);
    return stops > 0.0f ? lerp(grey, float3(1.0f, 0.15f, 0.05f), stops) : lerp(grey, float3(0.05f, 0.3f, 1.0f), -stops);
}

// Octahedral texel wrapping across the map edges (identical to octahedtral_texel_wrap in pgi_update.hlsl, which can not be
// included here): a texel just outside an edge is the mirrored texel on the other side, for taps at most one texel outside.
func rt_lab_gradient_oct_texel_wrap(int2 index, int resolution) -> int2
{
    if (index.y >= resolution)
    {
        // Flip y on the edge of the texture
        index.y = (resolution - 1) - (index.y - resolution);

        // Flip x on the middle of the texture.
        index.x = resolution - 1 - index.x;
    }

    if (index.y < 0)
    {
        index.y = -index.y - 1;
        index.x = resolution - 1 - index.x;
    }

    if (index.x >= resolution)
    {
        // Flip x on the edge of the texture
        index.x = (resolution - 1) - (index.x - resolution);

        // Flip y on the middle of the texture.
        index.y = resolution - 1 - index.y;
    }

    if (index.x < 0)
    {
        index.x = -index.x - 1;
        index.y = resolution - 1 - index.y;
    }

    return index;
}

func rt_lab_gradient_oct_load_convolved(uint level, uint2 texel) -> float4
{
    return level == 0u ? rt_lab_gradient_oct_resolve_push.attach.convolved0.get()[texel]
        : (level == 1u ? rt_lab_gradient_oct_resolve_push.attach.convolved1.get()[texel] : rt_lab_gradient_oct_resolve_push.attach.convolved2.get()[texel]);
}

func rt_lab_gradient_oct_load_position(uint level, uint2 area) -> float4
{
    return level == 0u ? rt_lab_gradient_oct_resolve_push.attach.positions0.get()[area]
        : (level == 1u ? rt_lab_gradient_oct_resolve_push.attach.positions1.get()[area] : rt_lab_gradient_oct_resolve_push.attach.positions2.get()[area]);
}

// Number of areas of a level.
func rt_lab_gradient_oct_area_count(uint level) -> int2
{
    uint width;
    uint height;
    if (level == 0u)
    {
        rt_lab_gradient_oct_resolve_push.attach.positions0.get().GetDimensions(width, height);
    }
    else if (level == 1u)
    {
        rt_lab_gradient_oct_resolve_push.attach.positions1.get().GetDimensions(width, height);
    }
    else
    {
        rt_lab_gradient_oct_resolve_push.attach.positions2.get().GetDimensions(width, height);
    }
    return int2(width, height);
}

// The area's convolved map at a direction: bilinear over 4 manual taps (wrapped at the octahedral map edges), weighted by their support.
// Returns (E old, E new, support sum (<= 1), variance of the log change), support 0 when no texel in reach has samples.
func rt_lab_gradient_oct_lookup(uint level, uint2 area, float3 direction) -> float4
{
    const float2 coordinate = map_octahedral(direction) * float(RT_LAB_GRADIENT_OCT) - 0.5f;
    const int2 base = int2(floor(coordinate));
    const float2 fraction = coordinate - float2(base);
    float old_sum = 0.0f;
    float new_sum = 0.0f;
    float variance_sum = 0.0f;
    float weight_sum = 0.0f;
    float support_sum = 0.0f;
    for (uint i = 0u; i < 4u; ++i)
    {
        const int2 offset = int2(i & 1u, i >> 1u);
        const float2 bilinear = lerp(1.0f - fraction, fraction, float2(offset));
        const int2 texel = rt_lab_gradient_oct_texel_wrap(base + offset, RT_LAB_GRADIENT_OCT);
        const uint2 map_texel = area * RT_LAB_GRADIENT_OCT + uint2(texel);
        const float4 convolved = rt_lab_gradient_oct_load_convolved(level, map_texel);
        const float weight = bilinear.x * bilinear.y * convolved.z;
        old_sum += weight * convolved.x;
        new_sum += weight * convolved.y;
        variance_sum += weight * convolved.w;
        weight_sum += weight;
        support_sum += bilinear.x * bilinear.y * convolved.z;
    }
    return weight_sum > 0.0f ? float4(old_sum / weight_sum, new_sum / weight_sum, support_sum, variance_sum / weight_sum) : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// The area's UNCONVOLVED map at a direction: the bin the direction falls into, as mean old / new radiance (old, new, 1, 0), z = 0 for an empty bin.
func rt_lab_gradient_oct_raw(uint level, uint2 area, float3 direction) -> float4
{
    const uint2 map_texel = area * RT_LAB_GRADIENT_OCT + rt_lab_gradient_oct_texel(direction);
    const float4 sums = level == 0u ? rt_lab_gradient_oct_resolve_push.attach.sums0.get()[map_texel]
        : (level == 1u ? rt_lab_gradient_oct_resolve_push.attach.sums1.get()[map_texel] : rt_lab_gradient_oct_resolve_push.attach.sums2.get()[map_texel]);
    return sums.z > 0.0f ? float4(sums.x / sums.z, sums.y / sums.z, 1.0f, 0.0f) : float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// Exposed radiance as a heatmap, log scale from 1e-4 to 16 (display white = 1). Black = empty.
func rt_lab_gradient_oct_radiance_color(float radiance, float valid) -> float3
{
    if (valid <= 0.0f)
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return Heatmap(saturate(log2(max(radiance, 1e-4f) / 1e-4f) / log2(16.0f / 1e-4f)));
}

[shader("compute")]
[numthreads(RT_LAB_RAYS_X, RT_LAB_RAYS_Y, 1)]
func entry_rt_lab_gradient_oct_resolve(uint2 dtid : SV_DispatchThreadID)
{
    let push = rt_lab_gradient_oct_resolve_push;
    if (any(dtid >= push.size))
    {
        return;
    }
    const CameraInfo camera = push.attach.globals.view_camera;
    const int mode = push.attach.globals.settings.debug_draw_mode;
    const uint resolution_scale = max(push.attach.globals.settings.render_target_size.x / push.size.x, 1u);

    // == Resolve: the shrunk lighting change at the pixel's normal: the highest valid level as baseline, finer levels can only enlarge it ==
    float4 result = float4(0.0f, 0.0f, 0.0f, 0.0f);
    const bool half_res = push.size.x != push.attach.globals.settings.render_target_size.x;
    uint packed_normal;
    const float depth = rt_lab_load_geometry(push.attach.view_cam_depth.get(), push.attach.view_cam_face_normals.get(),
        push.attach.view_cam_half_res_depth.get(), push.attach.view_cam_half_res_face_normals.get(), camera, half_res, dtid, packed_normal);
    if (push.resolve != 0u && depth > 0.0f)
    {
        const float3 position = linear_depth_to_world_space(camera, rt_lab_ndc(dtid, rcp(float2(push.size))), depth);
        const float3 normal = uncompress_normal_octahedral_32(packed_normal);
        const float pixel_width = rt_lab_pixel_width_ws(camera, float(push.lab_height), depth);
        bool any_valid = false;
        float best_change = 0.0f;
        float best_bias = 0.0f;
        float best_loss = 0.0f;
        float best_level = 0.0f;
        // Per level the result is the one of the 4 nearest areas that pass the world distance (position) test with the largest history adjustment
        // (loss = change^2 + bias, the max reaches over the areas instead of averaging them); the pixel takes the largest adjustment of the 3 levels
        // (a level without a passing area does not count; level 0 is only evaluated when levels 2 and 1 have no passing area). Output .x = the winner's shrunk change, .y = its bias: the cut is made of x^2 + y.
        for (int level_index = 2; level_index >= 0; --level_index)
        {
            const uint level = uint(level_index);
            if (level == 0u && any_valid)
            {
                // The finest level only when the coarser ones have nothing for this pixel (noisy: 16 samples per area).
                break;
            }
            const uint area_pixels = (RT_LAB_GRADIENT_CELL * RT_LAB_GRADIENT_OCT_AREA) << (2u * level);
            const float2 coordinate = (float2(dtid) + 0.5f) / float(area_pixels) - 0.5f;
            const int2 base = int2(floor(coordinate));
            const int2 areas = rt_lab_gradient_oct_area_count(level);
            const float sigma = max(float(area_pixels) * pixel_width * push.distance_scale, 1e-4f);
            bool level_valid = false;
            float level_change = 0.0f;
            float level_bias = 0.0f;
            float level_loss = 0.0f;
            for (uint i = 0u; i < 4u; ++i)
            {
                const int2 area = base + int2(i & 1u, i >> 1u);
                if (any(area < 0) || any(area >= areas))
                {
                    continue;
                }
                const float4 area_position = rt_lab_gradient_oct_load_position(level, uint2(area));
                if (area_position.w <= 0.0f || rt_lab_gradient_oct_distance_weight(position, area_position.xyz, sigma) < RT_LAB_GRADIENT_OCT_GATE_MIN)
                {
                    continue;
                }
                const float4 lookup = rt_lab_gradient_oct_lookup(level, uint2(area), normal);
                if (lookup.z < RT_LAB_GRADIENT_OCT_SUPPORT_MIN)
                {
                    continue;
                }
                const float change = log(max(lookup.y, RT_LAB_GRADIENT_OCT_FLOOR) / max(lookup.x, RT_LAB_GRADIENT_OCT_FLOOR));
                // James-Stein shrink: the squared change minus the variance of its estimate is the evidence of a real change.
                const float shrink = change * change > 1e-12f ? max(0.0f, 1.0f - lookup.w / (change * change)) : 0.0f;
                const float shrunk = change * shrink;
                const float bias = (change - shrunk) * (change - shrunk) + lookup.w;
                // The history cut is made of loss = change^2 + bias (rt_lab_gradient2_history_keep, monotonic): the largest loss is the largest adjustment.
                const float loss = shrunk * shrunk + bias;
                if (!level_valid || loss > level_loss)
                {
                    level_valid = true;
                    level_change = shrunk;
                    level_bias = bias;
                    level_loss = loss;
                }
            }
            if (level_valid && (!any_valid || level_loss > best_loss))
            {
                // The largest history adjustment (loss) of the 3 levels wins.
                any_valid = true;
                best_change = level_change;
                best_bias = level_bias;
                best_loss = level_loss;
                best_level = float(level);
            }
        }
        if (any_valid)
        {
            result = float4(clamp(best_change, -RT_LAB_GRADIENT_OCT_MAX_LOG, RT_LAB_GRADIENT_OCT_MAX_LOG), min(best_bias, 4.0f), best_level, 1.0f);
        }
    }
    if (push.resolve != 0u)
    {
        push.attach.gradient2.get()[dtid] = result;
    }

    // == Debug views ================================================================================================
    if (mode >= DEBUG_DRAW_MODE_RT_LAB_GRADIENT2_CHANGE && mode <= DEBUG_DRAW_MODE_RT_LAB_GRADIENT2_BIAS)
    {
        float3 color = float3(0.0f, 0.0f, 0.0f);
        if (result.w > 0.5f)
        {
            if (mode == DEBUG_DRAW_MODE_RT_LAB_GRADIENT2_CHANGE)
            {
                color = rt_lab_gradient_oct_color(float4(1.0f, exp(result.x), 1.0f, 0.0f));
            }
            else if (mode == DEBUG_DRAW_MODE_RT_LAB_GRADIENT2_LEVEL)
            {
                color = result.z < 0.5f ? float3(1.0f, 0.2f, 0.1f) : (result.z < 1.5f ? float3(0.2f, 1.0f, 0.2f) : float3(0.2f, 0.4f, 1.0f));
            }
            else
            {
                color = Heatmap(saturate(sqrt(result.y)));
            }
        }
        write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
            float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend), resolution_scale);
        return;
    }
    if (mode < DEBUG_DRAW_MODE_RT_LAB_GRADIENT_OCT_MAP0 || mode > DEBUG_DRAW_MODE_RT_LAB_GRADIENT_OCT_RAW_DIFFERENCE2)
    {
        return;
    }
    // The area maps as little balls: a half sphere facing the camera, centered in the area, each pixel shows the map at the world
    // space direction of the ball's surface normal there. Outside the ball nothing is drawn.
    // Views: MAP0..2 = the convolved map of level 0..2, RAW_BEFORE0..2 = the unconvolved mean old radiance, RAW_DIFFERENCE0..2 = the unconvolved ratio.
    const bool raw = mode >= DEBUG_DRAW_MODE_RT_LAB_GRADIENT_OCT_RAW_BEFORE0;
    const uint raw_index = uint(max(mode - DEBUG_DRAW_MODE_RT_LAB_GRADIENT_OCT_RAW_BEFORE0, 0));
    const bool raw_difference = raw && raw_index >= 3u;
    const uint ball_level = raw ? raw_index % 3u : uint(mode - DEBUG_DRAW_MODE_RT_LAB_GRADIENT_OCT_MAP0);
    uint ball_area_pixels = RT_LAB_GRADIENT_CELL * RT_LAB_GRADIENT_OCT_AREA;
    for (uint i = 0u; i < ball_level; ++i)
    {
        ball_area_pixels *= RT_LAB_GRADIENT_OCT_AREA;
    }
    const uint2 ball_area = dtid / ball_area_pixels;
    const float2 local = (float2(dtid) + 0.5f) - (float2(ball_area) + 0.5f) * float(ball_area_pixels);
    const float2 disc = local / (0.45f * float(ball_area_pixels));
    const float disc_length_squared = dot(disc, disc);
    if (disc_length_squared > 1.0f)
    {
        return;
    }
    const float height = sqrt(1.0f - disc_length_squared);
    // Camera basis from the view rays at the screen center and edges: forward, image right, image down.
    const float3 forward = normalize(linear_depth_to_world_space(camera, float2(0.0f, 0.0f), 1.0f) - camera.position);
    const float3 right = normalize(normalize(linear_depth_to_world_space(camera, float2(1.0f, 0.0f), 1.0f) - camera.position) -
                                   normalize(linear_depth_to_world_space(camera, float2(-1.0f, 0.0f), 1.0f) - camera.position));
    const float3 down = normalize(normalize(linear_depth_to_world_space(camera, float2(0.0f, 1.0f), 1.0f) - camera.position) -
                                  normalize(linear_depth_to_world_space(camera, float2(0.0f, -1.0f), 1.0f) - camera.position));
    const float3 ball_normal = normalize(right * disc.x + down * disc.y - forward * height);
    float3 color;
    if (raw)
    {
        const float4 bin = rt_lab_gradient_oct_raw(ball_level, ball_area, ball_normal);
        color = raw_difference ? rt_lab_gradient_oct_color(bin) : rt_lab_gradient_oct_radiance_color(bin.x, bin.z);
    }
    else
    {
        color = rt_lab_gradient_oct_color(rt_lab_gradient_oct_lookup(ball_level, ball_area, ball_normal));
    }
    color *= lerp(0.5f, 1.0f, height);
    write_debug_image(push.attach.debug_image.get(), push.attach.globals.settings.debug_visualization_tile, dtid,
        float4(color, 1.0f + push.attach.globals.settings.debug_visualization_blend), resolution_scale);
}
