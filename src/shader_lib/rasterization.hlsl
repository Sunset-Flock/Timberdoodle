#pragma once

#include "daxa/daxa.inl"
#include "misc.hlsl"

#define SUBPIXEL_BITS 12
#define SUBPIXEL_SAMPLES (1 << SUBPIXEL_BITS)

void atomic_visbuffer_write(RWTexture2D<uint64_t> atomic_visbuffer, int2 index, uint32_t triangle_id, float depth) {
    const daxa::u64 visdepth = (daxa::u64(asuint(depth)) << 32) | daxa::u64(triangle_id);
    AtomicMaxU64(atomic_visbuffer[index], visdepth);
}

void rasterize_triangle(RWTexture2D<uint64_t> atomic_visbuffer, in float3[3] triangle, int2 viewport_size, uint32_t triangle_id) {
    const float3 v01 = triangle[1].xyz - triangle[0].xyz;
    const float3 v02 = triangle[2].xyz - triangle[0].xyz;
    const float det_xy = v01.x * v02.y - v01.y * v02.x;
    if (det_xy >= 0.0) {
        return;
    }

    const float inv_det = 1.0 / det_xy;
    float2 grad_z = float2(
        (v01.z * v02.y - v01.y * v02.z) * inv_det,
        (v01.x * v02.z - v01.z * v02.x) * inv_det);

    float2 vert_0 = triangle[0].xy;
    float2 vert_1 = triangle[1].xy;
    float2 vert_2 = triangle[2].xy;

    const float2 min_subpixel = min(min(vert_0, vert_1), vert_2);
    const float2 max_subpixel = max(max(vert_0, vert_1), vert_2);

    int2 min_pixel = int2(floor((min_subpixel + (SUBPIXEL_SAMPLES / 2) - 1) * (1.0 / float(SUBPIXEL_SAMPLES))));
    int2 max_pixel = int2(floor((max_subpixel - (SUBPIXEL_SAMPLES / 2) - 1) * (1.0 / float(SUBPIXEL_SAMPLES))));

    min_pixel = max(min_pixel, (int2)0);
    max_pixel = min(max_pixel, viewport_size.xy - 1);
    if (any(min_pixel > max_pixel)) {
        return;
    }

    max_pixel = min(max_pixel, min_pixel + 63);

    const float2 edge_01 = -v01.xy;
    const float2 edge_12 = vert_1 - vert_2;
    const float2 edge_20 = v02.xy;

    const float2 base_subpixel = float2(min_pixel) * SUBPIXEL_SAMPLES + (SUBPIXEL_SAMPLES / 2);
    vert_0 -= base_subpixel;
    vert_1 -= base_subpixel;
    vert_2 -= base_subpixel;

    float hec_0 = edge_01.y * vert_0.x - edge_01.x * vert_0.y;
    float hec_1 = edge_12.y * vert_1.x - edge_12.x * vert_1.y;
    float hec_2 = edge_20.y * vert_2.x - edge_20.x * vert_2.y;

    hec_0 -= saturate(edge_01.y + saturate(1.0 - edge_01.x));
    hec_1 -= saturate(edge_12.y + saturate(1.0 - edge_12.x));
    hec_2 -= saturate(edge_20.y + saturate(1.0 - edge_20.x));

    const float z_0 = triangle[0].z - (grad_z.x * vert_0.x + grad_z.y * vert_0.y);
    grad_z *= SUBPIXEL_SAMPLES;

    float hec_y_0 = hec_0 * (1.0 / float(SUBPIXEL_SAMPLES));
    float hec_y_1 = hec_1 * (1.0 / float(SUBPIXEL_SAMPLES));
    float hec_y_2 = hec_2 * (1.0 / float(SUBPIXEL_SAMPLES));
    float z_y = z_0;

    if (WaveActiveAnyTrue(max_pixel.x - min_pixel.x > 4)) {
        const float3 edge_012 = float3(edge_01.y, edge_12.y, edge_20.y);
        const bool3 is_open_edge = edge_012 < float3(0.0);
        const float3 inv_edge_012 = float3(
            edge_012.x == 0 ? 1e8 : (1.0 / edge_012.x),
            edge_012.y == 0 ? 1e8 : (1.0 / edge_012.y),
            edge_012.z == 0 ? 1e8 : (1.0 / edge_012.z));
        int y = min_pixel.y;
        while (true) {
            const float3 cross_x = float3(hec_y_0, hec_y_1, hec_y_2) * inv_edge_012;
            const float3 min_x = float3(
                is_open_edge.x ? cross_x.x : 0.0,
                is_open_edge.y ? cross_x.y : 0.0,
                is_open_edge.z ? cross_x.z : 0.0);
            const float3 max_x = float3(
                is_open_edge.x ? max_pixel.x - min_pixel.x : cross_x.x,
                is_open_edge.y ? max_pixel.x - min_pixel.x : cross_x.y,
                is_open_edge.z ? max_pixel.x - min_pixel.x : cross_x.z);
            float x_0 = ceil(max(max(min_x.x, min_x.y), min_x.z));
            float x_1 = min(min(max_x.x, max_x.y), max_x.z);
            float z_x = z_y + grad_z.x * x_0;

            x_0 += min_pixel.x;
            x_1 += min_pixel.x;
            for (float x = x_0; x <= x_1; ++x) {
                atomic_visbuffer_write(atomic_visbuffer, int2(x, y), triangle_id, z_x);
                z_x += grad_z.x;
            }

            if (y >= max_pixel.y) {
                break;
            }
            hec_y_0 += edge_01.x;
            hec_y_1 += edge_12.x;
            hec_y_2 += edge_20.x;
            z_y += grad_z.y;
            ++y;
        }
    } else {
        int y = min_pixel.y;
        while (true) {
            int x = min_pixel.x;
            if (min(min(hec_y_0, hec_y_1), hec_y_2) >= 0.0) {
                atomic_visbuffer_write(atomic_visbuffer, int2(x, y), triangle_id, z_y);
            }

            if (x < max_pixel.x) {
                float hec_x_0 = hec_y_0 - edge_01.y;
                float hec_x_1 = hec_y_1 - edge_12.y;
                float hec_x_2 = hec_y_2 - edge_20.y;
                float z_x = z_y + grad_z.x;
                ++x;

                while (true) {
                    if (min(min(hec_x_0, hec_x_1), hec_x_2) >= 0.0) {
                        atomic_visbuffer_write(atomic_visbuffer, int2(x, y), triangle_id, z_x);
                    }

                    if (x >= max_pixel.x) {
                        break;
                    }

                    hec_x_0 -= edge_01.y;
                    hec_x_1 -= edge_12.y;
                    hec_x_2 -= edge_20.y;
                    z_x += grad_z.x;
                    ++x;
                }
            }

            if (y >= max_pixel.y) {
                break;
            }

            hec_y_0 += edge_01.x;
            hec_y_1 += edge_12.x;
            hec_y_2 += edge_20.x;
            z_y += grad_z.y;
            ++y;
        }
    }
}


// Unused. Keep code for future reference if we need to compute rasterize again.
#if 0
groupshared float4 gs_clip_vertex_positions[MAX_VERTICES_PER_MESHLET];
func generic_mesh_compute_raster(
    DrawVisbufferPush push,
    in GPUMesh mesh,
    in uint meshlet_thread_index,
    in uint meshlet_instance_index,
    in MeshletInstance meshlet_instance,
    in bool cull_backfaces,
    in bool cull_hiz_occluded)
{
    const GPUMesh mesh = deref_i(push.meshes, meshlet_instance.mesh_index);
    if (mesh.mesh_buffer.value == 0) // Unloaded Mesh
    {
        return;
    }
    const Meshlet meshlet = deref_i(mesh.meshlets, meshlet_instance.meshlet_index);
    daxa_BufferPtr(daxa_u32) micro_index_buffer = deref_i(push.meshes, meshlet_instance.mesh_index).micro_indices;
    const bool observer_pass = push.draw_data.observer;
    const bool visbuffer_two_pass_cull = push.attach.globals.settings.enable_visbuffer_two_pass_culling;
    cull_hiz_occluded = cull_hiz_occluded && !(observer_pass && !visbuffer_two_pass_cull);
    const daxa_f32mat4x4 view_proj = 
        observer_pass ? 
        deref(push.attach.globals).view_camera.view_proj : 
        deref(push.attach.globals).main_camera.view_proj;

    if (meshlet_instance_index >= MAX_MESHLET_INSTANCES)
    {
        printf("GPU ERROR: Invalid meshlet passed to mesh shader! Meshlet instance index %i exceeded max meshlet instance count %i\n", meshlet_instance_index, MAX_MESHLET_INSTANCES);
    }

    const daxa_f32mat4x3 model_mat4x3 = deref_i(push.entity_combined_transforms, meshlet_instance.entity_index);
    const daxa_f32mat4x4 model_mat = mat_4x3_to_4x4(model_mat4x3);
    {
        const uint in_meshlet_vertex_index = meshlet_thread_index;
        if (in_meshlet_vertex_index < meshlet.vertex_count)
        {
            // Very slow fetch, as its incoherent memory address across warps.
            const uint in_mesh_vertex_index = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + in_meshlet_vertex_index);
            if (in_mesh_vertex_index < mesh.vertex_count)
            {
                // Very slow fetch, as its incoherent memory address across warps.
                const daxa_f32vec4 vertex_position = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index), 1);
                const daxa_f32vec4 pos = mul(view_proj, mul(model_mat, vertex_position));

                gs_clip_vertex_positions[in_meshlet_vertex_index] = pos;
            }
            else
            {
                gs_clip_vertex_positions[in_meshlet_vertex_index] = float4(-1,-1,-1,-1);
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    {
        const uint in_meshlet_triangle_index = meshlet_thread_index;
        uint3 tri_in_meshlet_vertex_indices = uint3(0,0,0);
        if (in_meshlet_triangle_index < meshlet.triangle_count)
        {
            tri_in_meshlet_vertex_indices = uint3(
                get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 0),
                get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 1),
                get_micro_index(micro_index_buffer, meshlet.micro_indices_offset + in_meshlet_triangle_index * 3 + 2)
            );
        }
        float4[3] tri_vert_clip_positions = float4[3](
            gs_clip_vertex_positions[tri_in_meshlet_vertex_indices[0]],
            gs_clip_vertex_positions[tri_in_meshlet_vertex_indices[1]],
            gs_clip_vertex_positions[tri_in_meshlet_vertex_indices[2]]
        );

        if (in_meshlet_triangle_index < meshlet.triangle_count)
        {
            // From: https://zeux.io/2023/04/28/triangle-backface-culling/#fnref:3
            bool cull_primitive = false;

            // Observer culls triangles from the perspective of the main camera.
            if (push.draw_data.observer)
            {        
                for (uint c = 0; c < 3; ++c)
                {
                    const uint in_mesh_vertex_index = deref_i(mesh.indirect_vertices, meshlet.indirect_vertex_offset + tri_in_meshlet_vertex_indices[c]);
                    const daxa_f32vec4 vertex_position = daxa_f32vec4(deref_i(mesh.vertex_positions, in_mesh_vertex_index), 1);
                    let main_camera_view_proj = push.attach.globals.main_camera.view_proj;
                    const daxa_f32vec4 pos = mul(main_camera_view_proj, mul(model_mat, vertex_position));
                    tri_vert_clip_positions[c] = pos;
                }
            }

            if (push.attach.globals.settings.enable_triangle_cull)
            {
                if (cull_backfaces)
                {
                    cull_primitive = is_triangle_backfacing(tri_vert_clip_positions);
                }
                if (!cull_primitive)
                {
                    const float3[3] tri_vert_ndc_positions = float3[3](
                        tri_vert_clip_positions[0].xyz / (tri_vert_clip_positions[0].w),
                        tri_vert_clip_positions[1].xyz / (tri_vert_clip_positions[1].w),
                        tri_vert_clip_positions[2].xyz / (tri_vert_clip_positions[2].w)
                    );

                    float2 ndc_min = min(min(tri_vert_ndc_positions[0].xy, tri_vert_ndc_positions[1].xy), tri_vert_ndc_positions[2].xy);
                    float2 ndc_max = max(max(tri_vert_ndc_positions[0].xy, tri_vert_ndc_positions[1].xy), tri_vert_ndc_positions[2].xy);
                    let cull_micro_poly_invisible = is_triangle_invisible_micro_triangle( ndc_min, ndc_max, float2(push.attach.globals.settings.render_target_size));
                    cull_primitive = cull_micro_poly_invisible;

                    const float2 ndc_size = (ndc_max - ndc_min);
                    const float2 ndc_pixel_size = 0.5f * ndc_size * push.attach.globals.settings.render_target_size;
                    const float ndc_pixel_area_size = ndc_pixel_size.x * ndc_pixel_size.y;
                    bool large_triangle = ndc_pixel_area_size > 128;
                    if (large_triangle && push.attach.globals.settings.enable_triangle_cull && (push.attach.hiz.value != 0) && !cull_primitive && cull_hiz_occluded)
                    {
                        let is_hiz_occluded = is_triangle_hiz_occluded(
                            push.attach.globals.debug,
                            push.attach.globals.main_camera,
                            tri_vert_ndc_positions,
                            push.attach.globals.cull_data,
                            push.attach.hiz);
                        cull_primitive = is_hiz_occluded;
                    }
                }
            }
            
            if (!cull_primitive)
            {
                uint visibility_id = TRIANGLE_ID_MAKE(meshlet_instance_index, in_meshlet_triangle_index);

                const uint2 viewport_size = push.attach.globals.settings.render_target_size;
                const float2 scale = float2(0.5, 0.5) * float2(viewport_size) * float(SUBPIXEL_SAMPLES);
                const float2 bias = (0.5 * float2(viewport_size)) * float(SUBPIXEL_SAMPLES) + 0.5;

                float3[3] tri_vert_ndc_positions = float3[3](
                    tri_vert_clip_positions[0].xyz * rcp(tri_vert_clip_positions[0].w),
                    tri_vert_clip_positions[1].xyz * rcp(tri_vert_clip_positions[1].w),
                    tri_vert_clip_positions[2].xyz * rcp(tri_vert_clip_positions[2].w)
                );
                tri_vert_ndc_positions[0].xy = floor(tri_vert_ndc_positions[0].xy * scale + bias);
                tri_vert_ndc_positions[1].xy = floor(tri_vert_ndc_positions[1].xy * scale + bias);
                tri_vert_ndc_positions[2].xy = floor(tri_vert_ndc_positions[2].xy * scale + bias);

                rasterize_triangle(RWTexture2D<daxa::u64>::get_formatted(push.attach.atomic_visbuffer), tri_vert_ndc_positions, viewport_size, visibility_id);
            }
        }
    }
} 
#endif