#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/geometry.inl"
#include "../shader_lib/vsm_util.glsl"

/// ===== Shader Only Geometry Data and functions =====

struct TriangleGeometry
{
    uint entity_index;
    uint mesh_index;
    uint material_index;
    uint in_mesh_group_index;
    uint mesh_instance_index;
    uint3 vertex_indices;
    float3 barycentrics;
};

struct TriangleGeometryPoint
{
    float3 world_position;
    float3 world_tangent;
    float3 world_bitangent;
    float3 world_normal;
    float3 face_normal;
    daxa_f32vec2 uv;
    daxa_f32vec2 uv_ddx;    // Only roughly approximated in rt
    daxa_f32vec2 uv_ddy;    // Only roughly approximated in rt
};

#define MATERIAL_FLAG_NONE (0u)
#define MATERIAL_FLAG_ALPHA_DISCARD (1u << 0u)
#define MATERIAL_FLAG_DOUBLE_SIDED (1u << 1u)
#define MATERIAL_FLAG_BLEND (1u << 2u)

struct MaterialPointData
{
    float3 emissive;
    float3 albedo;
    float alpha;
    float3 normal;
    float3 geometry_normal;
    float3 face_normal;
    float3 position;
    uint material_flags;
};

interface LightVisibilityTesterI
{
    float sun_light(MaterialPointData material_point, float3 incoming_ray);
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index);
    float spot_light(MaterialPointData material_point, float3 incoming_ray, uint light_index);
};

float3 flip_normal_to_incoming(float3 face_normal, float3 normal, float3 incoming_ray)
{
    return sign(dot(face_normal, -incoming_ray)) * normal;
}

void geom_compute_uv_tangent(float3 tri_vert_positions[3], float2 tri_vert_uvs[3], out float3 ret_tangent, out float3 ret_bitangent)
{
    float3 d_p1 = tri_vert_positions[1].xyz - tri_vert_positions[0].xyz;
    float3 d_p2 = tri_vert_positions[2].xyz - tri_vert_positions[0].xyz;
    float2 d_uv1 = tri_vert_uvs[1] - tri_vert_uvs[0];
    float2 d_uv2 = tri_vert_uvs[2] - tri_vert_uvs[0];
    float r = 1.0f / (d_uv1.x * d_uv2.y - d_uv1.y * d_uv2.x);
    ret_tangent = normalize((d_p1 * d_uv2.y - d_p2 * d_uv1.y) * r);
    ret_bitangent = normalize((-d_p1 * d_uv2.y + d_p2 * d_uv1.y) * r);
}

float3 geom_compute_arb_tangent(float3 tri_vert_positions[3])
{
    return normalize(tri_vert_positions[1] - tri_vert_positions[0]);
}