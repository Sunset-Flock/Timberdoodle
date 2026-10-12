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
#define MATERIAL_FLAG_METAL (1u << 3u)

struct MaterialPointData
{
    float3 emissive;
    float3 albedo;
    float alpha;
    float roughness; // perceptual roughness [0,1] (glTF), alpha = roughness^2
    float metalness;
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

float3 flip_face_normal_to_incoming(float3 face_normal, float3 incoming_ray)
{
    return sign(dot(face_normal, -incoming_ray)) * face_normal;
}

float3 flip_normal_on_face_normal(float3 normal, float3 face_normal)
{
    return sign(dot(face_normal, normal)) * normal;
}

void geom_compute_uv_tangent(float3 tri_vert_positions[3], float2 tri_vert_uvs[3], out float3 ret_tangent, out float3 ret_bitangent)
{
    float3 d_p1 = tri_vert_positions[1].xyz - tri_vert_positions[0].xyz;
    float3 d_p2 = tri_vert_positions[2].xyz - tri_vert_positions[0].xyz;
    float2 d_uv1 = tri_vert_uvs[1] - tri_vert_uvs[0];
    float2 d_uv2 = tri_vert_uvs[2] - tri_vert_uvs[0];
    float det = d_uv1.x * d_uv2.y - d_uv1.y * d_uv2.x;
    // Only the sign of 1/det matters (normalized afterwards); dividing by a zero-area uv triangle gave inf/NaN tangents
    // -> NaN mapped normals -> black shading.
    float s = det < 0.0f ? -1.0f : 1.0f;
    float3 t = (d_p1 * d_uv2.y - d_p2 * d_uv1.y) * s;
    float3 b = (d_p2 * d_uv1.x - d_p1 * d_uv2.x) * s;
    float uv_scale = dot(d_uv1, d_uv1) + dot(d_uv2, d_uv2);
    float3 face_n = cross(d_p1, d_p2);
    // Degenerate uv triangle (zero area or collinear uvs): fall back to an arbitrary but orthonormal frame.
    if (abs(det) <= 1e-7f * uv_scale || dot(t, t) < 1e-30f || dot(b, b) < 1e-30f)
    {
        t = dot(d_p1, d_p1) > 1e-30f ? d_p1 : d_p2;
        b = cross(face_n, t);
        if (dot(b, b) < 1e-30f)
        {
            t = float3(1.0f, 0.0f, 0.0f);
            b = float3(0.0f, 1.0f, 0.0f);
        }
    }
    ret_tangent = normalize(t);
    ret_bitangent = normalize(b);
}

float3 geom_compute_arb_tangent(float3 tri_vert_positions[3])
{
    return normalize(tri_vert_positions[1] - tri_vert_positions[0]);
}