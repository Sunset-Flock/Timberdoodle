#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/geometry.inl"

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
    float3 world_normal;
    float3 face_normal;
    daxa_f32vec2 uv;
    daxa_f32vec2 uv_ddx;    // Only roughly approximated in rt
    daxa_f32vec2 uv_ddy;    // Only roughly approximated in rt
};

#define MATERIAL_FLAG_NONE (0u)
#define MATERIAL_FLAG_ALPHA_DISCARD (1u << 0u)
#define MATERIAL_FLAG_DOUBLE_SIDED (1u << 1u)

struct MaterialPointData
{
    float3 albedo;
    float alpha;
    float3 normal;
    float3 geometry_normal;
    float3 position;
    uint material_flags;
};

interface LightVisibilityTesterI
{
    float sun_light(MaterialPointData material_point, float3 incoming_ray);
    float point_light(MaterialPointData material_point, float3 incoming_ray, uint light_index);
};