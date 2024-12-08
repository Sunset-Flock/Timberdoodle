#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/geometry.inl"

/// ===== Shader Only Geometry Data and functions =====

struct TriangleGeometry
{
    GPUMaterial* material;
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
    daxa_f32vec2 uv;
    daxa_f32vec2 uv_ddx;
    daxa_f32vec2 uv_ddy;
};