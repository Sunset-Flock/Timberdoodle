#pragma once

#include "../shader_shared/visbuffer.inl"

/**
 * DESCRIPTION:
 * Visbuffer is used as an indirection into all relevant rendering data.
 * As the visbuffer is only 32 bit per pixel, it SIGNIFICANTLY reduces used bandwidth when drawing and shading.
 * In effect it is a form of bandwidth compression.
 * To get rendering data from the visbuffer we need to follow several indirections, examples:
 * * albedo texture: vis id -> meshlet instance -> material, contains albedo image id
 * * vertex positions: vis id -> meshlet instance -> mesh -> mesh buffer id, mesh vertex positions offset
 * * transform: vis id -> meshlet instance -> entity transform array
 * 
 * As you can see the meshlet instance is storing all kinds of indices into relevant parts of the data.
 * With the triangle index and the meshlet index in triangle id and meshlet instance, 
 * we also know exactly what part of each mesh we are from the vis id.
*/

void encode_triangle_id(uint instantiated_meshlet_index, uint triangle_index, out uint id)
{
    id = (instantiated_meshlet_index << 7) | (triangle_index);
}

uint meshlet_instance_index_from_triangle_id(uint id)
{
    return id >> 7;
}

uint triangle_index_from_triangle_id(uint id)
{
    return id & 0x7F;
}

void decode_triangle_id(uint id, out uint instantiated_meshlet_index, out uint triangle_index)
{
    instantiated_meshlet_index = meshlet_instance_index_from_triangle_id(id);
    triangle_index = triangle_index_from_triangle_id(id);
}