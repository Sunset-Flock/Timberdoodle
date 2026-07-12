#pragma once

#extension GL_EXT_debug_printf : enable

#include "shader_shared/debug.inl"

#define GPU_ASSERT_ENABLE 1

#if GPU_ASSERT_ENABLE

static bool _GPU_ASSERT_FAIL_BOOL = false;
#define GPU_ASSERT_FAIL _GPU_ASSERT_FAIL_BOOL

#define GPU_ASSERT(COND)\
{\
    if (!COND)\
    {\
        printf("GPU ASSERT FAILED IN " __FILE__ ":%i: " #COND "\n", __LINE__);\
        GPU_ASSERT_FAIL = true;\
    }\
}

#define GPU_ASSERT_COMPARE_INT(A, OP, B)\
{\
    int a = A;\
    int b = B;\
    bool COND = a OP b;\
    if (!COND)\
    {\
        printf("GPU ASSERT FAILED IN " __FILE__ ":%i: " #A " " #OP " " #B ": %i " #OP " %i\n", __LINE__, a, b);\
        GPU_ASSERT_FAIL = true;\
    }\
}

#else

#define GPU_ASSERT_FAIL false
#define GPU_ASSERT(COND)
#define GPU_ASSERT_COMPARE_INT(A, OP, B)

#endif

#define DEBUG_DRAW(DEBUG_DATA, field, VALUE)\
{\
    const uint capacity = deref(DEBUG_DATA).field ## _draws.draw_capacity;\
    const uint index = atomicAdd(deref(DEBUG_DATA).field ## _draws.draw_requests, 1);\
    if (index < capacity)\
    {\
        atomicAdd(deref(DEBUG_DATA).field ## _draws.draw_indirect.instance_count, 1);\
        deref_i(deref(DEBUG_DATA).field ## _draws.draws, index) = draw;\
    }\
}

void debug_draw_line(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugLineDraw draw)
{
    DEBUG_DRAW(debug_info, line, draw)
}

daxa_u32 debug_alloc_line_draws(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, daxa_u32 amount)
{
    const uint capacity = deref(debug_info).line_draws.draw_capacity;
    const uint offset = atomicAdd(deref(debug_info).line_draws.draw_requests, amount);
    if ((offset + amount) < capacity)
    {
        atomicAdd(deref(debug_info).line_draws.draw_indirect.instance_count, amount);
        return offset;
    }
    return ~0u;
}

void debug_draw_circle(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugCircleDraw draw)
{
    DEBUG_DRAW(debug_info, circle, draw)
}

void debug_draw_rectangle(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugRectangleDraw draw)
{
    DEBUG_DRAW(debug_info, rectangle, draw)
}

void debug_draw_aabb(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugAABBDraw draw)
{
    DEBUG_DRAW(debug_info, aabb, draw)
}

void debug_draw_box(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugBoxDraw draw)
{
    DEBUG_DRAW(debug_info, box, draw)
}

void debug_draw_cone(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugConeDraw draw)
{
    DEBUG_DRAW(debug_info, cone, draw)
}

void debug_draw_sphere(daxa_RWBufferPtr(ShaderDebugBufferHead) debug_info, ShaderDebugSphereDraw draw)
{
    DEBUG_DRAW(debug_info, sphere, draw)
}

#define DEBUG_INDEX(INDEX, MIN_INDEX, MAX_INDEX)                                                   \
    if (INDEX < MIN_INDEX || INDEX > MAX_INDEX)                                                    \
    {                                                                                              \
        debugPrintfEXT("index out of bounds: %i, range: [%i,%i]\n", INDEX, MIN_INDEX, MAX_INDEX);  \
    }                                                                                              

    
// Returns true if the pixel at `local` (within a tile of size `tile_size`) lies outside a
// rounded rectangle that is inset from the tile edges by `cutoff`, with corner radius `corner_radius`.
// Used to give each debug tile a slight gap and rounded corners so adjacent tiles are visually separated.
bool debug_image_tile_is_cutoff(uint2 local, uint2 tile_size, float cutoff, float corner_radius)
{
    const float2 pixel_center = float2(local) + 0.5f;
    const float2 half_size = float2(tile_size) * 0.5f;
    const float2 p = pixel_center - half_size;
    const float2 q = abs(p) - (half_size - cutoff - corner_radius);
    const float dist = length(max(q, float2(0.0f, 0.0f))) - corner_radius;
    return dist > 0.0f;
}

// Alpha ranges for the debug image:
//   [0, 1): normal alpha — written as-is into the debug image, composited at end of frame with standard alpha blending.
//   [1, 2): blend-over — at end of frame the value is blended over the final rendered image; blend weight = alpha - 1.
//   [2, 3): blend-over with tonemapping — same as [1,2) but the color is treated as an exposure-scaled linear value
//           that will be tonemapped together with the scene color before display.
// slot: the debug_visualization_tile value directly. -1 = full screen, >= 0 = tile index (0..15).
void write_debug_image(RWTexture2D<float4> tex, int slot, uint2 sv_position, float4 color, uint resolution_scale = 1, bool enable_rounding = true)
{
    // Small, fixed-size rounding/cutoff (a few pixels total) applied to each tile's footprint.
    const float TILE_EDGE_CUTOFF = 3.0f;
    const float TILE_CORNER_RADIUS = 5.0f;

    uint width, height;
    tex.GetDimensions(width, height);
    const uint2 full_res = uint2(width, height);
    if (slot < 0)
    {
        const uint2 dst_position = sv_position * resolution_scale;
        for (uint y = 0; y < resolution_scale; ++y)
        {
            for (uint x = 0; x < resolution_scale; ++x)
            {
                const uint2 dst = dst_position + uint2(x, y);
                if (all(dst < full_res))
                {
                    const float alpha = tex[dst].a;
                    if (color.a > alpha)
                    {
                        tex[dst] = color;
                    }
                }
            }
        }
        return;
    }
    const uint2 slot_size = full_res / 4;
    const uint2 dst_in_slot = (sv_position * resolution_scale) / 4;
    if (slot >= 16 || any(dst_in_slot >= slot_size))
    {
        return;
    }
    if (enable_rounding && debug_image_tile_is_cutoff(dst_in_slot, slot_size, TILE_EDGE_CUTOFF, TILE_CORNER_RADIUS))
    {
        return;
    }
    const uint2 slot_coord = uint2(slot % 4, slot / 4);
    const uint2 dst_position = slot_coord * slot_size + dst_in_slot;
    const float alpha = tex[dst_position].a;
    if (color.a > alpha)
    {
        tex[dst_position] = color;
    }
}