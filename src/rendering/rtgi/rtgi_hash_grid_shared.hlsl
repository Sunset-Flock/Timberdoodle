#pragma once

#include "shader_shared/rtgi.inl"

// Hash grid key / lookup helpers, shared by the hash grid passes (rtgi_hash_grid.hlsl) and the temporal accumulate
// (disocclusion fill).

// == Cell keys ================================================================================================
// A cell is (cascade level, integer cell coordinate at that level's cell size, face normal bucket). The level
// follows the distance to the camera, so a cell always covers roughly hash_grid_cell_size_px half-res pixels.
// Distance (not view depth) keeps the level stable under camera rotation. The normal bucket (dominant axis + sign)
// keeps the two sides of thin walls and corners apart; the SH carries the direction within a bucket.

struct RtgiHashGridKey
{
    int3 coord;
    uint level;
    uint normal_bucket;
};

// Half-res pixel width at distance 1, the same projection approximation as rtgi_half_res_pixel_width_ws.
func rtgi_hash_grid_pixel_width_per_distance(float inv_half_res_height) -> float
{
    return inv_half_res_height * 2.0f;
}

func rtgi_hash_grid_level(float distance, float pixel_width_per_distance, RtgiSettings settings) -> uint
{
    const float target_cell_size = distance * pixel_width_per_distance * settings.hash_grid_cell_size_px;
    const float levels = ceil(log2(max(target_cell_size / max(settings.hash_grid_min_cell_size, 1e-4f), 1.0f)));
    return uint(min(levels, float(RTGI_HASH_GRID_MAX_LEVEL)));
}

func rtgi_hash_grid_cell_size(uint level, RtgiSettings settings) -> float
{
    return max(settings.hash_grid_min_cell_size, 1e-4f) * float(1u << level);
}

func rtgi_hash_grid_normal_bucket(float3 normal) -> uint
{
    const float3 a = abs(normal);
    const uint axis = (a.x >= a.y && a.x >= a.z) ? 0u : (a.y >= a.z ? 1u : 2u);
    return axis * 2u + (normal[axis] < 0.0f ? 1u : 0u);
}

func rtgi_hash_grid_make_key(float3 position_ws, float3 normal_ws, float3 camera_position, float pixel_width_per_distance, RtgiSettings settings) -> RtgiHashGridKey
{
    RtgiHashGridKey key;
    key.level = rtgi_hash_grid_level(length(position_ws - camera_position), pixel_width_per_distance, settings);
    key.coord = int3(floor(position_ws / rtgi_hash_grid_cell_size(key.level, settings)));
    key.normal_bucket = rtgi_hash_grid_normal_bucket(normal_ws);
    return key;
}

// lowbias32 (Chris Wellons), good avalanche for chained word hashing.
func rtgi_hash_grid_hash_u32(uint x) -> uint
{
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

func rtgi_hash_grid_hash_key(RtgiHashGridKey key, uint seed) -> uint
{
    uint h = rtgi_hash_grid_hash_u32(seed ^ uint(key.coord.x));
    h = rtgi_hash_grid_hash_u32(h ^ uint(key.coord.y));
    h = rtgi_hash_grid_hash_u32(h ^ uint(key.coord.z));
    h = rtgi_hash_grid_hash_u32(h ^ (key.level | (key.normal_bucket << 8u)));
    return h;
}

// Two independent hashes: the slot hash picks the probe window, the checksum is what a slot stores as its key.
// Two different keys only alias when both 32-bit hashes collide within the same probe window.
func rtgi_hash_grid_slot_hash(RtgiHashGridKey key) -> uint
{
    return rtgi_hash_grid_hash_key(key, 0x9E3779B9u);
}

func rtgi_hash_grid_checksum(RtgiHashGridKey key) -> uint
{
    return max(rtgi_hash_grid_hash_key(key, 0x85EBCA6Bu), 1u); // 0 == empty slot
}

func rtgi_hash_grid_probe_slot(uint slot_hash, uint probe) -> uint
{
    return (slot_hash + probe) & (RTGI_HASH_GRID_CAPACITY - 1u);
}

// Lookup without insertion. Freed cells leave holes in probe chains, so the whole probe window is scanned instead
// of stopping at the first empty slot.
func rtgi_hash_grid_find(RtgiHashGridCell* cells, RtgiHashGridKey key) -> uint
{
    const uint slot_hash = rtgi_hash_grid_slot_hash(key);
    const uint checksum = rtgi_hash_grid_checksum(key);
    for (uint probe = 0u; probe < RTGI_HASH_GRID_MAX_PROBES; ++probe)
    {
        const uint slot = rtgi_hash_grid_probe_slot(slot_hash, probe);
        if (cells[slot].key == checksum)
        {
            return slot;
        }
    }
    return RTGI_HASH_GRID_INVALID_CELL;
}

// Lookup, inserting the key into the first free slot of its probe window when it is not present.
// Concurrent inserters of the same key all claim free slots in probe order, so they converge on the same slot
// (a failed claim that returns our own checksum means another thread inserted it first).
// Rare races (a hole opening between the scan and the claim) can create a duplicate cell; the duplicate stops
// getting queries and is freed after hash_grid_max_untouched_frames.
// out_inserted = this thread created the cell (won the claim of an empty slot); true for exactly one caller per new cell.
func rtgi_hash_grid_find_or_insert(RtgiHashGridCell* cells, RtgiHashGridKey key, out bool out_inserted) -> uint
{
    out_inserted = false;
    const uint slot_hash = rtgi_hash_grid_slot_hash(key);
    const uint checksum = rtgi_hash_grid_checksum(key);
    uint first_free_probe = RTGI_HASH_GRID_MAX_PROBES;
    for (uint probe = 0u; probe < RTGI_HASH_GRID_MAX_PROBES; ++probe)
    {
        const uint slot = rtgi_hash_grid_probe_slot(slot_hash, probe);
        const uint slot_key = cells[slot].key;
        if (slot_key == checksum)
        {
            return slot;
        }
        if (slot_key == 0u && first_free_probe == RTGI_HASH_GRID_MAX_PROBES)
        {
            first_free_probe = probe;
        }
    }
    for (uint probe = first_free_probe; probe < RTGI_HASH_GRID_MAX_PROBES; ++probe)
    {
        const uint slot = rtgi_hash_grid_probe_slot(slot_hash, probe);
        uint prev_key = 0u;
        InterlockedCompareExchange(cells[slot].key, 0u, checksum, prev_key);
        if (prev_key == 0u || prev_key == checksum)
        {
            out_inserted = prev_key == 0u;
            return slot;
        }
    }
    return RTGI_HASH_GRID_INVALID_CELL;
}

func rtgi_hash_grid_pack_pixel(uint2 pixel) -> uint
{
    return (pixel.x & 0xFFFFu) | ((pixel.y & 0xFFFFu) << 16u);
}

func rtgi_hash_grid_unpack_pixel(uint packed) -> uint2
{
    return uint2(packed & 0xFFFFu, packed >> 16u);
}

// Lookup at a random in-plane offset: the position moves within the surface plane, uniform over a square one cell
// wide (+-half a cell per tangent axis). Falls back to the unjittered cell when the jittered position lands in a cell
// that does not exist. Smooths the cell blocks stochastically (per pixel and frame, no filtering). out_key = the key
// of the returned cell.
func rtgi_hash_grid_find_jittered(RtgiHashGridCell* cells, float3 position_ws, float3 face_normal, float3 camera_position,
    float pixel_width_per_distance, RtgiSettings settings, uint seed, out RtgiHashGridKey out_key) -> uint
{
    out_key = rtgi_hash_grid_make_key(position_ws, face_normal, camera_position, pixel_width_per_distance, settings);
    const float cell_size = rtgi_hash_grid_cell_size(out_key.level, settings);
    const float3 helper = abs(face_normal.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    const float3 tangent = normalize(cross(helper, face_normal));
    const float3 bitangent = cross(face_normal, tangent);
    const float2 offset = (float2(seed & 0xFFFFu, seed >> 16u) * (1.0f / 65536.0f) - 0.5f) * cell_size;
    const float3 jittered_position_ws = position_ws + tangent * offset.x + bitangent * offset.y;
    const RtgiHashGridKey jittered_key = rtgi_hash_grid_make_key(jittered_position_ws, face_normal, camera_position, pixel_width_per_distance, settings);
    const uint jittered_cell = rtgi_hash_grid_find(cells, jittered_key);
    if (jittered_cell != RTGI_HASH_GRID_INVALID_CELL)
    {
        out_key = jittered_key;
        return jittered_cell;
    }
    return rtgi_hash_grid_find(cells, out_key);
}
