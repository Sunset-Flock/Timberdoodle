#include "daxa/daxa.inl"
#include "shader_lib/aurora_util.glsl"

#include "aurora.inl"
#define PI 3.1415926535897932384626433832795
#define EARTH_RADIUS 6371.0
[[vk::push_constant]] DistributeBeamOriginsH::AttachmentShaderBlob push;

uint hash(uint x)
{
    // https://nullprogram.com/blog/2018/07/31/
    x ^= x >> 16;
    x *= 0x7FEB352D;
    x ^= x >> 15;
    x *= 0x846CA68B;
    x ^= x >> 16;
    return x;
}

static uint _rand_state;
void set_seed(uint seed)
{
    _rand_state = seed;
}

float get_random_zero_one()
{
    _rand_state = hash(_rand_state);
    return 2.0 - asfloat((_rand_state >> 9) | 0x3F800000);
}

float get_random_neg_one_one()
{
    let zero_one_rand = get_random_zero_one();
    return (zero_one_rand * 2.0) - 1.0;
}

float get_height_alpha_d(float height)
{
    let B_100 = push.aurora_globals.B_0 * pow((EARTH_RADIUS / (EARTH_RADIUS + 100)), 3.0);
    let B_height = push.aurora_globals.B_0 * pow((EARTH_RADIUS / (EARTH_RADIUS + height)), 3.0);
    let alpha_d = acos(sqrt(B_height/B_100));
    return alpha_d;
}

[numthreads(DISTRIBUTE_BEAM_ORIGINS_WG, 1, 1)]
[shader("compute")]
void main(
    uint3 svdtid : SV_DispatchThreadID
)
{
    const int per_layer_beams = push.aurora_globals.beam_count / push.aurora_globals.layers;
    const int layer_idx = svdtid.x / per_layer_beams;

    int in_layer_idx = svdtid.x % per_layer_beams;
    let min_normalized_offset_step = 1.0 / float(per_layer_beams);

    let normalized_offset = float(in_layer_idx) / per_layer_beams;
    set_seed(uint((svdtid.x + 1) * 12533957));
    let randomized_normalized_offset = normalized_offset + (get_random_neg_one_one() * min_normalized_offset_step);

    let aurora_vec = push.aurora_globals.end - push.aurora_globals.start;
    let start_segment_buff_idx = beam_segment_buffer_idx(svdtid.x, 0, push.aurora_globals);

    let layer_freq_offset = sin((2 * PI * randomized_normalized_offset * push.aurora_globals.frequency) + (push.aurora_globals.phase_shift_per_layer * layer_idx)) * 5;
    let per_layer_offset = push.aurora_globals.width / push.aurora_globals.layers;
    let layer_width_offset = per_layer_offset * layer_idx;
    let layer_random_offset = per_layer_offset * get_random_neg_one_one();
    let y_offset_vec = cross(normalize(float3(aurora_vec, push.aurora_globals.height)), push.aurora_globals.B);

    let y_offset = layer_freq_offset + layer_width_offset + layer_random_offset;

    let position = float3(push.aurora_globals.start + randomized_normalized_offset * aurora_vec * 2.0, push.aurora_globals.height);
    float3 prev_offset_position = position + y_offset * y_offset_vec * push.aurora_globals.offset_strength;
    let relative_dist_offset_per_colision = 1.0 / push.aurora_globals.beam_path_segment_count;
    if(svdtid.x < push.aurora_globals.beam_count)
    {
        push.beam_paths[start_segment_buff_idx] = prev_offset_position;
        for(int segment = 1; segment < push.aurora_globals.beam_path_segment_count; segment++)
        {
            let step_offset = (push.aurora_globals.beam_path_length * relative_dist_offset_per_colision) * get_random_zero_one();
            let alpha_angular_offset = max(get_height_alpha_d(prev_offset_position.z) - (segment * push.aurora_globals.angle_offset_per_collision * 0.01), 0.0);
            let beta_angular_offset = get_random_zero_one() * 2 * PI;

            let offset_length = length(step_offset);

            let B_normal = push.aurora_globals.B;
            let B_tangent = y_offset_vec;
            let B_bitangent = cross(B_normal, B_tangent);

            let alpha_ofset = sin(alpha_angular_offset) * offset_length;
            let beta_offset = float2(cos(beta_angular_offset), sin(beta_angular_offset));
            let final_tan_plane_offset = beta_offset * alpha_ofset;
            let final_world_offset = step_offset * B_normal + final_tan_plane_offset.x * B_tangent + final_tan_plane_offset.y * B_bitangent;

            prev_offset_position = prev_offset_position + final_world_offset;
            push.beam_paths[start_segment_buff_idx + segment] = prev_offset_position;
        }
    }
}