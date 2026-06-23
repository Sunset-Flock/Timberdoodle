#include <daxa/daxa.inl>

#include "test.inl"

#include "shader_lib/misc.hlsl"

[[vk::push_constant]] CompressedSamplingTestPush push;

static const float RAMDOM_WALK_MAX_TEXEL_STEP_SIZE = 8.0f;
static float3 RANDOM_WALK_MAX_STEP_SIZE = float3(0,0,0);
static float3 line_walk_direction = float3(0,0,0);
static float3 sample_position = float3(0,0,0);

func init(uint3 dtid)
{
    switch (push.test_indexing_type)
    {
        case TEST_INDEXING_TYPE_ZERO:
        {
            sample_position = float3(0,0,0);
            break;
        }
        case TEST_INDEXING_TYPE_DTID:
        {
            sample_position = (float3(dtid) + 0.5f) * rcp(float3(push.tex_size));
            break;
        }
        case TEST_INDEXING_TYPE_RANDOM:
        {
            sample_position = float3(rand(), rand(), rand());
            break;
        }
        case TEST_INDEXING_TYPE_RANDOM_WALK:
        {
            sample_position = (float3(dtid) + 0.5f) * rcp(float3(push.tex_size));
            RANDOM_WALK_MAX_STEP_SIZE = RAMDOM_WALK_MAX_TEXEL_STEP_SIZE * rcp(float3(push.tex_size));
            break;
        }
        case TEST_INDEXING_TYPE_RANDOM_WALK_LINE:
        {
            sample_position = (float3(dtid) + 0.5f) * rcp(float3(push.tex_size));
            RANDOM_WALK_MAX_STEP_SIZE = RAMDOM_WALK_MAX_TEXEL_STEP_SIZE * rcp(float3(push.tex_size));
            line_walk_direction = normalize(float3(rand(), rand(), rand()));
            break;
        }
        default: break;
    }
}

__generic<TestTextureType test_tex_type, TestIndexingType test_indexing_type>
func next_uv() -> float3
{
    float3 uv = {};
    switch (test_indexing_type)
    {
        case TEST_INDEXING_TYPE_ZERO:
        {
            uv = float3(0,0,0);
            break;
        }
        case TEST_INDEXING_TYPE_DTID:
        {
            uv = sample_position;
            break;
        }
        case TEST_INDEXING_TYPE_RANDOM:
        {
            uv = float3(rand(), rand(), rand());
            break;
        }
        case TEST_INDEXING_TYPE_RANDOM_WALK:
        {
            const float3 step = float3(rand(), rand(), rand()) * RANDOM_WALK_MAX_STEP_SIZE;
            uv = sample_position + step;
            sample_position = uv;
            break;
        }
        case TEST_INDEXING_TYPE_RANDOM_WALK_LINE:
        {
            const float3 step = line_walk_direction * rand() * RANDOM_WALK_MAX_STEP_SIZE.z;
            uv = sample_position + step;
            sample_position = uv;
            break;
        }
    }
    uv = clamp(uv, float3(0,0,0), float3(1,1,1));
    if (test_tex_type == TEST_TEXTURE_TYPE_2D_ARRAY)
    {
        uv.z *= push.tex_size.z;
    }
    return uv;
}

__generic<bool SAMPLER_LINEAR>
func sample_3d_emulated(Texture2DArray<float4> tex, SamplerState s, float3 uv) -> float4
{
    if (SAMPLER_LINEAR)
    {
        let level_low = uint(floor(uv.z * float(push.tex_size.z)));
        let level_high = level_low + 1;
        let level_lerp_factor = frac(uv.z * float(push.tex_size.z));
        let low_sample = tex.SampleLevel(s, float3(uv.xy, level_low), 0.0f);
        let high_sample = tex.SampleLevel(s, float3(uv.xy, level_high), 0.0f);
        return lerp(low_sample, high_sample, level_lerp_factor);
    }
    else
    {
        let level = uint(round(uv.z * float(push.tex_size.z)));
        return tex.SampleLevel(s, float3(uv.xy, level), 0.0f);
    }
}

[shader("compute")]
[numthreads(COMPRESSED_SAMPLING_TEST_DISPATCH_X, COMPRESSED_SAMPLING_TEST_DISPATCH_Y)]
func entry_compressed_sampling_test(uint3 dtid : SV_DispatchThreadID)
{
    rand_seed(dtid.x * push.tex_size.y + dtid.y);

    init(dtid);

    float4 result = (float4)0;

    let test_sampler = push.test_sampler_id.get();

    // The outer switches here cause shader compilers to instantiate all code paths separately.
    // This allows it to optimize each inner loop for the indexing and texture type.
    switch (push.test_tex_type)
    {
        case TEST_TEXTURE_TYPE_2D:
        {
            let tex = Texture2D<float4>::get(push.image);
            switch (push.test_indexing_type)
            {
                case TEST_INDEXING_TYPE_ZERO: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D, TEST_INDEXING_TYPE_ZERO>().xy, 0.0f); } break;
                case TEST_INDEXING_TYPE_DTID: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D, TEST_INDEXING_TYPE_DTID>().xy, 0.0f); } break;
                case TEST_INDEXING_TYPE_RANDOM: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D, TEST_INDEXING_TYPE_RANDOM>().xy, 0.0f); } break;
                case TEST_INDEXING_TYPE_RANDOM_WALK: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D, TEST_INDEXING_TYPE_RANDOM_WALK>().xy, 0.0f); } break;
                case TEST_INDEXING_TYPE_RANDOM_WALK_LINE: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D, TEST_INDEXING_TYPE_RANDOM_WALK_LINE>().xy, 0.0f); } break;
            }
        }
        break;
        case TEST_TEXTURE_TYPE_2D_ARRAY:
        {
            let tex = Texture2DArray<float4>::get(push.image);
            switch (push.test_indexing_type)
            {
                case TEST_INDEXING_TYPE_ZERO: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D_ARRAY, TEST_INDEXING_TYPE_ZERO>(), 0.0f); } break;
                case TEST_INDEXING_TYPE_DTID: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D_ARRAY, TEST_INDEXING_TYPE_DTID>(), 0.0f); } break;
                case TEST_INDEXING_TYPE_RANDOM: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D_ARRAY, TEST_INDEXING_TYPE_RANDOM>(), 0.0f); } break;
                case TEST_INDEXING_TYPE_RANDOM_WALK: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D_ARRAY, TEST_INDEXING_TYPE_RANDOM_WALK>(), 0.0f); } break;
                case TEST_INDEXING_TYPE_RANDOM_WALK_LINE: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_2D_ARRAY, TEST_INDEXING_TYPE_RANDOM_WALK_LINE>(), 0.0f); } break;
            }
        }
        break;
        case TEST_TEXTURE_TYPE_3D:
        {
            let tex = Texture3D<float4>::get(push.image);
            switch (push.test_indexing_type)
            {
                case TEST_INDEXING_TYPE_ZERO: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_ZERO>(), 0.0f); } break;
                case TEST_INDEXING_TYPE_DTID: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_DTID>(), 0.0f); } break;
                case TEST_INDEXING_TYPE_RANDOM: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_RANDOM>(), 0.0f); } break;
                case TEST_INDEXING_TYPE_RANDOM_WALK: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_RANDOM_WALK>(), 0.0f); } break;
                case TEST_INDEXING_TYPE_RANDOM_WALK_LINE: for (uint iter = 0; iter < push.iterations; ++iter) { result += tex.SampleLevel(test_sampler,next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_RANDOM_WALK_LINE>(), 0.0f); } break;
            }
        }
        case TEST_TEXTURE_TYPE_3D_EMULATED:
        {
            let tex = Texture2DArray<float4>::get(push.image);
            if (push.sampler_linear)
            {
                switch (push.test_indexing_type)
                {
                    case TEST_INDEXING_TYPE_ZERO: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<true>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_ZERO>()); } break;
                    case TEST_INDEXING_TYPE_DTID: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<true>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_DTID>()); } break;
                    case TEST_INDEXING_TYPE_RANDOM: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<true>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_RANDOM>()); } break;
                    case TEST_INDEXING_TYPE_RANDOM_WALK: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<true>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_RANDOM_WALK>()); } break;
                    case TEST_INDEXING_TYPE_RANDOM_WALK_LINE: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<true>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_RANDOM_WALK_LINE>()); } break;
                }
            }
            else
            {
                switch (push.test_indexing_type)
                {
                    case TEST_INDEXING_TYPE_ZERO: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<false>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_ZERO>()); } break;
                    case TEST_INDEXING_TYPE_DTID: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<false>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_DTID>()); } break;
                    case TEST_INDEXING_TYPE_RANDOM: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<false>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_RANDOM>()); } break;
                    case TEST_INDEXING_TYPE_RANDOM_WALK: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<false>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_RANDOM_WALK>()); } break;
                    case TEST_INDEXING_TYPE_RANDOM_WALK_LINE: for (uint iter = 0; iter < push.iterations; ++iter) { result += sample_3d_emulated<false>(tex, test_sampler, next_uv<TEST_TEXTURE_TYPE_3D, TEST_INDEXING_TYPE_RANDOM_WALK_LINE>()); } break;
                }
            }
        }
        break;
    }

    // Prevent compiler from removing all code
    if (any(isnan(result)))
    {
        *push.dummy = dot(float4(1), result);
    }
}