#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

#include "../../shader_shared/shared.inl"
#include "../../shader_shared/globals.inl"
#include "../../shader_shared/rtgi.inl"

#define COMPRESSED_SAMPLING_TEST_DISPATCH_X 8
#define COMPRESSED_SAMPLING_TEST_DISPATCH_Y 8

#if !defined(__cplusplus)
[UnscopedEnum]
#endif
enum TestSamplerType
{
    TEST_SAMPLER_TYPE_LINEAR_CLAMP,
    TEST_SAMPLER_TYPE_LINEAR_REPEAT,
    TEST_SAMPLER_TYPE_NEAREST_CLAMP,
    TEST_SAMPLER_TYPE_NEAREST_REPEAT,
    TEST_SAMPLER_TYPE_COUNT
};

#if !defined(__cplusplus)
[UnscopedEnum]
#endif
enum TestTextureType
{
    TEST_TEXTURE_TYPE_2D,
    TEST_TEXTURE_TYPE_2D_ARRAY,
    TEST_TEXTURE_TYPE_3D,
    TEST_TEXTURE_TYPE_3D_EMULATED,
    TEST_TEXTURE_TYPE_COUNT,
};

#if !defined(__cplusplus)
[UnscopedEnum]
#endif
enum TestIndexingType
{
    TEST_INDEXING_TYPE_ZERO,
    TEST_INDEXING_TYPE_DTID,
    TEST_INDEXING_TYPE_RANDOM,
    TEST_INDEXING_TYPE_RANDOM_WALK,
    TEST_INDEXING_TYPE_RANDOM_WALK_LINE,
    TEST_INDEXING_TYPE_COUNT,
};

struct CompressedSamplingTestPush
{
    TestIndexingType test_indexing_type;
    TestTextureType test_tex_type;
    daxa_ImageViewId image;
    daxa_SamplerId test_sampler_id;
    daxa_u32vec3 tex_size;
    daxa_f32* dummy;
    daxa_u32 iterations;
    daxa_b32 sampler_linear; 
};