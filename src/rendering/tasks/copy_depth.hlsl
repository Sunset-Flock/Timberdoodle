#pragma once

#include <daxa/daxa.inl>
#include <daxa/utils/task_graph.inl>

DAXA_DECL_COMPUTE_TASK_HEAD_BEGIN(CopyDepthH)
DAXA_TH_IMAGE_TYPED(SAMPLED, daxa::Texture2DId<daxa_f32>, depth_src)
DAXA_TH_IMAGE_TYPED(WRITE, daxa::RWTexture2DId<daxa_f32>, depth_dst_f32)
DAXA_DECL_TASK_HEAD_END

struct CopyDepthPush
{
    CopyDepthH::AttachmentShaderBlob attachments;
    daxa_f32vec2 size;
};

#if DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG

[[vk::push_constant]] CopyDepthPush push;

[[shader("compute")]]
[numthreads(8,8,1)]
func entry_copy_depth(uint2 index : SV_DispatchThreadID)
{
    if (any(index >= push.size))
    {
        return;
    }
    push.attachments.depth_dst_f32.get()[index] = push.attachments.depth_src.get()[index];
}

#endif