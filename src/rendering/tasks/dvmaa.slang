#include "daxa/daxa.inl"

#include "dvmaa.inl"

[[vk::push_constant]] DVMResolveVisImagePush resolve_vis_push;
[numthreads(DVM_WG_X, DVM_WG_Y, 1)]
void entry_resolve_vis_image(
    uint3 svtid : SV_DispatchThreadID
){
    DVMResolveVisImagePush push = resolve_vis_push;
    if (any(svtid.xy > push.resolution)) return;

    Texture2DMS<uint4> msaa_vis_img = daxa_Texture2DMS(uint4, push.attachments.dvm_vis_image);
    RWTexture2D<uint4> vis_img = daxa_RWTexture2D(uint4, push.attachments.vis_image);
    vis_img[svtid.xy].x = msaa_vis_img.Load(svtid.xy, push.resolve_sample).x;
    Texture2DMS<float4> msaa_depth_img = daxa_Texture2DMS(float4, push.attachments.dvm_depth_image);
    RWTexture2D<float4> depth_img = daxa_RWTexture2D(float4, push.attachments.depth_image);
    depth_img[svtid.xy].x = msaa_depth_img.Load(svtid.xy, push.resolve_sample).x;
}