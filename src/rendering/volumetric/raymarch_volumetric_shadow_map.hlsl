#include <daxa/daxa.inl>

#include "clouds.inl"

[[vk::push_constant]] RaymarchCloudVolumetricShadowMapPush raymarch_cloud_shadowmap_push;

[shader("compute")]
[numthreads(RAYMARCH_VOLUMETRIC_SHADOW_MAP_DISPATCH_X, 1, 1)]
func entry_raymarch(uint3 dtid : SV_DispatchThreadID)
{
    let push = raymarch_cloud_shadowmap_push;
    let thread_index = dtid.x;

    const uint cells = push.volumetric_resolution.x * push.volumetric_resolution.y * push.volumetric_resolution.z;
    if(thread_index < cells)
    {
        const uint cell_z_index = thread_index / (256 * 256);
        const uint in_z_index = thread_index - (cell_z_index * (push.volumetric_resolution.x * push.volumetric_resolution.y));
        const uint cell_y_index = in_z_index / push.volumetric_resolution.y;
        const uint cell_x_index = in_z_index - cell_y_index * push.volumetric_resolution.y;
        push.attach.cloud_volumetric_shadow_map.get()[uint3(cell_x_index, cell_y_index, cell_z_index)] = float2(cell_z_index, cell_x_index);
    }
}