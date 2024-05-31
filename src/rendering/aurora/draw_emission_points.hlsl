#include "daxa/daxa.inl"
#include "shader_lib/aurora_util.glsl"

#include "aurora.inl"
[[vk::push_constant]] DrawEmissionPointsH::AttachmentShaderBlob raster_push;

struct Info
{
    float2 center          : float2;
    float radius_pix       : float;
    float height           : float;
    float to_camera_dist   : float;
};

struct VertexStageOutput
{
    Info      info            : Info;
    float4    sv_position     : SV_Position;
    float     point_size      : SV_PointSize;
};
[shader("vertex")]
VertexStageOutput vert_main(
    uint svvid : SV_VertexID,
)
{
    let push = raster_push;

    let position = push.beam_paths[svvid];
    VertexStageOutput output;
    output.sv_position = mul(deref(push.globals).camera.view_proj, float4(position, 1));

    output.info.center = (0.5 * output.sv_position.xy/output.sv_position.w + 0.5) * push.aurora_globals.aurora_image_resolution;
    output.point_size = push.aurora_globals.aurora_image_resolution.y * -push.globals.camera.proj[1][1] * 0.06 / output.sv_position.w;
    output.info.radius_pix = output.point_size / 2.0;
    output.info.height = position.z;
    output.info.to_camera_dist = length(push.globals.camera.position - position);
    return output;
}

float3 get_emission_color_intensity(float height, daxa_BufferPtr(float3) colors, daxa_BufferPtr(float) intensities)
{
    float color_lut_idx_f = max(height - 100.0, 0) / 5.0;
    let color_lut_idx_i = int(floor(color_lut_idx_f));
    let lower_color = colors[color_lut_idx_i];
    let upper_color = colors[color_lut_idx_i + 1];

    let color_interp_factor = frac(color_lut_idx_f);
    let final_color = (1.0 - color_interp_factor) * lower_color + color_interp_factor * upper_color;

    let emission_intensity_idx_f = max(height - 100.0, 0);
    let emission_intensity_idx_i = int(floor(emission_intensity_idx_f));
    let lower_intensity = intensities[emission_intensity_idx_i];
    let upper_intensity = intensities[emission_intensity_idx_i + 1];

    let emission_interp_factor = frac(emission_intensity_idx_f);
    let final_intensity = (1.0 - emission_interp_factor) * lower_intensity + emission_interp_factor * upper_intensity;

    return final_color * final_intensity;
}

[shader("fragment")]
float4 frag_main(
    Info info : Info,
    float4 pos : SV_Position
) : SV_Target
{
    let push = raster_push;
    float2 coord = (pos.xy - info.center) / info.radius_pix;
    float dist_from_center = length(coord);
    if(dist_from_center > 1.0)
    {
        return float4(0.0);
    }
    
    let emission_color = get_emission_color_intensity(info.height, push.aurora_globals.emission_colors, push.aurora_globals.emission_intensities);
    let emission_color_factor = 1.0 / info.to_camera_dist;

    return float4(emission_color * emission_color_factor * pow((1.0 - dist_from_center), 2.0), 1.0);
}