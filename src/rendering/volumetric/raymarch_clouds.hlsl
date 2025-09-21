#include <daxa/daxa.inl>

#include "clouds.inl"

#include "shader_lib/sky_util.glsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/volumetric.hlsl"
#include "shader_lib/misc.hlsl"

[[vk::push_constant]] RaymarchCloudsPush raymarch_clouds_push;

//
// Function to remap a value from one range to another. It is slightly cheaper than SetRange
//
#define ValueRemapFuncionDef(DATA_TYPE) \
	DATA_TYPE ValueRemap(DATA_TYPE inValue, DATA_TYPE inOldMin, DATA_TYPE inOldMax, DATA_TYPE inMin, DATA_TYPE inMax) \
	{ \
		DATA_TYPE old_min_max_range = (inOldMax - inOldMin); \
		DATA_TYPE clamped_normalized = saturate((inValue - inOldMin) / old_min_max_range); \
		return inMin + (clamped_normalized*(inMax - inMin)); \
	}

ValueRemapFuncionDef(float)
ValueRemapFuncionDef(float2)
ValueRemapFuncionDef(float3)
ValueRemapFuncionDef(float4)

struct RayAABBResult
{
    float near;
    float far;
};

// Returns the near and far intersection points.
// Ray missing the aabb is implied by (near >= far).
func intersect_ray_with_aabb(
    const float3 ray_origin,
    const float3 ray_dir,
    const float3 aabb_pos,
    const float3 aabb_size
) -> RayAABBResult
{
    const float3 aabb_min = aabb_pos - (aabb_size * 0.5f);
    const float3 aabb_max = aabb_pos + (aabb_size * 0.5f);

    const float3 t_min = (aabb_min - ray_origin) / ray_dir;
    const float3 t_max = (aabb_max - ray_origin) / ray_dir;
    const float3 t1 = min(t_min, t_max);
    const float3 t2 = max(t_min, t_max);
    const float t_near = max(max(t1.x, t1.y), t1.z);
    const float t_far = min(min(t2.x, t2.y), t2.z);

    // t_near and t_far can be negative in the inverse of the view direction.
    // To remove the side effect of reporting a false hit I max the t_near with 0.0f.
    // This also has the added benefit of having near be as 0 when inside of the AABB.
    return RayAABBResult(max(t_near, 0.0), t_far);
}

func debug_draw_step(
    const float3 step_start,
    const float3 step_end,
    const bool inside_cloud,
    daxa_RWBufferPtr(ShaderDebugBufferHead) debug
) -> void
{
    // const float sphere_radius = length(step_end - step_start);
    const float sphere_radius = 3.0;
    const float3 debug_line_color = float3(1.0f, 0.0f, 0.0f);
    const float3 debug_step_color = inside_cloud ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.5f, 0.0f);
    ShaderDebugSphereDraw start_sphere_draw = {
        step_start,                                 // position
        sphere_radius,                              // radius
        DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE,   // coord_space
        debug_step_color                            // color
    };

    ShaderDebugSphereDraw end_sphere_draw = start_sphere_draw;
    end_sphere_draw.position = step_end;

    ShaderDebugLineDraw line_draw = {
        step_start,                                 // start
        step_end,                                   // end
        debug_line_color,                           // color
        DEBUG_SHADER_DRAW_COORD_SPACE_WORLDSPACE    // coord_space
    };

    debug_draw_sphere(debug, start_sphere_draw);
    debug_draw_sphere(debug, end_sphere_draw);
    debug_draw_line(debug, line_draw);
}

//
// Function to erode a value given an erosion amount. A simplified version of SetRange.
//
float erode_base_density(float in_value, float in_old_min)
{
	// derrived from Set-Range, this function uses the oldMin to erode or inflate the input value. - inValues inflate while + inValues erode
	const float old_min_max_range = (1.0 - in_old_min);
	float clamped_normalized = saturate((in_value - in_old_min) / old_min_max_range);
	return (clamped_normalized);
}

struct CloudData
{
    float sdf;
    float normalized_sdf;
    float density;
    float eroded_density;
};

CloudData get_cloud_data(float3 position, float3 cloud_aabb_min, float3 cloud_aabb_scaled_size, float min_step_size)
{
    let push = raymarch_clouds_push;

    struct CloudModelingData
    {
        float density;
        float detail_type;
        float density_scale;
        float sdf;
    };

    // Absolute means position in bounds [(0, 0, 0);(cloud_scale, cloud_scale, cloud_scale)]
    const float3 in_cloud_aabb_absolute_position = position - cloud_aabb_min;
    // Relative means position in bounds [(0, 0, 0);(1, 1, 1)]
    const float3 in_cloud_aabb_relative_position = in_cloud_aabb_absolute_position / cloud_aabb_scaled_size;

    CloudModelingData modeling_data = reinterpret<CloudModelingData, float4>(push.attach.cloud_data_field.get().SampleLevel(
        push.attach.globals.samplers.linear_clamp.get(),
        in_cloud_aabb_relative_position,
        0,
        int3(0)
    ));

    modeling_data.sdf = modeling_data.detail_type / 512;
    modeling_data.density_scale = pow(saturate(0.0f + (in_cloud_aabb_relative_position.z)), 0.5);
    modeling_data.detail_type = pow(saturate(in_cloud_aabb_relative_position.z), 0.5);

    modeling_data.density = saturate(modeling_data.density * modeling_data.density_scale);

    const float density = modeling_data.density;
    float final_uprezzed_density = 0.0f;
    if(density > 0.0f)
    {
        struct NoiseData
        {
            float hf_billows;
            float hf_whisps;
            float lf_billows;
            float lf_whisps;
        };

        const float3 noise_offset = float3(1.0f, 1.0f, 0.0f) * push.attach.globals.total_elapsed_us * 0.00000002;
        const NoiseData noise = reinterpret<NoiseData, float4>(
            push.attach.cloud_detail_noise.get().SampleLevel(push.attach.globals.samplers.clouds_noise_sampler.get(), position * 0.002 + noise_offset, 0, int3(0)));
    
        const float wispy_noise = lerp(noise.lf_whisps, noise.hf_whisps, modeling_data.density);

        const float billowy_type_gradient = pow(modeling_data.density, 0.25f);
        const float billowy_noise = lerp(noise.lf_billows * 0.3, noise.hf_billows * 0.3, billowy_type_gradient);

        const float noise_composite = lerp(wispy_noise, billowy_noise, modeling_data.detail_type);

        const float upprezzed_density = erode_base_density(modeling_data.density, noise_composite);
        const float powered_density_scale = pow(modeling_data.density_scale, 4.0f);
        const float scaled_uprezzed_density = upprezzed_density * powered_density_scale;
        const float sharpened_scaled_uprezzed_density = pow(scaled_uprezzed_density, lerp(0.3, 0.6, max(0.0001, powered_density_scale)));
        final_uprezzed_density = sharpened_scaled_uprezzed_density;
        // final_uprezzed_density = modeling_data.density;
    }


    // I would really like an assert here - x and y should be the same and z should be smaller.
    // I guess I could also do max(x, y, z) here but eh, I'll just hope the data come in correct format.
    const float scaled_step_size = modeling_data.sdf * cloud_aabb_scaled_size.x;
    const float clamped_step_size = max(min_step_size, scaled_step_size);

    // return CloudData(scaled_step_size, modeling_data.sdf, density, final_uprezzed_density);
    return CloudData(clamped_step_size, modeling_data.sdf, density, final_uprezzed_density);
}

struct SecondaryTransmittance
{
    float3 clouds_trasmittance;
    float3 atmosphere_trasmittance;
    float density;
};

SecondaryTransmittance integrate_secondary_transmittance(float3 position, float max_distance, float3 cloud_aabb_min, float3 cloud_aabb_scaled_size, float density_scale)
{
    let push = raymarch_clouds_push;
    const float3 sun_direction = push.attach.globals.sky_settings.sun_direction;

    float current_distance = 0.00f;
    float cloud_density = 0.0f;

    for(int step = 0; step < push.attach.globals.volumetric_settings.secondary_steps; ++step)
    {
        if(current_distance >= max_distance) { break; }

        const float3 current_ray_pos = position + current_distance * sun_direction;

        let cloud_data = get_cloud_data(current_ray_pos, cloud_aabb_min, cloud_aabb_scaled_size, 100.0f);

#if defined(DEBUG_RAYMARCH)
            // Debug draws the step here.
            const float3 step_start = current_ray_pos;
            const float3 step_end = step_start + (cloud_data.sdf * sun_direction);
            const bool inside_cloud = cloud_data.sdf < 0.0f;
            debug_draw_step(step_start, step_end, inside_cloud, push.attach.globals.debug);
#endif

        cloud_density += cloud_data.eroded_density * cloud_data.sdf;

        current_distance += cloud_data.sdf;
    }

    const float3 clouds_extinction = density_scale * cloud_density;
    const float3 clouds_transmittance = exp(-clouds_extinction);

    const float bottom_atmosphere_offset = raymarch_clouds_push.attach.globals.sky_settings.atmosphere_bottom + BASE_HEIGHT_OFFSET;
    const float3 atmosphere_position = (position * M_TO_KM_SCALE) + float3(0.0f, 0.0f, bottom_atmosphere_offset);

    const float zenith_cos_angle = dot(push.attach.globals.sky_settings.sun_direction, normalize(atmosphere_position));

    TransmittanceParams transmittance_lut_params = TransmittanceParams(length(atmosphere_position), zenith_cos_angle);
    float2 transmittance_texture_uv = transmittance_lut_to_uv(
        transmittance_lut_params,
        raymarch_clouds_push.attach.globals.sky_settings.atmosphere_bottom,
        raymarch_clouds_push.attach.globals.sky_settings.atmosphere_top
    );

    const float3 atmosphere_transmittance = raymarch_clouds_push.attach.transmittance.get().SampleLevel(
        raymarch_clouds_push.attach.globals.samplers.linear_clamp.get(),
        transmittance_texture_uv,
        0,
    ).rgb;

    return SecondaryTransmittance(clouds_transmittance, atmosphere_transmittance, cloud_density);
}

float get_phase_value(const float cos_theta, const float extinction, const float density)
{
    let volumetric_settings = raymarch_clouds_push.attach.globals.volumetric_settings;
    float phase_value = 1.0f / (4.0f * PI);

    //float density_modulated_g = (volumetric_settings.g + clamp(pow(density, 0.5), 0.00, 1.0f - volumetric_settings.g - 0.05));
    // float density_modulated_g = volumetric_settings.g + lerp(-0.2, 0.5, pow(density, 0.5));
    float density_modulated_g = lerp(0.98, 0.01, pow(density, 0.6f));
    const bool modulate_g = (volumetric_settings.use_density_modulated_g == 1u);
    const float g = modulate_g ? density_modulated_g : volumetric_settings.g;

    switch(volumetric_settings.phase_function_model)
    {
        case HENYEY_GREENSTEIN:
        {
            phase_value = henyey_greenstein_phase(g, cos_theta);
            break;
        }
        case HENYEY_GREENSTEIN_OCTAVES:
        {
            phase_value = multi_octave_hg(
                g,
                cos_theta,
                extinction,
                volumetric_settings.w_0,
                volumetric_settings.w_1,
                volumetric_settings.octaves
            );
            break;
        }
        case DRAINE:
        {
            phase_value = draine_phase(
                volumetric_settings.diameter,
                g,
                cos_theta
            );
            break;
        }
    }
    return phase_value;
}

[shader("compute")]
#if defined(DEBUG_RAYMARCH)
[numthreads(RAYMARCH_CLOUDS_DEBUG_DISPATCH_X, RAYMARCH_CLOUDS_DEBUG_DISPATCH_Y)]
#else
[numthreads(RAYMARCH_CLOUDS_DISPATCH_X, RAYMARCH_CLOUDS_DISPATCH_Y)]
#endif
func entry_raymarch(uint2 svdtid : SV_DispatchThreadID)
{

    let push = raymarch_clouds_push;
    rand_seed(asuint(svdtid.x + svdtid.y * 13136.1235f) * push.attach.globals.frame_index);
#if defined(DEBUG_RAYMARCH)
    const float2 pixel_index = float2(push.attach.globals.volumetric_settings.debug_pixel) * 0.5;
#else
    const float2 pixel_index = svdtid.xy;
#endif

    const bool pixel_in_bounds = 
        all(lessThan(pixel_index, push.clouds_resolution)) &&
        all(greaterThanEqual(pixel_index, float2(0.0f)));

    if(pixel_in_bounds)
    {
        float depth = 10000.0f;
        for (int offset = 0; offset < 4; ++offset)
        {
            const int offset_x = offset & 0b1;
            const int offset_y = offset / 2;
            const float offset_depth = push.attach.depth.get()[uint2(pixel_index) * 3 + uint2(offset_x, offset_y)].x;
            depth = min(depth, offset_depth);
        }

        const float clamped_depth = max(0.000001f, depth);

#if defined(DEBUG_RAYMARCH)
        // Debug raymarch always steps the ray from the main camera so that we can properly observe it.
        const float3 world_position = sv_xy_to_world_space(
            push.attach.globals.main_camera.inv_screen_size * 2.0,
            push.attach.globals.main_camera.inv_view_proj,
            float3(pixel_index, clamped_depth)
        );
        const float3 ray_origin = push.attach.globals.main_camera.position;
#else
        const float3 world_position = sv_xy_to_world_space(
            push.attach.globals.view_camera.inv_screen_size * 2.0,
            push.attach.globals.view_camera.inv_view_proj,
            float3(pixel_index, clamped_depth)
        );
        const float3 ray_origin = push.attach.globals.view_camera.position;
#endif
        const float3 ray_direction = normalize(world_position - ray_origin);
        const float3 sun_direction = push.attach.globals.sky_settings.sun_direction;

        const float3 cloud_aabb_position = push.attach.globals.volumetric_settings.position;
        const float3 cloud_aabb_scale = push.attach.globals.volumetric_settings.scale;
        const float3 cloud_aabb_base_size = float3(push.attach.globals.volumetric_settings.size);
        const float3 cloud_aabb_scaled_size = cloud_aabb_base_size * cloud_aabb_scale;

        let intersection = intersect_ray_with_aabb(
            ray_origin,
            ray_direction,
            cloud_aabb_position,
            cloud_aabb_scaled_size
        );

        const float world_intersection_depth = length(world_position - ray_origin);
        const float intersection_far_distance = min(intersection.far, world_intersection_depth);

        float accumulated_transmittance = float(1.0f);
        float3 accumualted_scattered_light = float3(0.0f, 0.0f, 0.0f);

        const bool hit = (intersection.near < intersection_far_distance);
        if (hit) 
        {
            const float4 compressed_indirect_lighting = push.attach.sky_ibl.get().SampleLevel(
                push.attach.globals.samplers.linear_clamp.get(),
                sun_direction,
                0);

            const float3 ambient_light = compressed_indirect_lighting.rgb * compressed_indirect_lighting.a * PI;// * 2.0f;
            const float3 cloud_aabb_min = cloud_aabb_position - (cloud_aabb_scaled_size * 0.5f);
            const float3 ray_start = ray_origin + intersection.near * ray_direction;

            const float end_distance = intersection_far_distance - intersection.near;

            const float3 sun_luminance = push.attach.globals.sky_settings.sun_brightness * sun_color.rgb;

            float current_distance = 0.0f;//rand() * 100.0f;

            // TODO: 
            // Currently I:
            //    1) take a sample of both the density and the SDF
            //    2) integrate
            //    3) move the raymarch forward
            //
            // Once the density and SDF are separated, I should:
            //    1) sample the SDF
            //    2) sample the density - probably in 2/3 of the step distance
            //    3) integrate
            //    4) move the raymarch forward

            const float cloud_albedo = push.attach.globals.volumetric_settings.clouds_albedo;
            const float cloud_density_scale = push.attach.globals.volumetric_settings.clouds_density_scale;

            // Figure out the phase function values. We can do this before we start the entire loop,
            // because neither sun_direction nor world_direction change during the raymarch process.
            // Thus the phase values also remain constant.
            const float cos_theta = dot(sun_direction, ray_direction);
            //const float mie_phase_value = hg_draine_phase(cos_theta, 10.0f);

            int step_cnt = 0;
            while(current_distance < end_distance && (accumulated_transmittance.x > 0.0) && step_cnt < 500)
            {
                step_cnt += 1;
                const float3 current_ray_pos = ray_start + current_distance * ray_direction;

                const float distance_from_camera = current_distance + intersection.near;
                const float inside_cloud_step_size = max(1.0f, max(pow(distance_from_camera, 0.5), 0.001f)) * 0.16;
                let cloud_data = get_cloud_data(current_ray_pos, cloud_aabb_min, cloud_aabb_scaled_size, inside_cloud_step_size);

                if(cloud_data.eroded_density < 0.0f)
                {
                    current_distance += cloud_data.sdf;
                    // current_distance += inside_cloud_step_size;
#if defined(DEBUG_RAYMARCH)
                    // Debug draws the step here.
                    const float3 step_start = current_ray_pos;
                    const float3 step_end = ray_start + (current_distance * ray_direction);
                    const bool inside_cloud = cloud_data.density > 0.0f;
                    debug_draw_step(step_start, step_end, inside_cloud, push.attach.globals.debug);
#endif
                    continue;
                }

                const float extinction = cloud_data.eroded_density * cloud_density_scale;
                const float scattering = extinction * cloud_albedo;
                const float transmittance = exp(-extinction * cloud_data.sdf);

                let secondary_intersection = intersect_ray_with_aabb(
                    current_ray_pos,
                    sun_direction,
                    cloud_aabb_position,
                    cloud_aabb_scaled_size
                );
                const bool secondary_intersection_hit = secondary_intersection.near < secondary_intersection.far;

                if(secondary_intersection_hit)
                {
                    const float max_secondary_raymarch_distance = secondary_intersection.far - secondary_intersection.near;

                    const float phase_value = get_phase_value(cos_theta, extinction, cloud_data.eroded_density);

                    let volumetric_settings = push.attach.globals.volumetric_settings;

                    let secondary_transmittance = integrate_secondary_transmittance(
                        current_ray_pos,
                        max_secondary_raymarch_distance,
                        cloud_aabb_min,
                        cloud_aabb_scaled_size,
                        cloud_density_scale
                    );

                    const float ms_distance_factor_scattering = 0.65;
                    const float ms_phase_value = multi_octave_hg(
                        ms_distance_factor_scattering,
                        cos_theta,
                        extinction,
                        volumetric_settings.w_0,
                        volumetric_settings.w_1,
                        volumetric_settings.octaves
                    );

                    const float3 ambient_luminance = pow(1.0f - cloud_data.density, 0.5f) * ambient_light;

                    const float distance_factor = ValueRemap(cloud_data.normalized_sdf, -0.01f, 0.0f, 0.025f, 0.025f);
                    const float ms_transmittance = exp(-secondary_transmittance.density * ValueRemap(cos_theta, 0.0f, 0.9f, 0.025f, distance_factor));
                    const float3 multiscattered_approx_luminance = cloud_data.density * ms_transmittance;

                    const float3 luminance = 
                        1 * sun_luminance * phase_value * secondary_transmittance.clouds_trasmittance * secondary_transmittance.atmosphere_trasmittance +
                        1 * multiscattered_approx_luminance * sun_luminance * ms_phase_value * secondary_transmittance.atmosphere_trasmittance +
                        1 * ambient_luminance;

                    const float3 luminance_times_scattering = luminance * scattering;

                    // Avoid division by zero.
                    const float clamped_extinction = max(extinction, 0.0000001f);
                    const float3 integrated_scattered_light = (luminance_times_scattering - luminance_times_scattering * transmittance) / clamped_extinction;

                    accumualted_scattered_light += accumulated_transmittance * integrated_scattered_light;
                }

                accumulated_transmittance *= transmittance;

#if defined(DEBUG_RAYMARCH)
                // Debug draws the step here.
                const float3 step_start = current_ray_pos;
                const float3 step_end = step_start + (cloud_data.sdf * ray_direction);
                const bool inside_cloud = cloud_data.density > 0.0f;
                debug_draw_step(step_start, step_end, inside_cloud, push.attach.globals.debug);
#endif
                current_distance += cloud_data.sdf;
            }
        }
#if defined(DEBUG_RAYMARCH)
        // Debug raymarch terminates here to avoid any writes to the target texture.
        return;
#endif
        push.attach.clouds_raymarched_result.get()[svdtid] = float4(accumualted_scattered_light, accumulated_transmittance);
    }
}