#include <daxa/daxa.inl>

#include "clouds.inl"

#include "shader_lib/sky_util.glsl"
#include "shader_lib/transform.hlsl"
#include "shader_lib/debug.glsl"
#include "shader_lib/volumetric.hlsl"
#include "shader_lib/misc.hlsl"

[[vk::push_constant]] RaymarchCloudsPush raymarch_clouds_push;

#define USE_COMPRESSED_FIELDS 1

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

func debug_draw_step(
    const float3 step_start,
    const float3 step_end,
    const bool inside_cloud,
    daxa_RWBufferPtr(ShaderDebugBufferHead) debug,
    const float3 sphere_color = float3(0.0f)
) -> void
{
    // const float sphere_radius = length(step_end - step_start);
    const float sphere_radius = 1.0f;
    const float3 debug_line_color = inside_cloud ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.5f, 0.0f);
    const float3 debug_step_color = sphere_color;
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

struct GetCloudDataInfo
{
    float3 cloud_aabb_relative_position;
    CloudVolumeInstance * cloud_volume;
    float to_camera_distance;
};


struct CloudData
{
    float sdf;
    float density;
    float scaled_density;
    float eroded_density;
};

CloudData get_cloud_data(const GetCloudDataInfo info)
{
    let push = raymarch_clouds_push;

    struct CloudModelingData
    {
        float density;
        float detail_type;
        float density_scale;
        float sdf;
    };

    CloudModelingData modeling_data;
#if USE_COMPRESSED_FIELDS

    float3 compressed_sdf = Texture3D<float3>::get(info.cloud_volume->cloud_sdf_texture).SampleLevel(
        push.attach.globals.samplers.linear_clamp.get(), info.cloud_aabb_relative_position, 0
    );

    modeling_data.sdf = ((dot(compressed_sdf, float3(0.96414679f, 0.03518212f, 0.00067109f)) * (512.0f + 32.0f)) - 32.0f);

    if(modeling_data.sdf > 0.0f)
    {
        modeling_data.density = 0.0f;
    }
    else
    {
        float3 field_data = Texture3D<float3>::get(info.cloud_volume->cloud_data_texture).SampleLevel(
            push.attach.globals.samplers.linear_clamp.get(),
            float3(info.cloud_aabb_relative_position.xyz), 0,
        ).rgb;

        // float3 field_data_0 = Texture3D<float3>::get(info.cloud_volume->compressed_field_data).SampleLevel(
        //     push.attach.globals.samplers.linear_clamp.get(),
        //     float3(info.cloud_aabb_relative_position.xy, floor(info.cloud_aabb_relative_position.z * 64)), 0,
        // ).rgb;

        // float3 field_data_1 = Texture3D<float3>::get(info.cloud_volume->compressed_field_data).SampleLevel(
        //     push.attach.globals.samplers.linear_clamp.get(),
        //     float3(info.cloud_aabb_relative_position.xy, ceil(info.cloud_aabb_relative_position.z * 64)), 0,
        // ).rgb;

        // float3 field_data = lerp(field_data_0, field_data_1, frac(info.cloud_aabb_relative_position.z * 64));

        modeling_data.density = field_data.r;
        modeling_data.detail_type = field_data.g;
        modeling_data.density_scale = field_data.b;
    }

#else
    modeling_data = reinterpret<CloudModelingData, float4>(Texture3D<float4>::get(info.cloud_volume->cloud_data_texture).SampleLevel(
        push.attach.globals.samplers.linear_clamp.get(),
        info.cloud_aabb_relative_position,
        0
    ).rgba);
#endif

    modeling_data.sdf = modeling_data.sdf / 512.0f;

    modeling_data.density_scale = saturate(modeling_data.density_scale);
    modeling_data.detail_type = modeling_data.detail_type;

    const float density = modeling_data.density;
    float final_uprezzed_density = 0.0f;
    float scaled_density = 0.0f;

    struct NoiseData
    {
        float hf_billows;
        float hf_whisps;
        float lf_billows;
        float lf_whisps;
    };
    const float3 noise_offset = float3(1.0f, 1.0f, 0.0f) * push.attach.globals.total_elapsed_us * 0.00000002;
    const NoiseData noise = reinterpret<NoiseData>(//float4(0.0f));
        (Texture3D<float4>::get(info.cloud_volume->detail_noise_texture).SampleLevel(push.attach.globals.samplers.linear_repeat.get(), info.cloud_aabb_relative_position * float3(512, 512, 64) * 0.08 + noise_offset, 0, int3(0))));

    if(density > 0.0f)
    {
    
        const float wispy_noise = lerp(noise.lf_whisps, noise.hf_whisps, modeling_data.density);

        const float billowy_type_gradient = pow(modeling_data.density, 0.25f);
        const float billowy_noise = lerp(noise.lf_billows * 0.3, noise.hf_billows * 0.3, billowy_type_gradient);

        float noise_composite = lerp(wispy_noise, billowy_noise, modeling_data.detail_type);

        const float upprezzed_density = erode_base_density(modeling_data.density, noise_composite);
        const float powered_density_scale = pow(saturate(modeling_data.density_scale), 4.0f);
        const float scaled_uprezzed_density = upprezzed_density * powered_density_scale;
        const float sharpened_scaled_uprezzed_density = pow(scaled_uprezzed_density, lerp(0.3, 0.6, max(0.00001, powered_density_scale)));
        final_uprezzed_density = sharpened_scaled_uprezzed_density;
        scaled_density = pow(density * powered_density_scale, lerp(0.3, 0.6, powered_density_scale));
        // final_uprezzed_density = modeling_data.density;
    }

    return CloudData(modeling_data.sdf, density, scaled_density, final_uprezzed_density);
}

struct SecondaryTransmittance
{
    float3 clouds_transmittance;
    float3 atmosphere_transmittance;
    float density;
};

SecondaryTransmittance integrate_secondary_transmittance(float3 position, float max_distance, float3 cloud_aabb_min, float3 cloud_aabb_scaled_size, float density_scale, CloudVolumeInstance * cloud_volume)
{
    let push = raymarch_clouds_push;
    const float3 sun_direction = push.attach.globals.sky_settings.sun_direction;

    float current_distance = 0.00f;
    float cloud_density = 0.0f;

    for(int step = 0; step < push.attach.globals.volumetric_settings.secondary_steps; ++step)
    {
        if(current_distance >= max_distance) { break; }

        const float3 current_ray_pos = position + current_distance * sun_direction;

        let cloud_data = CloudData();//get_cloud_data(current_ray_pos, cloud_aabb_min, cloud_aabb_scaled_size, 50.0f, 0.0f, cloud_volume);

#if defined(DEBUG_RAYMARCH)
            // Debug draws the step here.
            const float3 step_start = current_ray_pos;
            const float3 step_end = step_start + (cloud_data.sdf * sun_direction);
            const bool inside_cloud = cloud_data.sdf < 0.0f;
            // debug_draw_step(step_start, step_end, inside_cloud, push.attach.globals.debug);
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


struct TraceCloudAABBIntersectionInfo
{
    float3 ray_origin;
    float3 ray_direction;
    AABB * cloud_instance_aabbs;
    uint cloud_instance_count;
};
struct NearestCloudAABBIntersection
{
    uint cloud_instance_index;
    float near_distance;
    float far_distance;
};

NearestCloudAABBIntersection trace_nearest_cloud_aabb(TraceCloudAABBIntersectionInfo trace_info)
{
    NearestCloudAABBIntersection nearest_intersection;
    nearest_intersection.cloud_instance_index = uint::maxValue;
    nearest_intersection.near_distance = float::maxValue;
    nearest_intersection.far_distance = float::minValue;

    for(uint i = 0; i < trace_info.cloud_instance_count; ++i)
    {
        const AABB cloud_aabb = trace_info.cloud_instance_aabbs[i];
        const RayAABBResult intersection = intersect_ray_with_aabb(
            trace_info.ray_origin,
            trace_info.ray_direction,
            cloud_aabb
        );

        const bool ray_hit = (intersection.near < intersection.far);

        if(ray_hit && intersection.near < nearest_intersection.near_distance)
        {
            nearest_intersection.near_distance = intersection.near;
            nearest_intersection.far_distance = intersection.far;
            nearest_intersection.cloud_instance_index = i;
        }
    }

    return nearest_intersection;
}

struct CloudAABBRelativeRay
{
    float3 origin;
    float3 direction;
};

struct WorldRay
{
    float3 origin;
    float3 direction;
};

WorldRay cloud_aabb_relative_ray_to_world_ray(const CloudAABBRelativeRay relative_ray, const AABB cloud_aabb)
{
    const float3 cloud_aabb_min = cloud_aabb.center - cloud_aabb.size * 0.5f;
    const float3 world_origin = cloud_aabb_min + relative_ray.origin * cloud_aabb.size;
    const float3 world_direction = normalize(relative_ray.direction * cloud_aabb.size);
    return WorldRay(world_origin, world_direction);
};

CloudAABBRelativeRay world_ray_to_cloud_aabb_relative_ray(const WorldRay world_ray, const AABB cloud_aabb)
{
    const float3 cloud_aabb_min = cloud_aabb.center - cloud_aabb.size * 0.5f;
    const float3 relative_origin = (world_ray.origin - cloud_aabb_min) * rcp(cloud_aabb.size);
    const float3 relative_direction = normalize(world_ray.direction * rcp(cloud_aabb.size));
    return CloudAABBRelativeRay(relative_origin, relative_direction);
};

float cloud_aabb_relative_distance_to_world_distance(const float cloud_aabb_relative_distance, const AABB cloud_aabb, const float3 ray_direction)
{
    const float3 world_space_step = cloud_aabb.size * ray_direction;
    const float world_space_step_length = length(world_space_step);
    return cloud_aabb_relative_distance * world_space_step_length;
}

float world_distance_to_cloud_aabb_distance(const float world_distance, const AABB cloud_aabb, const float3 ray_direction)
{
    const float world_to_aabb_unit_step_length = length(ray_direction * rcp(cloud_aabb.size));
    return world_distance * world_to_aabb_unit_step_length;
}

struct MarchCloudHitInfo
{
    CloudAABBRelativeRay ray;
    float cloud_aabb_max_distance;
    uint max_steps;

    CloudVolumeInstance * instance;
#if defined(DEBUG_RAYMARCH)
    AABB * cloud_instance_aabb;
#endif
};

float march_until_cloud_hit(const MarchCloudHitInfo trace_info)
{
    float cloud_aabb_relative_distance = 0.0f;
    float steps = 0.0f;
    for(uint step_count = 0; step_count < trace_info.max_steps; ++step_count)
    {
        steps = step_count;
        if(cloud_aabb_relative_distance >= trace_info.cloud_aabb_max_distance) { break; }

        const float3 current_ray_relative_pos = trace_info.ray.origin + cloud_aabb_relative_distance * trace_info.ray.direction;

#if USE_COMPRESSED_FIELDS
        const float3 compressed_sdf = Texture3D<float3>::get(trace_info.instance->cloud_sdf_texture)
            .SampleLevel(raymarch_clouds_push.attach.globals.samplers.linear_clamp.get(), current_ray_relative_pos, 0);

        const float sdf = dot(compressed_sdf, float3(0.96414679f, 0.03518212f, 0.00067109f));
        const float renormalized_sdf = (sdf * (512.0f + 32.0f) - 32.0f) / 512.0f;
#else // USE_COMPRESSED_FIELDS
        const float sdf = Texture3D<float4>::get(trace_info.instance->cloud_data_texture)
            .SampleLevel(raymarch_clouds_push.attach.globals.samplers.linear_clamp.get(), current_ray_relative_pos, 0).a;
#endif
        // The SDF is rescaled and stored to be between [0, 1] but the values going in are [512, -32].
        // Thus anything over 32 / (512 + 32) is negative and so inside of a cloud;

        if(renormalized_sdf <= -(0.0f / 512.0f)) { break; }
        cloud_aabb_relative_distance += max(renormalized_sdf, 0.001f);
#if defined(DEBUG_RAYMARCH)
        {
            const float3 step_start = cloud_aabb_relative_ray_to_world_ray(CloudAABBRelativeRay(current_ray_relative_pos, float3(0.0f)), *trace_info.cloud_instance_aabb).origin;
            const float3 cloud_relative_new_ray_origin = trace_info.ray.origin + cloud_aabb_relative_distance * trace_info.ray.direction;
            const float3 step_end = cloud_aabb_relative_ray_to_world_ray(CloudAABBRelativeRay(cloud_relative_new_ray_origin, float3(0.0f)), *trace_info.cloud_instance_aabb).origin;
            const float3 sphere_color = max(renormalized_sdf, 0.001f) == renormalized_sdf ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
            debug_draw_step(step_start, step_end, false, raymarch_clouds_push.attach.globals.debug, sphere_color);
        }
#endif
    }

    return cloud_aabb_relative_distance;
}

struct SecondaryMarchThroughCloudInfo
{
    CloudAABBRelativeRay ray;
    uint steps;

    CloudVolumeInstance * instance;

    float cloud_aabb_step_size;
    float world_step_size;
};

struct SecondaryMarchThroughCloudResult
{
    float transmittance;
    float density;
};

SecondaryMarchThroughCloudResult secondary_march_through_cloud(const SecondaryMarchThroughCloudInfo march_cloud_info)
{
    SecondaryMarchThroughCloudResult result;
    result.transmittance = 1.0f;
    result.density = 0.0f;

    GetCloudDataInfo info;
    info.cloud_volume = march_cloud_info.instance;
    info.to_camera_distance = 0.0f;

    float cloud_aabb_relative_distance_marched = 0.0f;

    CloudData cloud_data;
    for(uint step = 0; step < march_cloud_info.steps; ++step)
    {
        info.cloud_aabb_relative_position = march_cloud_info.ray.origin + march_cloud_info.ray.direction * cloud_aabb_relative_distance_marched;
        cloud_data = get_cloud_data(info);
        cloud_aabb_relative_distance_marched += march_cloud_info.cloud_aabb_step_size;

        if(cloud_data.density == 0.0f) { break; }

        const float extinction = cloud_data.density * march_cloud_info.instance->density_scale;
        const float per_step_transmittance = exp(-extinction * march_cloud_info.world_step_size);
        result.transmittance *= per_step_transmittance;
        result.density += cloud_data.density;
    }
    return result;
};

struct PrimaryMarchThroughCloudInfo
{
    CloudAABBRelativeRay ray;
    float cloud_aabb_max_distance;
    float cloud_aabb_step_size;
    float world_step_size;
    float sun_dot;

    float phase_value;

    float3 sun_light;
    float3 ambient_light;

    SecondaryMarchThroughCloudInfo secondary_march_info;

    CloudVolumeInstance * instance;
    AABB * cloud_aabb;
};

struct PrimaryMarchThroughCloudResult
{
    float3 scattered_light;
    float transmittance;
    float cloud_aabb_relative_distance_marched;
};

PrimaryMarchThroughCloudResult primary_march_through_cloud(const PrimaryMarchThroughCloudInfo march_cloud_info)
{
    GetCloudDataInfo info;
    info.cloud_volume = march_cloud_info.instance;
    info.to_camera_distance = 0.0f;

    PrimaryMarchThroughCloudResult result;
    result.scattered_light = float3(0.0f);
    result.transmittance = 1.0f;
    result.cloud_aabb_relative_distance_marched = 0.0f;

    const float bottom_atmosphere_offset = raymarch_clouds_push.attach.globals.sky_settings.atmosphere_bottom + BASE_HEIGHT_OFFSET;
    const float3 world_secondary_ray_direction = cloud_aabb_relative_ray_to_world_ray(march_cloud_info.secondary_march_info.ray, *march_cloud_info.cloud_aabb).direction;
    WorldRay world_ray = WorldRay(cloud_aabb_relative_ray_to_world_ray(march_cloud_info.ray, *march_cloud_info.cloud_aabb));

    CloudData cloud_data;
    bool hit_cloud = false;
    for(uint step = 0; step < 512; ++step)
    {
        info.cloud_aabb_relative_position = march_cloud_info.ray.origin + march_cloud_info.ray.direction * result.cloud_aabb_relative_distance_marched;
        cloud_data = get_cloud_data(info);

        result.cloud_aabb_relative_distance_marched += march_cloud_info.cloud_aabb_step_size;

        if(cloud_data.density == 0.0f) 
        {
            if(hit_cloud) { break; }
            else          { continue; }
        }
        hit_cloud = true;

        const float extinction = cloud_data.eroded_density * march_cloud_info.instance->density_scale;
        const float scattering = extinction * march_cloud_info.instance->albedo;
        const float per_step_transmittance = exp(-extinction * march_cloud_info.world_step_size);

        SecondaryMarchThroughCloudInfo secondary_march_info = march_cloud_info.secondary_march_info;
        secondary_march_info.ray.origin = info.cloud_aabb_relative_position;

        const SecondaryMarchThroughCloudResult secondary_raymarch_result = secondary_march_through_cloud(secondary_march_info);

        // ============================================== ATMOSPHERE TRANSMITTANCE =================================================
        // ==================== TODO(saky) Remove this and fold it into the shadow volume grid once we have it =====================
        float3 atmosphere_transmittance = float3(1.0f);
        {
            const float3 in_world_position = world_ray.origin + world_ray.direction * march_cloud_info.world_step_size * step;
            const float3 atmosphere_position = (in_world_position * M_TO_KM_SCALE) + float3(0.0f, 0.0f, bottom_atmosphere_offset);
            const float zenith_cos_angle = dot(world_secondary_ray_direction, normalize(atmosphere_position));

            TransmittanceParams transmittance_lut_params = TransmittanceParams(length(atmosphere_position), zenith_cos_angle);
            float2 transmittance_texture_uv = transmittance_lut_to_uv(
                transmittance_lut_params,
                raymarch_clouds_push.attach.globals.sky_settings.atmosphere_bottom,
                raymarch_clouds_push.attach.globals.sky_settings.atmosphere_top
            );

            atmosphere_transmittance = raymarch_clouds_push.attach.transmittance.get().SampleLevel(
                raymarch_clouds_push.attach.globals.samplers.linear_clamp.get(), transmittance_texture_uv, 0,).rgb;
        }

        const float3 primary_luminance = 1.0f * march_cloud_info.sun_light * march_cloud_info.phase_value * secondary_raymarch_result.transmittance * atmosphere_transmittance;
        const float3 ambient_luminance = 1.0f * pow(max(1.0f - cloud_data.scaled_density, 0.0f), 1.0f) * march_cloud_info.ambient_light;

        const float distance_factor = ValueRemap(cloud_data.sdf, -0.01, 0.0f, 0.1f, 0.25f);
        const float ms_transmittance = exp(-secondary_raymarch_result.density * ValueRemap(march_cloud_info.sun_dot, 0.0f, 0.9f, 0.25f, distance_factor));
        const float3 multiscattered_luminance = 0.2f * cloud_data.density * ms_transmittance * atmosphere_transmittance * march_cloud_info.sun_light;

        const float3 luminance = primary_luminance + multiscattered_luminance + ambient_luminance;
        const float3 luminance_times_scattering = luminance * scattering;

        const float3 integrated_scattered_light = (luminance_times_scattering - (luminance_times_scattering * per_step_transmittance)) / max(extinction, 0.00001f);

        result.scattered_light += integrated_scattered_light * result.transmittance;
        result.transmittance *= per_step_transmittance;
    }
    return result;
}

bool cloud_volume_instance_complete(CloudVolumeInstance * instance)
{
    if (
        instance->cloud_data_texture.is_empty() ||
        instance->detail_noise_texture.is_empty()
#if USE_COMPRESSED_FIELDS
        || instance->cloud_sdf_texture.is_empty()
#endif // USE_COMPRESSED_FIELDS
    ) {
        return false;
    }
    return true;
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

#if defined(DEBUG_RAYMARCH)
    const float2 pixel_index = float2(push.attach.globals.volumetric_settings.debug_pixel) * 0.5;
#else
    const float2 pixel_index = svdtid.xy;
#endif

    const bool pixel_in_bounds = 
        all(lessThan(pixel_index, push.clouds_resolution)) &&
        all(greaterThanEqual(pixel_index, float2(0.0f)));

    CloudVolumeInstancesBufferHead * cloud_volumes_head = push.attach.cloud_volumes;

    rand_seed(asuint(svdtid.x + svdtid.y * 13136.1235f) * push.attach.globals.frame_index);

    if(pixel_in_bounds)
    {
        float depth = 0.0f;
        for (int offset = 0; offset < 4; ++offset)
        {
            const int offset_x = offset & 0b1;
            const int offset_y = offset / 2;
            const float offset_depth = push.attach.depth.get()[uint2(pixel_index) * 2 + uint2(offset_x, offset_y)].x;
            depth = max(depth, offset_depth);
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

        TraceCloudAABBIntersectionInfo trace_info;
        trace_info.ray_origin = ray_origin;
        trace_info.ray_direction = ray_direction;
        trace_info.cloud_instance_aabbs = cloud_volumes_head->instance_aabbs;
        trace_info.cloud_instance_count = cloud_volumes_head->count;

        const NearestCloudAABBIntersection intersection = trace_nearest_cloud_aabb(trace_info);

        float accumulated_transmittance = float(1.0f);
        float3 accumulated_scattered_light = float3(0.0f, 0.0f, 0.0f);

        if (intersection.cloud_instance_index != uint::maxValue) 
        {
            let cloud_instance_aabb = cloud_volumes_head->instance_aabbs[intersection.cloud_instance_index];
            let cloud_instance = &cloud_volumes_head->instances[intersection.cloud_instance_index];
            const CloudAABBRelativeRay cloud_relative_start_ray = world_ray_to_cloud_aabb_relative_ray(WorldRay(ray_origin + intersection.near_distance * ray_direction, ray_direction), cloud_instance_aabb);

            const float3 sun_direction = push.attach.globals.sky_settings.sun_direction;
            const float3 cloud_aabb_relative_sun_direction = normalize(sun_direction * rcp(cloud_instance_aabb.size));
            const float4 compressed_indirect_lighting = push.attach.sky_ibl.get().SampleLevel(push.attach.globals.samplers.linear_clamp.get(), sun_direction, 0);

            const float3 ambient_light = compressed_indirect_lighting.rgb * compressed_indirect_lighting.a * PI * 1.0f;
            const float3 sun_light = push.attach.globals.sky_settings.sun_brightness * sun_color.rgb;

            const float primary_ray_world_to_aabb_distance_scaling_factor = world_distance_to_cloud_aabb_distance(1.0f, cloud_volumes_head->instance_aabbs[intersection.cloud_instance_index], ray_direction);
            const float secondary_ray_world_to_aabb_distance_scaling_factor = world_distance_to_cloud_aabb_distance(1.0f, cloud_volumes_head->instance_aabbs[intersection.cloud_instance_index], sun_direction);
            // Trace until we either hit the end of the cloud AABB or we hit something in the scene.
            const float world_intersection_depth = length(world_position - ray_origin);
            const float ray_end_distance = min(intersection.far_distance, world_intersection_depth) - intersection.near_distance;
            const float cloud_aabb_relative_end_distance = ray_end_distance * primary_ray_world_to_aabb_distance_scaling_factor;

            MarchCloudHitInfo march_cloud_hit_info;
            march_cloud_hit_info.ray = cloud_relative_start_ray;
            march_cloud_hit_info.cloud_aabb_max_distance = cloud_aabb_relative_end_distance;
            march_cloud_hit_info.max_steps = 512;
            march_cloud_hit_info.instance = cloud_instance;
#if defined(DEBUG_RAYMARCH)
            march_cloud_hit_info.cloud_instance_aabb = &cloud_volumes_head->instance_aabbs[intersection.cloud_instance_index];
#endif

            float cloud_aabb_relative_distance_along_ray = march_until_cloud_hit(march_cloud_hit_info);

#if defined(DEBUG_RAYMARCH)
        {
            const float3 step_start = cloud_aabb_relative_ray_to_world_ray(march_cloud_hit_info.ray, cloud_instance_aabb).origin;
            const float3 cloud_relative_new_ray_origin = march_cloud_hit_info.ray.origin + march_cloud_hit_info.ray.direction * cloud_aabb_relative_distance_along_ray;
            const float3 step_end = cloud_aabb_relative_ray_to_world_ray(CloudAABBRelativeRay(cloud_relative_new_ray_origin, float3(0.0f)), cloud_instance_aabb).origin;
            debug_draw_step(step_start, step_end, false, push.attach.globals.debug);
        }
#endif
            uint step_count = 0;
            while(cloud_aabb_relative_distance_along_ray < cloud_aabb_relative_end_distance && accumulated_transmittance > 0.0f)
            {
                step_count += 1;
                PrimaryMarchThroughCloudInfo march_through_cloud_info;
                march_through_cloud_info.ray.direction = cloud_relative_start_ray.direction; 
                march_through_cloud_info.ray.origin = cloud_relative_start_ray.origin + cloud_relative_start_ray.direction * cloud_aabb_relative_distance_along_ray;
                march_through_cloud_info.cloud_aabb_step_size = 0.0035f;
                march_through_cloud_info.world_step_size = march_through_cloud_info.cloud_aabb_step_size * rcp(primary_ray_world_to_aabb_distance_scaling_factor);
                march_through_cloud_info.instance = cloud_instance;
                march_through_cloud_info.phase_value = get_phase_value(dot(sun_direction, ray_direction), 1.0f, 1.0f);
                march_through_cloud_info.sun_dot = dot(sun_direction, ray_direction);
                march_through_cloud_info.sun_light = sun_light;
                march_through_cloud_info.ambient_light = ambient_light;
                march_through_cloud_info.cloud_aabb = &cloud_volumes_head->instance_aabbs[intersection.cloud_instance_index];

                march_through_cloud_info.secondary_march_info.ray.direction = cloud_aabb_relative_sun_direction;
                march_through_cloud_info.secondary_march_info.steps = push.attach.globals.volumetric_settings.secondary_steps;
                march_through_cloud_info.secondary_march_info.instance = cloud_instance;
                march_through_cloud_info.secondary_march_info.cloud_aabb_step_size = 0.004f;
                march_through_cloud_info.secondary_march_info.world_step_size = march_through_cloud_info.secondary_march_info.cloud_aabb_step_size * rcp(secondary_ray_world_to_aabb_distance_scaling_factor);

                const PrimaryMarchThroughCloudResult march_result = primary_march_through_cloud(march_through_cloud_info);
                accumulated_scattered_light += march_result.scattered_light * accumulated_transmittance;
                accumulated_transmittance *= march_result.transmittance;

                // Advance for the distance we stepped through the cloud.
                cloud_aabb_relative_distance_along_ray += march_result.cloud_aabb_relative_distance_marched;

                // Calculate intersection with the next cloud inside of this cloud instance.
                march_cloud_hit_info.ray.origin = cloud_relative_start_ray.origin + cloud_relative_start_ray.direction * cloud_aabb_relative_distance_along_ray;
                march_cloud_hit_info.cloud_aabb_max_distance = cloud_aabb_relative_end_distance - cloud_aabb_relative_distance_along_ray;
                const float dist_until_cloud_hit = march_until_cloud_hit(march_cloud_hit_info);
                cloud_aabb_relative_distance_along_ray += dist_until_cloud_hit;

#if defined(DEBUG_RAYMARCH)
                {
                    const float3 step_start = cloud_aabb_relative_ray_to_world_ray(march_through_cloud_info.ray, cloud_instance_aabb).origin;
                    const float3 step_end = cloud_aabb_relative_ray_to_world_ray(march_cloud_hit_info.ray, cloud_instance_aabb).origin;
                    debug_draw_step(step_start, step_end, true, push.attach.globals.debug);
                }
#endif

#if defined(DEBUG_RAYMARCH)
                {
                    const float3 step_start = cloud_aabb_relative_ray_to_world_ray(march_cloud_hit_info.ray, cloud_instance_aabb).origin;
                    const float3 cloud_relative_new_ray_origin = march_cloud_hit_info.ray.origin + march_cloud_hit_info.ray.direction * dist_until_cloud_hit;
                    const float3 step_end = cloud_aabb_relative_ray_to_world_ray(CloudAABBRelativeRay(cloud_relative_new_ray_origin, float3(0.0f)), cloud_instance_aabb).origin;
                    debug_draw_step(step_start, step_end, false, push.attach.globals.debug);
                }
#endif
            }
        }
        push.attach.clouds_raymarched_result.get()[svdtid] = float4(accumulated_scattered_light, accumulated_transmittance);
    }
#if 0 
            const float cloud_albedo = cloud_volume->albedo;
            const float cloud_density_scale = cloud_volume->density_scale;

            // Figure out the phase function values. We can do this before we start the entire loop,
            // because neither sun_direction nor world_direction change during the raymarch process.
            // Thus the phase values also remain constant.
            const float cos_theta = dot(sun_direction, ray_direction);
            //const float mie_phase_value = hg_draine_phase(cos_theta, 10.0f);

            int step_cnt = 0;
            while(current_distance < end_distance && (accumulated_transmittance.x > 0.0) && step_cnt < 1000)
            {
                step_cnt += 1;
                const float3 current_ray_pos = ray_start + current_distance * ray_direction;

                const float distance_from_camera = current_distance + intersection.near;
                const float inside_cloud_step_size = max(1.0f, max(pow(distance_from_camera, 0.5), 0.001f)) * 0.2;// * rand();// * 0.1 * rand();
                let cloud_data = get_cloud_data(current_ray_pos, cloud_bottom_left_corner, cloud_aabb_size, inside_cloud_step_size, current_distance, cloud_volume);

                if(cloud_data.eroded_density <= 0.0f)
                {
                    current_distance += cloud_data.sdf;
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
                    cloud_aabb_size
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
                        cloud_bottom_left_corner,
                        cloud_aabb_size,
                        cloud_density_scale,
                        cloud_volume
                    );

                    const float ms_distance_factor_scattering = 0.4;
                    const float ms_phase_value = multi_octave_hg(
                        ms_distance_factor_scattering,
                        cos_theta,
                        extinction,
                        volumetric_settings.w_0,
                        volumetric_settings.w_1,
                        volumetric_settings.octaves
                    );

                    const float3 ambient_luminance = max(pow(1.0f - cloud_data.scaled_density, 1.0f), 0.0) * ambient_light;

                    const float distance_factor = ValueRemap(cloud_data.normalized_sdf, -0.010, 0.0f, 0.01f, 0.05f);
                    const float ms_transmittance = exp(-secondary_transmittance.density * ValueRemap(cos_theta, 0.0f, 0.9f, 0.05f, distance_factor));
                    // const float ms_transmittance = exp(-secondary_transmittance.density * distance_factor);
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
                current_distance += cloud_data.sdf;// + rand() * 50;
            }
        }
#if defined(DEBUG_RAYMARCH)
        // Debug raymarch terminates here to avoid any writes to the target texture.
        return;
#endif
        push.attach.clouds_raymarched_result.get()[svdtid] = float4(accumualted_scattered_light, accumulated_transmittance);
    }
#endif
}