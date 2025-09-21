#include <daxa/daxa.inl>
#include "shader_lib/sky_util.glsl"
#include "shader_lib/volumetric.hlsl"
#include "sky.inl"

[[vk::push_constant]] ComputeTransmittanceH::AttachmentShaderBlob transmittance_push;
[[vk::push_constant]] ComputeMultiscatteringH::AttachmentShaderBlob multiscattering_push;
[[vk::push_constant]] ComputeSkyH::AttachmentShaderBlob sky_push;
[[vk::push_constant]] SkyIntoCubemapH::AttachmentShaderBlob sky_cubemap_push;

float3 integrate_transmittance(const float3 world_position, const float3 world_direction, const uint step_count)
{
    const SkySettings * settings = &transmittance_push.globals.sky_settings;
    // The integration length is the distance in which the starting ray intersects top of the atmosphere.
    // This gives us ray sphere intersection, where the sphere is placed in the origin of the coordinate system
    // and has radius of the top atmosphere boundary.
    // 
    // As opposed to multiscattering or sky raymarching loops, we do not need to check the bottom intersection distance.
    // This is because the calling function must guarantee we are inside of the atmosphere and the ray won't intersect
    // the atmosphere bottom (ground). In our case this property is inherent to our LUT parameterization.
    const float integration_length = ray_sphere_intersect_nearest(
        world_position, world_direction, float3(0.0f), settings->atmosphere_top
    );

    // We use uniform steps for transmittance calculation
    const float step_length = integration_length / step_count;
    float3 optical_depth = float3(0.0f);

    // Raymarching loop
    for(int step = 0; step < step_count; ++step)
    {
        const float3 step_position = world_position + (step_length * step * world_direction);
        const float3 atmosphere_extinction = sample_medium(settings, step_position).medium_extinction;
        optical_depth += atmosphere_extinction * step_length;
    }
    return optical_depth;
}

[numthreads(TRANSMITTANCE_X, TRANSMITTANCE_Y, 1)]
[shader("compute")]
void compute_transmittance_lut(
    uint3 svdtid : SV_DispatchThreadID
)
{
    const SkySettings * settings = &transmittance_push.globals.sky_settings;
    if(all(lessThan(svdtid.xy, settings->transmittance_dimensions.xy)))
    {
        // Convert uv coordinates of the texture into starting height and angle in which we will raymarch
        const float2 uv = float2(svdtid.xy) / settings->transmittance_dimensions.xy;
        const TransmittanceParams mapping = uv_to_transmittance_lut_params(
            uv, settings->atmosphere_bottom, settings->atmosphere_top
        );

        // Now convert height and angle into starting position and direction
        const float3 world_position = float3(0.0f, 0.0f, mapping.height);
        const float3 world_direction = normalize( 
            float3(safe_sqrt(1.0f - pow(mapping.zenith_cos_angle, 2)), 0.0f, mapping.zenith_cos_angle)
        );
        
        // Raymarch and accumulate the optical depth, the transmittance is then given by e^(-optical_depth)
        const float3 transmittance = exp(-integrate_transmittance(world_position, world_direction, settings->transmittance_step_count));
        transmittance_push.transmittance.get()[svdtid.xy] = transmittance;
    }
}

// Calculate the length of the raymarching loop
// Intersects a ray with the provided parameters with bottom and top of the atmosphere
// and returns the smaller of the two distances. 
// When the ray intersects neither atmosphere top nor bottom return float.maxValue
float get_integration_length(const float3 world_position, const float3 world_direction, SkySettings * settings)
{
    const float atmo_bottom_intersect_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, float3(0.0f), settings->atmosphere_bottom);
    const float atmo_top_intersect_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, float3(0.0f), settings->atmosphere_top);

    const float capped_atmo_bottom_dist = (atmo_bottom_intersect_distance == -1.0f) ? float.maxValue : atmo_bottom_intersect_distance;
    const float capped_atmo_top_dist = (atmo_top_intersect_distance == -1.0f) ? float.maxValue : atmo_top_intersect_distance;
    return min(capped_atmo_bottom_dist, capped_atmo_top_dist);
}

struct GetMultipleScatteringInfo
{
    float3 world_position;
    float view_zenith_cos_angle;
    SkySettings * settings;
    daxa::SamplerId sampler;
    daxa::Texture2DId<vector<float, 3>> multiscattering_id;
};

float3 get_multiple_scattering(const GetMultipleScatteringInfo info)
{
    const float atmosphere_thickness = info.settings->atmosphere_top - info.settings->atmosphere_bottom;
    const float relative_atmosphere_height = length(info.world_position) - info.settings->atmosphere_bottom;
    const float2 uv = float2( info.view_zenith_cos_angle * 0.5 + 0.5, relative_atmosphere_height / atmosphere_thickness);
    const float2 clamped_uv = clamp(uv, 0.0f, 1.0f);
    const float2 subuv = float2(
        from_unit_to_subuv(uv.x, info.settings->multiscattering_dimensions.x),
        from_unit_to_subuv(uv.y, info.settings->multiscattering_dimensions.y));
                                                
    const float3 multiscattering_contribution = info.multiscattering_id.get()
        .SampleLevel(SamplerState::get(info.sampler), subuv, 0).rgb;

    return multiscattering_contribution;
}

struct LuminanceRaymarchResult
{
    float3 luminance;
    float padd;
    float3 multiscattering;
    float padd_1;
};

struct IntegrateLuminanceInfo
{
    SkySettings * settings;
    float3 world_position;
    float3 world_direction;
    float3 sun_direction;
    int step_count;
    daxa::SamplerId sampler;
    daxa::Texture2DId<vector<float, 3>> transmittance_id;
    daxa::Texture2DId<vector<float, 3>> multiscattering_id;
};

LuminanceRaymarchResult integrate_luminance(const IntegrateLuminanceInfo info)
{
    const SkySettings * settings = info.settings;
    LuminanceRaymarchResult result = LuminanceRaymarchResult(float3(0.0f), 1.0f, float3(0.0f), 1.0f);

    const float integration_length = get_integration_length(info.world_position, info.world_direction, settings);
    // We do not intersect the atmosphere at all.
    if(integration_length == float.maxValue) { return result; }

    // We use uniform steps for transmittance calculation
    // const float step_length = integration_length / info.step_count;
    float step_length = integration_length / info.step_count;
    float3 optical_depth = float3(0.0f);

    // Figure out the phase function values. We can do this before we start the entire loop,
    // because neither sun_direction nor world_direction change during the raymarch process.
    // Thus the phase values also remain constant.
    const float cos_theta = dot(info.sun_direction, info.world_direction);
    const float mie_phase_value = hg_draine_phase(cos_theta, 3.6);
    const float rayleigh_phase_value = rayleigh_phase(cos_theta);

    float3 accumulated_transmittance = float3(1.0f);
    float old_ray_shift = 0.0f;
    // Raymarching loop
    for(int step = 0; step < info.step_count; ++step)
    {
        float new_ray_shift = integration_length * (float(step) + 0.3) / info.step_count;
        step_length = new_ray_shift - old_ray_shift;
        old_ray_shift = new_ray_shift;
        const float3 step_position = info.world_position + new_ray_shift * info.world_direction;
        // const float3 step_position = info.world_position + (step_length * step * info.world_direction);

        // The height is the distance from the atmosphere sphere origin, 
        // which is just the length of the position vector.
        const float height = length(step_position);
        // The zenith angle of the sun. The zenith angle is the angle between the sun and the vertical.
        // The vertical in our case is just vector from the atmosphere origin and current position.
        // Because atmosphere origin is in the origin of the current coordinate system, we can normalize
        // the current position to obtain the vertical.
        const float sun_angle = dot(info.sun_direction, normalize(step_position));
        TransmittanceParams transmittance_lut_params = TransmittanceParams(length(step_position), sun_angle);

        // Translate transmittance parameters into the uv coordinates of transmittance lut.
        const float2 transmittance_uv = transmittance_lut_to_uv(
            transmittance_lut_params, settings->atmosphere_bottom, settings->atmosphere_top);
        // Transmittance from the current position towards the sun
        const float3 transmittance_to_sun = info.transmittance_id.get()
            .SampleLevel(SamplerState::get(info.sampler), transmittance_uv, 0).rgb;

        const MediumSample medium_sample = sample_medium(settings, step_position);
        const float3 step_transmittance_increase = exp(-medium_sample.medium_extinction * step_length);

        // If the ray from the current position towards the sun intersects the bottom of the atmosphere
        // we are in the planets shadow. We assume planets surface is given by the bottom atmosphere radius.
        const float planet_intersection_distance = ray_sphere_intersect_nearest(
            step_position, info.sun_direction, float3(0.0f), settings->atmosphere_bottom);
        const float in_planet_shadow = (planet_intersection_distance == -1.0f) ? 1.0f : 0.0f;

        // This should only be used for sky lut NOT when we are building the multiscattering lut

        float3 multiscattering_lut_value = 0.0f;
        if(! info.multiscattering_id.id.is_empty()){
            multiscattering_lut_value = get_multiple_scattering(GetMultipleScatteringInfo(
                step_position,
                sun_angle,
                settings,
                info.sampler,
                info.multiscattering_id));
        }

        const float3 multiscattering_contribution =
            multiscattering_lut_value * (medium_sample.mie_scattering + medium_sample.rayleigh_scattering);

        // Multiply scattering intensities with respective phase functions.
        const float3 phase_times_scattering = 
            medium_sample.mie_scattering * mie_phase_value +
            medium_sample.rayleigh_scattering * rayleigh_phase_value;

        // Calculate the unitless light increase factor.
        const float3 medium_scattering = medium_sample.mie_scattering + medium_sample.rayleigh_scattering;
        const float3 unitless_light = in_planet_shadow * transmittance_to_sun * phase_times_scattering + multiscattering_contribution;
        
        //  Analytic integration of both multiscattering contribution and inscattering contribution.
        float3 multiscattered_contribution_integral = 
            (medium_scattering - medium_scattering * step_transmittance_increase) / medium_sample.medium_extinction;
        float3 inscattered_contribution_integral = 
            (unitless_light - unitless_light * step_transmittance_increase) / medium_sample.medium_extinction;
        
        // clear invalid values in case the extinction drops to zero (this can happen due to fp imprecision etc...)
        const bool3 extinction_zero = equal(medium_sample.medium_extinction, float3(0.0f));
        multiscattered_contribution_integral = select(extinction_zero, float3(0.0f), multiscattered_contribution_integral);
        inscattered_contribution_integral = select(extinction_zero, float3(0.0f), inscattered_contribution_integral);

        result.multiscattering += accumulated_transmittance * multiscattered_contribution_integral;
        result.luminance += accumulated_transmittance * inscattered_contribution_integral;
        accumulated_transmittance *= step_transmittance_increase;
    }
    return result;
}

static groupshared LuminanceRaymarchResult shared_data[2];
[numthreads(MULTISCATTERING_X, MULTISCATTERING_Y, MULTISCATTERING_Z)]
[shader("compute")]
void compute_multiscattering_lut(
    uint3 svdtid : SV_DispatchThreadID,
    uint gidx : SV_GroupIndex,
)
{
    const float SPHERE_SAMPLES = MULTISCATTERING_Z;
    const float GOLDEN_RATIO = 1.6180339f;

    const SkySettings * settings = &multiscattering_push.globals.sky_settings;
    float2 uv = (float2(svdtid.xy) + float2(0.5f)) / settings->multiscattering_dimensions.xy;
    uv = float2(from_subuv_to_unit(uv.x, settings->multiscattering_dimensions.x),
                from_subuv_to_unit(uv.y, settings->multiscattering_dimensions.y));

    const float sun_cos_zenith_angle = uv.x * 2.0f - 1.0f;
    const float3 sun_direction = float3(
        0.0f, safe_sqrt(clamp(1.0 - pow(sun_cos_zenith_angle, 2.0f), 0.0f, 1.0f)), sun_cos_zenith_angle);

    const float atmosphere_thickness = settings->atmosphere_top - settings->atmosphere_bottom;
    const float view_height = settings->atmosphere_bottom + uv.y * atmosphere_thickness + PLANET_RADIUS_OFFSET;

    // Fibbonaci lattice - Distribute points on a sphere (in our case these are directions in which we will raymrach)
    // http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/ */
    // MULTISCATTERING_Z is the number of sphere samples we want to take
    const float theta = acos(1.0f - 2.0f * (svdtid.z + 0.5f) / SPHERE_SAMPLES);
    const float phi = (2 * PI * svdtid.z) / GOLDEN_RATIO;

    const float3 world_position = float3(0.0f, 0.0f, view_height);
    const float3 world_direction = float3(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
    IntegrateLuminanceInfo info = IntegrateLuminanceInfo(
        settings,
        world_position,
        world_direction,
        sun_direction,
        settings->multiscattering_step_count,
        multiscattering_push.globals.samplers.linear_clamp,
        multiscattering_push.transmittance,
        daxa::Texture2DId<vector<float, 3>>(0),
    );
    LuminanceRaymarchResult result = integrate_luminance(info);

    // Here we do a reduction which takes all values stored in shared buffers and sums them 
    // all into a single value (which will be stored in the 0th and first element of the shared buffer)
    result.multiscattering = WaveActiveSum(result.multiscattering);
    result.luminance = WaveActiveSum(result.luminance);

    GroupMemoryBarrierWithGroupSync();
    if(WaveIsFirstLane())
    {
        const uint index = gidx > 0 ? 1 : 0;
        shared_data[index] = result;
    }
    GroupMemoryBarrierWithGroupSync();

    // Only a single thread writes the results into the texture.
    if(gidx == 0)
    {
        float3 multiscattering;
        float3 luminance_;
        const float3 multiscattering_sum = (shared_data[0].multiscattering + shared_data[1].multiscattering) / SPHERE_SAMPLES;
        const float3 luminance_sum = (shared_data[0].luminance + shared_data[1].luminance) / SPHERE_SAMPLES;

        const float3 r = multiscattering_sum;
        const float3 sum_of_all_multiscattering_events_contribution = float3(1.0 / (1.0 - r.x), 1.0 / (1.0 - r.y), 1.0 / (1.0 - r.z));
        const float3 luminance = luminance_sum * sum_of_all_multiscattering_events_contribution;

        multiscattering_push.multiscattering.get()[svdtid.xy] = luminance;
    }
}

[numthreads(SKY_X, SKY_Y, 1)]
[shader("compute")]
void compute_sky_lut(
    uint3 svdtid : SV_DispatchThreadID,
)
{
    const SkySettings * settings = &sky_push.globals.sky_settings;
    if(all(lessThan(svdtid.xy, settings->sky_dimensions.xy)))
    {
        // In game height is in meters, atmosphere uses kilometers so we need to traslate.
        float3 world_position = sky_push.globals.main_camera.position * M_TO_KM_SCALE;
        // Push the camera above the atmosphere with a set offset
        world_position.z += settings->atmosphere_bottom + BASE_HEIGHT_OFFSET;
        const float camera_height = length(world_position);

        const float2 uv = float2(svdtid.xy) / settings->sky_dimensions.xy;
        const SkyviewParams skyview_params = uv_to_skyview_lut_params(
            uv,
            settings->atmosphere_bottom,
            settings->atmosphere_top,
            settings->sky_dimensions,
            camera_height);

        const float3 world_direction = float3(
                cos(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
                sin(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
                cos(skyview_params.view_zenith_angle));

        const float3x3 camera_basis = build_orthonormal_basis(world_position / camera_height);
        const float3 normalized_world_position = float3(0, 0, camera_height);

        const float sun_zenith_cos_angle = dot(float3(0.0f, 0.0f, 1.0f), mul(settings->sun_direction, camera_basis));
        // sin^2 + cos^2 = 1 -> sqrt(1 - cos^2) = sin
        // rotate the sun direction so that we are aligned with the y = 0 axis
        const float3 local_sun_direction = normalize(float3(
            safe_sqrt(1.0 - sun_zenith_cos_angle * sun_zenith_cos_angle),
            0.0,
            sun_zenith_cos_angle));

        if (!move_to_top_atmosphere(world_position, world_direction, settings->atmosphere_bottom, settings->atmosphere_top))
        {
            /* No intersection with the atmosphere */
            sky_push.sky.get()[svdtid.xy] = float4(0.0f, 0.0f, 0.0f, 1.0f);
        } 
        else 
        {
            IntegrateLuminanceInfo info = IntegrateLuminanceInfo(
                settings,
                world_position,
                world_direction,
                local_sun_direction,
                settings->sky_step_count,
                sky_push.globals.samplers.linear_clamp,
                sky_push.transmittance,
                sky_push.multiscattering,
            );
            LuminanceRaymarchResult result = integrate_luminance(info);
            const float3 inv_luminance = 1.0 / max(result.luminance, float3(1.0 / 1048576.0));
            const float inv_mult = min(1048576.0, max(inv_luminance.x, max(inv_luminance.y, inv_luminance.z)));
            sky_push.sky.get()[svdtid.xy] = float4(result.luminance * inv_mult, 1.0f/inv_mult);
        }
    }
}
