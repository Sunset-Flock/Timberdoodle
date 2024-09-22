#include <daxa/daxa.inl>
#include "shader_lib/sky_util.glsl"
#include "sky.inl"

#if defined(TRANSMITTANCE)
DAXA_DECL_PUSH_CONSTANT(ComputeTransmittanceH, push)
layout(local_size_x = TRANSMITTANCE_X_DISPATCH, local_size_y = TRANSMITTANCE_Y_DISPATCH) in;

vec3 integrate_transmittance(vec3 world_position, vec3 world_direction, uint sample_count)
{
    daxa_BufferPtr(SkySettings) settings = deref(push.globals).sky_settings_ptr;
    /* The length of ray between position and nearest atmosphere top boundary */
    float integration_length = ray_sphere_intersect_nearest(
        world_position,
        world_direction,
        vec3(0.0, 0.0, 0.0),
        deref(settings).atmosphere_top);

    float integration_step = integration_length / float(sample_count);

    /* Result of the integration */
    vec3 optical_depth = vec3(0.0, 0.0, 0.0);

    for (int i = 0; i < sample_count; i++)
    {
        /* Move along the world direction ray to new position */
        vec3 new_pos = world_position + i * integration_step * world_direction;
        vec3 atmosphere_extinction = sample_medium(settings, new_pos).medium_extinction;
        optical_depth += atmosphere_extinction * integration_step;
    }
    return optical_depth;
}

void main()
{
    daxa_BufferPtr(SkySettings) settings = deref(push.globals).sky_settings_ptr;
    if (all(lessThan(gl_GlobalInvocationID.xy, deref(settings).transmittance_dimensions)))
    {
        vec2 uv = vec2(gl_GlobalInvocationID.xy) / vec2(deref(settings).transmittance_dimensions.xy);

        TransmittanceParams mapping = uv_to_transmittance_lut_params(uv, deref(settings).atmosphere_bottom, deref(settings).atmosphere_top);

        vec3 world_position = vec3(0.0, 0.0, mapping.height);
        vec3 world_direction = vec3(
            safe_sqrt(1.0 - mapping.zenith_cos_angle * mapping.zenith_cos_angle),
            0.0,
            mapping.zenith_cos_angle);

        vec3 transmittance = exp(-integrate_transmittance(world_position, world_direction, 400));
        imageStore(daxa_image2D(push.transmittance), ivec2(gl_GlobalInvocationID.xy), vec4(transmittance, 1.0));
    }
}
#endif // TRANSMITTANCE

#if defined(MULTISCATTERING)
DAXA_DECL_PUSH_CONSTANT(ComputeMultiscatteringPush, push)
/* This number should match the number of local threads -> z dimension */
const float SPHERE_SAMPLES = 64.0;
const float GOLDEN_RATIO = 1.6180339;
const float uniformPhase = 1.0 / (4.0 * PI);

layout(local_size_x = 1, local_size_y = 1, local_size_z = uint(SPHERE_SAMPLES)) in;

shared vec3 multiscatt_shared[64];
shared vec3 luminance_shared[64];

struct RaymarchResult
{
    vec3 luminance;
    vec3 multiscattering;
};

RaymarchResult integrate_scattered_luminance(vec3 world_position, vec3 world_direction, vec3 sun_direction, float sample_count)
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attach.globals).sky_settings_ptr;
    RaymarchResult result = RaymarchResult(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));
    vec3 planet_zero = vec3(0.0, 0.0, 0.0);
    float planet_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(settings).atmosphere_bottom);
    float atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(settings).atmosphere_top);

    float integration_length;
    /* ============================= CALCULATE INTERSECTIONS ============================ */
    if ((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance == -1.0))
    {
        /* ray does not intersect planet or atmosphere -> no point in raymarching*/
        return result;
    }
    else if ((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance > 0.0))
    {
        /* ray intersects only atmosphere */
        integration_length = atmosphere_intersection_distance;
    }
    else if ((planet_intersection_distance > 0.0) && (atmosphere_intersection_distance == -1.0))
    {
        /* ray intersects only planet */
        integration_length = planet_intersection_distance;
    }
    else
    {
        /* ray intersects both planet and atmosphere -> return the first intersection */
        integration_length = min(planet_intersection_distance, atmosphere_intersection_distance);
    }
    float integration_step = integration_length / float(sample_count);

    /* stores accumulated transmittance during the raymarch process */
    vec3 accum_transmittance = vec3(1.0, 1.0, 1.0);
    /* stores accumulated light contribution during the raymarch process */
    vec3 accum_light = vec3(0.0, 0.0, 0.0);
    float old_ray_shift = 0;

    /* ============================= RAYMARCH ==========================================  */
    for (int i = 0; i < sample_count; i++)
    {
        /* Sampling at 1/3rd of the integration step gives better results for exponential
           functions */
        float new_ray_shift = integration_length * (float(i) + 0.3) / sample_count;
        integration_step = new_ray_shift - old_ray_shift;
        vec3 new_position = world_position + new_ray_shift * world_direction;
        old_ray_shift = new_ray_shift;

        /* Raymarch shifts the angle to the sun a bit recalculate */
        vec3 up_vector = normalize(new_position);
        TransmittanceParams transmittance_lut_params = TransmittanceParams(length(new_position), dot(sun_direction, up_vector));

        /* uv coordinates later used to sample transmittance texture */
        vec2 trans_texture_uv = transmittance_lut_to_uv(transmittance_lut_params, deref(settings).atmosphere_bottom, deref(settings).atmosphere_top);

        vec3 transmittance_to_sun = texture(daxa_sampler2D(push.attach.transmittance, push.sampler_id), trans_texture_uv).rgb;

        MediumSample m_sample = sample_medium(settings, new_position);
        vec3 medium_scattering = m_sample.mie_scattering + m_sample.rayleigh_scattering;
        vec3 medium_extinction = m_sample.medium_extinction;

        /* TODO: This probably should be a texture lookup altho might be slow*/
        vec3 trans_increase_over_integration_step = exp(-(medium_extinction * integration_step));
        /* Check if current position is in earth's shadow */
        float earth_intersection_distance = ray_sphere_intersect_nearest(
            new_position, sun_direction, planet_zero + PLANET_RADIUS_OFFSET * up_vector, deref(settings).atmosphere_bottom);
        float in_earth_shadow = earth_intersection_distance == -1.0 ? 1.0 : 0.0;

        /* Light arriving from the sun to this point */
        vec3 sunLight = in_earth_shadow * transmittance_to_sun * medium_scattering * uniformPhase;
        vec3 multiscattered_cont_int = (medium_scattering - medium_scattering * trans_increase_over_integration_step) / medium_extinction;
        vec3 inscatteredContInt = (sunLight - sunLight * trans_increase_over_integration_step) / medium_extinction;

        if (medium_extinction.r == 0.0)
        {
            multiscattered_cont_int.r = 0.0;
            inscatteredContInt.r = 0.0;
        }
        if (medium_extinction.g == 0.0)
        {
            multiscattered_cont_int.g = 0.0;
            inscatteredContInt.g = 0.0;
        }
        if (medium_extinction.b == 0.0)
        {
            multiscattered_cont_int.b = 0.0;
            inscatteredContInt.b = 0.0;
        }

        result.multiscattering += accum_transmittance * multiscattered_cont_int;
        accum_light += accum_transmittance * inscatteredContInt;
        accum_transmittance *= trans_increase_over_integration_step;
    }
    result.luminance = accum_light;
    return result;
    /* TODO: Check for bounced light off the earth */
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 64) in;
void main()
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attach.globals).sky_settings_ptr;
    const float sample_count = 20;

    vec2 uv = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5, 0.5)) / vec2(deref(settings).multiscattering_dimensions.xy);
    uv = vec2(from_subuv_to_unit(uv.x, deref(settings).multiscattering_dimensions.x),
        from_subuv_to_unit(uv.y, deref(settings).multiscattering_dimensions.y));

    /* Mapping uv to multiscattering LUT parameters
    TODO -> Is the range from 0.0 to -1.0 really needed? */
    float sun_cos_zenith_angle = uv.x * 2.0 - 1.0;
    vec3 sun_direction = vec3(
        0.0,
        safe_sqrt(clamp(1.0 - sun_cos_zenith_angle * sun_cos_zenith_angle, 0.0, 1.0)),
        sun_cos_zenith_angle);

    float view_height =
        deref(settings).atmosphere_bottom +
        clamp(uv.y + PLANET_RADIUS_OFFSET, 0.0, 1.0) *
            (deref(settings).atmosphere_top - deref(settings).atmosphere_bottom - PLANET_RADIUS_OFFSET);

    vec3 world_position = vec3(0.0, 0.0, view_height);

    float sample_idx = gl_LocalInvocationID.z;
    // local thread dependent raymarch
    {
#define USE_HILL_SAMPLING 0
#if USE_HILL_SAMPLING
#define SQRTSAMPLECOUNT 8
        const float sqrt_sample = float(SQRTSAMPLECOUNT);
        float i = 0.5 + float(sample_idx / SQRTSAMPLECOUNT);
        float j = 0.5 + mod(sample_idx, SQRTSAMPLECOUNT);
        float randA = i / sqrt_sample;
        float randB = j / sqrt_sample;

        float theta = 2.0 * PI * randA;
        float phi = PI * randB;
#else
        /* Fibbonaci lattice -> http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/ */
        float theta = acos(1.0 - 2.0 * (sample_idx + 0.5) / SPHERE_SAMPLES);
        float phi = (2 * PI * sample_idx) / GOLDEN_RATIO;
#endif

        vec3 world_direction = vec3(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
        RaymarchResult result = integrate_scattered_luminance(world_position, world_direction, sun_direction, sample_count);

        multiscatt_shared[gl_LocalInvocationID.z] = result.multiscattering / SPHERE_SAMPLES;
        luminance_shared[gl_LocalInvocationID.z] = result.luminance / SPHERE_SAMPLES;
    }

    groupMemoryBarrier();
    barrier();

    if (gl_LocalInvocationID.z < 32)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 32];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 32];
    }

    groupMemoryBarrier();
    barrier();

    if (gl_LocalInvocationID.z < 16)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 16];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 16];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z < 8)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 8];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 8];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z < 4)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 4];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 4];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z < 2)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 2];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 2];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z < 1)
    {
        multiscatt_shared[gl_LocalInvocationID.z] += multiscatt_shared[gl_LocalInvocationID.z + 1];
        luminance_shared[gl_LocalInvocationID.z] += luminance_shared[gl_LocalInvocationID.z + 1];
    }
    groupMemoryBarrier();
    barrier();
    if (gl_LocalInvocationID.z == 0)
    {
        vec3 multiscatt_sum = multiscatt_shared[0];
        vec3 inscattered_luminance_sum = luminance_shared[0];

        const vec3 r = multiscatt_sum;
        const vec3 sum_of_all_multiscattering_events_contribution = vec3(1.0 / (1.0 - r.x), 1.0 / (1.0 - r.y), 1.0 / (1.0 - r.z));
        vec3 lum = inscattered_luminance_sum * sum_of_all_multiscattering_events_contribution;

        imageStore(daxa_image2D(push.attach.multiscattering), ivec2(gl_GlobalInvocationID.xy), vec4(lum, 1.0));
    }
}
#endif // MULTISCATTERING

#if defined(SKY)
DAXA_DECL_PUSH_CONSTANT(ComputeSkyPush, push)

/* ============================= PHASE FUNCTIONS ============================ */
float cornette_shanks_mie_phase_function(float g, float cos_theta)
{
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + cos_theta * cos_theta) / pow(1.0 + g * g - 2.0 * g * -cos_theta, 1.5);
}
#define TAU 2 * PI
float klein_nishina_phase(float cos_theta, float e)
{
    return e / (TAU * (e * (1.0 - cos_theta) + 1.0) * log(2.0 * e + 1.0));
}

float rayleigh_phase(float cos_theta)
{
    float factor = 3.0 / (16.0 * PI);
    return factor * (1.0 + cos_theta * cos_theta);
}
// https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf
float draine_phase(float alpha, float g, float cos_theta)
{
    return (1.0 / (4.0 * PI)) *
           ((1.0 - (g * g)) / pow((1.0 + (g * g) - (2.0 * g * cos_theta)), 3.0 / 2.0)) *
           ((1.0 + (alpha * cos_theta * cos_theta)) / (1.0 + (alpha * (1.0 / 3.0) * (1.0 + (2.0 * g * g)))));
}

float hg_draine_phase(float cos_theta, float diameter)
{
    const float g_hg = exp(-(0.0990567 / (diameter - 1.67154)));
    const float g_d = exp(-(2.20679 / (diameter + 3.91029)) - 0.428934);
    const float alpha = exp(3.62489 - (0.599085 / (diameter + 5.52825)));
    const float w_d = exp(-(0.599085 / (diameter - 0.641583)) - 0.665888);
    return (1 - w_d) * draine_phase(0, g_hg, cos_theta) + w_d * draine_phase(alpha, g_d, cos_theta);
}
/* ========================================================================== */

vec3 get_multiple_scattering(vec3 world_position, float view_zenith_cos_angle)
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attach.globals).sky_settings_ptr;
    vec2 uv = clamp(vec2(view_zenith_cos_angle * 0.5 + 0.5,
                        (length(world_position) - deref(settings).atmosphere_bottom) /
                            (deref(settings).atmosphere_top - deref(settings).atmosphere_bottom)),
        0.0, 1.0);
    uv = vec2(from_unit_to_subuv(uv.x, deref(settings).multiscattering_dimensions.x),
        from_unit_to_subuv(uv.y, deref(settings).multiscattering_dimensions.y));

    return texture(daxa_sampler2D(push.attach.multiscattering, push.sampler_id), uv).rgb;
}

vec3 integrate_scattered_luminance(vec3 world_position, vec3 world_direction, vec3 sun_direction, int sample_count)
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attach.globals).sky_settings_ptr;
    vec3 planet_zero = vec3(0.0, 0.0, 0.0);
    float planet_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(settings).atmosphere_bottom);
    float atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        world_position, world_direction, planet_zero, deref(settings).atmosphere_top);

    float integration_length;
    /* ============================= CALCULATE INTERSECTIONS ============================ */
    if ((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance == -1.0))
    {
        /* ray does not intersect planet or atmosphere -> no point in raymarching*/
        return vec3(0.0, 0.0, 0.0);
    }
    else if ((planet_intersection_distance == -1.0) && (atmosphere_intersection_distance > 0.0))
    {
        /* ray intersects only atmosphere */
        integration_length = atmosphere_intersection_distance;
    }
    else if ((planet_intersection_distance > 0.0) && (atmosphere_intersection_distance == -1.0))
    {
        /* ray intersects only planet */
        integration_length = planet_intersection_distance;
    }
    else
    {
        /* ray intersects both planet and atmosphere -> return the first intersection */
        integration_length = min(planet_intersection_distance, atmosphere_intersection_distance);
    }

    float cos_theta = dot(sun_direction, world_direction);
    // float mie_phase_value = klein_nishina_phase(cos_theta, 2800.0);
    float mie_phase_value = hg_draine_phase(cos_theta, 3.6);
    // float mie_phase_value = cornette_shanks_mie_phase_function(deref(settings).mie_phase_function_g, -cos_theta);
    float rayleigh_phase_value = rayleigh_phase(cos_theta);

    vec3 accum_transmittance = vec3(1.0, 1.0, 1.0);
    vec3 accum_light = vec3(0.0, 0.0, 0.0);
    /* ============================= RAYMARCH ============================ */
    for (int i = 0; i < sample_count; i++)
    {
        /* Step size computation */
        float step_0 = float(i) / sample_count;
        float step_1 = float(i + 1) / sample_count;

        /* Nonuniform step size*/
        step_0 *= step_0;
        step_1 *= step_1;

        step_0 = step_0 * integration_length;
        step_1 = step_1 > 1.0 ? integration_length : step_1 * integration_length;
        /* Sample at one third of the integrated interval -> better results for exponential functions */
        float integration_step = step_0 + (step_1 - step_0) * 0.3;
        float d_int_step = step_1 - step_0;

        /* Position shift */
        vec3 new_position = world_position + integration_step * world_direction;
        MediumSample m_sample = sample_medium(settings, new_position);
        vec3 medium_extinction = m_sample.medium_extinction;

        vec3 up_vector = normalize(new_position);
        TransmittanceParams transmittance_lut_params = TransmittanceParams(length(new_position), dot(sun_direction, up_vector));

        /* uv coordinates later used to sample transmittance texture */
        vec2 trans_texture_uv = transmittance_lut_to_uv(transmittance_lut_params, deref(settings).atmosphere_bottom, deref(settings).atmosphere_top);
        vec3 transmittance_to_sun = texture(daxa_sampler2D(push.attach.transmittance, push.sampler_id), trans_texture_uv).rgb;

        vec3 phase_times_scattering = m_sample.mie_scattering * mie_phase_value + m_sample.rayleigh_scattering * rayleigh_phase_value;

        float earth_intersection_distance = ray_sphere_intersect_nearest(
            new_position, sun_direction, planet_zero, deref(settings).atmosphere_bottom);
        float in_earth_shadow = earth_intersection_distance == -1.0 ? 1.0 : 0.0;

        vec3 multiscattered_luminance = get_multiple_scattering(new_position, dot(sun_direction, up_vector));

        /* Light arriving from the sun to this point */
        vec3 sun_light =
            ((in_earth_shadow * transmittance_to_sun * phase_times_scattering) +
                (multiscattered_luminance * (m_sample.rayleigh_scattering + m_sample.mie_scattering))); // * deref(settings).sun_brightness;

        /* TODO: This probably should be a texture lookup*/
        vec3 trans_increase_over_integration_step = exp(-(medium_extinction * d_int_step));

        vec3 sun_light_integ = (sun_light - sun_light * trans_increase_over_integration_step) / medium_extinction;

        if (medium_extinction.r == 0.0) { sun_light_integ.r = 0.0; }
        if (medium_extinction.g == 0.0) { sun_light_integ.g = 0.0; }
        if (medium_extinction.b == 0.0) { sun_light_integ.b = 0.0; }

        accum_light += accum_transmittance * sun_light_integ;
        accum_transmittance *= trans_increase_over_integration_step;
    }
    return accum_light;
}

layout(local_size_x = SKY_X_DISPATCH, local_size_y = SKY_Y_DISPATCH) in;
void main()
{
    daxa_BufferPtr(SkySettings) settings = deref(push.attach.globals).sky_settings_ptr;
    if (all(lessThan(gl_GlobalInvocationID.xy, deref(settings).sky_dimensions)))
    {
        vec3 world_position = deref(push.attach.globals).camera.position * M_TO_KM_SCALE;
        world_position.z += deref(settings).atmosphere_bottom + BASE_HEIGHT_OFFSET;
        const float camera_height = length(world_position);

        vec2 uv = vec2(gl_GlobalInvocationID.xy) / vec2(deref(settings).sky_dimensions.xy);
        SkyviewParams skyview_params = uv_to_skyview_lut_params(
            uv,
            deref(settings).atmosphere_bottom,
            deref(settings).atmosphere_top,
            deref(settings).sky_dimensions,
            camera_height);
        vec3 ray_direction = vec3( cos(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
                sin(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
                cos(skyview_params.view_zenith_angle));

        const mat3 camera_basis = build_orthonormal_basis(world_position / camera_height);
        world_position = vec3(0, 0, camera_height);

        float sun_zenith_cos_angle = dot(vec3(0, 0, 1), deref(settings).sun_direction * camera_basis);
        // sin^2 + cos^2 = 1 -> sqrt(1 - cos^2) = sin
        // rotate the sun direction so that we are aligned with the y = 0 axis
        vec3 local_sun_direction = normalize(vec3(
            safe_sqrt(1.0 - sun_zenith_cos_angle * sun_zenith_cos_angle),
            0.0,
            sun_zenith_cos_angle));

        vec3 world_direction = vec3(
            cos(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
            sin(skyview_params.light_view_angle) * sin(skyview_params.view_zenith_angle),
            cos(skyview_params.view_zenith_angle));

        if (!move_to_top_atmosphere(world_position, world_direction, deref(settings).atmosphere_bottom, deref(settings).atmosphere_top))
        {
            /* No intersection with the atmosphere */
            imageStore(daxa_image2D(push.attach.sky), ivec2(gl_GlobalInvocationID.xy), vec4(0.0, 0.0, 0.0, 1.0));
            return;
        }
        const vec3 luminance = integrate_scattered_luminance(world_position, world_direction, local_sun_direction, 50);
        const vec3 inv_luminance = 1.0 / max(luminance, vec3(1.0 / 1048576.0));
        const float inv_mult = min(1048576.0, max(inv_luminance.x, max(inv_luminance.y, inv_luminance.z)));
        imageStore(daxa_image2D(push.attach.sky), ivec2(gl_GlobalInvocationID.xy), vec4(luminance * inv_mult, 1.0/inv_mult));
    }
}
#endif // SKY
#if defined(CUBEMAP)
DAXA_DECL_PUSH_CONSTANT(SkyIntoCubemapH, push)
layout(local_size_x = IBL_CUBE_X, local_size_y = IBL_CUBE_Y, local_size_z = IBL_CUBE_RES) in;

float radical_inverse_vdc(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, uint n) {
    return vec2(float(i + 1) / n, radical_inverse_vdc(i + 1));
}

mat3 CUBE_MAP_FACE_ROTATION(uint face) 
{
    switch (face) {
    case 0: return mat3(+0, +0, -1, +0, -1, +0, -1, +0, +0);
    case 1: return mat3(+0, +0, +1, +0, -1, +0, +1, +0, +0);
    case 2: return mat3(+1, +0, +0, +0, +0, +1, +0, -1, +0);
    case 3: return mat3(+1, +0, +0, +0, +0, -1, +0, +1, +0);
    case 4: return mat3(+1, +0, +0, +0, -1, +0, +0, +0, -1);
    default: return mat3(-1, +0, +0, +0, -1, +0, +0, +0, +1);
    }
}

uint _rand_state;
void rand_seed(uint seed) {
    _rand_state = seed;
}

float rand() {
    // https://www.pcg-random.org/
    _rand_state = _rand_state * 747796405u + 2891336453u;
    uint result = ((_rand_state >> ((_rand_state >> 28u) + 4u)) ^ _rand_state) * 277803737u;
    result = (result >> 22u) ^ result;
    return result / 4294967295.0;
}

float rand_normal_dist() {
    float theta = 2.0 * PI * rand();
    float rho = sqrt(-2.0 * log(rand()));
    return rho * cos(theta);
}

vec3 rand_dir() {
    return normalize(vec3(
        rand_normal_dist(),
        rand_normal_dist(),
        rand_normal_dist()));
}

vec3 rand_hemi_dir(vec3 nrm) {
    vec3 result = rand_dir();
    return result * sign(dot(nrm, result));
}

void main() {
    const uvec2 wg_base_pix_pos = gl_WorkGroupID.xy * uvec2(IBL_CUBE_X, IBL_CUBE_Y);
    const uvec2 sg_pix_pos = wg_base_pix_pos + uvec2(gl_SubgroupID % IBL_CUBE_X, gl_SubgroupID / IBL_CUBE_X);
    uint face = gl_WorkGroupID.z;
    vec2 uv = (vec2(sg_pix_pos) + vec2(0.5)) / IBL_CUBE_RES;

    vec3 output_dir = normalize(CUBE_MAP_FACE_ROTATION(face) * vec3(uv * 2 - 1, -1.0));
    const mat3 basis = build_orthonormal_basis(output_dir);

    daxa_BufferPtr(SkySettings) sky_settings_ptr = deref(push.globals).sky_settings_ptr;
    // Because the atmosphere is using km as it's default units and we want one unit in world
    // space to be one meter we need to scale the position by a factor to get from meters -> kilometers
    const vec3 camera_position = deref(push.globals).camera.position * M_TO_KM_SCALE;
    vec3 world_camera_position = camera_position + vec3(0.0, 0.0, deref(sky_settings_ptr).atmosphere_bottom + BASE_HEIGHT_OFFSET);
    const float height = length(world_camera_position);

    vec3 accumulated_result = vec3(0);

    // We hardcode the subgroup size to be 32
    const uint sample_count = 128;
    const uint subgroup_size = 32;
    const uint iter_count = sample_count / subgroup_size;
    const uint global_thread_index = (gl_GlobalInvocationID.x * IBL_CUBE_RES * IBL_CUBE_RES + gl_GlobalInvocationID.y * IBL_CUBE_RES + gl_GlobalInvocationID.z);
    const uint seed = global_thread_index + deref(push.globals).frame_index * IBL_CUBE_RES * IBL_CUBE_RES * 6;

    for (uint i = 0; i < iter_count; ++i) {
        rand_seed((i * subgroup_size + gl_SubgroupInvocationID + seed * sample_count));
        vec3 input_dir = rand_hemi_dir(output_dir);
        // TODO: Now that we sample the atmosphere directly, computing this IBL is really slow.
        // We should cache the IBL cubemap, and only re-render it when necessary.
        const vec3 result = get_atmosphere_illuminance_along_ray(
            sky_settings_ptr,
            push.transmittance,
            push.sky,
            deref(push.globals).samplers.linear_clamp,
            input_dir,
            world_camera_position
        );
        const vec3 cos_weighed_result = result * dot(output_dir, input_dir);
        accumulated_result += subgroupInclusiveAdd(cos_weighed_result);
    }
    // Only last thread in each subgroup contains the correct accumulated result
    if(gl_SubgroupInvocationID == 31)
    {
        const vec3 this_frame_luminance = accumulated_result / sample_count;
        const vec4 compressed_accumulated_luminance = imageLoad(daxa_image2DArray(push.ibl_cube), ivec3(sg_pix_pos, gl_WorkGroupID.z));
        // Could be nan for some reason
        const vec3 unsafe_accumulated_luminance = compressed_accumulated_luminance.rgb * compressed_accumulated_luminance.a;
        const vec3 accumulated_luminance = isnan(unsafe_accumulated_luminance.x) ? vec3(0.0) : unsafe_accumulated_luminance;
        const vec3 luminance = 0.995 * accumulated_luminance + 0.005 * this_frame_luminance;
        const vec3 inv_luminance = 1.0 / max(luminance, vec3(1.0 / 1048576.0));
        const float inv_mult = min(1048576.0, max(inv_luminance.x, max(inv_luminance.y, inv_luminance.z)));
        imageStore(daxa_image2DArray(push.ibl_cube), ivec3(sg_pix_pos, gl_WorkGroupID.z), vec4(luminance * inv_mult, 1.0/inv_mult));
    }
}
#endif //CUBEMAP