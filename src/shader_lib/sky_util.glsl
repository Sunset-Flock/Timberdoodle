#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/shared.inl"

const float PLANET_RADIUS_OFFSET = 0.01;
const float BASE_HEIGHT_OFFSET = 3.0;
const float PI = 3.1415926535897932384626433832795;
const float M_TO_KM_SCALE = 0.001;

/* Return sqrt clamped to 0 */
float safe_sqrt(float x)
{
    return sqrt(max(0, x));
}

float from_subuv_to_unit(float u, float resolution)
{
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

float from_unit_to_subuv(float u, float resolution)
{
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

struct TransmittanceParams
{
    float height;
    float zenith_cos_angle;
};

///	Transmittance LUT uses not uniform mapping -> transfer from mapping to texture uv
///	@param parameters
/// @param atmosphere_bottom - bottom radius of the atmosphere in km
/// @param atmosphere_top - top radius of the atmosphere in km
///	@return - uv of the corresponding texel
vec2 transmittance_lut_to_uv(TransmittanceParams parameters, float atmosphere_bottom, float atmosphere_top)
{
    float H = safe_sqrt(atmosphere_top * atmosphere_top - atmosphere_bottom * atmosphere_bottom);
    float rho = safe_sqrt(parameters.height * parameters.height - atmosphere_bottom * atmosphere_bottom);

    float discriminant = parameters.height * parameters.height *
                             (parameters.zenith_cos_angle * parameters.zenith_cos_angle - 1.0) +
                         atmosphere_top * atmosphere_top;
    /* Distance to top atmosphere boundary */
    float d = max(0.0, (-parameters.height * parameters.zenith_cos_angle + safe_sqrt(discriminant)));

    float d_min = atmosphere_top - parameters.height;
    float d_max = rho + H;
    float mu = (d - d_min) / (d_max - d_min);
    float r = rho / H;

    return vec2(mu, r);
}

/// Transmittance LUT uses not uniform mapping -> transfer from uv to this mapping
/// @param uv - uv in the range [0,1]
/// @param atmosphere_bottom - bottom radius of the atmosphere in km
/// @param atmosphere_top - top radius of the atmosphere in km
/// @return - TransmittanceParams structure
TransmittanceParams uv_to_transmittance_lut_params(vec2 uv, float atmosphere_bottom, float atmosphere_top)
{
    TransmittanceParams params;
    float H = safe_sqrt(atmosphere_top * atmosphere_top - atmosphere_bottom * atmosphere_bottom.x);

    float rho = H * uv.y;
    params.height = safe_sqrt(rho * rho + atmosphere_bottom * atmosphere_bottom);

    float d_min = atmosphere_top - params.height;
    float d_max = rho + H;
    float d = d_min + uv.x * (d_max - d_min);

    params.zenith_cos_angle = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * params.height * d);
    params.zenith_cos_angle = clamp(params.zenith_cos_angle, -1.0, 1.0);

    return params;
}

struct SkyviewParams
{
    float view_zenith_angle;
    float light_view_angle;
};

/// Get skyview LUT uv from skyview parameters
/// @param intersects_ground - true if ray intersects ground false otherwise
/// @param params - SkyviewParams structure
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
/// @param skyview_dimensions - skyViewLUT dimensions
/// @param view_height - view_height in world coordinates -> distance from planet center
/// @return - uv for the skyview LUT sampling
vec2 skyview_lut_params_to_uv(bool intersects_ground, SkyviewParams params, float atmosphere_bottom, float atmosphere_top, vec2 skyview_dimensions, float view_height)
{
    vec2 uv;
    float beta = asin(atmosphere_bottom / view_height);
    float zenith_horizon_angle = PI - beta;

    if (!intersects_ground)
    {
        float coord = params.view_zenith_angle / zenith_horizon_angle;
        coord = (1.0 - safe_sqrt(1.0 - coord)) / 2.0;
        uv.y = coord;
    }
    else
    {
        float coord = (params.view_zenith_angle - zenith_horizon_angle) / beta;
        coord = (safe_sqrt(coord) + 1.0) / 2.0;
        uv.y = coord;
    }
    uv.x = safe_sqrt(params.light_view_angle / PI);
    uv = vec2(from_unit_to_subuv(uv.x, skyview_dimensions.x),
        from_unit_to_subuv(uv.y, skyview_dimensions.y));
    return uv;
}

/// Get parameters used for skyview LUT computation from uv coords
/// @param uv - texel uv in the range [0,1]
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
/// @param skyview dimensions
/// @param view_height - view_height in world coordinates -> distance from planet center
/// @return - SkyviewParams structure
SkyviewParams uv_to_skyview_lut_params(vec2 uv, float atmosphere_bottom, float atmosphere_top, vec2 skyview_dimensions, float view_height)
{
    /* Constrain uvs to valid sub texel range
    (avoid zenith derivative issue making LUT usage visible) */
    uv = vec2(from_subuv_to_unit(uv.x, skyview_dimensions.x),
        from_subuv_to_unit(uv.y, skyview_dimensions.y));

    float beta = asin(atmosphere_bottom / view_height);
    float zenith_horizon_angle = PI - beta;

    float view_zenith_angle;
    float light_view_angle;
    /* Nonuniform mapping near the horizon to avoid artefacts */
    if (uv.y < 0.5)
    {
        float coord = 1.0 - (1.0 - 2.0 * uv.y) * (1.0 - 2.0 * uv.y);
        view_zenith_angle = zenith_horizon_angle * coord;
    }
    else
    {
        float coord = (uv.y * 2.0 - 1.0) * (uv.y * 2.0 - 1.0);
        view_zenith_angle = zenith_horizon_angle + beta * coord;
    }
    light_view_angle = (uv.x * uv.x) * PI;
    return SkyviewParams(view_zenith_angle, light_view_angle);
}

/// Return distance of the first intersection between ray and sphere
/// @param r0 - ray origin
/// @param rd - normalized ray direction
/// @param s0 - sphere center
/// @param sR - sphere radius
/// @return return distance of intersection or -1.0 if there is no intersection
float ray_sphere_intersect_nearest(vec3 r0, vec3 rd, vec3 s0, float sR)
{
    float a = dot(rd, rd);
    vec3 s0_r0 = r0 - s0;
    float b = 2.0 * dot(rd, s0_r0);
    float c = dot(s0_r0, s0_r0) - (sR * sR);
    float delta = b * b - 4.0 * a * c;
    if (delta < 0.0 || a == 0.0)
    {
        return -1.0;
    }
    float sol0 = (-b - safe_sqrt(delta)) / (2.0 * a);
    float sol1 = (-b + safe_sqrt(delta)) / (2.0 * a);
    if (sol0 < 0.0 && sol1 < 0.0)
    {
        return -1.0;
    }
    if (sol0 < 0.0)
    {
        return max(0.0, sol1);
    }
    else if (sol1 < 0.0)
    {
        return max(0.0, sol0);
    }
    return max(0.0, min(sol0, sol1));
}

/// Moves to the nearest intersection with top of the atmosphere in the direction specified in
/// world_direction
/// @param world_position - current world position -> will be changed to new pos at the top of
/// 		the atmosphere if there exists such intersection
/// @param world_direction - the direction in which the shift will be done
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
bool move_to_top_atmosphere(inout vec3 world_position, vec3 world_direction,
    float atmosphere_bottom, float atmosphere_top)
{
    vec3 planet_origin = vec3(0.0, 0.0, 0.0);
    /* Check if the world_position is outside of the atmosphere */
    if (length(world_position) > atmosphere_top)
    {
        float dist_to_top_atmo_intersection = ray_sphere_intersect_nearest(
            world_position, world_direction, planet_origin, atmosphere_top);

        /* No intersection with the atmosphere */
        if (dist_to_top_atmo_intersection == -1.0) { return false; }
        else
        {
            // bias the world position to be slightly inside the sphere
            const float BIAS = uintBitsToFloat(0x3f800040); // uintBitsToFloat(0x3f800040) == 1.00000762939453125
            world_position += world_direction * (dist_to_top_atmo_intersection * BIAS);
            vec3 up_offset = normalize(world_position) * -PLANET_RADIUS_OFFSET;
            world_position += up_offset;
        }
    }
    /* Position is in or at the top of the atmosphere */
    return true;
}

float sample_profile_density(daxa_BufferPtr(DensityProfileLayer) profile, float above_surface_height)
{
    int layer_index = -1;
    float curr_layer_end = 0.0;
    for(int i = 0; i < PROFILE_LAYER_COUNT; i++)
    {
        curr_layer_end += deref(profile[i]).layer_width;
        if(above_surface_height < curr_layer_end)
        {
            layer_index = i;
            break;
        }
    }
    // Not in any layer
    if(layer_index == -1) { return 0.0; }
    return deref(profile[layer_index]).exp_term * exp(deref(profile[layer_index]).exp_scale * above_surface_height) +
           deref(profile[layer_index]).lin_term * above_surface_height +
           deref(profile[layer_index]).const_term;
    return 0.0;
}

struct MediumSample
{
    vec3 mie_scattering;
    vec3 rayleigh_scattering;
    vec3 medium_extinction;
};
/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return atmosphere extinction at the desired point
MediumSample sample_medium(daxa_BufferPtr(SkySettings) params, vec3 position)
{
    const float above_surface_height = length(position) - deref(params).atmosphere_bottom;

    daxa_BufferPtr(DensityProfileLayer) mie_density_ptr = deref(params).mie_density_ptr;
    daxa_BufferPtr(DensityProfileLayer) rayleigh_density_ptr = deref(params).rayleigh_density_ptr;
    daxa_BufferPtr(DensityProfileLayer) absorption_density_ptr = deref(params).absorption_density_ptr;
    const float density_mie = max(0.0,sample_profile_density(mie_density_ptr, above_surface_height));
    const float density_ray = max(0.0,sample_profile_density(rayleigh_density_ptr, above_surface_height));
    const float density_ozo = max(0.0,sample_profile_density(absorption_density_ptr, above_surface_height));

    const vec3 mie_extinction = deref(params).mie_extinction * density_mie;
    const vec3 ray_extinction = deref(params).rayleigh_scattering * density_ray;
    const vec3 ozo_extinction = deref(params).absorption_extinction * density_ozo;
    const vec3 medium_extinction = mie_extinction + ray_extinction + ozo_extinction;

    const vec3 mie_scattering = deref(params).mie_scattering * density_mie;
    const vec3 ray_scattering = deref(params).rayleigh_scattering * density_ray;

    return MediumSample(mie_scattering, ray_scattering, medium_extinction);
}

#define LAYERS 5.0

// License: Unknown, author: Unknown, found: don't remember
float tanh_approx(float x) {
    //  Found this somewhere on the interwebs
    //  return tanh(x);
    float x2 = x * x;
    return clamp(x * (27.0 + x2) / (27.0 + 9.0 * x2), -1.0, 1.0);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
vec2 mod2(inout vec2 p, vec2 size) {
    vec2 c = floor((p + size * 0.5) / size);
    p = mod(p + size * 0.5, size) - size * 0.5;
    return c;
}

// License: Unknown, author: Unknown, found: don't remember
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453123);
}

vec3 toSpherical(vec3 p) {
    float r = length(p);
    float t = acos(p.z / r);
    float ph = atan(p.y, p.x);
    return vec3(r, t, ph);
}

// License: CC BY-NC-SA 3.0, author: Stephane Cuillerdier - Aiekick/2015 (twitter:@aiekick), found: https://www.shadertoy.com/view/Mt3GW2
vec3 blackbody(float Temp) {
    vec3 col = vec3(255.);
    col.x = 56100000. * pow(Temp, (-3. / 2.)) + 148.;
    col.y = 100.04 * log(Temp) - 623.6;
    if (Temp > 6500.)
        col.y = 35200000. * pow(Temp, (-3. / 2.)) + 184.;
    col.z = 194.18 * log(Temp) - 1448.6;
    col = clamp(col, 0., 255.) / 255.;
    if (Temp < 1000.)
        col *= Temp / 1000.;
    return col;
}

// https://www.shadertoy.com/view/stBcW1
// License CC0: Stars and galaxy
// Bit of sunday tinkering lead to stars and a galaxy
// Didn't turn out as I envisioned but it turned out to something
// that I liked so sharing it.
vec3 stars(vec3 ro, vec3 rd, vec2 sp, float hh) {
    vec3 col = vec3(0.0);

    const float m = LAYERS;
    hh = tanh_approx(20.0 * hh);

    for (float i = 0.0; i < m; ++i) {
        vec2 pp = sp + 0.5 * i;
        float s = i / (m - 1.0);
        vec2 dim = vec2(mix(0.05, 0.003, s) * PI);
        vec2 np = mod2(pp, dim);
        vec2 h = hash2(np + 127.0 + i);
        vec2 o = -1.0 + 2.0 * h;
        float y = sin(sp.x);
        pp += o * dim * 0.5;
        pp.y *= y;
        float l = length(pp);

        float h1 = fract(h.x * 1667.0);
        float h2 = fract(h.x * 1887.0);
        float h3 = fract(h.x * 2997.0);

        vec3 scol = mix(8.0 * h2, 0.25 * h2 * h2, s) * blackbody(mix(3000.0, 22000.0, h1 * h1));

        vec3 ccol = col + exp(-(mix(6000.0, 2000.0, hh) / mix(2.0, 0.25, s)) * max(l - 0.001, 0.0)) * scol;
        col = h3 < y ? ccol : col;
    }

    return col;
}

vec3 get_star_radiance(vec3 view_direction) {
    vec3 ro = vec3(0.0, 0.0, 0.0);
    vec2 sp = toSpherical(view_direction.xzy).yz;
    float sf = 0.0;

    return stars(ro, view_direction, sp, sf) * 0.001;
}