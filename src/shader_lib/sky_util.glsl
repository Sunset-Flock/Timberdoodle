#pragma once

#include <daxa/daxa.inl>
#include "../shader_shared/shared.inl"
#include "../shader_shared/globals.inl"

#define PLANET_RADIUS_OFFSET 0.01
#define BASE_HEIGHT_OFFSET 3.0
#define PI 3.1415926535897932384626433832795
#define M_TO_KM_SCALE 0.001
#define sun_color (daxa_f32vec4(255.0, 240.0, 233.0, 255.0)/255.0)

// Building an Orthonormal Basis, Revisited
// http://jcgt.org/published/0006/01/01/
daxa_f32mat3x3 build_orthonormal_basis(daxa_f32vec3 n) {
    daxa_f32vec3 b1;
    daxa_f32vec3 b2;

    if (n.z < 0.0) {
        const daxa_f32 a = 1.0 / (1.0 - n.z);
        const daxa_f32 b = n.x * n.y * a;
        b1 = daxa_f32vec3(1.0 - n.x * n.x * a, -b, n.x);
        b2 = daxa_f32vec3(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        const daxa_f32 a = 1.0 / (1.0 + n.z);
        const daxa_f32 b = -n.x * n.y * a;
        b1 = daxa_f32vec3(1.0 - n.x * n.x * a, b, -n.x);
        b2 = daxa_f32vec3(b, 1.0 - n.y * n.y * a, -n.y);
    }
#if (DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG)
    return transpose(daxa_f32mat3x3(b1, b2, n));
#else
    return daxa_f32mat3x3(b1, b2, n);
#endif
}

/* Return sqrt clamped to 0 */
daxa_f32 safe_sqrt(daxa_f32 x)
{
    return sqrt(max(0, x));
}

daxa_f32 from_subuv_to_unit(daxa_f32 u, daxa_f32 resolution)
{
    return (u - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

daxa_f32 from_unit_to_subuv(daxa_f32 u, daxa_f32 resolution)
{
    return (u + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

struct TransmittanceParams
{
    daxa_f32 height;
    daxa_f32 zenith_cos_angle;
};

///	Transmittance LUT uses not uniform mapping -> transfer from mapping to texture uv
///	@param parameters
/// @param atmosphere_bottom - bottom radius of the atmosphere in km
/// @param atmosphere_top - top radius of the atmosphere in km
///	@return - uv of the corresponding texel
daxa_f32vec2 transmittance_lut_to_uv(TransmittanceParams parameters, daxa_f32 atmosphere_bottom, daxa_f32 atmosphere_top)
{
    daxa_f32 H = safe_sqrt(atmosphere_top * atmosphere_top - atmosphere_bottom * atmosphere_bottom);
    daxa_f32 rho = safe_sqrt(parameters.height * parameters.height - atmosphere_bottom * atmosphere_bottom);

    daxa_f32 discriminant = parameters.height * parameters.height *
                             (parameters.zenith_cos_angle * parameters.zenith_cos_angle - 1.0) +
                         atmosphere_top * atmosphere_top;
    /* Distance to top atmosphere boundary */
    daxa_f32 d = max(0.0, (-parameters.height * parameters.zenith_cos_angle + safe_sqrt(discriminant)));

    daxa_f32 d_min = atmosphere_top - parameters.height;
    daxa_f32 d_max = rho + H;
    daxa_f32 mu = (d - d_min) / (d_max - d_min);
    daxa_f32 r = rho / H;

    return daxa_f32vec2(mu, r);
}

/// Transmittance LUT uses not uniform mapping -> transfer from uv to this mapping
/// @param uv - uv in the range [0,1]
/// @param atmosphere_bottom - bottom radius of the atmosphere in km
/// @param atmosphere_top - top radius of the atmosphere in km
/// @return - TransmittanceParams structure
TransmittanceParams uv_to_transmittance_lut_params(daxa_f32vec2 uv, daxa_f32 atmosphere_bottom, daxa_f32 atmosphere_top)
{
    TransmittanceParams params;
    daxa_f32 H = safe_sqrt(atmosphere_top * atmosphere_top - atmosphere_bottom * atmosphere_bottom.x);

    daxa_f32 rho = H * uv.y;
    params.height = safe_sqrt(rho * rho + atmosphere_bottom * atmosphere_bottom);

    daxa_f32 d_min = atmosphere_top - params.height;
    daxa_f32 d_max = rho + H;
    daxa_f32 d = d_min + uv.x * (d_max - d_min);

    params.zenith_cos_angle = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * params.height * d);
    params.zenith_cos_angle = clamp(params.zenith_cos_angle, -1.0, 1.0);

    return params;
}

struct SkyviewParams
{
    daxa_f32 view_zenith_angle;
    daxa_f32 light_view_angle;
};

/// Get skyview LUT uv from skyview parameters
/// @param daxa_i32ersects_ground - true if ray daxa_i32ersects ground false otherwise
/// @param params - SkyviewParams structure
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
/// @param skyview_dimensions - skyViewLUT dimensions
/// @param view_height - view_height in world coordinates -> distance from planet center
/// @return - uv for the skyview LUT sampling
daxa_f32vec2 skyview_lut_params_to_uv(bool intersects_ground, SkyviewParams params, daxa_f32 atmosphere_bottom, daxa_f32 atmosphere_top, daxa_f32vec2 skyview_dimensions, daxa_f32 view_height)
{
    daxa_f32vec2 uv;
    if(view_height < atmosphere_top)
    {
        daxa_f32 beta = asin(atmosphere_bottom / view_height);
        daxa_f32 zenith_horizon_angle = PI - beta;

        if (!intersects_ground)
        {
            daxa_f32 coord = params.view_zenith_angle / zenith_horizon_angle;
            coord = (1.0 - safe_sqrt(1.0 - coord)) / 2.0;
            uv.y = coord;
        }
        else
        {
            daxa_f32 coord = (params.view_zenith_angle - zenith_horizon_angle) / beta;
            coord = (safe_sqrt(coord) + 1.0) / 2.0;
            uv.y = coord;
        }
    } else {
        daxa_f32 beta = asin(atmosphere_top / view_height);
        daxa_f32 zenith_horizon_angle = PI - beta;
        daxa_f32 coord = (params.view_zenith_angle - zenith_horizon_angle) / beta;
        coord = safe_sqrt(coord);
        uv.y = coord;
    }
    uv.x = safe_sqrt(params.light_view_angle / PI);
    uv = daxa_f32vec2(from_unit_to_subuv(uv.x, skyview_dimensions.x),
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
SkyviewParams uv_to_skyview_lut_params(daxa_f32vec2 uv, daxa_f32 atmosphere_bottom, daxa_f32 atmosphere_top, daxa_f32vec2 skyview_dimensions, daxa_f32 view_height) 
{
    /* Constrain uvs to valid sub texel range
    (avoid zenith derivative issue making LUT usage visible) */
    uv = daxa_f32vec2(from_subuv_to_unit(uv.x, skyview_dimensions.x),
        from_subuv_to_unit(uv.y, skyview_dimensions.y));

    daxa_f32 view_zenith_angle;
    daxa_f32 light_view_angle;
    if(view_height < atmosphere_top)
    {
        daxa_f32 beta = asin(atmosphere_bottom / view_height);
        daxa_f32 zenith_horizon_angle = PI - beta;

        /* Nonuniform mapping near the horizon to avoid artefacts */
        if (uv.y < 0.5)
        {
            daxa_f32 coord = 1.0 - (1.0 - 2.0 * uv.y) * (1.0 - 2.0 * uv.y);
            view_zenith_angle = zenith_horizon_angle * coord;
        }
        else
        {
            daxa_f32 coord = (uv.y * 2.0 - 1.0) * (uv.y * 2.0 - 1.0);
            view_zenith_angle = zenith_horizon_angle + beta * coord;
        }
    } else {
        daxa_f32 beta = asin(atmosphere_top / view_height);
        daxa_f32 zenith_horizon_angle = PI - beta;
        daxa_f32 coord = uv.y * uv.y;
        view_zenith_angle = zenith_horizon_angle + beta * coord;
    }
    light_view_angle = (uv.x * uv.x) * PI;
    return SkyviewParams(view_zenith_angle, light_view_angle);
}

/// Return distance of the first daxa_i32ersection between ray and sphere
/// @param r0 - ray origin
/// @param rd - normalized ray direction
/// @param s0 - sphere center
/// @param sR - sphere radius
/// @return return distance of daxa_i32ersection or -1.0 if there is no daxa_i32ersection
daxa_f32 ray_sphere_intersect_nearest(daxa_f32vec3 r0, daxa_f32vec3 rd, daxa_f32vec3 s0, daxa_f32 sR)
{
    daxa_f32 a = dot(rd, rd);
    daxa_f32vec3 s0_r0 = r0 - s0;
    daxa_f32 b = 2.0 * dot(rd, s0_r0);
    daxa_f32 c = dot(s0_r0, s0_r0) - (sR * sR);
    daxa_f32 delta = b * b - 4.0 * a * c;
    if (delta < 0.0 || a == 0.0)
    {
        return -1.0;
    }
    daxa_f32 sol0 = (-b - safe_sqrt(delta)) / (2.0 * a);
    daxa_f32 sol1 = (-b + safe_sqrt(delta)) / (2.0 * a);
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

/// Moves to the nearest daxa_i32ersection with top of the atmosphere in the direction specified in
/// world_direction
/// @param world_position - current world position -> will be changed to new pos at the top of
/// 		the atmosphere if there exists such daxa_i32ersection
/// @param world_direction - the direction in which the shift will be done
/// @param atmosphere_bottom - bottom of the atmosphere in km
/// @param atmosphere_top - top of the atmosphere in km
bool move_to_top_atmosphere(inout daxa_f32vec3 world_position, daxa_f32vec3 world_direction,
    daxa_f32 atmosphere_bottom, daxa_f32 atmosphere_top)
{
    daxa_f32vec3 planet_origin = daxa_f32vec3(0.0, 0.0, 0.0);
    /* Check if the world_position is outside of the atmosphere */
    if (length(world_position) > atmosphere_top)
    {
        daxa_f32 dist_to_top_atmo_daxa_i32ersection = ray_sphere_intersect_nearest(
            world_position, world_direction, planet_origin, atmosphere_top);

        /* No daxa_i32ersection with the atmosphere */
        if (dist_to_top_atmo_daxa_i32ersection == -1.0) { return false; }
        else
        {
            // bias the world position to be slightly inside the sphere
            const daxa_f32 BIAS = uintBitsToFloat(0x3f800040); // udaxa_i32BitsTodaxa_f32(0x3f800040) == 1.00000762939453125
            world_position += world_direction * (dist_to_top_atmo_daxa_i32ersection * BIAS);
            daxa_f32vec3 up_offset = normalize(world_position) * -PLANET_RADIUS_OFFSET;
            world_position += up_offset;
        }
    }
    /* Position is in or at the top of the atmosphere */
    return true;
}

daxa_f32 sample_profile_density(daxa_BufferPtr(DensityProfileLayer) profile, daxa_f32 above_surface_height)
{
    daxa_i32 layer_index = -1;
    daxa_f32 curr_layer_end = 0.0;
    for(daxa_i32 i = 0; i < PROFILE_LAYER_COUNT; i++)
    {
        curr_layer_end += deref_i(profile, i).layer_width;
        if(above_surface_height < curr_layer_end)
        {
            layer_index = i;
            break;
        }
    }
    // Not in any layer
    if(layer_index == -1) { return 0.0; }
    return deref_i(profile, layer_index).exp_term * exp(deref_i(profile, layer_index).exp_scale * above_surface_height) +
           deref_i(profile, layer_index).lin_term * above_surface_height +
           deref_i(profile, layer_index).const_term;
    return 0.0;
}

struct MediumSample
{
    daxa_f32vec3 mie_scattering;
    daxa_f32vec3 rayleigh_scattering;
    daxa_f32vec3 medium_extinction;
};
/// @param params - buffer reference to the atmosphere parameters buffer
/// @param position - position in the world where the sample is to be taken
/// @return atmosphere extinction at the desired podaxa_i32
MediumSample sample_medium(daxa_BufferPtr(SkySettings) params, daxa_f32vec3 position)
{
    const daxa_f32 above_surface_height = length(position) - deref(params).atmosphere_bottom;

    daxa_BufferPtr(DensityProfileLayer) mie_density_ptr = deref(params).mie_density_ptr;
    daxa_BufferPtr(DensityProfileLayer) rayleigh_density_ptr = deref(params).rayleigh_density_ptr;
    daxa_BufferPtr(DensityProfileLayer) absorption_density_ptr = deref(params).absorption_density_ptr;
    const daxa_f32 density_mie = max(0.0,sample_profile_density(mie_density_ptr, above_surface_height));
    const daxa_f32 density_ray = max(0.0,sample_profile_density(rayleigh_density_ptr, above_surface_height));
    const daxa_f32 density_ozo = max(0.0,sample_profile_density(absorption_density_ptr, above_surface_height));

    const daxa_f32vec3 mie_extinction = deref(params).mie_extinction * density_mie;
    const daxa_f32vec3 ray_extinction = deref(params).rayleigh_scattering * density_ray;
    const daxa_f32vec3 ozo_extinction = deref(params).absorption_extinction * density_ozo;
    const daxa_f32vec3 medium_extinction = mie_extinction + ray_extinction + ozo_extinction;

    const daxa_f32vec3 mie_scattering = deref(params).mie_scattering * density_mie;
    const daxa_f32vec3 ray_scattering = deref(params).rayleigh_scattering * density_ray;

    return MediumSample(mie_scattering, ray_scattering, medium_extinction);
}

#define LAYERS 5.0

// License: Unknown, author: Unknown, found: don't remember
daxa_f32 tanh_approx(daxa_f32 x) {
    //  Found this somewhere on the daxa_i32erwebs
    //  return tanh(x);
    daxa_f32 x2 = x * x;
    return clamp(x * (27.0 + x2) / (27.0 + 9.0 * x2), -1.0, 1.0);
}

// License: MIT OR CC-BY-NC-4.0, author: mercury, found: https://mercury.sexy/hg_sdf/
daxa_f32vec2 mod2(inout daxa_f32vec2 p, daxa_f32vec2 size) {
    daxa_f32vec2 c = floor((p + size * 0.5) / size);
    p = _mod(p + size * 0.5, size) - size * 0.5;
    return c;
}

// License: Unknown, author: Unknown, found: don't remember
daxa_f32vec2 hash2(daxa_f32vec2 p) {
    p = daxa_f32vec2(dot(p, daxa_f32vec2(127.1, 311.7)), dot(p, daxa_f32vec2(269.5, 183.3)));
    return _frac(sin(p) * 43758.5453123);
}

daxa_f32vec3 toSpherical(daxa_f32vec3 p) {
    daxa_f32 r = length(p);
    daxa_f32 t = acos(p.z / r);
    daxa_f32 ph = atan(p.y / p.x);
    return daxa_f32vec3(r, t, ph);
}

// License: CC BY-NC-SA 3.0, author: Stephane Cuillerdier - Aiekick/2015 (twitter:@aiekick), found: https://www.shadertoy.com/view/Mt3GW2
daxa_f32vec3 blackbody(daxa_f32 Temp) {
    daxa_f32vec3 col = daxa_f32vec3(255.);
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
daxa_f32vec3 stars(daxa_f32vec3 ro, daxa_f32vec3 rd, daxa_f32vec2 sp, daxa_f32 hh) {
    daxa_f32vec3 col = daxa_f32vec3(0.0);

    const daxa_f32 m = LAYERS;
    hh = tanh_approx(20.0 * hh);

    for (daxa_f32 i = 0.0; i < m; ++i) {
        daxa_f32vec2 pp = sp + 0.5 * i;
        daxa_f32 s = i / (m - 1.0);
        daxa_f32vec2 dim = daxa_f32vec2(mix(0.05, 0.003, s) * PI);
        daxa_f32vec2 np = mod2(pp, dim);
        daxa_f32vec2 h = hash2(np + 127.0 + i);
        daxa_f32vec2 o = -1.0 + 2.0 * h;
        daxa_f32 y = sin(sp.x);
        pp += o * dim * 0.5;
        pp.y *= y;
        daxa_f32 l = length(pp);

        daxa_f32 h1 = _frac(h.x * 1667.0);
        daxa_f32 h2 = _frac(h.x * 1887.0);
        daxa_f32 h3 = _frac(h.x * 2997.0);

        daxa_f32vec3 scol = mix(8.0 * h2, 0.25 * h2 * h2, s) * blackbody(mix(3000.0, 22000.0, h1 * h1));

        daxa_f32vec3 ccol = col + exp(-(mix(6000.0, 2000.0, hh) / mix(2.0, 0.25, s)) * max(l - 0.001, 0.0)) * scol;
        col = h3 < y ? ccol : col;
    }

    return col;
}

daxa_f32vec3 get_star_radiance(daxa_f32vec3 view_direction) {
    daxa_f32vec3 ro = daxa_f32vec3(0.0, 0.0, 0.0);
    daxa_f32vec2 sp = toSpherical(view_direction.xzy).yz;
    daxa_f32 sf = 0.0;

    return stars(ro, view_direction, sp, sf) * 0.001;
}

daxa_f32vec3 get_sun_illuminance(
    daxa_BufferPtr(SkySettings) settings,
    daxa_ImageViewId transmittance,
    daxa_SamplerId lin_sampler,
    daxa_f32vec3 view_direction,
    daxa_f32 height,
    daxa_f32 zenith_cos_angle
    )
{
    const daxa_f32 sun_solid_angle = 0.25 * PI / 180.0;
    const daxa_f32 min_sun_cos_theta = cos(sun_solid_angle);

    const daxa_f32vec3 sun_direction = deref(settings).sun_direction;
    const daxa_f32 cos_theta = dot(view_direction, sun_direction);
    if(cos_theta >= min_sun_cos_theta) 
    {
        TransmittanceParams transmittance_lut_params = TransmittanceParams(height, zenith_cos_angle);
        daxa_f32vec2 transmittance_texture_uv = transmittance_lut_to_uv(
            transmittance_lut_params,
            deref(settings).atmosphere_bottom,
            deref(settings).atmosphere_top
        );
        
#if (DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG)
        daxa_f32vec3 transmittance_to_sun = Texture2D<float4>::get(transmittance).SampleLevel(SamplerState::get(lin_sampler), transmittance_texture_uv, 0).rgb;
#else
        daxa_f32vec3 transmittance_to_sun = texture( daxa_sampler2D( transmittance, lin_sampler), transmittance_texture_uv).rgb;
#endif
        return transmittance_to_sun * sun_color.rgb * deref(settings).sun_brightness;
    }
    return daxa_f32vec3(0.0);
}

// TODO(SAKY): precalculate value on cpu
daxa_f32vec3 get_atmo_position(daxa_BufferPtr(RenderGlobalData) globals)
{
    const daxa_f32vec3 atmo_camera_position = deref(globals).view_camera.position * M_TO_KM_SCALE;
    const daxa_f32vec3 bottom_atmo_offset = daxa_f32vec3(0,0, deref(globals).sky_settings.atmosphere_bottom + BASE_HEIGHT_OFFSET);
    const daxa_f32vec3 bottom_atmo_offset_camera_position = atmo_camera_position + bottom_atmo_offset;
    return bottom_atmo_offset_camera_position;
}

daxa_f32vec3 get_atmosphere_illuminance_along_ray(
    SkySettings settings,
    daxa_ImageViewId transmittance,
    daxa_ImageViewId sky,
    daxa_SamplerId lin_sampler,
    daxa_f32vec3 ray,
    daxa_f32vec3 atmo_position
)
{
    const daxa_f32 height = length(atmo_position);
    const daxa_f32mat3x3 basis = build_orthonormal_basis(atmo_position / height);
    ray = mul(ray, basis);
    const daxa_f32vec3 sun_direction = mul(settings.sun_direction, basis);

    const daxa_f32vec3 world_up = daxa_f32vec3(0.0, 0.0, 1.0);
    const daxa_f32 view_zenith_angle = acos(dot(ray, world_up));
    const daxa_f32 light_view_angle = acos(clamp(dot(
        normalize(daxa_f32vec3(sun_direction.xy, 0.0)),
        normalize(daxa_f32vec3(ray.xy, 0.0))
        ),-1.0, 1.0)
    );

    const daxa_f32 bottom_atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        daxa_f32vec3(0.0, 0.0, height),
        ray,
        daxa_f32vec3(0.0),
        settings.atmosphere_bottom
    );

    const daxa_f32 top_atmosphere_intersection_distance = ray_sphere_intersect_nearest(
        daxa_f32vec3(0.0, 0.0, height),
        ray,
        daxa_f32vec3(0.0),
        settings.atmosphere_top
    );

    const bool intersects_ground = bottom_atmosphere_intersection_distance >= 0.0;
    const bool intersects_sky = top_atmosphere_intersection_distance >= 0.0;

    daxa_f32vec3 atmosphere_transmittance = daxa_f32vec3(1.0);
    daxa_f32vec3 atmosphere_scattering_illuminance = daxa_f32vec3(0.0);

    if(intersects_sky)
    {
        daxa_f32vec2 sky_uv = skyview_lut_params_to_uv(
            intersects_ground,
            SkyviewParams(view_zenith_angle, light_view_angle),
            settings.atmosphere_bottom,
            settings.atmosphere_top,
            daxa_f32vec2(settings.sky_dimensions),
            height
        );

#if (DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG)
        const daxa_f32vec4 unitless_atmosphere_illuminance_mult = Texture2D<float4>::get(sky).SampleLevel(SamplerState::get(lin_sampler), sky_uv, 0);
#else
        const daxa_f32vec4 unitless_atmosphere_illuminance_mult = texture(daxa_sampler2D(sky, lin_sampler) , sky_uv).rgba;
#endif
        const daxa_f32vec3 unitless_atmosphere_illuminance = unitless_atmosphere_illuminance_mult.rgb * unitless_atmosphere_illuminance_mult.a;
        const daxa_f32vec3 sun_color_weighed_atmosphere_illuminance = sun_color.rgb * unitless_atmosphere_illuminance;
        atmosphere_scattering_illuminance = sun_color_weighed_atmosphere_illuminance * settings.sun_brightness;

        TransmittanceParams transmittance_lut_params = TransmittanceParams(height, dot(ray, world_up));
        daxa_f32vec2 transmittance_texture_uv = transmittance_lut_to_uv(
            transmittance_lut_params,
            settings.atmosphere_bottom,
            settings.atmosphere_top
        );

#if (DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG)
        atmosphere_transmittance = Texture2D<float4>::get(transmittance).SampleLevel(SamplerState::get(lin_sampler), transmittance_texture_uv, 0).rgb;
#else
        atmosphere_transmittance = texture( daxa_sampler2D(transmittance, lin_sampler), transmittance_texture_uv).rgb;
#endif
    }

    const daxa_f32mat3x3 sun_basis = build_orthonormal_basis(normalize(sun_direction));
    const daxa_f32vec3 stars_color = atmosphere_transmittance * get_star_radiance(mul(ray, sun_basis)) * daxa_f32(!intersects_ground);

    return atmosphere_scattering_illuminance + stars_color;
}