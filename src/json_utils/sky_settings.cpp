#include "sky_settings.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

auto load_sky_settings(std::filesystem::path const & path_to_settings) -> SkySettings
{
    SkySettings settings = {};
    auto json = nlohmann::json::parse(std::ifstream(path_to_settings));
    auto read_val = [&json](auto const name, auto & val)
    {
        val = json[name];
    };

    auto read_vec = [&json](auto const name, auto & val)
    {
        val.x = json[name]["x"];
        val.y = json[name]["y"];
        if constexpr (requires(decltype(val) x) { x.z; }) val.z = json[name]["z"];
        if constexpr (requires(decltype(val) x) { x.w; }) val.w = json[name]["w"];
    };

    auto read_density_profile_layer = [&json](auto const name, auto const layer, DensityProfileLayer & val)
    {
        val.layer_width = json[name][layer]["layer_width"];
        val.exp_term = json[name][layer]["exp_term"];
        val.exp_scale = json[name][layer]["exp_scale"];
        val.lin_term = json[name][layer]["lin_term"];
        val.const_term = json[name][layer]["const_term"];
    };
    read_vec("transmittance_dimensions", settings.transmittance_dimensions);
    read_vec("multiscattering_dimensions", settings.multiscattering_dimensions);
    read_vec("sky_dimensions", settings.sky_dimensions);
    read_val("transmittance_step_count", settings.transmittance_step_count);
    read_val("multiscattering_step_count", settings.multiscattering_step_count);
    read_val("sky_step_count", settings.sky_step_count);

    f32vec2 sun_angle = {};
    read_vec("sun_angle", sun_angle);

    read_val("atmosphere_bottom", settings.atmosphere_bottom);
    read_val("atmosphere_top", settings.atmosphere_top);

    // Mie
    read_vec("mie_scattering", settings.mie_scattering);
    read_vec("mie_extinction", settings.mie_extinction);
    read_val("mie_scale_height", settings.mie_scale_height);
    read_val("mie_phase_function_g", settings.mie_phase_function_g);
    read_density_profile_layer("mie_density", 0, settings.mie_density[0]);
    read_density_profile_layer("mie_density", 1, settings.mie_density[1]);

    // Rayleigh
    read_vec("rayleigh_scattering", settings.rayleigh_scattering);
    read_val("rayleigh_scale_height", settings.rayleigh_scale_height);
    read_density_profile_layer("rayleigh_density", 0, settings.rayleigh_density[0]);
    read_density_profile_layer("rayleigh_density", 1, settings.rayleigh_density[1]);

    // Absorption
    read_vec("absorption_extinction", settings.absorption_extinction);
    read_density_profile_layer("absorption_density", 0, settings.absorption_density[0]);
    read_density_profile_layer("absorption_density", 1, settings.absorption_density[1]);

    settings.sun_direction =
        {
            daxa_f32(glm::cos(glm::radians(sun_angle.x)) * glm::sin(glm::radians(sun_angle.y))),
            daxa_f32(glm::sin(glm::radians(sun_angle.x)) * glm::sin(glm::radians(sun_angle.y))),
            daxa_f32(glm::cos(glm::radians(sun_angle.y))),
        };
    settings.sun_brightness = 10.0f;
    return settings;
}
