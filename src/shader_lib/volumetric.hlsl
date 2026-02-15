#pragma once

#include "daxa/daxa.inl"

#if DAXA_LANGUAGE == DAXA_LANGUAGE_CPP
#include "../shader_shared/shared.inl"
#elif DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG
#include "shader_shared/shared.inl"
#endif

// ============================================== PHASE FUNCTIONS ==============================================
SHARED_FUNCTION float rayleigh_phase(float cos_theta)
{
    float factor = 3.0f / (16.0f * float(PI));
    return factor * (1.0f + cos_theta * cos_theta);
}

SHARED_FUNCTION float henyey_greenstein_phase(float g, float cos_theta)
{
#if DAXA_LANGUAGE == DAXA_LANGUAGE_CPP
    return (1.0f / (4.0f * float(PI))) * ((1.0f - (g * g)) / std::pow((1.0f + (g * g) - (2.0f * g * cos_theta)), 3.0f / 2.0f));
#elif DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG
    return (1.0f / (4.0f * PI)) * ((1.0f - (g * g)) / pow((1.0f + (g * g) - (2.0f * g * cos_theta)), 3.0f / 2.0f));
#endif
}

// https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf
SHARED_FUNCTION float draine_phase(float alpha, float g, float cos_theta)
{
    return henyey_greenstein_phase(g, cos_theta) *
           ((1.0f + (alpha * cos_theta * cos_theta)) / (1.0f + (alpha * (1.0f / 3.0f) * (1.0f + (2.0f * g * g)))));
}

SHARED_FUNCTION float hg_draine_phase(float cos_theta, float diameter)
{
    const float g_hg = exp(-(0.0990567f / (diameter - 1.67154f)));
    const float g_d = exp(-(2.20679f / (diameter + 3.91029f)) - 0.428934f);
    const float alpha = exp(3.62489f - (8.29288f / (diameter + 5.52825f)));
    const float w_d = exp(-(0.599085f / (diameter - 0.641583f)) - 0.665888f);

    return ((1.0f - w_d) * draine_phase(0, g_hg, cos_theta)) + (w_d * draine_phase(alpha, g_d, cos_theta));
}

// Creating the atmospheric world of red dead redemption 2 - https://www.youtube.com/watch?v=9-HTvoBi0Iw&t=7100s
SHARED_FUNCTION float multi_octave_hg(float g, float cos_theta, float extinction, float w_0, float w_1, int octaves)
{
    const float base = henyey_greenstein_phase(g, cos_theta) * w_0;
    const float octaves_weight = (w_1 * extinction) / (octaves - 1);

    float octaves_phase = 0.0f;
    for(int octave = 1; octave <= octaves; ++octave)
    {
#if DAXA_LANGUAGE == DAXA_LANGUAGE_CPP
        octaves_phase += henyey_greenstein_phase(std::powf((2.0f/3.0f), float(octave)) * g, cos_theta);
#elif DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG
        octaves_phase += henyey_greenstein_phase(pow((2.0f/3.0f), octave) * g, cos_theta);
#endif
    }

    const float combined_octaves = base + octaves_weight * octaves_phase;
#if DAXA_LANGUAGE == DAXA_LANGUAGE_CPP
    const float backscatter_clamp = (1.0f / float(PI));// * std::max(0.2f, extinction * 0.1f);
    return std::max(combined_octaves, backscatter_clamp);
#elif DAXA_LANGUAGE == DAXA_LANGUAGE_SLANG
    const float backscatter_clamp = (1.0f / float(PI));// * max(0.2, extinction * 0.1);
    return max(combined_octaves, backscatter_clamp);
#endif
}