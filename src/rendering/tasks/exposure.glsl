#pragma once

#define AE_BASE_EV (-2.0)
#define AE_BASE_EV_INFLUENCE (0.4f)
#define AUTO_EXPOSURE_BIAS 0
// Dark side: 0 = old atan curve (brightening capped at ~+7 EV), 1 = 1:1 up to the dark limit knee. In between: blend.
#define AE_DARK_CURVE_BLEND 0.5

// Maps the desired EV (-log2 of the scene luminance; higher = darker scene, more brightening) to the applied EV.
// Bright side (below the center -AE_BASE_EV): soft atan compression as before. Dark side: follows the desired EV 1:1
// up to a knee at 3/4 of the way to dark_limit_ev, then saturates softly at dark_limit_ev (tanh), blended with the old
// symmetric atan curve by AE_DARK_CURVE_BLEND. The atan curve alone capped brightening at ~+7 EV (desired 10 -> applied
// 6, dark scenes ~4 stops too dark); 1:1 alone brightened dark scenes too much.
float eval_exposure_compensation_curve_ev(float ev, float dark_limit_ev)
{
    const float center = -AE_BASE_EV;
    const float d = ev - center;
    const float atan_curve = d * 0.1 + atan(AE_BASE_EV_INFLUENCE * d) / AE_BASE_EV_INFLUENCE + center;
    if (d <= 0.0)
    {
        return atan_curve;
    }
    const float range = max(dark_limit_ev - center, 0.01);
    const float knee = range * 0.75;
    float dark_curve = center + d;
    if (d > knee)
    {
        const float rest = range - knee;
        dark_curve = center + knee + rest * tanh((d - knee) / rest);
    }
    return mix(atan_curve, dark_curve, AE_DARK_CURVE_BLEND);
}

float auto_exposure_get_ev_smoothed(daxa_RWBufferPtr(AutoExposureHistogram) self)
{
	return (deref(self).ev_slow + deref(self).ev_fast) * 0.5 + AUTO_EXPOSURE_BIAS;
}
