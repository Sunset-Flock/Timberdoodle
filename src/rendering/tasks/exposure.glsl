#pragma once

#define AE_BASE_EV (-2.0)
#define AE_BASE_EV_INFLUENCE (0.4f)
#define AUTO_EXPOSURE_BIAS 0

float eval_exposure_compensation_curve_ev(float ev)
{
    ev = (ev + AE_BASE_EV) * 0.1 + atan(AE_BASE_EV_INFLUENCE * (ev + AE_BASE_EV)) / AE_BASE_EV_INFLUENCE - AE_BASE_EV;
    return ev;
}

float auto_exposure_get_ev_smoothed(daxa_RWBufferPtr(AutoExposureHistogram) self)
{
	return (deref(self).ev_slow + deref(self).ev_fast) * 0.5 + AUTO_EXPOSURE_BIAS;
}
