#pragma once

#include "daxa/daxa.inl"
#include "shared.inl"

#define RTGI_DIFFUSE_PIXEL_SCALE_DIV 2

struct RtgiSettings
{
    daxa_i32 enabled TIDO_DEFAULT_VALUE(1);
    daxa_i32 history_frames TIDO_DEFAULT_VALUE(32);
};