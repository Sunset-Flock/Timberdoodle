#pragma once

#include "timberdoodle.hpp"
using namespace tido::types;

struct BezierSegment
{
    f32vec3 s = {};
    f32vec3 c1 = {};
    f32vec3 c2 = {};
    f32vec3 e = {};
};

struct BezierCurve
{
  public:
    BezierCurve();
    void load_from_file(std::filesystem::path const & path);
    auto as_line_strip() -> std::vector<f32vec3>;

    std::vector<BezierSegment> bezier_segments = {};
};