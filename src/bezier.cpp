#include "bezier.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

BezierCurve::BezierCurve()
{
}

void BezierCurve::load_from_file(std::filesystem::path const & path)
{
    auto json = nlohmann::json::parse(std::ifstream(path));

    for(auto const & path : json["paths"])
    {
        for(auto const & segment : path)
        {
            auto read_segment_point = [&](auto const & seg_name, auto & dst)
            {
                dst.x = segment[seg_name]["x"];
                dst.y = segment[seg_name]["y"];
                dst.z = segment[seg_name]["z"];
            };
            auto & curr_segment = bezier_segments.emplace_back();
            read_segment_point("s", curr_segment.s);
            read_segment_point("c1", curr_segment.c1);
            read_segment_point("c2", curr_segment.c2);
            read_segment_point("e", curr_segment.e);
        }
    }
}

auto BezierCurve::as_line_strip() -> std::vector<f32vec3>
{
    std::vector<f32vec3> ret;
    ret.reserve(bezier_segments.size() + 1);
    ret.push_back(bezier_segments.front().s);
    for(auto const & segment : bezier_segments)
    {
        ret.push_back(segment.e);
    }
    return ret;
}