#include "camera_animation.hpp"
#include <nlohmann/json.hpp>
#include <fstream>

auto load_camera_animation(std::filesystem::path const & path) -> std::vector<CameraAnimationKeyframe>
{
    std::vector<CameraAnimationKeyframe> keyframes = {};
    auto file = std::ifstream(path);
    if (!file.is_open()) return keyframes;
    auto json = nlohmann::json::parse(file);
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
            auto read_rotation = [&](auto const & rot_name, auto & dst)
            {
                dst = {
                    segment[rot_name]["x"],
                    segment[rot_name]["y"],
                    segment[rot_name]["z"],
                    segment[rot_name]["w"]
                };
            };
            auto & curr_keyframe = keyframes.emplace_back();
            read_rotation("rot", curr_keyframe.rotation);
            read_segment_point("s", curr_keyframe.position);
            curr_keyframe.transition_time = segment["time"];
        }
    }
    return keyframes;
}

void export_camera_animation(std::filesystem::path const & path, std::vector<CameraAnimationKeyframe> const & keyframes)
{
    auto json = nlohmann::json {};

    auto path_ = nlohmann::json{};

    for (auto const & segment : keyframes)
    {
        auto path_keyframe = nlohmann::json{};
        path_keyframe["s"]["x"] = segment.position.x;
        path_keyframe["s"]["y"] = segment.position.y;
        path_keyframe["s"]["z"] = segment.position.z;

        path_keyframe["rot"]["x"] = segment.rotation.w;
        path_keyframe["rot"]["y"] = segment.rotation.x;
        path_keyframe["rot"]["z"] = segment.rotation.y;
        path_keyframe["rot"]["w"] = segment.rotation.z;

        path_keyframe["time"] = segment.transition_time;

        path_.push_back(path_keyframe);
    }

    json["_version"] = 1;
    json["paths"].push_back(path_);
    auto f = std::ofstream(path);
    f << std::setw(4) << json;
}
