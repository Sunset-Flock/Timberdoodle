#pragma once

#include <cstring>
#include <functional>
#include <imgui.h>
#include <imgui_internal.h>
#include "../ui_shared.hpp"
#include "../../timberdoodle.hpp"

namespace tido
{
    namespace ui
    {
        struct State
        {
            bool hovered = {};
            bool pressed = {};
            bool held = {};
        };

        struct BetterDragFloatInfo
        {
            std::string text = {};
            f32 * value = {};
            f32 speed = {};
            f32 min = {};
            f32 max = {};
            std::string format = {};
            f32 text_width = {};
            i32 left_offset = {};
            bool clip = {};
        };
        inline auto better_drag_float(BetterDragFloatInfo const & info)
        {
            ImGui::PushID(info.text.c_str());
            ImGui::BeginChild("##", {info.text_width, ImGui::CalcTextSize("x").y + 2});
            ImGui::SetNextItemWidth(-1);
            ImGui::Text("%s", info.text.c_str());
            ImGui::EndChild();
            ImGui::SameLine();
            ImGui::SetNextItemWidth(s_cast<f32>(info.left_offset));
            ImGuiSliderFlags flags = info.clip ? ImGuiSliderFlags_AlwaysClamp : ImGuiSliderFlags_None;
            ImGui::DragFloat("##", info.value, info.speed, info.min, info.max, info.format.c_str(), flags);
            ImGui::PopID();
        };
        inline auto button_like_behavior(ImRect bb, ImGuiID id) -> State
        {
            State ret_state = {};
            ImGuiButtonFlags const flags = ImGuiButtonFlags_AllowOverlap | ImGuiButtonFlags_PressedOnClick;
            ret_state.pressed = ImGui::ButtonBehavior(bb, id, &ret_state.hovered, &ret_state.held, flags);
            return ret_state;
        };

        inline void draw_with_bg_rect(std::function<void()> draw_func, int right_pad = 0, ImVec4 col = bg_1)
        {
            auto const absolute_start_pos = ImGui::GetCurrentWindow()->DC.CursorPos;

            auto * draw_list = ImGui::GetWindowDrawList();
            // The number of indices that were recorded before we begin the drawing of the current element
            auto const prefix_index_count = draw_list->IdxBuffer.size();
            // Draw the elements that user wants. Note that this call needs to be before we draw the background rectangle
            // because we don't actually know beforehand how big the background rectangle should be. Thus we draw the contents
            // measure the size and draw the background rectagle. We then reorder the draw order manually.
            draw_func();
            // Get the number of indices that we have after we draw the content provided by the user
            auto const after_content_index_count = draw_list->IdxBuffer.size();
            auto const absolute_end_pos = ImGui::GetCurrentWindow()->DC.CursorPos;
            auto const size = ImVec2{
                ImGui::GetContentRegionAvail().x - right_pad,
                absolute_end_pos.y - absolute_start_pos.y,
            };
            draw_list->AddRectFilled(
                absolute_start_pos,
                {absolute_start_pos.x + size.x, absolute_start_pos.y + size.y},
                ImGui::GetColorU32(col),
                3.0f);

            // The number of indices we get after drawing the content and the background rectangle
            auto const after_bg_square_index_count = draw_list->IdxBuffer.size();

            ImVector<ImDrawIdx> new_indices = {};
            // The number of indices that were added by drawing the user provided content
            auto const content_index_count = after_content_index_count - prefix_index_count;
            // The number of indices that were added by drawing the background square
            auto const bg_square_index_count = after_bg_square_index_count - after_content_index_count;

            /// NOTE: Here we reorder the indices in the index buffer. We can view the contents of the index buffer as follows:
            //            [Previous indices][User specified content indices (draw_func())][Background square indices]
            //        ImGui draws these elements in the order that they are presented from the index buffer. We want the background
            //        rectangle to be behind the user contents. So we reorder the index buffer to be as follows:
            //            [Previous indices][Background square indices][User specified content indices (draw_func())]
            new_indices.resize(after_bg_square_index_count);
            // Copy the prefix indices
            std::memcpy(new_indices.Data, draw_list->IdxBuffer.Data, sizeof(ImDrawIdx) * prefix_index_count);
            // Copy the background square indices
            std::memcpy(new_indices.Data + prefix_index_count, draw_list->IdxBuffer.Data + after_content_index_count, sizeof(ImDrawIdx) * bg_square_index_count);
            // Copy the user content indices
            std::memcpy(new_indices.Data + prefix_index_count + bg_square_index_count, draw_list->IdxBuffer.Data + prefix_index_count, sizeof(ImDrawIdx) * content_index_count);
            // Rewrite the old index buffer with the new reordered index buffer
            draw_list->IdxBuffer = new_indices;
            draw_list->_IdxWritePtr = draw_list->IdxBuffer.end();
        }

        static constexpr inline auto icon_to_color(ICONS icon) -> ImVec4
        {
            switch (icon)
            {
                case ICONS::CAMERA:     [[fallthrough]];
                case ICONS::COLLECTION: [[fallthrough]];
                case ICONS::LIGHT:      [[fallthrough]];
                case ICONS::SUN:
                    return ImVec4(1.00f, 0.62f, 0.37f, 1.00f);
                case ICONS::MESH: [[fallthrough]];
                case ICONS::MESHGROUP:
                    return ImVec4(0.40f, 1.00f, 0.52f, 1.00f);
                case ICONS::MATERIAL:
                    return ImVec4(1.00f, 0.37f, 0.37f, 1.00f);
                default:
                    return ImGui::GetStyleColorVec4(ImGuiCol_Text);
            }
        }
    } // namespace ui
} // namespace tido