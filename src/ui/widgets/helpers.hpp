#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <functional>
#include <span>
#include <string_view>
#include <unordered_map>
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

        // Drop-in replacement for ImGui::Combo(label, current_item, items, items_count): a normal
        // combo box (closed button showing the current selection) whose popup has an extra first
        // row — a text filter. Typing there hides every item whose name doesn't contain the typed
        // substring (case-insensitive). The filter text is kept in internal per-widget storage
        // keyed off `label`, so callers don't need to own a buffer. Returns true when the
        // selection changed.
        inline bool filter_combo(char const * label, i32 * current_item, char const * const * items, i32 items_count)
        {
            bool changed = false;
            ImGuiID const id = ImGui::GetID(label);
            static std::unordered_map<ImGuiID, std::array<char, 128>> s_filter_bufs;
            std::array<char, 128> & filter_buf = s_filter_bufs[id];

            char const * preview = (*current_item >= 0 && *current_item < items_count)
                ? items[s_cast<usize>(*current_item)]
                : "";
            if (ImGui::BeginCombo(label, preview))
            {
                if (ImGui::IsWindowAppearing())
                {
                    ImGui::SetKeyboardFocusHere();
                }
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                ImGui::InputTextWithHint("##filter", "filter...", filter_buf.data(), filter_buf.size());
                ImGui::Separator();

                std::string_view const filter{filter_buf.data()};
                bool any = false;
                for (i32 i = 0; i < items_count; ++i)
                {
                    std::string_view const name = items[s_cast<usize>(i)];
                    bool match = filter.empty();
                    if (!match)
                    {
                        auto it = std::search(name.begin(), name.end(), filter.begin(), filter.end(),
                            [](char a, char b) { return std::tolower(s_cast<unsigned char>(a)) == std::tolower(s_cast<unsigned char>(b)); });
                        match = it != name.end();
                    }
                    if (!match) continue;
                    any = true;
                    bool const selected = (i == *current_item);
                    if (ImGui::Selectable(items[s_cast<usize>(i)], selected))
                    {
                        if (*current_item != i) changed = true;
                        *current_item = i;
                    }
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                if (!any)
                {
                    ImGui::TextDisabled("(no match)");
                }
                ImGui::EndCombo();
            }
            else
            {
                // Start with a clean filter next time the popup opens.
                filter_buf[0] = '\0';
            }
            return changed;
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