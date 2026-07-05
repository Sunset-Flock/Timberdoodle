#pragma once

#include <algorithm>
#include <cctype>
#include <cstring>
#include <functional>
#include <span>
#include <string_view>
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

        // Searchable combo widget. Renders a text input; clicking it or typing opens a filtered
        // dropdown popup listing all items whose names contain the typed substring (case-insensitive).
        // Pressing Tab accepts the first visible match. Returns true when the selection changed.
        //
        // `label`       — ImGui ID / right-side label (same semantics as ImGui::Combo)
        // `current_idx` — in/out selected index into `items`
        // `search_buf`  — caller-owned char array used for the search text (persistent across frames)
        // `buf_size`    — size of search_buf in bytes
        // `items`       — array of null-terminated strings
        inline bool searchable_combo(
            const char * label,
            i32 & current_idx,
            char * search_buf,
            i32 buf_size,
            std::span<const char * const> items)
        {
            bool changed = false;
            ImGui::PushID(label);

            // We track popup state in a per-widget ImGui storage slot keyed by the current ID.
            ImGuiID popup_key = ImGui::GetID("##sc_open");
            ImGuiStorage * storage = ImGui::GetStateStorage();
            bool popup_open = storage->GetBool(popup_key, false);

            ImGui::SetNextItemWidth(ImGui::CalcItemWidth());
            if (ImGui::InputText("##search", search_buf, s_cast<usize>(buf_size),
                ImGuiInputTextFlags_AutoSelectAll))
            {
                popup_open = true;
            }
            if (ImGui::IsItemActivated())
                popup_open = true;

            // Draw current selection name as disabled hint when the bar is empty
            if (search_buf[0] == '\0' && current_idx >= 0 && current_idx < s_cast<i32>(items.size()))
            {
                ImVec2 rmin = ImGui::GetItemRectMin();
                float pad = ImGui::GetStyle().FramePadding.x;
                ImGui::GetWindowDrawList()->AddText(
                    ImVec2(rmin.x + pad, rmin.y + ImGui::GetStyle().FramePadding.y),
                    ImGui::GetColorU32(ImGuiCol_TextDisabled),
                    items[s_cast<usize>(current_idx)]);
            }

            // Label to the right (standard ImGui convention)
            ImGui::SameLine();
            ImGui::TextUnformatted(label);

            // Tab: autocomplete search text to the first matching item's full name, then select it.
            if (popup_open && (ImGui::IsItemFocused() || ImGui::GetID("##search") == ImGui::GetActiveID()))
            {
                if (ImGui::IsKeyPressed(ImGuiKey_Tab, false))
                {
                    std::string_view filter = search_buf;
                    for (i32 i = 0; i < s_cast<i32>(items.size()); ++i)
                    {
                        std::string_view name = items[s_cast<usize>(i)];
                        bool match = filter.empty();
                        if (!match)
                        {
                            auto it = std::search(name.begin(), name.end(), filter.begin(), filter.end(),
                                [](char a, char b){ return std::toupper(s_cast<unsigned char>(a)) == std::toupper(s_cast<unsigned char>(b)); });
                            match = it != name.end();
                        }
                        if (match)
                        {
                            // Write the full name into the search buffer so the user sees what matched,
                            // and select it. Truncate safely to buf_size-1.
                            const usize copy_len = std::min(name.size(), s_cast<usize>(buf_size - 1));
                            std::copy(name.begin(), name.begin() + s_cast<std::ptrdiff_t>(copy_len), search_buf);
                            search_buf[copy_len] = '\0';
                            if (current_idx != i) changed = true;
                            current_idx = i;
                            popup_open = false;
                            break;
                        }
                    }
                }
            }

            if (popup_open)
            {
                float item_w = ImGui::CalcItemWidth();
                // We already called SameLine+TextUnformatted so GetItemRectMin is the label.
                // Re-derive input left edge from the label rect.
                float input_left  = ImGui::GetItemRectMin().x - item_w - ImGui::GetStyle().ItemSpacing.x;
                float input_top   = ImGui::GetItemRectMin().y;
                float input_bot   = ImGui::GetItemRectMax().y;
                float screen_bot  = ImGui::GetMainViewport()->WorkPos.y + ImGui::GetMainViewport()->WorkSize.y;
                float space_below = screen_bot - input_bot;
                float space_above = input_top - ImGui::GetMainViewport()->WorkPos.y;
                float max_height  = 300.0f;
                // Open above when there is more room there than below
                bool open_above = space_below < max_height && space_above > space_below;
                ImVec2 popup_pos = open_above
                    ? ImVec2(input_left, input_top)   // SetNextWindowPos pivot (1,1) anchors bottom-left to input top-left
                    : ImVec2(input_left, input_bot);
                ImGui::SetNextWindowPos(popup_pos, ImGuiCond_Always, open_above ? ImVec2(0.0f, 1.0f) : ImVec2(0.0f, 0.0f));
                ImGui::SetNextWindowSizeConstraints(ImVec2(item_w, 0.0f), ImVec2(item_w, max_height));
                ImGui::SetNextWindowBgAlpha(1.0f);
                if (ImGui::Begin("##sc_popup", nullptr,
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                    ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                    ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
                {
                    std::string_view filter = search_buf;
                    bool any = false;
                    for (i32 i = 0; i < s_cast<i32>(items.size()); ++i)
                    {
                        std::string_view name = items[s_cast<usize>(i)];
                        bool match = filter.empty();
                        if (!match)
                        {
                            auto it = std::search(name.begin(), name.end(), filter.begin(), filter.end(),
                                [](char a, char b){ return std::toupper(s_cast<unsigned char>(a)) == std::toupper(s_cast<unsigned char>(b)); });
                            match = it != name.end();
                        }
                        if (!match) continue;
                        any = true;
                        bool selected = (i == current_idx);
                        if (ImGui::Selectable(items[s_cast<usize>(i)], selected))
                        {
                            if (current_idx != i) changed = true;
                            current_idx = i;
                            search_buf[0] = '\0';
                            popup_open = false;
                        }
                        if (selected) ImGui::SetItemDefaultFocus();
                    }
                    if (!any)
                        ImGui::TextDisabled("(no match)");
                }
                ImGui::End();

                // Close when focus leaves both the input and the popup
                if (!ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow))
                {
                    popup_open = false;
                    search_buf[0] = '\0';
                }
            }

            storage->SetBool(popup_key, popup_open);
            ImGui::PopID();
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