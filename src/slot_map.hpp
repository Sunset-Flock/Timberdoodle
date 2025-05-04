#pragma once

#include "timberdoodle.hpp"
#include <vector>
#include <optional>
namespace tido
{
    using namespace types;
    template <typename T>
    struct SlotMap
    {
        private:
            // TODO: It better to not use optional here.
            std::vector<std::optional<T>>
                _slots = {};
            std::vector<u32>
                _versions = {};
            std::vector<size_t>
                _free_list = {};

        public:
            struct Id
            {
                u32 index = {};
                u32 version = {};
            };
            static inline constexpr Id EMPTY_ID = { 0, 0 };
            auto create_slot(T&& v = {}) -> Id
            {
                if (_free_list.size() > 0)
                {
                    u32 const index = _free_list.back();
                    _free_list.pop_back();
                    _slots[index] = std::move(v);
                    return Id{index, _versions[index]};
                }
                else
                {
                    u32 const index = s_cast<u32>(_slots.size());
                    _slots.emplace_back(std::move(v));
                    _versions.emplace_back(1u);
                    return Id{index, 1u};
                }
            }
            auto destroy_slot(Id id) -> bool
            {
                if (this->is_id_valid(id))
                {
                    _slots[s_cast<size_t>(id.index)] = std::nullopt;
                    _versions[s_cast<size_t>(id.index)] += 1;
                    if (_versions[s_cast<size_t>(id.index)] < std::numeric_limits<u32>::max())
                    {
                        _free_list.push_back(id.index);
                    }
                    return true;
                }
                return false;
            }
            auto slot_by_index(size_t index) -> T *
            {
                if (index < this->_slots.size())
                {
                    return &_slots[index].value();
                }
                return nullptr;
            }
            auto slot(Id id) -> T *
            {
                if (this->is_id_valid(id))
                {
                    return &_slots[s_cast<size_t>(id.index)].value();
                }
                return nullptr;
            }
            auto slot(Id id) const -> T const *
            {
                if (this->is_id_valid(id))
                {
                    return &_slots[s_cast<size_t>(id.index)].value();
                }
                return nullptr;
            }
            auto is_id_valid(Id id) const -> bool
            {
                auto const uz_index = s_cast<size_t>(id.index);
                return uz_index < _slots.size() && _versions[uz_index] == id.version;
            }
            auto id_from_index(size_t index) -> Id
            {
                if (index < _slots.size())
                {
                    return {static_cast<u32>(index), _versions[index]};
                }
                return {};
            }
            auto size() const -> usize
            {
                return _slots.size() - _free_list.size();
            }
            auto capacity() const -> usize
            {
                return _slots.size();
            }

            void clear()
            {
                _slots.clear();
                _versions.clear();
                _free_list.clear();
            }
        };
}