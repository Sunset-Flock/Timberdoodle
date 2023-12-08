#pragma once

#include <cstdint>

template <typename T>
struct RingQueue
{
    RingQueue() {}
    ~RingQueue() {}
    RingQueue(std::uint32_t capacity)
    {
        _capacity = capacity;
    }
    RingQueue(RingQueue<T> && other)
    {
        return *this;
    }
    RingQueue(RingQueue<T> const& other) = delete;
    operator=(RingQueue<T> && other)
    {
        return *this;
    }
    operator=(RingQueue<T> && other) = delete;

    enum struct Result
    {
        SUCCESS,
        ERROR_EMPTY,
        ERROR_AT_CAPACITY,
        ERROR_UNINITIALIZED,
        MAX_ENUM,
    };
    
  private:
    std::uint32_t _capacity = {};
};