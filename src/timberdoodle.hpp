#pragma once

#define GLM_FORCE_DEPTH_ZERO_TO_ONE 1
#define GLM_ENABLE_EXPERIMENTAL 1
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define DAXA_REMOVE_DEPRECATED 0
#include <daxa/daxa.hpp>
#include <daxa/utils/task_graph.hpp>
#include <daxa/utils/pipeline_manager.hpp>
#include <daxa/utils/mem.hpp>
#include <fmt/format.h>
#include <cinttypes>

namespace tido
{
    inline namespace types
    {
        using u8 = std::uint8_t;
        using u16 = std::uint16_t;
        using u32 = std::uint32_t;
        using u64 = std::uint64_t;
        using usize = std::size_t;

        using i8 = std::int8_t;
        using i16 = std::int16_t;
        using i32 = std::int32_t;
        using i64 = std::int64_t;
        using isize = std::ptrdiff_t;

        using b32 = u32;
        using f32 = float;
        using f64 = double;

        using u32vec2 = glm::uvec2;
        using u32vec3 = glm::uvec3;
        using u32vec4 = glm::uvec4;

        using i32vec2 = glm::ivec2;
        using i32vec3 = glm::ivec3;
        using i32vec4 = glm::ivec4;

        using f32vec2 = glm::vec2;
        using f32vec3 = glm::vec3;
        using f32vec4 = glm::vec4;

        using f64vec2 = glm::dvec2;
        using f64vec3 = glm::dvec3;
        using f64vec4 = glm::dvec4;

        using u32mat2x2 = glm::umat2x2;
        using u32mat2x3 = glm::umat2x3;
        using u32mat2x4 = glm::umat2x4;
        using u32mat3x2 = glm::umat3x2;
        using u32mat3x3 = glm::umat3x3;
        using u32mat3x4 = glm::umat3x4;
        using u32mat4x2 = glm::umat4x2;
        using u32mat4x3 = glm::umat4x3;
        using u32mat4x4 = glm::umat4x4;

        using i32mat2x2 = glm::imat2x2;
        using i32mat2x3 = glm::imat2x3;
        using i32mat2x4 = glm::imat2x4;
        using i32mat3x2 = glm::imat3x2;
        using i32mat3x3 = glm::imat3x3;
        using i32mat3x4 = glm::imat3x4;
        using i32mat4x2 = glm::imat4x2;
        using i32mat4x3 = glm::imat4x3;
        using i32mat4x4 = glm::imat4x4;

        using f32mat2x2 = glm::mat2x2;
        using f32mat2x3 = glm::mat2x3;
        using f32mat2x4 = glm::mat2x4;
        using f32mat3x2 = glm::mat3x2;
        using f32mat3x3 = glm::mat3x3;
        using f32mat3x4 = glm::mat3x4;
        using f32mat4x2 = glm::mat4x2;
        using f32mat4x3 = glm::mat4x3;
        using f32mat4x4 = glm::mat4x4;

        using f64mat2x2 = glm::dmat2x2;
        using f64mat2x3 = glm::dmat2x3;
        using f64mat2x4 = glm::dmat2x4;
        using f64mat3x2 = glm::dmat3x2;
        using f64mat3x3 = glm::dmat3x3;
        using f64mat3x4 = glm::dmat3x4;
        using f64mat4x2 = glm::dmat4x2;
        using f64mat4x3 = glm::dmat4x3;
        using f64mat4x4 = glm::dmat4x4;
    } // namespace types
} // namespace tido

#define s_cast static_cast
#define d_cast dynamic_cast
#define r_cast reinterpret_cast

#ifdef _DEBUG
#include <fmt/format.h>
#define DEBUG_MSG(M) fmt::println("{}", M);
#define DBG_ASSERT_TRUE_M(X, M)                                                                                        \
    [&]                                                                                                                \
    {                                                                                                                  \
        if (!(X))                                                                                                      \
        {                                                                                                              \
            fmt::println("ASSERTION FAILURE: {}", M);                                                                  \
            throw std::runtime_error("DEBUG ASSERTION FAILURE");                                                       \
        }                                                                                                              \
    }()
#else
#define DEBUG_MSG(M)
#define DBG_ASSERT_TRUE_M(X, M)
#endif

#ifndef defer
struct defer_dummy
{
};
template <class F> struct deferrer
{
    F f;
    ~deferrer() { f(); }
};
template <class F> deferrer<F> operator*(defer_dummy, F f)
{
    return {f};
}
#define DEFER_(LINE) zz_defer##LINE
#define DEFER(LINE) DEFER_(LINE)
#define defer auto DEFER(__LINE__) = defer_dummy{} * [&]()
#endif // defer

// I write c++ to erode my sanity
#define SANE_STATIC_BEGIN(NAME) \
    inline static auto const & NAME() { \
    static const auto value =

#define SANE_STATIC_END \
    return value; } 

inline constexpr auto find_msb(daxa::u32 v) -> daxa::u32
{
    daxa::u32 index = 0;
    while (v != 0)
    {
        v = v >> 1;
        index = index + 1;
    }
    return index;
}

inline constexpr auto find_next_lower_po2(daxa::u32 v) -> daxa::u32
{
    auto const msb = find_msb(v);
    return 1u << ((msb == 0 ? 1 : msb) - 1);
}