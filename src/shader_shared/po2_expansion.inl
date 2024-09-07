#pragma once

#include <daxa/daxa.inl>

#include "shared.inl"

#if DAXA_LANGUAGE == DAXA_LANGUAGE_GLSL
    #error "po2_expansion headers only available in c++ and slang-hlsl!"
#endif

#define PO2_WORK_EXPANSION_BUCKET_COUNT 32

#if defined(__cplusplus)
inline
#endif
daxa::u32 capacity_of_bucket(daxa::u32 max_src_items, daxa::u32 max_dst_items, daxa::u32 bucket_index)
{
    return 
    #if defined(__cplusplus)
        std::min
    #else
        min
    #endif
        (max_src_items, max_dst_items >> bucket_index);
}

struct Po2WorkExpansionBufferHead
{
    // Values need to be reset and written every frame:
    DispatchIndirectStruct bucket_dispatches[PO2_WORK_EXPANSION_BUCKET_COUNT];
    daxa::u32 bucket_sizes[PO2_WORK_EXPANSION_BUCKET_COUNT];
    // Values stay constant after initialization:
    daxa::u32 * buckets[PO2_WORK_EXPANSION_BUCKET_COUNT];
    daxa::u32 max_src_items;
    daxa::u32 max_dst_items;
    daxa::u32 dst_workgroup_size_log2;
    daxa::u32 buffer_size;

    #if defined(__cplusplus)
        static void create_in_place(
            daxa::DeviceAddress device_address, 
            daxa::u32 max_src_items,
            daxa::u32 max_dst_items,
            daxa::u32 dst_workgroup_size,
            DispatchIndirectStruct dispatch_clear,
            Po2WorkExpansionBufferHead* out)
        {
            daxa::f64 max_dst_items_integer_part;
            daxa::f64 max_dst_items_fractional_part;
            max_dst_items_fractional_part = std::modf(std::log2(static_cast<daxa::f64>(max_dst_items)), &max_dst_items_integer_part);
            DAXA_DBG_ASSERT_TRUE_M(max_dst_items_fractional_part < 0.0000001, "max_dst_items must be a power of two");
            out->max_src_items = max_src_items;
            out->max_dst_items = max_dst_items;
            out->dst_workgroup_size_log2 = static_cast<daxa::u32>(std::log2(static_cast<daxa::f32>(dst_workgroup_size)));
            daxa::u32 size = sizeof(Po2WorkExpansionBufferHead);
            for (daxa::u32 i = 0; i < PO2_WORK_EXPANSION_BUCKET_COUNT; ++i)
            {
                out->buckets[i] = reinterpret_cast<daxa::u32*>(reinterpret_cast<daxa::u8*>(device_address) + size);
                size += capacity_of_bucket(max_src_items, max_dst_items, i) * sizeof(daxa::u32);
                out->bucket_dispatches[i] = dispatch_clear;
                out->bucket_sizes[i] = {};
            }
            out->buffer_size = size;
        }

        static auto create(
            daxa::DeviceAddress device_address, 
            daxa::u32 max_src_items,
            daxa::u32 max_dst_items,
            daxa::u32 dst_workgroup_size,
            DispatchIndirectStruct dispatch_clear) -> Po2WorkExpansionBufferHead
        {
            Po2WorkExpansionBufferHead ret = {};
            create_in_place(device_address, max_src_items, max_dst_items, dst_workgroup_size, dispatch_clear, &ret);
            return ret;
        }

        static auto calc_buffer_size(daxa::u32 max_src_items, daxa::u32 max_dst_items) -> daxa::u32
        {
            daxa::f64 max_dst_items_integer_part;
            daxa::f64 max_dst_items_fractional_part;
            max_dst_items_fractional_part = std::modf(std::log2(static_cast<daxa::f64>(max_dst_items)), &max_dst_items_integer_part);
            DAXA_DBG_ASSERT_TRUE_M(max_dst_items_fractional_part < 0.0000001, "max_dst_items must be a power of two");
            daxa::u32 ret = {};
            ret = sizeof(Po2WorkExpansionBufferHead);
            for (daxa::u32 i = 0; i < PO2_WORK_EXPANSION_BUCKET_COUNT; ++i)
            {
                auto const cap = capacity_of_bucket(max_src_items, max_dst_items, i) * sizeof(daxa::u32);
                ret += cap;
            }
            return ret;
        }
    #endif
};    