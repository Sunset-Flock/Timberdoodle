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
daxa::u32 capacity_of_bucket(daxa::u32 max_src_items, daxa::u32 bucket_index)
{
    return max_src_items;
}

struct Po2WorkExpansionBufferHead
{
    DispatchIndirectStruct dispatch;
    daxa::u32 src_work_item_count;
    daxa::u32 dst_work_item_count;
    daxa::u32 buffer_size;
    daxa::u32 buckets_capacity;
    daxa::u32 bucket_sizes[PO2_WORK_EXPANSION_BUCKET_COUNT];
    daxa::u32 * buckets[PO2_WORK_EXPANSION_BUCKET_COUNT];

    #if defined(__cplusplus)
        static void create_in_place(
            daxa::DeviceAddress device_address, 
            daxa::u32 max_src_items,
            DispatchIndirectStruct dispatch_clear,
            Po2WorkExpansionBufferHead* out)
        {
            daxa::u32 size = sizeof(Po2WorkExpansionBufferHead);
            for (daxa::u32 i = 0; i < PO2_WORK_EXPANSION_BUCKET_COUNT; ++i)
            {
                out->buckets[i] = reinterpret_cast<daxa::u32*>(reinterpret_cast<daxa::u8*>(device_address) + size);
                out->bucket_sizes[i] = 0;
                size += capacity_of_bucket(max_src_items, i) * sizeof(daxa::u32);
            }
            out->buffer_size = size;
            out->buckets_capacity = max_src_items;
            out->dispatch = dispatch_clear;
            // We want each bucket to get whole warps to work on items.
            // This means if bucket 3 has 17 items, it will still get 32 threads assigned to it.
            // This is VERY IMPORTANT in order to make the bucket search fast by distributing it across the warp with wave intrinsics.
            // We simply add a warp size for each bucket beforehand when we clear to make sure there are enough threads for each bucket.
            out->dispatch.x = WARP_SIZE * 32;
        }

        static auto create(
            daxa::DeviceAddress device_address, 
            daxa::u32 max_src_items,
            DispatchIndirectStruct dispatch_clear) -> Po2WorkExpansionBufferHead
        {
            Po2WorkExpansionBufferHead ret = {};
            create_in_place(device_address, max_src_items, dispatch_clear, &ret);
            return ret;
        }

        static auto calc_buffer_size(daxa::u32 max_src_items, daxa::u32 max_dst_items) -> daxa::u32
        {
            daxa::u32 ret = {};
            ret = sizeof(Po2WorkExpansionBufferHead);
            for (daxa::u32 i = 0; i < PO2_WORK_EXPANSION_BUCKET_COUNT; ++i)
            {
                auto const cap = capacity_of_bucket(max_src_items, i) * sizeof(daxa::u32);
                ret += cap;
            }
            return ret;
        }
    #endif
};    

struct PrefixSumExpansionBufferHead
{
    DispatchIndirectStruct dispatch;
    // Upper 32bit store src item count, lower 32bit store dst item count
    daxa::u32 dwig_capacity;
    daxa::u64 merged_src_item_dst_item_count;
    // DWIG = DstWorkItemGroup
    daxa::u32* dwig_inclusive_dst_work_item_count_prefix_sum;
    daxa::u32* dwig_src_work_items;

    #if defined(__cplusplus)
        static daxa::u32 calc_buffer_size(daxa::u32 max_dst_work_item_groups)
        {
            return sizeof(PrefixSumExpansionBufferHead) + sizeof(daxa::u32) * max_dst_work_item_groups * 2;
        }

        static auto create(
            daxa::DeviceAddress device_address, 
            daxa::u32 max_dst_work_item_groups,
            DispatchIndirectStruct dispatch_clear) -> PrefixSumExpansionBufferHead
        {
            PrefixSumExpansionBufferHead ret = {};
            ret.dispatch = dispatch_clear;
            ret.dwig_capacity = max_dst_work_item_groups;
            ret.merged_src_item_dst_item_count = 0;
            ret.dwig_inclusive_dst_work_item_count_prefix_sum = 
                reinterpret_cast<daxa::u32*>(device_address + static_cast<daxa::DeviceAddress>(sizeof(PrefixSumExpansionBufferHead)) + sizeof(daxa::u32) * max_dst_work_item_groups * 0);
            ret.dwig_src_work_items = 
                reinterpret_cast<daxa::u32*>(device_address + static_cast<daxa::DeviceAddress>(sizeof(PrefixSumExpansionBufferHead)) + sizeof(daxa::u32) * max_dst_work_item_groups * 1);
            return ret;
        }
    #endif
};