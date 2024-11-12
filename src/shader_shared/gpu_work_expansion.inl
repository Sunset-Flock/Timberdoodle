#pragma once

#include <daxa/daxa.inl>

#include "shared.inl"

#if DAXA_LANGUAGE == DAXA_LANGUAGE_GLSL
    #error "po2_expansion headers only available in c++ and slang-hlsl!"
#endif

#define PO2_WORK_EXPANSION_BUCKET_COUNT 32

// Default 270 million. 
// Lower max factor will result in less memory usage
#define WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS (1u << 28u)

#if defined(__cplusplus)
inline
#endif
daxa::u32 capacity_of_bucket(daxa::u32 expansions_max, daxa::u32 bucket_index)
{
    #if defined(__cplusplus)
        return std::min(expansions_max, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS >> bucket_index);
    #else
        return min(expansions_max, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS >> bucket_index);
    #endif
}

struct WorkExpansion
{
    daxa::u32 src_item_index;
    daxa::u32 expansion_count;
    daxa::u32 in_bucket_first_thread;
};

struct Po2PackedWorkExpansionBufferHead
{
    DispatchIndirectStruct dispatch;

    daxa::u32 expansions_max;
    daxa::u32 expansions_count;
    daxa::u32 expansions_thread_count;
    WorkExpansion* expansions; 

    daxa::u32 bucket_arg_counts[PO2_WORK_EXPANSION_BUCKET_COUNT];
    daxa::u32 bucket_thread_counts[PO2_WORK_EXPANSION_BUCKET_COUNT];
    daxa_u32* buckets[PO2_WORK_EXPANSION_BUCKET_COUNT];          // uints point to expansions, each arg can have to expansions.

    daxa::u32 buffer_size;

    #if defined(__cplusplus)
        static void create_in_place(
            daxa::DeviceAddress device_address, 
            daxa::u32 expansions_max,
            DispatchIndirectStruct dispatch_clear,
            Po2PackedWorkExpansionBufferHead* out)
        {
            out->dispatch = dispatch_clear;
            daxa::u32 size = sizeof(Po2PackedWorkExpansionBufferHead);
            out->expansions_max = expansions_max;
            out->expansions_count = 0;
            out->expansions_thread_count = 0;
            out->expansions = reinterpret_cast<WorkExpansion*>(reinterpret_cast<daxa::u8*>(device_address) + size);
            size += sizeof(WorkExpansion) * expansions_max;
            for (daxa::u32 i = 0; i < PO2_WORK_EXPANSION_BUCKET_COUNT; ++i)
            {
                out->bucket_arg_counts[i] = 0;
                out->bucket_thread_counts[i] = 0;
                out->buckets[i] = reinterpret_cast<daxa_u32*>(reinterpret_cast<daxa::u8*>(device_address) + size);
                size += capacity_of_bucket(expansions_max, i) * sizeof(daxa_u32) * 2;
            }
            out->buffer_size = size;
        }

        static auto create(
            daxa::DeviceAddress device_address, 
            daxa::u32 max_src_items,
            DispatchIndirectStruct dispatch_clear) -> Po2PackedWorkExpansionBufferHead
        {
            Po2PackedWorkExpansionBufferHead ret = {};
            create_in_place(device_address, max_src_items, dispatch_clear, &ret);
            return ret;
        }

        static auto calc_buffer_size(daxa::u32 expansions_max) -> daxa::u32
        {
            daxa::u32 ret = {};
            ret = sizeof(Po2PackedWorkExpansionBufferHead);
            ret += sizeof(WorkExpansion) * expansions_max;
            for (daxa::u32 i = 0; i < PO2_WORK_EXPANSION_BUCKET_COUNT; ++i)
            {
                auto const cap = capacity_of_bucket(expansions_max, i) * sizeof(daxa_u32) * 2;
                ret += cap;
            }
            return ret;
        }
    #endif
};    

struct PrefixSumWorkExpansionBufferHead
{
    DispatchIndirectStruct dispatch;
    // Upper 32bit store src item count, lower 32bit store dst item count
    daxa::u64 merged_expansion_count_thread_count;
    daxa::u32 expansion_count;
    daxa::u32 expansions_max;
    daxa::u32* expansions_inclusive_prefix_sum;
    daxa::u32* expansions_src_work_item;
    daxa::u32* expansions_expansion_factor;

    #if defined(__cplusplus)
        static daxa::u32 calc_buffer_size(daxa::u32 max_expansions)
        {
            return sizeof(PrefixSumWorkExpansionBufferHead) + sizeof(daxa_u32) * max_expansions * 3;
        }

        static auto create(
            daxa::DeviceAddress device_address, 
            daxa::u32 max_expansions,
            DispatchIndirectStruct dispatch_clear) -> PrefixSumWorkExpansionBufferHead
        {
            PrefixSumWorkExpansionBufferHead ret = {};
            ret.dispatch = dispatch_clear;
            ret.expansions_max = max_expansions;
            ret.merged_expansion_count_thread_count = 0;
            ret.expansions_inclusive_prefix_sum = 
                reinterpret_cast<daxa_u32*>(device_address + static_cast<daxa::DeviceAddress>(sizeof(PrefixSumWorkExpansionBufferHead)) + max_expansions * sizeof(daxa::u32) * 0);
            ret.expansions_src_work_item = 
                reinterpret_cast<daxa_u32*>(device_address + static_cast<daxa::DeviceAddress>(sizeof(PrefixSumWorkExpansionBufferHead)) + max_expansions * sizeof(daxa::u32) * 1);
            ret.expansions_expansion_factor = 
                reinterpret_cast<daxa_u32*>(device_address + static_cast<daxa::DeviceAddress>(sizeof(PrefixSumWorkExpansionBufferHead)) + max_expansions * sizeof(daxa::u32) * 2);
            return ret;
        }
    #endif
};