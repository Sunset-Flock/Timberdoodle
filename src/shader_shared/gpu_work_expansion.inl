#pragma once

#include <daxa/daxa.inl>

#include "shared.inl"

#if DAXA_LANGUAGE == DAXA_LANGUAGE_GLSL
    #error "po2_expansion headers only available in c++ and slang-hlsl!"
#endif

#define PO2_WORK_EXPANSION_BUCKET_COUNT 32
// pairs of two buckets grow into a single double ended stack
#define PO2_WORK_EXPANSION_BUCKET_ARRAY_COUNT (PO2_WORK_EXPANSION_BUCKET_COUNT/2)

// Default 270 million. 
// Lower max factor will result in less memory usage
#define WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS (1u << 28u)

// pairs of two buckets grow into a single double ended stack
#if defined(__cplusplus)
inline
#endif
daxa::u32 po2_expansion_max_bucket_entries(daxa::u32 expansions_max, daxa::u32 bucket)
{
    #if defined(__cplusplus)
        return std::min(expansions_max, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS >> bucket) * 2;
    #else
        return min(expansions_max, WORK_EXPANSION_PO2_MAX_TOTAL_EXPANDED_THREADS >> bucket) * 2;
    #endif
}

struct Po2BucketWorkExpansion
{
    daxa_u32vec2 payload;
    daxa::u32 work_item_count;
    daxa::u32 first_thread_in_bucket;
};

struct Po2BucketWorkExpansionBufferHead
{
    DispatchIndirectStruct dispatch;

    daxa::u32 expansions_max;
    daxa::u32 expansion_count;
    daxa::u32 expansions_thread_count;
    Po2BucketWorkExpansion* expansions; 

    daxa::u32 bucket_arg_counts[PO2_WORK_EXPANSION_BUCKET_COUNT];
    daxa::u32 bucket_thread_counts[PO2_WORK_EXPANSION_BUCKET_COUNT];
    daxa::u32* bucket_arg_array_ptrs[PO2_WORK_EXPANSION_BUCKET_COUNT];

    daxa::u32 buffer_size;

    #if defined(__cplusplus)
        static void create_in_place(
            daxa::DeviceAddress device_address, 
            daxa::u32 expansions_max,
            DispatchIndirectStruct dispatch_clear,
            Po2BucketWorkExpansionBufferHead* out)
        {
            out->dispatch = dispatch_clear;
            daxa::u32 total_size = sizeof(Po2BucketWorkExpansionBufferHead);
            out->expansions_max = expansions_max;
            out->expansion_count = 0;
            out->expansions_thread_count = 0;
            out->expansions = reinterpret_cast<Po2BucketWorkExpansion*>(device_address + total_size);
            total_size += sizeof(Po2BucketWorkExpansion) * expansions_max;
            for (daxa::u32 bucket_pair_i = 0; bucket_pair_i < PO2_WORK_EXPANSION_BUCKET_ARRAY_COUNT; ++bucket_pair_i)
            {
                u32 first_bucket = bucket_pair_i * 2u;
                u32 second_bucket = bucket_pair_i * 2u + 1u;
                out->bucket_arg_counts[first_bucket] = 0u;
                out->bucket_arg_counts[second_bucket] = 0u;
                out->bucket_thread_counts[first_bucket] = 0u;
                out->bucket_thread_counts[second_bucket] = 0u;
                u32 first_bucket_max_entries = po2_expansion_max_bucket_entries(expansions_max, first_bucket);
                [[maybe_unused]] u32 second_bucket_max_entries = po2_expansion_max_bucket_entries(expansions_max, second_bucket);
                u32 bucket_pair_size = first_bucket_max_entries * sizeof(daxa_u32);                                                                    // lower index buckets can always have >= elements.
                out->bucket_arg_array_ptrs[first_bucket] = reinterpret_cast<u32*>(device_address + total_size);                                        // points to first element in bucket pair array
                out->bucket_arg_array_ptrs[second_bucket] = reinterpret_cast<u32*>(device_address + total_size + bucket_pair_size - sizeof(daxa_u32)); // points to last element in bucket pair array
                total_size += bucket_pair_size;
            }
            out->buffer_size = total_size;
        }

        static auto create(
            daxa::DeviceAddress device_address, 
            daxa::u32 max_src_items,
            DispatchIndirectStruct dispatch_clear) -> Po2BucketWorkExpansionBufferHead
        {
            Po2BucketWorkExpansionBufferHead ret = {};
            create_in_place(device_address, max_src_items, dispatch_clear, &ret);
            return ret;
        }

        static auto calc_buffer_size(daxa::u32 expansions_max) -> daxa::u32
        {
            daxa::u32 ret = {};
            ret = sizeof(Po2BucketWorkExpansionBufferHead);
            ret += sizeof(Po2BucketWorkExpansion) * expansions_max;
            for (daxa::u32 bucket_pair_i = 0; bucket_pair_i < PO2_WORK_EXPANSION_BUCKET_ARRAY_COUNT; ++bucket_pair_i)
            {
                u32 first_bucket = bucket_pair_i * 2;   // lower index buckets can always have >= elements.
                ret += po2_expansion_max_bucket_entries(expansions_max, first_bucket) * sizeof(daxa_u32);
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
    daxa_u32vec2* expansions_payloads;
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
            ret.expansions_payloads = 
                reinterpret_cast<daxa_u32vec2*>(device_address + static_cast<daxa::DeviceAddress>(sizeof(PrefixSumWorkExpansionBufferHead)) + max_expansions * sizeof(daxa::u32) * 1);
            ret.expansions_expansion_factor = 
                reinterpret_cast<daxa_u32*>(device_address + static_cast<daxa::DeviceAddress>(sizeof(PrefixSumWorkExpansionBufferHead)) + max_expansions * sizeof(daxa_u32vec2) * 2);
            return ret;
        }
    #endif
};