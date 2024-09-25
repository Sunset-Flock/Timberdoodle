#include "daxa_helper.hpp"

namespace tido
{
    auto tido::make_task_buffer(daxa::Device & device, u32 size, std::string_view name, daxa::MemoryFlags flags) -> daxa::TaskBuffer
    {
        return daxa::TaskBuffer{
            device,
            daxa::BufferInfo{
                .size = size,
                .allocate_info = flags,
                .name = name,
            },
        };
    }

    auto upgrade_compute_pipeline_compile_info(daxa::ComputePipelineCompileInfo const & old) -> daxa::ComputePipelineCompileInfo2
    {
        daxa::ComputePipelineCompileInfo2 info = {};
        info.source = std::move(old.shader_info.source);
        info.entry_point = std::move(old.shader_info.compile_options.entry_point);
        info.language = std::move(old.shader_info.compile_options.language);
        info.defines = std::move(old.shader_info.compile_options.defines);
        info.enable_debug_info = std::move(old.shader_info.compile_options.enable_debug_info);
        info.create_flags = std::move(old.shader_info.compile_options.create_flags);
        info.required_subgroup_size = std::move(old.shader_info.compile_options.required_subgroup_size);
        info.push_constant_size = std::move(old.push_constant_size);
        info.name = std::move(old.name);
        return info;
    }

    auto channel_count_of_format(daxa::Format format) -> u32
    {
        switch(format)
        {
            case daxa::Format::UNDEFINED: return 0;
            case daxa::Format::R4G4_UNORM_PACK8: return 2;
            case daxa::Format::R4G4B4A4_UNORM_PACK16: return 4;
            case daxa::Format::B4G4R4A4_UNORM_PACK16: return 4;
            case daxa::Format::R5G6B5_UNORM_PACK16: return 3;
            case daxa::Format::B5G6R5_UNORM_PACK16: return 3;
            case daxa::Format::R5G5B5A1_UNORM_PACK16: return 4;
            case daxa::Format::B5G5R5A1_UNORM_PACK16: return 4;
            case daxa::Format::A1R5G5B5_UNORM_PACK16: return 4;
            case daxa::Format::R8_UNORM: return 1;
            case daxa::Format::R8_SNORM: return 1;
            case daxa::Format::R8_USCALED: return 1;
            case daxa::Format::R8_SSCALED: return 1;
            case daxa::Format::R8_UINT: return 1;
            case daxa::Format::R8_SINT: return 1;
            case daxa::Format::R8_SRGB: return 1;
            case daxa::Format::R8G8_UNORM: return 2;
            case daxa::Format::R8G8_SNORM: return 2;
            case daxa::Format::R8G8_USCALED: return 2;
            case daxa::Format::R8G8_SSCALED: return 2;
            case daxa::Format::R8G8_UINT: return 2;
            case daxa::Format::R8G8_SINT: return 2;
            case daxa::Format::R8G8_SRGB: return 2;
            case daxa::Format::R8G8B8_UNORM: return 3;
            case daxa::Format::R8G8B8_SNORM: return 3;
            case daxa::Format::R8G8B8_USCALED: return 3;
            case daxa::Format::R8G8B8_SSCALED: return 3;
            case daxa::Format::R8G8B8_UINT: return 3;
            case daxa::Format::R8G8B8_SINT: return 3;
            case daxa::Format::R8G8B8_SRGB: return 3;
            case daxa::Format::B8G8R8_UNORM: return 3;
            case daxa::Format::B8G8R8_SNORM: return 3;
            case daxa::Format::B8G8R8_USCALED: return 3;
            case daxa::Format::B8G8R8_SSCALED: return 3;
            case daxa::Format::B8G8R8_UINT: return 3;
            case daxa::Format::B8G8R8_SINT: return 3;
            case daxa::Format::B8G8R8_SRGB: return 3;
            case daxa::Format::R8G8B8A8_UNORM: return 4;
            case daxa::Format::R8G8B8A8_SNORM: return 4;
            case daxa::Format::R8G8B8A8_USCALED: return 4;
            case daxa::Format::R8G8B8A8_SSCALED: return 4;
            case daxa::Format::R8G8B8A8_UINT: return 4;
            case daxa::Format::R8G8B8A8_SINT: return 4;
            case daxa::Format::R8G8B8A8_SRGB: return 4;
            case daxa::Format::B8G8R8A8_UNORM: return 4;
            case daxa::Format::B8G8R8A8_SNORM: return 4;
            case daxa::Format::B8G8R8A8_USCALED: return 4;
            case daxa::Format::B8G8R8A8_SSCALED: return 4;
            case daxa::Format::B8G8R8A8_UINT: return 4;
            case daxa::Format::B8G8R8A8_SINT: return 4;
            case daxa::Format::B8G8R8A8_SRGB: return 4;
            case daxa::Format::A8B8G8R8_UNORM_PACK32: return 4;
            case daxa::Format::A8B8G8R8_SNORM_PACK32: return 4;
            case daxa::Format::A8B8G8R8_USCALED_PACK32: return 4;
            case daxa::Format::A8B8G8R8_SSCALED_PACK32: return 4;
            case daxa::Format::A8B8G8R8_UINT_PACK32: return 4;
            case daxa::Format::A8B8G8R8_SINT_PACK32: return 4;
            case daxa::Format::A8B8G8R8_SRGB_PACK32: return 4;
            case daxa::Format::A2R10G10B10_UNORM_PACK32: return 4;
            case daxa::Format::A2R10G10B10_SNORM_PACK32: return 4;
            case daxa::Format::A2R10G10B10_USCALED_PACK32: return 4;
            case daxa::Format::A2R10G10B10_SSCALED_PACK32: return 4;
            case daxa::Format::A2R10G10B10_UINT_PACK32: return 4;
            case daxa::Format::A2R10G10B10_SINT_PACK32: return 4;
            case daxa::Format::A2B10G10R10_UNORM_PACK32: return 4;
            case daxa::Format::A2B10G10R10_SNORM_PACK32: return 4;
            case daxa::Format::A2B10G10R10_USCALED_PACK32: return 4;
            case daxa::Format::A2B10G10R10_SSCALED_PACK32: return 4;
            case daxa::Format::A2B10G10R10_UINT_PACK32: return 4;
            case daxa::Format::A2B10G10R10_SINT_PACK32: return 4;
            case daxa::Format::R16_UNORM: return 1;
            case daxa::Format::R16_SNORM: return 1;
            case daxa::Format::R16_USCALED: return 1;
            case daxa::Format::R16_SSCALED: return 1;
            case daxa::Format::R16_UINT: return 1;
            case daxa::Format::R16_SINT: return 1;
            case daxa::Format::R16_SFLOAT: return 1;
            case daxa::Format::R16G16_UNORM: return 2;
            case daxa::Format::R16G16_SNORM: return 2;
            case daxa::Format::R16G16_USCALED: return 2;
            case daxa::Format::R16G16_SSCALED: return 2;
            case daxa::Format::R16G16_UINT: return 2;
            case daxa::Format::R16G16_SINT: return 2;
            case daxa::Format::R16G16_SFLOAT: return 2;
            case daxa::Format::R16G16B16_UNORM: return 3;
            case daxa::Format::R16G16B16_SNORM: return 3;
            case daxa::Format::R16G16B16_USCALED: return 3;
            case daxa::Format::R16G16B16_SSCALED: return 3;
            case daxa::Format::R16G16B16_UINT: return 3;
            case daxa::Format::R16G16B16_SINT: return 3;
            case daxa::Format::R16G16B16_SFLOAT: return 3;
            case daxa::Format::R16G16B16A16_UNORM: return 4;
            case daxa::Format::R16G16B16A16_SNORM: return 4;
            case daxa::Format::R16G16B16A16_USCALED: return 4;
            case daxa::Format::R16G16B16A16_SSCALED: return 4;
            case daxa::Format::R16G16B16A16_UINT: return 4;
            case daxa::Format::R16G16B16A16_SINT: return 4;
            case daxa::Format::R16G16B16A16_SFLOAT: return 4;
            case daxa::Format::R32_UINT: return 1;
            case daxa::Format::R32_SINT: return 1;
            case daxa::Format::R32_SFLOAT: return 1;
            case daxa::Format::R32G32_UINT: return 2;
            case daxa::Format::R32G32_SINT: return 2;
            case daxa::Format::R32G32_SFLOAT: return 2;
            case daxa::Format::R32G32B32_UINT: return 3;
            case daxa::Format::R32G32B32_SINT: return 3;
            case daxa::Format::R32G32B32_SFLOAT: return 3;
            case daxa::Format::R32G32B32A32_UINT: return 4;
            case daxa::Format::R32G32B32A32_SINT: return 4;
            case daxa::Format::R32G32B32A32_SFLOAT: return 4;
            case daxa::Format::R64_UINT: return 1;
            case daxa::Format::R64_SINT: return 1;
            case daxa::Format::R64_SFLOAT: return 1;
            case daxa::Format::R64G64_UINT: return 2;
            case daxa::Format::R64G64_SINT: return 2;
            case daxa::Format::R64G64_SFLOAT: return 2;
            case daxa::Format::R64G64B64_UINT: return 3;
            case daxa::Format::R64G64B64_SINT: return 3;
            case daxa::Format::R64G64B64_SFLOAT: return 3;
            case daxa::Format::R64G64B64A64_UINT: return 4;
            case daxa::Format::R64G64B64A64_SINT: return 4;
            case daxa::Format::R64G64B64A64_SFLOAT: return 4;
            case daxa::Format::B10G11R11_UFLOAT_PACK32: return 3;
            case daxa::Format::E5B9G9R9_UFLOAT_PACK32: return 4;
            case daxa::Format::D16_UNORM: return 1;
            case daxa::Format::X8_D24_UNORM_PACK32: return 2;
            case daxa::Format::D32_SFLOAT: return 1;
            case daxa::Format::S8_UINT: return 1;
            case daxa::Format::D16_UNORM_S8_UINT: return 2;
            case daxa::Format::D24_UNORM_S8_UINT: return 2;
            case daxa::Format::D32_SFLOAT_S8_UINT: return 2;
            case daxa::Format::BC1_RGB_UNORM_BLOCK: return 3;
            case daxa::Format::BC1_RGB_SRGB_BLOCK: return 3;
            case daxa::Format::BC1_RGBA_UNORM_BLOCK: return 4;
            case daxa::Format::BC1_RGBA_SRGB_BLOCK: return 4;
            case daxa::Format::BC2_UNORM_BLOCK: return 1;
            case daxa::Format::BC2_SRGB_BLOCK: return 1;
            case daxa::Format::BC3_UNORM_BLOCK: return 1;
            case daxa::Format::BC3_SRGB_BLOCK: return 1;
            case daxa::Format::BC4_UNORM_BLOCK: return 1;
            case daxa::Format::BC4_SNORM_BLOCK: return 1;
            case daxa::Format::BC5_UNORM_BLOCK: return 1;
            case daxa::Format::BC5_SNORM_BLOCK: return 1;
            default: return 0;
        }
        return 0;
    }

    auto scalar_kind_of_format(daxa::Format format) -> ScalarKind
    {
        switch (format)
        {
            case daxa::Format::UNDEFINED: return ScalarKind::FLOAT;
            case daxa::Format::R4G4_UNORM_PACK8: return ScalarKind::FLOAT;
            case daxa::Format::R4G4B4A4_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::B4G4R4A4_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::R5G6B5_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::B5G6R5_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::R5G5B5A1_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::B5G5R5A1_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::A1R5G5B5_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::R8_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::R8_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::R8_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::R8_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::R8_UINT: return ScalarKind::UINT;
            case daxa::Format::R8_SINT: return ScalarKind::INT;
            case daxa::Format::R8_SRGB: return ScalarKind::FLOAT;
            case daxa::Format::R8G8_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::R8G8_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::R8G8_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::R8G8_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::R8G8_UINT: return ScalarKind::UINT;
            case daxa::Format::R8G8_SINT: return ScalarKind::INT;
            case daxa::Format::R8G8_SRGB: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8_UINT: return ScalarKind::UINT;
            case daxa::Format::R8G8B8_SINT: return ScalarKind::INT;
            case daxa::Format::R8G8B8_SRGB: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8_UINT: return ScalarKind::UINT;
            case daxa::Format::B8G8R8_SINT: return ScalarKind::INT;
            case daxa::Format::B8G8R8_SRGB: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8A8_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8A8_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8A8_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8A8_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::R8G8B8A8_UINT: return ScalarKind::UINT;
            case daxa::Format::R8G8B8A8_SINT: return ScalarKind::INT;
            case daxa::Format::R8G8B8A8_SRGB: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8A8_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8A8_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8A8_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8A8_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8A8_UINT: return ScalarKind::UINT;
            case daxa::Format::B8G8R8A8_SINT: return ScalarKind::INT;
            case daxa::Format::B8G8R8A8_SRGB: return ScalarKind::FLOAT;
            case daxa::Format::A8B8G8R8_UNORM_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A8B8G8R8_SNORM_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A8B8G8R8_USCALED_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A8B8G8R8_SSCALED_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A8B8G8R8_UINT_PACK32: return ScalarKind::UINT;
            case daxa::Format::A8B8G8R8_SINT_PACK32: return ScalarKind::INT;
            case daxa::Format::A8B8G8R8_SRGB_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A2R10G10B10_UNORM_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A2R10G10B10_SNORM_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A2R10G10B10_USCALED_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A2R10G10B10_SSCALED_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A2R10G10B10_UINT_PACK32: return ScalarKind::UINT;
            case daxa::Format::A2R10G10B10_SINT_PACK32: return ScalarKind::INT;
            case daxa::Format::A2B10G10R10_UNORM_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A2B10G10R10_SNORM_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A2B10G10R10_USCALED_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A2B10G10R10_SSCALED_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::A2B10G10R10_UINT_PACK32: return ScalarKind::UINT;
            case daxa::Format::A2B10G10R10_SINT_PACK32: return ScalarKind::INT;
            case daxa::Format::R16_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::R16_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::R16_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::R16_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::R16_UINT: return ScalarKind::UINT;
            case daxa::Format::R16_SINT: return ScalarKind::INT;
            case daxa::Format::R16_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R16G16_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::R16G16_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::R16G16_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::R16G16_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::R16G16_UINT: return ScalarKind::UINT;
            case daxa::Format::R16G16_SINT: return ScalarKind::INT;
            case daxa::Format::R16G16_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16_UINT: return ScalarKind::UINT;
            case daxa::Format::R16G16B16_SINT: return ScalarKind::INT;
            case daxa::Format::R16G16B16_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16A16_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16A16_SNORM: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16A16_USCALED: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16A16_SSCALED: return ScalarKind::FLOAT;
            case daxa::Format::R16G16B16A16_UINT: return ScalarKind::UINT;
            case daxa::Format::R16G16B16A16_SINT: return ScalarKind::INT;
            case daxa::Format::R16G16B16A16_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R32_UINT: return ScalarKind::UINT;
            case daxa::Format::R32_SINT: return ScalarKind::INT;
            case daxa::Format::R32_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R32G32_UINT: return ScalarKind::UINT;
            case daxa::Format::R32G32_SINT: return ScalarKind::INT;
            case daxa::Format::R32G32_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R32G32B32_UINT: return ScalarKind::UINT;
            case daxa::Format::R32G32B32_SINT: return ScalarKind::INT;
            case daxa::Format::R32G32B32_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R32G32B32A32_UINT: return ScalarKind::UINT;
            case daxa::Format::R32G32B32A32_SINT: return ScalarKind::INT;
            case daxa::Format::R32G32B32A32_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R64_UINT: return ScalarKind::UINT;
            case daxa::Format::R64_SINT: return ScalarKind::INT;
            case daxa::Format::R64_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R64G64_UINT: return ScalarKind::UINT;
            case daxa::Format::R64G64_SINT: return ScalarKind::INT;
            case daxa::Format::R64G64_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R64G64B64_UINT: return ScalarKind::UINT;
            case daxa::Format::R64G64B64_SINT: return ScalarKind::INT;
            case daxa::Format::R64G64B64_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::R64G64B64A64_UINT: return ScalarKind::UINT;
            case daxa::Format::R64G64B64A64_SINT: return ScalarKind::INT;
            case daxa::Format::R64G64B64A64_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::B10G11R11_UFLOAT_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::E5B9G9R9_UFLOAT_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::D16_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::X8_D24_UNORM_PACK32: return ScalarKind::FLOAT;
            case daxa::Format::D32_SFLOAT: return ScalarKind::FLOAT;
            case daxa::Format::S8_UINT: return ScalarKind::UINT;
            case daxa::Format::D16_UNORM_S8_UINT: return ScalarKind::UINT;
            case daxa::Format::D24_UNORM_S8_UINT: return ScalarKind::UINT;
            case daxa::Format::D32_SFLOAT_S8_UINT: return ScalarKind::UINT;
            case daxa::Format::BC1_RGB_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC1_RGB_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC1_RGBA_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC1_RGBA_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC2_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC2_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC3_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC3_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC4_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC4_SNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC5_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC5_SNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC6H_UFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC6H_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC7_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::BC7_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ETC2_R8G8B8_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ETC2_R8G8B8_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ETC2_R8G8B8A1_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ETC2_R8G8B8A1_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ETC2_R8G8B8A8_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ETC2_R8G8B8A8_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::EAC_R11_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::EAC_R11_SNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::EAC_R11G11_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::EAC_R11G11_SNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_4x4_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_4x4_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_5x4_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_5x4_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_5x5_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_5x5_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_6x5_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_6x5_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_6x6_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_6x6_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_8x5_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_8x5_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_8x6_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_8x6_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_8x8_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_8x8_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x5_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x5_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x6_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x6_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x8_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x8_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x10_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x10_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_12x10_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_12x10_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_12x12_UNORM_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_12x12_SRGB_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::G8B8G8R8_422_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::B8G8R8G8_422_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G8_B8_R8_3PLANE_420_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G8_B8R8_2PLANE_420_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G8_B8_R8_3PLANE_422_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G8_B8R8_2PLANE_422_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G8_B8_R8_3PLANE_444_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::R10X6_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::R10X6G10X6_UNORM_2PACK16: return ScalarKind::FLOAT;
            case daxa::Format::R10X6G10X6B10X6A10X6_UNORM_4PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G10X6B10X6G10X6R10X6_422_UNORM_4PACK16: return ScalarKind::FLOAT;
            case daxa::Format::B10X6G10X6R10X6G10X6_422_UNORM_4PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::R12X4_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::R12X4G12X4_UNORM_2PACK16: return ScalarKind::FLOAT;
            case daxa::Format::R12X4G12X4B12X4A12X4_UNORM_4PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G12X4B12X4G12X4R12X4_422_UNORM_4PACK16: return ScalarKind::FLOAT;
            case daxa::Format::B12X4G12X4R12X4G12X4_422_UNORM_4PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G16B16G16R16_422_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::B16G16R16G16_422_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G16_B16_R16_3PLANE_420_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G16_B16R16_2PLANE_420_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G16_B16_R16_3PLANE_422_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G16_B16R16_2PLANE_422_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G16_B16_R16_3PLANE_444_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G8_B8R8_2PLANE_444_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::G10X6_B10X6R10X6_2PLANE_444_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G12X4_B12X4R12X4_2PLANE_444_UNORM_3PACK16: return ScalarKind::FLOAT;
            case daxa::Format::G16_B16R16_2PLANE_444_UNORM: return ScalarKind::FLOAT;
            case daxa::Format::A4R4G4B4_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::A4B4G4R4_UNORM_PACK16: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_4x4_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_5x4_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_5x5_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_6x5_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_6x6_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_8x5_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_8x6_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_8x8_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x5_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x6_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x8_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_10x10_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_12x10_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::ASTC_12x12_SFLOAT_BLOCK: return ScalarKind::FLOAT;
            case daxa::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG: return ScalarKind::FLOAT;
            case daxa::Format::PVRTC1_4BPP_UNORM_BLOCK_IMG: return ScalarKind::FLOAT;
            case daxa::Format::PVRTC2_2BPP_UNORM_BLOCK_IMG: return ScalarKind::FLOAT;
            case daxa::Format::PVRTC2_4BPP_UNORM_BLOCK_IMG: return ScalarKind::FLOAT;
            case daxa::Format::PVRTC1_2BPP_SRGB_BLOCK_IMG: return ScalarKind::FLOAT;
            case daxa::Format::PVRTC1_4BPP_SRGB_BLOCK_IMG: return ScalarKind::FLOAT;
            case daxa::Format::PVRTC2_2BPP_SRGB_BLOCK_IMG: return ScalarKind::FLOAT;
            case daxa::Format::PVRTC2_4BPP_SRGB_BLOCK_IMG: return ScalarKind::FLOAT;
            case daxa::Format::MAX_ENUM: return ScalarKind::FLOAT;
        }
        return ScalarKind::FLOAT;
    }
    
    auto is_format_depth_stencil(daxa::Format format) -> bool
    {
        switch(format)
        {
            case daxa::Format::D16_UNORM: return true;
            case daxa::Format::X8_D24_UNORM_PACK32: return true;
            case daxa::Format::D32_SFLOAT: return true;
            case daxa::Format::S8_UINT: return true;
            case daxa::Format::D16_UNORM_S8_UINT: return true;
            case daxa::Format::D24_UNORM_S8_UINT: return true;
            case daxa::Format::D32_SFLOAT_S8_UINT: return true;
        }
        return false;
    }    
    
    auto compute_shader_info(char const * ident) -> daxa::ComputePipelineCompileInfo2
    {
        std::string str(ident);
        auto offset = str.find(':');
        std::string path = str.substr(0, offset);
        std::string entry = str.substr(offset+1, str.size());
        daxa::ComputePipelineCompileInfo2 ret = {
            .source = daxa::ShaderFile{path},
            .entry_point = entry,
            .language = daxa::ShaderLanguage::SLANG,
            .defines = {},
            .enable_debug_info = {},
            .create_flags = {},
            .required_subgroup_size = {},
            .push_constant_size = {},
            .name = str,
        };
        return ret;
    }
} // namespace tido