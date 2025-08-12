#include "daxa_tg_debugger.hpp"

auto channel_count_of_format(daxa::Format format) -> daxa::u32
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

#include "daxa_tg_debugger.inl"

void debug_task(daxa::TaskInterface ti, DaxaTgDebugContext & tg_debug, daxa::ComputePipeline& pipeline, bool pre_task)
{
    if (!tg_debug.ui_open)
    {
        return;
    }

    // Construct Task Debug Info
    if (pre_task)
    {
        std::string name = std::string(ti.task_name);
        daxa::u32 name_counter = tg_debug.task_name_counters[name];
        tg_debug.task_name_counters.at(name) = name_counter + 1;
        if (name_counter > 0)
        {
            name += std::format(" (Nr. {})", name_counter + 1);
        }
        tg_debug.this_frame_debug_tasks.push_back(DaxaTgDebugContext::TgDebugTask{.task_index = ti.task_index, .task_name = name });
    }
    daxa::usize debug_task_index = tg_debug.this_frame_debug_tasks.size() - 1ull;
    auto& debug_task = tg_debug.this_frame_debug_tasks[debug_task_index];


    // Construct Image Attachment Debug Infos
    if (pre_task)
    {
        for (daxa::u32 i = 0; i < ti.attachment_infos.size(); ++i)
        {
            if (ti.attachment_infos[i].type != daxa::TaskAttachmentType::IMAGE)
                continue;
            
            std::string attachment_key = std::format("Task \"{}\" Attachment \"{}\"", debug_task.task_name, ti.attachment_infos[i].name());
            if (pre_task)
            {
                debug_task.attachments.push_back(ti.attachment_infos[i]);
            }
        }
    }


    // Perform Attachment inspector for all attachments with an active inspector:
    for (daxa::u32 i = 0; i < ti.attachment_infos.size(); ++i)
    {
        daxa::TaskImageAttachmentIndex src = {i};
        auto& attach_info = ti.get(src);

        std::string attachment_key = std::format("Task \"{}\" Attachment \"{}\"", debug_task.task_name, ti.attachment_infos[i].name());
        if (!tg_debug.inspector_states.contains(attachment_key))
            continue;
        DaxaTgDebugImageInspectorState& inspector_state = tg_debug.inspector_states.at(attachment_key);
        if (!inspector_state.active)
            continue;

        // Select if the inspector works on the image state from before or after the tasks execution
        if (inspector_state.pre_task != pre_task)
            continue;

        // Skip attachments with null resources
        if (ti.id(src).is_empty())
        {
            return;
        }

        // Gather attachment information
        auto attach_real_id = ti.id(src);
        auto attach_real_info = ti.info(src).value();
        auto attach_real_scalar_kind = scalar_kind_of_format(attach_real_info.format);


        // Go over all inspector resources, calculate the needed dimensions, create/destroy/recreate if required
        {
            // Readback buffer
            if (inspector_state.readback_buffer.is_empty())
            {
                inspector_state.readback_buffer = ti.device.create_buffer({
                    .size = sizeof(daxa_f32vec4) * 2 /*raw,color*/ * 4 /*frames in flight*/,
                    .allocate_info = daxa::MemoryFlagBits::HOST_ACCESS_RANDOM,
                    .name = std::string("readback buffer for ") + attachment_key,
                });
            }

            // Raw Image
            daxa::ImageInfo raw_copy_image_info = attach_real_info;
            if (raw_copy_image_info.format == daxa::Format::D32_SFLOAT)
                raw_copy_image_info.format = daxa::Format::R32_SFLOAT;
            if (raw_copy_image_info.format == daxa::Format::D16_UNORM)
                raw_copy_image_info.format = daxa::Format::R16_UNORM;
            raw_copy_image_info.usage = daxa::ImageUsageFlagBits::TRANSFER_DST | daxa::ImageUsageFlagBits::SHADER_STORAGE; // STORAGE is better than SAMPLED as it supports 64bit images.
            raw_copy_image_info.name = attachment_key + " raw image copy";
            if (inspector_state.raw_image_copy.is_empty())
            {
                inspector_state.freeze_image = false;
                inspector_state.raw_image_copy = ti.device.create_image(raw_copy_image_info);
            }
            else
            {
                daxa::ImageInfo prev_raw_copy_image_info = ti.device.image_info(inspector_state.raw_image_copy).value();
                auto const current_identical_to_prev = 
                    raw_copy_image_info.size.x == prev_raw_copy_image_info.size.x && 
                    raw_copy_image_info.size.y == prev_raw_copy_image_info.size.y &&
                    raw_copy_image_info.size.z == prev_raw_copy_image_info.size.z &&
                    raw_copy_image_info.mip_level_count == prev_raw_copy_image_info.mip_level_count &&
                    raw_copy_image_info.array_layer_count == prev_raw_copy_image_info.array_layer_count &&
                    raw_copy_image_info.format == prev_raw_copy_image_info.format;
                if (!current_identical_to_prev)
                {
                    inspector_state.freeze_image = false;
                    ti.device.destroy_image(inspector_state.raw_image_copy);
                    inspector_state.raw_image_copy = ti.device.create_image(raw_copy_image_info);
                }
            }

            // Display Image
            daxa::ImageInfo display_image_info = {};
            display_image_info.dimensions = 2u;
            display_image_info.size.x = std::max(1u, raw_copy_image_info.size.x >> inspector_state.mip);
            display_image_info.size.y = std::max(1u, raw_copy_image_info.size.y >> inspector_state.mip);
            display_image_info.size.z = 1u;
            display_image_info.mip_level_count = 1u;
            display_image_info.array_layer_count = 1u;
            display_image_info.sample_count = 1u;
            display_image_info.format = daxa::Format::R16G16B16A16_SFLOAT;
            display_image_info.name = attachment_key + " display image";
            display_image_info.usage = daxa::ImageUsageFlagBits::SHADER_SAMPLED | daxa::ImageUsageFlagBits::SHADER_STORAGE | daxa::ImageUsageFlagBits::TRANSFER_DST;

            if (inspector_state.display_image.is_empty())
            {
                inspector_state.freeze_image = false;
                inspector_state.display_image = ti.device.create_image(display_image_info);
            }
            else
            {
                auto const prev_size = ti.device.image_info(inspector_state.display_image).value().size;
                auto const size_changed = (display_image_info.size.x != prev_size.x || display_image_info.size.y != prev_size.y);
                if (size_changed)
                {
                    inspector_state.freeze_image = false;
                    ti.device.destroy_image(inspector_state.display_image);
                    inspector_state.display_image = ti.device.create_image(display_image_info);
                }
            }
        }
        

        // Copy attachment image to raw image
        if (!inspector_state.freeze_image)
        {
            // CLear before copy. Magenta marks mips/array layers that are not within the image slice of this attachment!
            daxa::ClearValue frozen_copy_clear = {};
            if (is_format_depth_stencil(attach_real_info.format))
            {
                frozen_copy_clear = daxa::DepthValue{ .depth = 0.0f, .stencil = 0u };
            }
            else
            {
                switch(attach_real_scalar_kind)
                {
                    case ScalarKind::FLOAT: frozen_copy_clear = std::array{1.0f,0.0f,1.0f,1.0f}; break;
                    case ScalarKind::INT: frozen_copy_clear = std::array{1,0,0,1}; break;
                    case ScalarKind::UINT: frozen_copy_clear = std::array{1u, 0u, 0u, 1u}; break;
                }
            }

            daxa::ImageMipArraySlice slice = {
                .level_count = attach_real_info.mip_level_count,
                .layer_count = attach_real_info.array_layer_count,
            };

            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                .image_slice = slice,
                .image_id = inspector_state.raw_image_copy,
            });

            ti.recorder.clear_image({
                .clear_value = std::array{1.0f,0.0f,1.0f,1.0f}, 
                .dst_image = inspector_state.raw_image_copy,
                .dst_slice = slice,
            });

            ti.recorder.pipeline_barrier(daxa::MemoryBarrierInfo{
                .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            });

            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .src_access = ti.get(src).access,
                .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
                .src_layout = ti.get(src).layout,
                .dst_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                .image_slice = ti.get(src).view.slice,
                .image_id = attach_real_id,
            });

            // Copy src image data to frozen image.
            for (daxa::u32 mip = slice.base_mip_level; mip < (slice.base_mip_level + slice.level_count); ++mip)
            {
                ti.recorder.copy_image_to_image({
                    .src_image = attach_real_id,
                    .dst_image = inspector_state.raw_image_copy,
                    .src_slice = daxa::ImageArraySlice::slice(slice, mip),
                    .dst_slice = daxa::ImageArraySlice::slice(slice, mip),
                    .extent = {
                        std::max(1u, attach_real_info.size.x >> mip),
                        std::max(1u, attach_real_info.size.y >> mip),
                        std::max(1u, attach_real_info.size.z >> mip)
                    },
                });
            }

            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_access = ti.get(src).access,
                .src_layout = daxa::ImageLayout::TRANSFER_SRC_OPTIMAL,
                .dst_layout = ti.get(src).layout,
                .image_slice = ti.get(src).view.slice,
                .image_id = attach_real_id,
            });
            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_access = daxa::AccessConsts::COMPUTE_SHADER_READ,
                .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                .dst_layout = daxa::ImageLayout::GENERAL,               // STORAGE always uses general in daxa
                .image_slice = slice,
                .image_id = inspector_state.raw_image_copy,
            });
        }
        auto const raw_image_copy_info = ti.device.info(inspector_state.raw_image_copy).value();
        auto const raw_image_copy = inspector_state.raw_image_copy;
        auto const display_image_info = ti.device.info(inspector_state.display_image).value();
        auto const scalar_kind = scalar_kind_of_format(raw_image_copy_info.format);
        inspector_state.runtime_image_info = raw_image_copy_info;
        inspector_state.attachment_info = attach_info;

        inspector_state.slice_valid = inspector_state.attachment_info.view.slice.contains(daxa::ImageMipArraySlice{
            .base_mip_level = inspector_state.mip,
            .base_array_layer = inspector_state.layer,
        });

        // Perform Inspector logic
        ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
            .dst_access = daxa::AccessConsts::TRANSFER_WRITE,
            .dst_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
            .image_id = inspector_state.display_image,
        });

        ti.recorder.clear_image({
            .clear_value = std::array{1.0f,0.0f,1.0f,1.0f}, 
            .dst_image = inspector_state.display_image,
        });

        if (inspector_state.slice_valid)
        {
            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
                .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                .dst_layout = daxa::ImageLayout::GENERAL,
                .image_slice = ti.device.image_view_info(inspector_state.display_image.default_view()).value().slice,
                .image_id = inspector_state.display_image,
            });

            ti.recorder.set_pipeline(pipeline);

            daxa::ImageViewInfo src_image_view_info = ti.device.image_view_info(raw_image_copy.default_view()).value();
            src_image_view_info.slice.level_count = 1;
            src_image_view_info.slice.layer_count = 1;
            src_image_view_info.slice.base_mip_level = inspector_state.mip;
            src_image_view_info.slice.base_array_layer = inspector_state.layer;
            daxa::ImageViewId src_view = ti.device.create_image_view(src_image_view_info);
            ti.recorder.destroy_image_view_deferred(src_view);
            ti.recorder.push_constant(DebugTaskDrawDebugDisplayPush{
                .src = src_view,
                .dst = inspector_state.display_image.default_view(),
                .src_size = { display_image_info.size.x, display_image_info.size.y },
                .image_view_type = static_cast<daxa::u32>(src_image_view_info.type),
                .format = static_cast<daxa::i32>(scalar_kind),
                .float_min = static_cast<daxa::f32>(inspector_state.min_v),
                .float_max = static_cast<daxa::f32>(inspector_state.max_v),
                .int_min = static_cast<daxa::i32>(inspector_state.min_v),
                .int_max = static_cast<daxa::i32>(inspector_state.max_v),
                .uint_min = static_cast<daxa::u32>(inspector_state.min_v),
                .uint_max = static_cast<daxa::u32>(inspector_state.max_v),
                .rainbow_ints = inspector_state.rainbow_ints,
                .enabled_channels = inspector_state.enabled_channels,
                .mouse_over_index = {
                    inspector_state.mouse_pos_relative_to_image_mip0.x >> inspector_state.mip,
                    inspector_state.mouse_pos_relative_to_image_mip0.y >> inspector_state.mip,
                },
                .readback_ptr = ti.device.device_address(inspector_state.readback_buffer).value(),
                .readback_index = tg_debug.readback_index,
            }); 
            ti.recorder.dispatch({
                (display_image_info.size.x + DEBUG_DRAW_CLONE_X - 1) / DEBUG_DRAW_CLONE_X,
                (display_image_info.size.y + DEBUG_DRAW_CLONE_Y - 1) / DEBUG_DRAW_CLONE_Y,
                1,
            });
            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .src_access = daxa::AccessConsts::COMPUTE_SHADER_WRITE,
                .dst_access = daxa::AccessConsts::FRAGMENT_SHADER_READ,
                .src_layout = daxa::ImageLayout::GENERAL,
                .dst_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
                .image_id = inspector_state.display_image,
            });
        }
        else // ui image slice NOT valid.
        {
            ti.recorder.pipeline_barrier_image_transition(daxa::ImageMemoryBarrierInfo{
                .src_access = daxa::AccessConsts::TRANSFER_WRITE,
                .dst_access = daxa::AccessConsts::FRAGMENT_SHADER_READ,
                .src_layout = daxa::ImageLayout::TRANSFER_DST_OPTIMAL,
                .dst_layout = daxa::ImageLayout::READ_ONLY_OPTIMAL,
                .image_id = inspector_state.display_image,
            });
        }
    }
}


#include <imgui.h>
#include <implot.h>

auto format_vec4_rows_float(daxa_f32vec4 vec) -> std::string
{
    return std::format("R: {:10.7}\nG: {:10.7}\nB: {:10.7}\nA: {:10.7}",
        vec.x,
        vec.y,
        vec.z,
        vec.w);
}

auto format_vec4_rows(Vec4Union vec_union, ScalarKind scalar_kind) -> std::string
{
    switch (scalar_kind)
    {
        case ScalarKind::FLOAT:
            return format_vec4_rows_float(vec_union._float);
        case ScalarKind::INT:
            return std::format("R: {:11}\nG: {:11}\nB: {:11}\nA: {:11}",
                vec_union._int.x,
                vec_union._int.y,
                vec_union._int.z,
                vec_union._int.w);
        case ScalarKind::UINT:
            return std::format("R: {:11}\nG: {:11}\nB: {:11}\nA: {:11}",
                vec_union._uint.x,
                vec_union._uint.y,
                vec_union._uint.z,
                vec_union._uint.w);
    }
    return std::string();
}

void tg_debug_image_inspector(
    daxa::ImGuiRenderer & imgui_renderer, 
    daxa::Device & device, 
    daxa::SamplerId lin_clamp_sampler, 
    daxa::SamplerId nearest_clamp_sampler, 
    DaxaTgDebugContext & tg_debug, 
    std::string active_attachment_key, 
    daxa::u32 frame_index)
{
    tg_debug.readback_index = frame_index % (tg_debug.frames_in_flight + 1);
    auto & state = tg_debug.inspector_states[active_attachment_key];
    if (ImGui::Begin(std::format("Inspector for {}", active_attachment_key.c_str()).c_str(), nullptr, {}))
    {
        // The ui update is staggered a frame.
        // This is because the ui gets information from the task graph with a delay of one frame.
        // Because of this we first shedule a draw for the previous frames debug image canvas.
        ImTextureID tex_id = {};
        daxa::ImageInfo clone_image_info = {};
        daxa::ImageInfo const & image_info = state.runtime_image_info;
        if (!state.display_image.is_empty())
        {
            clone_image_info = device.image_info(state.display_image).value();

            daxa::SamplerId sampler = lin_clamp_sampler;
            if (state.nearest_filtering)
            {
                sampler = nearest_clamp_sampler;
            }
            tex_id = imgui_renderer.create_texture_id({
                .image_view_id = state.display_image.default_view(),
                .sampler_id = sampler,
            });
            daxa::u32 active_channels_of_format = channel_count_of_format(image_info.format);

            // Now we actually process the ui.
            daxa::TaskImageAttachmentInfo const & attachment_info = state.attachment_info;
            auto slice = attachment_info.view.slice;

            if (ImGui::BeginTable("Some Inspected Image", 2, ImGuiTableFlags_NoHostExtendX | ImGuiTableFlags_SizingFixedFit))
            {
                ImGui::TableSetupColumn("Inspector settings", ImGuiTableFlags_NoHostExtendX | ImGuiTableFlags_SizingFixedFit);
                ImGui::TableSetupColumn("Image view", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableNextColumn();
                ImGui::SeparatorText("Inspector settings");
                ImGui::Checkbox("pre task", &state.pre_task);
                ImGui::SameLine();
                ImGui::Checkbox("freeze image", &state.freeze_image);
                ImGui::SetItemTooltip("make sure to NOT have freeze image set when switching this setting. The frozen image is either pre or post.");
                ImGui::PushItemWidth(80);
                daxa::i32 imip = state.mip;
                ImGui::InputInt("mip", &imip, 1);
                state.mip = imip;
                daxa::i32 ilayer = state.layer;
                ImGui::SameLine();
                ImGui::InputInt("layer", &ilayer, 1);
                state.layer = ilayer;
                ImGui::PopItemWidth();
                ImGui::PushItemWidth(180);
                ImGui::Text("selected mip size: (%i,%i,%i)", std::max(image_info.size.x >> state.mip, 1u), std::max(image_info.size.y >> state.mip, 1u), std::max(image_info.size.z >> state.mip, 1u));
                if (!state.slice_valid)
                    ImGui::PushStyleColor(ImGuiCol_Text, 0xFF0000FF);
                ImGui::Text(state.slice_valid ? "" : "SELECTED SLICE INVALID");
                if (!state.slice_valid)
                    ImGui::PopStyleColor();
                auto modes = std::array{
                    "Linear",
                    "Nearest",
                };
                ImGui::Combo("sampler", &state.nearest_filtering, modes.data(), modes.size());

                if (ImGui::BeginTable("Channels", 4, ImGuiTableFlags_NoBordersInBody | ImGuiTableFlags_SizingFixedFit))
                {
                    std::array channel_names = {"r", "g", "b", "a"};
                    std::array<bool, 4> channels = {};

                    daxa::u32 active_channel_count = 0;
                    daxa::i32 last_active_channel = -1;
                    for (daxa::u32 channel = 0; channel < 4; ++channel)
                    {
                        channels[channel] = std::bit_cast<std::array<daxa::u32, 4u>>(state.enabled_channels)[channel];
                        active_channel_count += channels[channel] ? 1u : 0u;
                        last_active_channel = channels[channel] ? channel : channels[channel];
                    }

                    for (daxa::u32 channel = 0; channel < 4; ++channel)
                    {
                        auto const disabled = channel >= active_channels_of_format;
                        ImGui::BeginDisabled(disabled);
                        if (disabled)
                            channels[channel] = false;
                        ImGui::TableNextColumn();
                        bool const clicked = ImGui::Checkbox(channel_names[channel], channels.data() + channel);
                        ImGui::EndDisabled();
                        if (disabled)
                            ImGui::SetItemTooltip("image format does not have this channel");
                    }

                    state.enabled_channels.x = channels[0];
                    state.enabled_channels.y = channels[1];
                    state.enabled_channels.z = channels[2];
                    state.enabled_channels.w = channels[3];
                    ImGui::EndTable();
                }
                ImGui::PopItemWidth();
                ImGui::PushItemWidth(90);
                ImGui::InputDouble("min", &state.min_v);
                ImGui::SetItemTooltip("min value only effects rgb, not alpha");
                ImGui::SameLine();
                ImGui::InputDouble("max", &state.max_v);
                ImGui::SetItemTooltip("max value only effects rgb, not alpha");

                Vec4Union readback_raw = {};
                daxa_f32vec4 readback_color = {};
                daxa_f32vec4 readback_color_min = {};
                daxa_f32vec4 readback_color_max = {};

                ScalarKind scalar_kind = scalar_kind_of_format(image_info.format);
                if (!state.readback_buffer.is_empty())
                {
                    switch (scalar_kind)
                    {
                        case ScalarKind::FLOAT: readback_raw._float = device.buffer_host_address_as<daxa_f32vec4>(state.readback_buffer).value()[tg_debug.readback_index * 2]; break;
                        case ScalarKind::INT:   readback_raw._int = device.buffer_host_address_as<daxa_i32vec4>(state.readback_buffer).value()[tg_debug.readback_index * 2]; break;
                        case ScalarKind::UINT:  readback_raw._uint = device.buffer_host_address_as<daxa_u32vec4>(state.readback_buffer).value()[tg_debug.readback_index * 2]; break;
                    }
                    auto floatvec_readback = device.buffer_host_address_as<daxa_f32vec4>(state.readback_buffer).value();
                    readback_color = floatvec_readback[tg_debug.readback_index * 2 + 1];
                }

                constexpr auto MOUSE_PICKER_FREEZE_COLOR = 0xFFBBFFFF;
                auto mouse_picker = [&](daxa_i32vec2 image_idx, bool frozen, Vec4Union readback_union)
                {
                    if (frozen)
                    {
                        ImGui::PushStyleColor(ImGuiCol_Text, MOUSE_PICKER_FREEZE_COLOR);
                    }
                    // ImGui::Dummy({0, 2});
                    constexpr auto MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH = 7;
                    constexpr auto MOUSE_PICKER_MAGNIFIER_DISPLAY_SIZE = ImVec2{70.0f, 70.0f};
                    daxa_i32vec2 image_idx_at_mip = {
                        image_idx.x >> state.mip,
                        image_idx.y >> state.mip,
                    };
                    ImVec2 magnify_start_uv = {
                        float(image_idx_at_mip.x - (MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH / 2)) * (1.0f / float(clone_image_info.size.x)),
                        float(image_idx_at_mip.y - (MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH / 2)) * (1.0f / float(clone_image_info.size.y)),
                    };
                    ImVec2 magnify_end_uv = {
                        float(image_idx_at_mip.x + MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH / 2 + 1) * (1.0f / float(clone_image_info.size.x)),
                        float(image_idx_at_mip.y + MOUSE_PICKER_MAGNIFIER_TEXEL_WIDTH / 2 + 1) * (1.0f / float(clone_image_info.size.y)),
                    };
                    if (tex_id && image_idx.x >= 0 && image_idx.y >= 0 && image_idx.x < image_info.size.x && image_idx.y < image_info.size.y)
                    {
                        ImGui::Image(tex_id, MOUSE_PICKER_MAGNIFIER_DISPLAY_SIZE, magnify_start_uv, magnify_end_uv);
                    }
                    if (frozen)
                    {
                        ImGui::PopStyleColor(1);
                    }
                    ImGui::SameLine();
                    daxa_i32vec2 index_at_mip = {
                        image_idx.x >> state.mip,
                        image_idx.y >> state.mip,
                    };
                    ImGui::Text("(%5d,%5d) %s\n%s",
                        index_at_mip.x,
                        index_at_mip.y,
                        frozen ? "FROZEN" : "      ",
                        format_vec4_rows(readback_union, scalar_kind).c_str());
                };

                ImGui::SeparatorText("Mouse Picker (?)\n");
                ImGui::SetItemTooltip(
                    "Usage:\n"
                    "  * left click on image to freeze selection, left click again to unfreeze\n"
                    "  * hold shift to replicate the selection on all other open inspector mouse pickers (also replicates freezes)\n"
                    "  * use middle mouse button to grab and move zoomed in image");
                if (state.display_image_hovered || state.freeze_image_hover_index || tg_debug.override_mouse_picker)
                {
                    mouse_picker(state.frozen_mouse_pos_relative_to_image_mip0, state.freeze_image_hover_index, state.frozen_readback_raw);
                }
                else
                {
                    mouse_picker(daxa_i32vec2{0, 0}, false, {});
                }

                ImGui::PopItemWidth();

                ImGui::TableNextColumn();
                ImGui::Text("slice used in task: %s", daxa::to_string(slice).c_str());
                ImGui::Text("size: (%i,%i,%i), mips: %i, layers: %i, format: %s",
                    image_info.size.x,
                    image_info.size.y,
                    image_info.size.z,
                    image_info.mip_level_count,
                    image_info.array_layer_count,
                    daxa::to_string(image_info.format).data());

                ImGui::SetNextItemWidth(60.0f);
                ImGui::InputFloat("scaling", &state.inspector_image_draw_scale);
                ImGui::SetItemTooltip("scaling of -1 causes automatic scaling to fit the current window size");

                ImGui::SameLine();
                ImGui::Checkbox("fix mip sizes", &state.fixed_display_mip_sizes);
                ImGui::SetItemTooltip("fixes all displayed mip sizes to be the scaled size of mip 0");
                
                if (tex_id)
                {
                    float const aspect = static_cast<float>(clone_image_info.size.y) / static_cast<float>(clone_image_info.size.x);
                    float const auto_scale = std::min(
                        ImGui::GetContentRegionAvail().x / static_cast<float>(image_info.size.x), 
                        ImGui::GetContentRegionAvail().y / static_cast<float>(image_info.size.y)
                    );
                    ImVec2 image_display_size = { auto_scale * static_cast<float>(image_info.size.x), auto_scale * static_cast<float>(image_info.size.y) };
                    if (state.inspector_image_draw_scale > 0.0f)
                    {
                        ImVec2 fixed_size_draw_size = {};
                        if (state.fixed_display_mip_sizes)
                        {
                            fixed_size_draw_size.x = static_cast<float>(image_info.size.x) * state.inspector_image_draw_scale;
                            fixed_size_draw_size.y = static_cast<float>(image_info.size.y) * state.inspector_image_draw_scale;
                        }
                        else
                        {
                            fixed_size_draw_size.x = static_cast<float>(clone_image_info.size.x) * state.inspector_image_draw_scale;
                            fixed_size_draw_size.y = static_cast<float>(clone_image_info.size.y) * state.inspector_image_draw_scale;
                        };

                        image_display_size = fixed_size_draw_size;
                    }

                    ImVec2 start_pos = ImGui::GetCursorScreenPos();
                    state.display_image_size = daxa_i32vec2(image_display_size.x, image_display_size.y);
                    ImGui::BeginChild("scrollable image", ImVec2(0, 0), {}, ImGuiWindowFlags_HorizontalScrollbar);
                    ImVec2 scroll_offset = ImVec2{ImGui::GetScrollX(), ImGui::GetScrollY()};
                    if (state.display_image_hovered && ImGui::IsKeyDown(ImGuiKey_MouseMiddle))
                    {
                        ImGui::SetScrollX(ImGui::GetScrollX() - ImGui::GetIO().MouseDelta.x);
                        ImGui::SetScrollY(ImGui::GetScrollY() - ImGui::GetIO().MouseDelta.y);
                    }
                    else if (state.display_image_hovered && ImGui::IsKeyDown(ImGuiKey_LeftShift) && (ImGui::GetIO().MouseWheel != 0.0f))
                    {
                        if (state.inspector_image_draw_scale <= 0.0f)
                        {
                            state.inspector_image_draw_scale = auto_scale;
                        }
                        state.inspector_image_draw_scale *= (1.0f + ImGui::GetIO().MouseWheel * 0.1f);
                    }
                    ImGui::Image(tex_id, image_display_size);
                    ImVec2 const mouse_pos = ImGui::GetMousePos();
                    ImVec2 const end_pos = ImVec2{start_pos.x + image_display_size.x, start_pos.y + image_display_size.y};

                    ImVec2 const clipped_display_image_size = {
                        end_pos.x - start_pos.x,
                        end_pos.y - start_pos.y,
                    };
                    state.display_image_hovered = ImGui::IsMouseHoveringRect(start_pos, end_pos) && (ImGui::IsItemHovered() || ImGui::IsItemClicked());
                    state.freeze_image_hover_index = state.freeze_image_hover_index ^ (state.display_image_hovered && ImGui::IsItemClicked());
                    state.mouse_pos_relative_to_display_image = daxa_i32vec2(mouse_pos.x - start_pos.x, mouse_pos.y - start_pos.y);
                    tg_debug.request_mouse_picker_override |= state.display_image_hovered && ImGui::IsKeyDown(ImGuiKey_LeftShift);

                    bool const override_other_inspectors = tg_debug.override_mouse_picker && state.display_image_hovered;
                    bool const get_overriden = tg_debug.override_mouse_picker && !state.display_image_hovered;
                    if (override_other_inspectors)
                    {
                        tg_debug.override_frozen_state = state.freeze_image_hover_index;
                        tg_debug.override_mouse_picker_uv = {
                            float(state.mouse_pos_relative_to_display_image.x) / clipped_display_image_size.x,
                            float(state.mouse_pos_relative_to_display_image.y) / clipped_display_image_size.y,
                        };
                    }
                    if (get_overriden)
                    {
                        state.freeze_image_hover_index = tg_debug.override_frozen_state;
                        state.mouse_pos_relative_to_display_image = {
                            static_cast<daxa::i32>(tg_debug.override_mouse_picker_uv.x * clipped_display_image_size.x),
                            static_cast<daxa::i32>(tg_debug.override_mouse_picker_uv.y * clipped_display_image_size.y),
                        };
                    }

                    state.mouse_pos_relative_to_image_mip0 = daxa_i32vec2(
                        ((state.mouse_pos_relative_to_display_image.x + scroll_offset.x) / static_cast<float>(state.display_image_size.x)) * static_cast<float>(image_info.size.x),
                        ((state.mouse_pos_relative_to_display_image.y + scroll_offset.y) / static_cast<float>(state.display_image_size.y)) * static_cast<float>(image_info.size.y));

                    float x = ImGui::GetScrollMaxX();
                    float y = ImGui::GetScrollMaxY();

                    if (!state.freeze_image_hover_index)
                    {
                        state.frozen_mouse_pos_relative_to_image_mip0 = state.mouse_pos_relative_to_image_mip0;
                        state.frozen_readback_raw = readback_raw;
                        state.frozen_readback_color = readback_color;
                    }
                    if (state.display_image_hovered)
                    {
                        ImGui::BeginTooltip();
                        mouse_picker(state.mouse_pos_relative_to_image_mip0, false, readback_raw);
                        ImGui::EndTooltip();
                    }
                    if (state.display_image_hovered || tg_debug.override_mouse_picker)
                    {
                        ImVec2 const frozen_mouse_pos_relative_to_display_image = {
                            float(state.mouse_pos_relative_to_image_mip0.x) / float(image_info.size.x) * state.display_image_size.x - scroll_offset.x,
                            float(state.mouse_pos_relative_to_image_mip0.y) / float(image_info.size.y) * state.display_image_size.y - scroll_offset.y,
                        };
                        ImVec2 const window_marker_pos = {
                            start_pos.x + frozen_mouse_pos_relative_to_display_image.x,
                            start_pos.y + frozen_mouse_pos_relative_to_display_image.y,
                        };
                        ImGui::GetWindowDrawList()->AddCircle(window_marker_pos, 5.0f, ImGui::GetColorU32(ImVec4{
                                                                                           readback_color.x > 0.5f ? 0.0f : 1.0f,
                                                                                           readback_color.y > 0.5f ? 0.0f : 1.0f,
                                                                                           readback_color.z > 0.5f ? 0.0f : 1.0f,
                                                                                           1.0f,
                                                                                       }));
                    }
                    if (state.freeze_image_hover_index)
                    {
                        ImVec2 const frozen_mouse_pos_relative_to_display_image = {
                            float(state.frozen_mouse_pos_relative_to_image_mip0.x) / float(image_info.size.x) * state.display_image_size.x - scroll_offset.x,
                            float(state.frozen_mouse_pos_relative_to_image_mip0.y) / float(image_info.size.y) * state.display_image_size.y - scroll_offset.y,
                        };
                        ImVec2 const window_marker_pos = {
                            start_pos.x + frozen_mouse_pos_relative_to_display_image.x,
                            start_pos.y + frozen_mouse_pos_relative_to_display_image.y,
                        };
                        auto inv_color = ImVec4{
                            state.frozen_readback_color.x > 0.5f ? 0.0f : 1.0f,
                            state.frozen_readback_color.y > 0.5f ? 0.0f : 1.0f,
                            state.frozen_readback_color.z > 0.5f ? 0.0f : 1.0f,
                            1.0f,
                        };
                        ImGui::GetWindowDrawList()->AddCircle(window_marker_pos, 5.0f, ImGui::GetColorU32(inv_color));
                    }
                    ImGui::EndChild();
                }
                ImGui::EndTable();
            }
        }
        ImGui::End();
    }
}

void tg_resource_debug_ui(
    daxa::ImGuiRenderer & imgui_renderer, 
    daxa::Device & device, 
    daxa::SamplerId lin_clamp_sampler, 
    daxa::SamplerId nearest_clamp_sampler, 
    DaxaTgDebugContext & tg_debug, 
    daxa::u32 frame_index, 
    bool ui_open)
{
    tg_debug.ui_open = ui_open;
    tg_debug.task_name_counters.clear();
    if (ui_open && ImGui::Begin("TG Debug Clones", nullptr, ImGuiWindowFlags_NoCollapse))
    {
        bool const clear_search = ImGui::Button("clear");
        if (clear_search)
            tg_debug.search_substr = {};
        ImGui::SameLine();
        ImGui::SetNextItemWidth(200);
        ImGui::InputText("Search for Task", tg_debug.search_substr.data(), tg_debug.search_substr.size());
        for (auto & c : tg_debug.search_substr)
            c = std::tolower(c);

        bool const search_used = tg_debug.search_substr[0] != '\0';

        ImGui::BeginChild("Tasks");
        for (auto task : tg_debug.this_frame_debug_tasks)
        {
            if (task.task_name.size() == 0 || task.task_name.c_str()[0] == 0)
                continue;

            if (search_used)
            {
                std::string compare_string = task.task_name;
                for (auto & c : compare_string)
                    c = std::tolower(c);
                if (!strstr(compare_string.c_str(), tg_debug.search_substr.data()))
                    continue;
            }

            if (ImGui::CollapsingHeader(task.task_name.c_str()))
            {
                for (auto attach : task.attachments)
                {
                    std::string attachment_key = std::format("Task \"{}\" Attachment \"{}\"", task.task_name, attach.name());
                    ImGui::PushID(attachment_key.c_str());
                    if (ImGui::Button(attach.name()))
                    {
                        bool already_active = tg_debug.inspector_states[attachment_key].active;
                        if (already_active)
                        {
                            auto iter = tg_debug.active_inspectors.find(attachment_key);
                            if (iter != tg_debug.active_inspectors.end())
                            {
                                tg_debug.active_inspectors.erase(iter);
                            }
                            tg_debug.inspector_states[attachment_key].active = false;
                        }
                        else
                        {
                            tg_debug.active_inspectors.emplace(attachment_key);
                            tg_debug.inspector_states[attachment_key].active = true;
                        }
                    }
                    ImGui::PopID();
                    ImGui::SameLine();
                    switch (attach.type)
                    {
                        case daxa::TaskAttachmentType::IMAGE:
                            ImGui::Text("| view: %s", daxa::to_string(attach.value.image.view.slice).c_str());
                            ImGui::SameLine();
                            ImGui::Text((std::format("| task access: {}", daxa::to_string(attach.value.image.task_access))).c_str());
                            break;
                        default: break;
                    }
                }
            }
        }
        ImGui::EndChild();

        ImGui::End();
        for (auto active_attachment_key : tg_debug.active_inspectors)
        {
            tg_debug_image_inspector(
                imgui_renderer, 
                device, 
                lin_clamp_sampler, 
                nearest_clamp_sampler, 
                tg_debug,
                active_attachment_key, 
                frame_index);
        }
    }
    if (tg_debug.request_mouse_picker_override)
    {
        tg_debug.override_mouse_picker = true;
    }
    else
    {
        tg_debug.override_mouse_picker = false;
    }
    tg_debug.request_mouse_picker_override = false;
    tg_debug.this_frame_debug_tasks.clear();
}