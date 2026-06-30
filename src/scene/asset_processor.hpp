#pragma once

#include <filesystem>
#include <meshoptimizer.h>
#include <fastgltf/types.hpp>
#include <mutex>

#include "../timberdoodle.hpp"
#include "../gpu_context.hpp"
#include "../shader_shared/geometry.inl"
#include "importers/openvdb_importer.hpp"
#include "optimizers/image_optimizer.hpp" // TextureMaterialType + image cook types (moved out of here)
#include <ktx.h>

using namespace tido::types;

using MeshIndex = size_t;
using ImageIndex = size_t;
static constexpr u32 TIDO_VOLUMETRIC_CLOUD_FILE_VERSION = 1;

enum TidoVolumetricCloudFileFormat : u8
{
    RAW = 0, // All fields are stored in 16bit float format.
    CLOUD_SDF_BC1_DATA_BC6 = 1, // SDF field is stored using custom BC1 compression, the three data fields are stored using BC6 compression.
    COUNT = 2, // SDF field is stored using custom BC1 compression, the three data fields are stored using BC6 compression.
};

inline constexpr auto to_string(TidoVolumetricCloudFileFormat format) -> std::string_view
{
    switch(format)
    {
        case TidoVolumetricCloudFileFormat::RAW: return "RAW";
        case TidoVolumetricCloudFileFormat::CLOUD_SDF_BC1_DATA_BC6: return "CLOUD_SDF_BC1_DATA_BC6";
        default: return "UNKNOWN";
    }
}

struct TidoVolumetricCloudDataHeader
{
    u32 version = {}; // Leave this as the first field so that changes in the header do not break version checking.
    std::array<char, 4> magic; // "TDVC"
    TidoVolumetricCloudFileFormat format;
    i32vec3 field_extents;

    u32 field_count;
};

struct AssetProcessor
{
    enum struct AssetLoadResultCode
    {
        SUCCESS,
        ERROR_MISSING_INDEX_BUFFER,
        ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR,
        ERROR_FAULTY_BUFFER_VIEW,
        ERROR_COULD_NOT_OPEN_GLTF,
        ERROR_COULD_NOT_READ_BUFFER_IN_GLTF,
        ERROR_COULD_NOT_OPEN_FILE,
        ERROR_COULD_NOT_READ_FILE,
        ERROR_INVALID_HEADER_MAGIC_CONSTANT_IN_FILE,
        ERROR_INVALID_FIELD_COUNT_IN_FILE,
        ERROR_COULD_NOT_OPEN_TEXTURE_FILE,
        ERROR_COULD_NOT_READ_TEXTURE_FILE,
        ERROR_COULD_NOT_READ_TEXTURE_FILE_FROM_MEMSTREAM,
        ERROR_UNSUPPORTED_TEXTURE_PIXEL_FORMAT,
        ERROR_UNKNOWN_FILETYPE_FORMAT,
        ERROR_UNSUPPORTED_READ_FOR_FILEFORMAT,
        ERROR_URI_FILE_OFFSET_NOT_SUPPORTED,
        ERROR_UNSUPPORTED_ABSOLUTE_PATH,
        ERROR_MISSING_VERTEX_POSITIONS,
        ERROR_FAULTY_GLTF_VERTEX_POSITIONS,
        ERROR_MISSING_VERTEX_TEXCOORD_0,
        ERROR_FAULTY_GLTF_VERTEX_TEXCOORD_0,
        ERROR_MISSING_VERTEX_NORMALS,
        ERROR_FAULTY_GLTF_VERTEX_NORMALS,
        ERROR_MISSING_VERTEX_TANGENTS,
        ERROR_FAULTY_GLTF_VERTEX_TANGENTS,
        ERROR_FAILED_TO_PROCESS_KTX,
        ERROR_LAYER_IMAGES_NOT_IDENTICAL_SIZE,
        ERROR_LAYER_IMAGES_NOT_IDENTICAL_FORMAT,
    };
    static auto to_string(AssetLoadResultCode code) -> std::string_view
    {
        switch (code)
        {
            case AssetLoadResultCode::SUCCESS: return "SUCCESS"; 
            case AssetLoadResultCode::ERROR_MISSING_INDEX_BUFFER: return "ERROR_MISSING_INDEX_BUFFER"; 
            case AssetLoadResultCode::ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR: return "ERROR_FAULTY_INDEX_BUFFER_GLTF_ACCESSOR"; 
            case AssetLoadResultCode::ERROR_FAULTY_BUFFER_VIEW: return "ERROR_FAULTY_BUFFER_VIEW"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_OPEN_GLTF: return "ERROR_COULD_NOT_OPEN_GLTF"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_BUFFER_IN_GLTF: return "ERROR_COULD_NOT_READ_BUFFER_IN_GLTF"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_OPEN_FILE: return "ERROR_COULD_NOT_OPEN_FILE";
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_FILE: return "ERROR_COULD_NOT_READ_FILE";
            case AssetLoadResultCode::ERROR_INVALID_HEADER_MAGIC_CONSTANT_IN_FILE: return "ERROR_INVALID_HEADER_MAGIC_CONSTANT_IN_FILE";
            case AssetLoadResultCode::ERROR_INVALID_FIELD_COUNT_IN_FILE: return "ERROR_INVALID_FIELD_COUNT_IN_FILE";
            case AssetLoadResultCode::ERROR_COULD_NOT_OPEN_TEXTURE_FILE: return "ERROR_COULD_NOT_OPEN_TEXTURE_FILE"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_TEXTURE_FILE: return "ERROR_COULD_NOT_READ_TEXTURE_FILE"; 
            case AssetLoadResultCode::ERROR_COULD_NOT_READ_TEXTURE_FILE_FROM_MEMSTREAM: return "ERROR_COULD_NOT_READ_TEXTURE_FILE_FROM_MEMSTREAM"; 
            case AssetLoadResultCode::ERROR_UNSUPPORTED_TEXTURE_PIXEL_FORMAT: return "ERROR_UNSUPPORTED_TEXTURE_PIXEL_FORMAT"; 
            case AssetLoadResultCode::ERROR_UNKNOWN_FILETYPE_FORMAT: return "ERROR_UNKNOWN_FILETYPE_FORMAT"; 
            case AssetLoadResultCode::ERROR_UNSUPPORTED_READ_FOR_FILEFORMAT: return "ERROR_UNSUPPORTED_READ_FOR_FILEFORMAT"; 
            case AssetLoadResultCode::ERROR_URI_FILE_OFFSET_NOT_SUPPORTED: return "ERROR_URI_FILE_OFFSET_NOT_SUPPORTED"; 
            case AssetLoadResultCode::ERROR_UNSUPPORTED_ABSOLUTE_PATH: return "ERROR_UNSUPPORTED_ABSOLUTE_PATH"; 
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_POSITIONS: return "ERROR_MISSING_VERTEX_POSITIONS"; 
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_POSITIONS: return "ERROR_FAULTY_GLTF_VERTEX_POSITIONS"; 
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_TEXCOORD_0: return "ERROR_MISSING_VERTEX_TEXCOORD_0"; 
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_TEXCOORD_0: return "ERROR_FAULTY_GLTF_VERTEX_TEXCOORD_0"; 
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_NORMALS: return "ERROR_MISSING_VERTEX_NORMALS"; 
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_NORMALS: return "ERROR_FAULTY_GLTF_VERTEX_NORMALS"; 
            case AssetLoadResultCode::ERROR_MISSING_VERTEX_TANGENTS: return "ERROR_MISSING_VERTEX_TANGENTS"; 
            case AssetLoadResultCode::ERROR_FAULTY_GLTF_VERTEX_TANGENTS: return "ERROR_FAULTY_GLTF_VERTEX_TANGENTS"; 
            default: return "UNKNOWN";
        }
    }
    AssetProcessor(daxa::Device device);
    AssetProcessor(AssetProcessor &&) = default;
    ~AssetProcessor();

    struct LoadNonManifestTextureInfo
    {
        std::filesystem::path const & filepath;
        u32 layers = 1;
        bool load_as_srgb = true;
    };
    using NonmanifestLoadRet = std::variant<AssetProcessor::AssetLoadResultCode, daxa::ImageId>;
    auto load_nonmanifest_texture(LoadNonManifestTextureInfo const & info) -> NonmanifestLoadRet;

    struct ConvertVDBTaskInfo
    {
        std::filesystem::path vdb_path;
        std::filesystem::path converted_result_path;
        TidoVolumetricCloudFileFormat output_format;
        std::vector<VDBGridInfo> grids_to_convert;
        
        ThreadPool * threadpool;
    };

    struct ConvertVDBTask : Task
    {
        std::string progress_message;
        Task* current_subtask = nullptr;

        bool result;
        std::string error_message;

        ConvertVDBTaskInfo info;
        std::vector<std::vector<u16>> grids_data;

        ConvertVDBTask(ConvertVDBTaskInfo const & info);
        virtual void callback([[maybe_unused]] u32 chunk_index, [[maybe_unused]] u32 thread_index) override;
    };

    /**
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel.
     */
    struct LoadedTextureInfo
    {
        daxa::ImageId image = {};
        u32 texture_manifest_index = {};
        bool secondary_texture = {};
        bool compressed_bc5_rg = {};
    };
    // NOTE: glTF image -> texture loading (load_raw_image / load_image) moved out of AssetProcessor:
    // raw loading lives in the importer (part 1), compression in the optimizer (part 2), and the GPU
    // upload in the streamer.

    struct LoadCloudVolumetricDataInfo
    {
        std::filesystem::path volumetric_data_path = {};
        u32 cloud_data_texture_manifest_index = {};
        u32 cloud_sdf_texture_manifest_index = {};
    };
    auto load_cloud_volumetric_data(LoadCloudVolumetricDataInfo const & info) -> AssetLoadResultCode;

    /**
     * Collects the cloud-volume textures that finished loading since the last call so the scene can make
     * them resident. NOTE: gltf meshes/textures no longer flow through here - they go straight into the
     * scene manifest (Scene::add_mesh / Scene::add_texture). Only the not-yet-ported cloud volume
     * path still queues here.
     * THREADSAFETY:
     * * internally synchronized, can be called on multiple threads in parallel
     * * optimally called once a frame
     */
    struct LoadedResources
    {
        std::vector<LoadedTextureInfo> uploaded_textures = {};
    };
    auto collect_loaded_resources() -> LoadedResources;

    void clear();

  private:
    static inline std::string_view const VERT_ATTRIB_POSITION_NAME = "POSITION";
    static inline std::string_view const VERT_ATTRIB_TEXCOORD0_NAME = "TEXCOORD_0";
    static inline std::string_view const VERT_ATTRIB_NORMAL_NAME = "NORMAL";
    static inline std::string_view const VERT_ATTRIB_TANGENT_NAME = "TANGENT";

    daxa::Device _device = {};
    // TODO: Replace with lockless queue.
    std::vector<LoadedTextureInfo> _upload_texture_queue = {};

    std::unique_ptr<std::mutex> _texture_upload_mutex = std::make_unique<std::mutex>();
};