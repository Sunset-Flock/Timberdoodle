cmake_minimum_required (VERSION 3.21)
include(cmake/vcpkg.cmake)

# Clone daxa if it's not present in the deps directory
if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/deps/daxa/CMakeLists.txt")
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/deps/meshoptimizer/CMakeLists.txt")
    find_package(Git REQUIRED)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        COMMAND_ERROR_IS_FATAL ANY)
endif()

project(TimberDoodle)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

# include_directories(${PROJECT_SOURCE_DIR}/deps/meshoptimizer)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/meshoptimizer)

set(DAXA_USE_VCPKG false)
set(DAXA_ENABLE_UTILS_IMGUI true)
set(DAXA_ENABLE_UTILS_MEM true)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG true)
set(DAXA_ENABLE_UTILS_PIPELINE_MANAGER_SLANG true)
set(DAXA_ENABLE_UTILS_TASK_GRAPH true)
set(DAXA_ENABLE_TESTS false)
add_subdirectory(${PROJECT_SOURCE_DIR}/deps/daxa)

add_executable(${PROJECT_NAME} 
    "src/main.cpp"
    "src/window.cpp"
    "src/camera.cpp"
    "src/application.cpp"
    "src/gpu_context.cpp"
    "src/ui/ui.cpp"
    "src/ui/widgets/scene_graph.cpp"
    "src/ui/widgets/property_viewer.cpp"
    "src/ui/widgets/camera_path_editor.cpp"
    "src/rendering/renderer.cpp"
    "src/rendering/pgi/pgi.cpp"
    "src/multithreading/thread_pool.cpp"
    "src/daxa_helper.cpp"
    "src/scene/scene.cpp"
    "src/scene/asset_processor.cpp"
    "src/json_handler.cpp"
    "src/rendering/rasterize_visbuffer/rasterize_visbuffer.cpp"
)
find_package(fmt CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)     
find_package(fastgltf CONFIG REQUIRED)
# find_package(meshoptimizer CONFIG REQUIRED)
find_package(freeimage CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)

# NOTE(grundlett): Really hacky. It appears there's no way to create plugin dll dependencies...
# see https://gitlab.kitware.com/cmake/cmake/-/issues/22993

find_program(SLANG_DLL slang.dll)
find_program(SLANG_GLSLANG_DLL slang-glslang.dll)
add_custom_command(TARGET TimberDoodle POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${SLANG_DLL} $<TARGET_FILE_DIR:TimberDoodle>
    COMMAND ${CMAKE_COMMAND} -E copy ${SLANG_GLSLANG_DLL} $<TARGET_FILE_DIR:TimberDoodle>
)

# FreeImage links OpenEXR, which adds /EHsc for its targets, even if we're using Clang
function(FIXUP_TARGET TGT_NAME)
    if(CMAKE_CXX_COMPILER_ID MATCHES Clang)
        get_property(A_OPTIONS TARGET ${TGT_NAME} PROPERTY INTERFACE_COMPILE_OPTIONS)
        string(REPLACE "/EHsc" "-Wl,/EHsc" A_OPTIONS "")
        set_property(TARGET ${TGT_NAME} PROPERTY INTERFACE_COMPILE_OPTIONS "${A_OPTIONS}")
    endif()
endfunction()

FIXUP_TARGET(Imath::Imath)
FIXUP_TARGET(OpenEXR::Iex)
FIXUP_TARGET(OpenEXR::IlmThread)
FIXUP_TARGET(OpenEXR::OpenEXRCore)
FIXUP_TARGET(OpenEXR::OpenEXR)
FIXUP_TARGET(OpenEXR::OpenEXRUtil)

find_package(ktx CONFIG REQUIRED)
option(KTX_FEATURE_TESTS "" OFF)
option(KTX_FEATURE_VULKAN "" OFF)
option(KTX_FEATURE_GL_UPLOAD "" OFF)
option(KTX_FEATURE_VK_UPLOAD "" OFF)
option(KTX_FEATURE_WRITE "" OFF)
option(KTX_FEATURE_TOOLS "" OFF)
option(KTX_FEATURE_STATIC_LIBRARY "" ON)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE
    fmt::fmt
    daxa::daxa
    glfw
    meshoptimizer
    fastgltf::fastgltf
    KTX::ktx
    freeimage::FreeImage
    nlohmann_json::nlohmann_json
    implot::implot
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(${PROJECT_NAME} PRIVATE Dwmapi)
endif()