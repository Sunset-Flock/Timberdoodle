cmake_minimum_required (VERSION 3.21)
project(TimberDoodle)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")

find_package(Git REQUIRED)

find_package(Git REQUIRED)
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/deps/timberdoodle_assets")
    execute_process(COMMAND ${GIT_EXECUTABLE} clone https://github.com/Sunset-Flock/Timberdoodle-assets.git deps/timberdoodle_assets
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMAND_ERROR_IS_FATAL ANY)
else()
    execute_process(COMMAND ${GIT_EXECUTABLE} pull
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/deps/timberdoodle_assets
        COMMAND_ERROR_IS_FATAL ANY)
endif()

add_executable(${PROJECT_NAME} 
    "src/main.cpp"
    "src/window.cpp"
    "src/application.cpp"
    "src/gpu_context.cpp"
    "src/ui/ui.cpp"
    "src/ui/widgets/scene_graph.cpp"
    "src/rendering/renderer.cpp"
    "src/multithreading/thread_pool.cpp"
    "src/scene/scene.cpp"
    "src/scene/asset_processor.cpp"
)
find_package(fmt CONFIG REQUIRED)
find_package(daxa CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)     
find_package(fastgltf CONFIG REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(freeimage CONFIG REQUIRED)

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
    meshoptimizer::meshoptimizer
    fastgltf::fastgltf
    KTX::ktx
    freeimage::FreeImage
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_link_libraries(${PROJECT_NAME} PRIVATE Dwmapi)
endif()