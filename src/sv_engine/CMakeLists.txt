cmake_minimum_required(VERSION 3.16)

if (POLICY CMP0092)
    cmake_policy(SET CMP0092 NEW)
endif()

project(sv_engine VERSION "1.0.0" LANGUAGES C CXX)

option(WITH_PROFILE_CUDA_TIME "Profile CUDARF cuda pipeline performance" OFF)
option(WITH_BUILD_NATIVE "Enable native x64-specific compile options" OFF)

set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_library(AVFORMAT avformat REQUIRED)
find_library(AVCODEC  avcodec  REQUIRED)
find_library(AVUTIL   avutil   REQUIRED)
find_library(SWSCALE  swscale  REQUIRED)


set(PUBLIC_HEADERS
    include/engine/engine.hpp
    include/engine/views_config.hpp
    include/engine/overlays_config.hpp
    include/engine/frame_packet.hpp
    include/engine/frame_source.hpp
    include/engine/video_source.hpp
    include/engine/image_frame_loader.hpp
    include/engine/png_source.hpp
)

set(SOURCE_FILES
    src/engine.cpp
    src/engine_init.cpp
    src/engine_frame.cpp
    src/engine_view_control.cpp
    src/world.cpp
    src/sv_gl_common.cpp

    src/video_io/png_source.cpp
    src/video_io/image_frame_loader.cpp

    src/views/view_3d.cpp
    src/views/viewpoint_registry.cpp
    src/views/view_navigation_state.cpp
    src/views/view_interaction_router.cpp
    src/views/surround_view_composer.cpp
    src/views/scene_pass_builder.cpp
    src/views/view_post_process_pipeline.cpp

    src/vehicle/vehicle_state.cpp

    src/overlays/vehicle/vehicle_controller.cpp
    src/overlays/underlay/underlay.cpp
    src/overlays/2d/sky_fade.cpp

    src/virtual_controls/viewpoints/viewpoints.cpp
    )

if(NOT TARGET spdlog::spdlog)
    find_package(spdlog REQUIRED CONFIG)
endif()

if(NOT TARGET engine_interface)
    find_package(engine_interface REQUIRED CONFIG)
endif()

find_package(CUDAToolkit REQUIRED)

set(rfTarget rf)
if(NOT TARGET ${rfTarget})
    find_package(rf REQUIRED CONFIG)
endif()

add_library(sv_engine SHARED ${PUBLIC_HEADERS} ${SOURCE_FILES})
if(WITH_PROFILE_CUDA_TIME)
  target_compile_definitions(${PROJECT_NAME} PRIVATE PROFILE_CUDA_TIME DUMP_FRAME_TIMING)
endif()

find_package(glm REQUIRED CONFIG)

target_compile_options(${PROJECT_NAME}
    PRIVATE
        -Wall
        $<$<BOOL:${WITH_BUILD_NATIVE}>:-msse4.1  -Wno-unused-variable -Wno-unused-function -Wno-sequence-point>
        -fPIC
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/
        $<$<BOOL:${WITH_BUILD_NATIVE}>:${CMAKE_CURRENT_SOURCE_DIR}/../neon-sse-wrap>
        ${GLM_INCLUDE_DIRS}
        src
        src/overlays
        )

target_link_libraries(${PROJECT_NAME}
        engine_interface
        spdlog::spdlog
        CUDA::cudart
        ${AVFORMAT}
        ${AVCODEC}
        ${AVUTIL}
        ${SWSCALE}
        ${rfTarget}
    )
    
set_target_properties(${PROJECT_NAME} PROPERTIES
    EXPORT_NAME sv_engine
    INTERFACE_LINK_LIBRARIES "engine_interface"
)
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    FILES ${PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/engine/
    COMPONENT ${PROJECT_NAME}_HEADERS
)
configure_package_config_file(
    cmake/${PROJECT_NAME}Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    PATH_VARS
        CMAKE_INSTALL_INCLUDEDIR
        CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${${PROJECT_NAME}_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
    EXPORT ${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    FILE ${PROJECT_NAME}Targets.cmake)
#    NAMESPACE CE::)
