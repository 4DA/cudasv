cmake_minimum_required(VERSION 3.16)

project(rf VERSION 0.9.0 LANGUAGES C CXX CUDA)

set(LIBRARY_NAME "rf")

# Namespace folder to put public interface in install include directory
set(${PROJECT_NAME}_NAMESPACE "rf")

string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" ${PROJECT_NAME}_BUILD ${${PROJECT_NAME}_VERSION})
string(REGEX MATCH "^[0-9]+\\.[0-9]+" ${PROJECT_NAME}_API ${${PROJECT_NAME}_BUILD})

message(STATUS "${PROJECT_NAME}  Version: ${${PROJECT_NAME}_VERSION}")
message(STATUS "${PROJECT_NAME}  Build: ${${PROJECT_NAME}_BUILD}")
message(STATUS "${PROJECT_NAME}  API: ${${PROJECT_NAME}_API}")

set(${PROJECT_NAME}_SOVERSION ${${PROJECT_NAME}_API})

option(WITH_ASAN "Enable address sanitizer" OFF)
option(WITH_PTX "Enable separate PTX file generation" OFF)


include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Add current build output to root search path, workaround for
# cmake issue: https://gitlab.kitware.com/cmake/cmake/issues/16870
# if fixed shall be removed and CMAKE_STAGING_PREFIX shall be used
# on cross-compilation targets
list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_INSTALL_PREFIX})


find_package(EGL REQUIRED)
find_package(OpenGLES3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(OPENGLES_INCLUDE_DIRS ${OPENGLES3_INCLUDE_DIRS})
set(OPENGLES_LIBRARIES ${OPENGLES3_LIBRARIES})

if(NOT TARGET spdlog::spdlog)
    find_package(spdlog REQUIRED CONFIG)
endif()
find_package(glfw3 REQUIRED)

#------------------------------------------------------------------------------
# Mandatory includes and libraries
#------------------------------------------------------------------------------

set(
    ${PROJECT_NAME}_LIBS
    ${OPENGLES_LIBRARIES}
    ${EGL_LIBRARIES}
    glfw
    )

set(
    ${PROJECT_NAME}_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/glcommon.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/glm_common.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/material.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/math.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/primitive_component.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/point_light_component.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/bounding_box.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/scene_component.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/virtual_camera.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/trs_transform.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/texture.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/scene.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/renderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/gltf_loader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/surround_view/projector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/ibl.hpp

    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/camera_control/rotator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/camera_control/camera_control.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/camera_control/viewpoint_animation.hpp
    )

set(
    ${PROJECT_NAME}_CUDARF_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/cudarf/cudarf.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/cudarf/cudarf_profile.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/cudarf/material.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/cudarf/draw_list_renderer.hpp
    )

set(
    ${PROJECT_NAME}_SOURCES
    ${${PROJECT_NAME}_PUBLIC_HEADERS}
    ${${PROJECT_NAME}_CUDARF_PUBLIC_HEADERS}

    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/surround_view/projector.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/draw_list_renderer.cpp


    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/gltf_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/trs_transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/scene_component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/primitive_component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/point_light_component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/bounding_box.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/material.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/virtual_camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/animation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/mesh_geometry.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/ibl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/utils.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/cudarf/cudarf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/cudarf/cudarf_pipe.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/cudarf/cudarf_blit.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/cudarf/cudarf_profile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/gltf_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/gltf/accessors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/gltf/animation_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/gltf/material_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/gltf/scene_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/cudarf/output.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/camera_control/camera_control.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/camera_control/viewpoint_animation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/camera_control/rotator.cpp
    )

set(
  ${PROJECT_NAME}_LIBS
  ${${PROJECT_NAME}_LIBS}
  CUDA::cudart
  CUDA::cuda_driver
  )

if (WITH_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -lasan")
endif()

add_library(
    ${LIBRARY_NAME} STATIC
    ${${PROJECT_NAME}_SOURCES}
    ${${PROJECT_NAME}_RESOURCES}
    )

target_compile_options(${LIBRARY_NAME}
    PRIVATE
        -Wall
        -Wextra
        -Wfloat-conversion
        -fPIC
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-fPIC>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20012>
)
target_compile_definitions(${LIBRARY_NAME} PRIVATE
    VERSION_STRING="${PROJECT_VERSION}"
    WITH_BUILTIN_LOGGING
)
target_compile_definitions(${LIBRARY_NAME} PUBLIC
    CUDARF_ENABLE_CUDA_PROFILING=$<BOOL:${WITH_PROFILE_CUDA_TIME}>
)
target_include_directories(${LIBRARY_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        src
        thirdparty
        ${CMAKE_CURRENT_SOURCE_DIR}/include/rf
        ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer
        ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/cudarf
        ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/camera_control
        ${EGL_INCLUDE_DIRS}
        ${GLM_INCLUDE_DIRS}
        ${OPENGLES_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/cudarf/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/
)

target_link_libraries(${LIBRARY_NAME}
        ${${PROJECT_NAME}_LIBS}
        spdlog::spdlog
)

set_target_properties(${LIBRARY_NAME}
    PROPERTIES
    VERSION ${${PROJECT_NAME}_BUILD}
    SOVERSION ${${PROJECT_NAME}_API}
    INTERFACE_LINK_LIBRARIES ""
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

# ------------------------------------------------------------------------------
# Custom target to generate PTX code
# ------------------------------------------------------------------------------
if (WITH_PTX)
  set(cuda_ptx_file "${CMAKE_CURRENT_BINARY_DIR}/cudarf_pipe.ptx")
  set(cuda_ptx_include_dirs
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/rf"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/rf/renderer/cudarf"
      "${CMAKE_CURRENT_SOURCE_DIR}/include/rf/camera_control"
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/cudarf"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty"
      "${EGL_INCLUDE_DIRS}"
      "${GLM_INCLUDE_DIRS}"
      "${OPENGLES_INCLUDE_DIRS}"
  )
  list(REMOVE_DUPLICATES cuda_ptx_include_dirs)
  set(cuda_ptx_include_flags "")
  foreach(include_dir IN LISTS cuda_ptx_include_dirs)
    list(APPEND cuda_ptx_include_flags "-I${include_dir}")
  endforeach()

  add_custom_command(
      OUTPUT "${cuda_ptx_file}"
      COMMAND ${CMAKE_CUDA_COMPILER}
              --std=c++20
              --ptx
              --src-in-ptx
              --lineinfo
              --compiler-options=-fPIC
              -Xcudafe=--diag_suppress=20012
              ${cuda_ptx_include_flags}
              "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/cudarf/cudarf_pipe.cu"
              -o "${cuda_ptx_file}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/renderer/cudarf/cudarf_pipe.cu"
      VERBATIM
  )

  add_custom_target(ptx ALL DEPENDS "${cuda_ptx_file}")
endif()

#------------------------------------------------------------------------------
# Installation
#------------------------------------------------------------------------------
install(
  TARGETS ${LIBRARY_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rf/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rf
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
    EXPORT ${PROJECT_NAME}Targets
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    FILE ${PROJECT_NAME}Targets.cmake
)
