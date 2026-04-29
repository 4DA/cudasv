cmake_minimum_required(VERSION 3.16)

if (POLICY CMP0092)
    cmake_policy(SET CMP0092 NEW)
endif()

project(sv_app VERSION "1.0.0" LANGUAGES C CXX)

list(APPEND CMAKE_FIND_ROOT_PATH ${CMAKE_INSTALL_PREFIX})
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../rf/cmake")

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT TARGET spdlog::spdlog)
  find_package(spdlog REQUIRED CONFIG)
endif()

if(NOT TARGET sv_engine)
  find_package(sv_engine REQUIRED)
endif()

find_package(OpenGLES3 REQUIRED)
find_package(Threads REQUIRED)
find_package(EGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED CONFIG)


add_executable(sv_app
  src/main.cpp
  src/app/cmdline.cpp
  src/app/test_scenario.cpp
  src/app/glfw_host.cpp
  src/app/demo_vehicle_signal_provider.cpp
  src/app/nuscenes_inspector.cpp
  src/app/app_runtime.cpp
  src/compat/render_bridge_4cam_contract.cpp
  src/compat/runtime_source_bridge_4cam.cpp
  src/config/config.cpp
  src/config/canonical_rig.cpp
  src/config/runtime_calibration_bridge_4cam.cpp
  src/sources/file_sequence_source.cpp
  src/sources/nuscenes_source.cpp
  src/sources/render_bridge_4cam.cpp
  src/sources/rigged_png_source_support.cpp
  src/sources/source_validation.cpp
  src/sources/source_factory.cpp
  src/conversion_utils.cpp
  )

target_compile_options(sv_app
  PRIVATE
  -Wall -Wno-unused-variable -Wno-unused-function -Wno-sequence-point
  )

target_include_directories(sv_app
  PRIVATE
  src
  src/config
  ${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/json/include
  ${GLM_INCLUDE_DIRS}
  )

target_link_libraries(sv_app
  PRIVATE
  spdlog::spdlog
  glfw
  Threads::Threads
  sv_engine
  EGL::EGL
  OpenGLES3::OpenGLES3
)

install(
  TARGETS sv_app
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
