cmake_minimum_required(VERSION 3.16)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(engine_interface LANGUAGES C CXX)

set(LIBRARY_NAME "engine_interface")
set(${PROJECT_NAME}_VERSION "1.0.0")

set(PROJECT_NAMESPACE "engine")

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(
    ${PROJECT_NAME}_INSTALL_INC_DIR
    ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAMESPACE}
)

set(
    ${PROJECT_NAME}_HEADERS
    engine/camera_rig.hpp
    engine/camera_config.hpp
    engine/extrinsics.hpp
    engine/intrinsics.hpp
    engine/vehicle_data.hpp
)

add_library(${LIBRARY_NAME} INTERFACE)
target_include_directories(${LIBRARY_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)


set_target_properties(${LIBRARY_NAME} PROPERTIES
    INTERFACE_POSITION_INDEPENDENT_CODE ON
)

install(
    FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/include/engine/camera_rig.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/engine/camera_config.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/engine/extrinsics.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/engine/intrinsics.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/engine/vehicle_data.hpp
    DESTINATION ${${PROJECT_NAME}_INSTALL_INC_DIR}
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

install(TARGETS engine_interface EXPORT ${PROJECT_NAME}Targets)
install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
