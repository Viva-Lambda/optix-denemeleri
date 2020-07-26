if (POLICY CMP0048) # The project() command manages VERSION variables.
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

cmake_minimum_required(VERSION 3.0)

project("OptixDenemeleri")
set(CMAKE_CXX_FLAGS "--std=c++17")


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/")

set(optix_LIBRARY "")
# optix cuda related
include("cmake/configure_optix.cmake")


# needed for creating a window
set( GLFW_SHARED_LIB
    "${PROJECT_SOURCE_DIR}/libs/glfw/src/libglfw.so.3.4")


# needed for loading models
set (ASSIMP_SHARED_LIB5
    "${PROJECT_SOURCE_DIR}/libs/assimp/libassimp.so.4")


set( ASSIMP_LIBS ${ASSIMP_SHARED_LIB5})


# general compiler flags for clang
set (FLAGS "-ldl -ggdb -Wall -Wextra -pedantic -Wextra-tokens")


# per executable configurations
add_subdirectory("${PROJECT_SOURCE_DIR}/src/ex01/")
