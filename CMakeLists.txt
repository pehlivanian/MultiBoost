cmake_minimum_required(VERSION 3.16)

project(InductiveBoost)

# Supress all warnings - to eliminate clutter when debugging compile errors
# add_definitions(-w)

# set(CMAKE_VERBOSE_MAKEFILE ON)

include(GNUInstallDirs)

if (CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "The C++ compiler verion is ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_CXX_FLAGS "-mavx2 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(src/cpp/src)
