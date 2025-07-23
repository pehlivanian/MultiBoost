cmake_minimum_required(VERSION 3.16)

project(InductiveBoost)

# Supress all warnings - to eliminate clutter when debugging compile errors
# add_definitions(-w)

set(CMAKE_VERBOSE_MAKEFILE OFF)

include(GNUInstallDirs)

if (CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "The C++ compiler verion is ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_CXX_FLAGS "-mavx2 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Requires C++20 for ThreadPool optimizations
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define project root directory
set(IB_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
message(STATUS "Project root directory: ${IB_PROJECT_ROOT}")

# Add definition for preprocessor
add_definitions(-DIB_PROJECT_ROOT="${IB_PROJECT_ROOT}")

# Define test data directory - this is also used in C++ code
set(IB_TEST_DATA_DIR "${IB_PROJECT_ROOT}/test_data")
add_definitions(-DIB_TEST_DATA_DIR="${IB_TEST_DATA_DIR}")

# Make test data available to the build
message(STATUS "Test data directory: ${IB_TEST_DATA_DIR}")

# Create digest directories needed at runtime
file(MAKE_DIRECTORY ${IB_PROJECT_ROOT}/digest/classify)
file(MAKE_DIRECTORY ${IB_PROJECT_ROOT}/digest/regress)
message(STATUS "Created digest directories for models")

# Custom target to ensure digest directories exist at build time
add_custom_target(create_digest_dirs ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${IB_PROJECT_ROOT}/digest/classify"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${IB_PROJECT_ROOT}/digest/regress"
    COMMENT "Ensuring digest directories exist")

# Install test data as part of the package
install(DIRECTORY ${IB_PROJECT_ROOT}/test_data/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/${CMAKE_PROJECT_NAME}/data
        FILES_MATCHING PATTERN "*.csv")

# Create and install digest directories
install(DIRECTORY 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/${CMAKE_PROJECT_NAME}/digest/classify)
install(DIRECTORY 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/${CMAKE_PROJECT_NAME}/digest/regress)

add_subdirectory(src/cpp/src)
