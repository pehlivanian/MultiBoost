cmake_minimum_required(VERSION 3.6)

project(InductiveBoost)

if (CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "The C++ compiler verion is ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 14)	
add_definitions(-DIS_CXX_14)	
message(STATUS "C++14-compliant CXX")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(OpenMP)
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_library(dataset OBJECT dataset.cpp)
add_library(pmlb OBJECT pmlb.cpp)
add_library(processor OBJECT processor.cpp)
add_library(decision_tree OBJECT decision_tree.cpp)
add_library(gradientboostclassifier gradientboostclassifier.cpp)
			  
add_executable(pmlb_driver pmlb_driver.cpp)
target_link_libraries(pmlb_driver PUBLIC ${ARMADILLO_LIBRARiES} "${OpenMP_CXX_FLAGS}")
