cmake_minimum_required(VERSION 3.6)

project(InductiveBoost)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(GNUInstallDirs)

if (CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "The C++ compiler verion is ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# autodiff requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(autodiff)
find_package(Eigen3 REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenMP)
find_package(BLAS REQUIRED)

include_directories(${ARMADILLO_INCLUDE_DIRS})

add_library(dataset OBJECT dataset.cpp)
add_library(pmlb OBJECT pmlb.cpp)
add_library(processor OBJECT processor.cpp)
add_library(decision_tree OBJECT decision_tree.cpp)
add_library(loss OBJECT loss.cpp)
target_link_libraries(loss PUBLIC autodiff::autodiff Eigen3::Eigen ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
add_library(LTSS OBJECT LTSS.cpp)
add_library(DP OBJECT DP.cpp)
target_link_libraries(DP PUBLIC LTSS Eigen3::Eigen)
add_library(gradientboostclassifier OBJECT gradientboostclassifier.cpp)
target_link_libraries(gradientboostclassifier PUBLIC autodiff::autodiff Eigen3::Eigen {ARMADILLO_LIBRARIES} DP LTSS)
			  
add_executable(pmlb_driver pmlb_driver.cpp)
target_link_libraries(pmlb_driver PUBLIC autodiff::autodiff Eigen3::Eigen ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
