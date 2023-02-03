cmake_minimum_required(VERSION 3.6)

project(InductiveBoost)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(GNUInstallDirs)

if (CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "The C++ compiler verion is ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_CXX_FLAGS "-mavx -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# autodiff not required
SET(USE_AUTODIFF OFF)

# gtests not required
SET(BUILD_TESTS OFF)

# benchmarks not required
SET(BUILD_BENCHMARKS OFF)

# OpenMP not required
SET(USE_OPENMP OFF)

# Eigen not reuqired
SET(USE_EIGEN OFF)

# gprof
if (PROFILE_RESULTS)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
  SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
endif()

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)

if (USE_OPENMP)
  find_package(OpenMP)
endif()

# if (BUILD_TESTS)
find_package(GTest REQUIRED)
# endif()

# if(BUILD_BENCHMARKS)
find_package(benchmark REQUIRED)
# endif()

if (USE_AUTODIFF)
  find_package(autodiff)
endif()

if (USE_EIGEN)
  find_package(Eigen3 REQUIRED)
  include_directories(/usr/local/include/eigen3)    
endif()

include_directories(${ARMADILLO_INCLUDE_DIRS})

add_library(pmlb OBJECT pmlb.cpp)
add_library(processor OBJECT processor.cpp)
add_library(loss OBJECT loss.cpp)
add_library(score2 OBJECT score2.cpp)
add_library(utils OBJECT utils.cpp)

if (USE_OPENMP)
  target_link_libraries(loss PUBLIC autodiff::autodiff ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
else()
  target_link_libraries(loss PUBLIC ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
endif()

add_library(LTSS OBJECT LTSS.cpp)
target_link_libraries(LTSS PUBLIC utils)
add_library(DP OBJECT DP.cpp)
target_link_libraries(DP PUBLIC LTSS utils)
add_library(gradientboostclassifier OBJECT gradientboostclassifier.cpp)
target_link_libraries(gradientboostclassifier PUBLIC LTSS DP ${ARMADILLO_LIBRARIES})

# DP solver example	
add_executable(DP_solver_ex DP_solver_ex.cpp)
target_link_libraries(DP_solver_ex DP LTSS pthread)
		  
# InductiveBoostClassifier driver on pmlb data
add_executable(pmlb_driver pmlb_driver.cpp)
if (USE_OPENMP)
  target_link_libraries(pmlb_driver pmlb processor loss utils gradientboostclassifier autodiff::autodiff ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
else()
  target_link_libraries(pmlb_driver pmlb processor loss utils gradientboostclassifier ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
endif()
# if (BUILD_TESTS)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
add_executable(gtest_all gtest_all.cpp)
target_link_libraries(gtest_all utils DP ${GTEST_LIBRARIES} pthread)
# endif()

# if (BUILD_BENCHMARKS)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks benchmark::benchmark pthread)
# endif()
