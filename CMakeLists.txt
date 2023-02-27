cmake_minimum_required(VERSION 3.6)

project(InductiveBoost)

set(CMAKE_VERBOSE_MAKEFILE ON)

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

# arma ex not required
option(BUILD_ARMA_EX "Build armadillo ex" OFF)

# ClassifierContext::Context creation
option(BUILD_CONTEXT_BUILDER "Build context builder" ON)

# DP_solver ex not required
option(BUILD_DP_SOLVER "Build DP solver ex" OFF)

# gtests not required, by default will build
option(BUILD_TESTS "Build unit tests" ON)

# benchmarks not required, by default will build
option(BUILD_BENCHMARKS "Build benchmark tests" ON)

# build Inductive Boost Classifier
option(BUILD_IBC "Build inductive boost classifier" ON)

# build incremental driver
option(BUILD_INC_DRIVER "Build incrmental fitter" ON)

# build incremental predictor
option(BUILD_INC_PREDICT "Build incrmental predictor" ON)

# build stepwise predictor
option(BUILD_STEPWISE_PREDICT "Build stepwise predictor" ON)

# build titanic example
option(BUILD_TITANIC "Build titanic ex" OFF)

# build multiclass pmlb driver
option(BUILD_MULTICLASS "Build multiclass inductive boost classifier" ON)

# Profile
option(PROFILE_RESULTS "gprof enabled" OFF)

# OpenMP not required
option(USE_OPENMP "Use OpenMP" OFF)

# autodiff not required
option(USE_AUTODIFF "Use autodiff for gradient, hessian calculations" OFF)

# Eigen not reuqired
option(USE_EIGEN "Use eigen, only needed for finding optimal t in DP calculation" OFF)

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

if (BUILD_CONTEXT_BUILDER OR BUILD_INC_DRIVER OR BUILD_INC_PREDICT)
  find_package(Boost COMPONENTS system program_options REQUIRED)	
  # find_package(Boost COMPONENTS system REQUIRED)
  include_directories(${Boost_INCLUDE_DIR})
  link_directories(${Boost_LIBRARY_DIR})
endif()

if (BUILD_TESTS)
  find_package(GTest REQUIRED)
endif()

if(BUILD_BENCHMARKS)
  find_package(benchmark REQUIRED)
endif()

if (USE_AUTODIFF)
  find_package(autodiff)
endif()

if (USE_EIGEN)
  find_package(Eigen3 REQUIRED)
  include_directories(/usr/local/include/eigen3)    
endif()

include_directories(${ARMADILLO_INCLUDE_DIRS})

add_library(loss OBJECT loss.cpp)
add_library(score2 OBJECT score2.cpp)
add_library(utils OBJECT utils.cpp)
add_library(replay OBJECT replay.cpp)

if (USE_OPENMP)
  target_link_libraries(loss PUBLIC autodiff::autodiff ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
else()
  target_link_libraries(loss PUBLIC ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
endif()

add_library(LTSS OBJECT LTSS.cpp)
target_link_libraries(LTSS PUBLIC utils)
add_library(DP OBJECT DP.cpp)
target_link_libraries(DP PUBLIC LTSS utils)
add_library(threadsafequeue OBJECT threadsafequeue.cpp)
target_link_libraries(threadsafequeue PUBLIC pthread)
add_library(gradientboostclassifier OBJECT gradientboostclassifier.cpp)
target_link_libraries(gradientboostclassifier PUBLIC utils LTSS DP ${ARMADILLO_LIBRARIES})
add_library(gradientboostmulticlassifier OBJECT gradientboostmulticlassifier.cpp)
target_link_libraries(gradientboostmulticlassifier PUBLIC threadsafequeue ${ARMADILLO_LIBRARIES} pthread)

# Armadillo example
if (BUILD_ARMA_EX)
  add_executable(arma_ex arma_ex.cpp)
  target_link_libraries(arma_ex ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
endif()

# Context creation
if (BUILD_CONTEXT_BUILDER)
  add_executable(createContext createContext.cpp)
  target_link_libraries(createContext LINK_PUBLIC ${Boost_LIBRARIES} utils)
endif()

# DP solver example	
if (BUILD_DP_SOLVER_EX)
  add_executable(DP_solver_ex DP_solver_ex.cpp)
  target_link_libraries(DP_solver_ex DP LTSS pthread)
endif()
		  
# InductiveBoostClassifier driver on pmlb data
if (BUILD_IBC)
  add_executable(pmlb_driver pmlb_driver.cpp)
  if (USE_OPENMP)
    target_link_libraries(pmlb_driver loss utils gradientboostclassifier autodiff::autodiff ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
  else()
    target_link_libraries(pmlb_driver loss utils gradientboostclassifier ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
endif()

# Incremental driver for sequential, serialized runs
if (BUILD_INC_DRIVER)
  add_executable(incremental_driver incremental_driver.cpp)
  if (USE_OPENMP)
    target_link_libraries(incremental_driver LINK_PUBLIC ${Boost_LIBRARIES} ${Boost_SYSTEM_LIBRARY} loss utils gradientboostclassifier autodiff::autodiff ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
  else()
    target_link_libraries(incremental_driver LINK_PUBLIC ${Boost_LIBRARIES} ${Boost_SYSTEM_LIBRARY} loss utils gradientboostclassifier ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
endif()

# Incremental predictor for sequential, serialized runs
if (BUILD_INC_PREDICT)
  add_executable(incremental_predict incremental_predict.cpp)
  if (USE_OPENMP)
    target_link_libraries(incremental_predict LINK_PUBLIC ${Boost_LIBRARIES} loss utils gradientboostclassifier autodiff::autodiff ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
  else()
    target_link_libraries(incremental_predict LINK_PUBLIC ${Boost_LIBRARIES} loss utils gradientboostclassifier ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
endif()

# Stepwise predictor for sequential, serialized runs
if (BUILD_STEPWISE_PREDICT)
  add_executable(stepwise_predict stepwise_predict.cpp)
  if (USE_OPENMP)
    target_link_libraries(stepwise_predict LINK_PUBLIC ${Boost_LIBRARIES} loss utils gradientboostclassifier autodiff::autodiff ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
  else()
    target_link_libraries(stepwise_predict LINK_PUBLIC ${Boost_LIBRARIES} loss utils gradientboostclassifier ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
endif()

# Titanic dataset driver on pmlb data
if (BUILD_TITANIC)
  add_executable(titanic_driver titanic_driver.cpp)
  if (USE_OPENMP)
    target_link_libraries(titanic_driver LINK_PUBLIC ${Boost_LIBRARIES} ${Boost_SYSTEM_LIBRARY} loss utils gradientboostclassifier autodiff::autodiff ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
  else()
    target_link_libraries(titanic_driver loss utils gradientboostclassifier ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
endif()

# Multiclass classification driver on pmlb data
if (BUILD_MULTICLASS)
  add_executable(multiple_class_pmlb_driver multiple_class_pmlb_driver.cpp)
  if (USE_OPENMP)
    target_link_libraries(multiple_class_pmlb_driver loss utils gradientboostclassifier gradientboostmulticlassifier autodiff::autodiff ${ARMADILLO_LIBRARIES} "${OpenMP_CXX_FLAGS}" ${BLAS_LIBRARIES})
  else()
    target_link_libraries(multiple_class_pmlb_driver loss utils gradientboostclassifier gradientboostmulticlassifier ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})
  endif()
endif()

if (BUILD_TESTS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_executable(gtest_all gtest_all.cpp)
  target_link_libraries(gtest_all LINK_PUBLIC ${Boost_SYSTEM_LIBRARY} utils replay DP ${GTEST_LIBRARIES} ${ARMADILLO_LIBRARIES} pthread)
endif()

if (BUILD_BENCHMARKS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
  add_executable(benchmarks benchmarks.cpp)
  target_link_libraries(benchmarks benchmark::benchmark pthread)
endif()

# arma ex not required
MESSAGE(STATUS "Build armadillo ex: " ${BUILD_ARMA_EX})
MESSAGE(STATUS "Build context builder: " ${BUILD_CONTEXT_BUILDER})
MESSAGE(STATUS "Build DP solver ex: " ${BUILD_DP_SOLVER})
MESSAGE(STATUS "Build google tests: " ${BUILD_TESTS})
MESSAGE(STATUS "Build google benchmarks: " ${BUILD_BENCHMARKS})
MESSAGE(STATUS "Build inductive boost classifier: " ${BUILD_IBC})
MESSAGE(STATUS "Build incremental fitter: " ${BUILD_INC_DRIVER})
MESSAGE(STATUS "Build incremental predictor: " ${BUILD_INC_PREDICT})
MESSAGE(STATUS "Build stepwise predictor: " ${BUILD_STEPWISE_PREDICT})
MESSAGE(STATUS "Build titanic ex: " ${BUILD_TITANIC})
MESSAGE(STATUS "Build multiclass inductive boost classifier: " ${BUILD_MULTICLASS})
MESSAGE(STATUS "Gprof enabled: " ${PROFILE_RESULTS})
MESSAGE(STATUS "Use OpenMP: " ${USE_OPENMP})
MESSAGE(STATUS "Use autodiff lib: " ${USE_AUTODIFF})
MESSAGE(STATUS "Use eigen lib: " ${USE_EIGEN})

