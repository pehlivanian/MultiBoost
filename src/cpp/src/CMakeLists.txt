cmake_minimum_required(VERSION 3.24)

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

# arma ex not required
option(BUILD_ARMA_EX "Build armadillo ex" ON)

# ClassifierContext::Context creation
option(BUILD_CONTEXT_BUILDER "Build context builder" ON)

# DP_solver ex not required
option(BUILD_DP_SOLVER_EX "Build DP solver ex" ON)

# gtests not required, by default will build
option(BUILD_TESTS "Build unit tests" ON)

# benchmarks not required, by default will build
option(BUILD_BENCHMARKS "Build benchmark tests" ON)

# build Inductive Boost Classifier
option(BUILD_SINGLECLASS "Build inductive boost classifier" ON)

# build Inductive Boost Regressor
option(BUILD_REGRESSOR "Build inductive boost regressor" ON)

# build incremental classifier
option(BUILD_INC_CLASSIFY "Build incrmental classifier" ON)

# build incremental predictor
option(BUILD_INC_PREDICT "Build incrmental predictor" ON)

# build stepwise classifier
option(BUILD_STEPWISE_CLASSIFY "Build stepwise classifier" ON)

# build OOS predictor
option(BUILD_OOS_PREDICTOR "Build OOS predictor" ON)

# build OOS classifier
option(BUILD_OOS_CLASSIFIER "Build OOS classifier" ON)

# build stepwise predictor
option(BUILD_STEPWISE_PREDICT "Build stepwise predictor" ON)

# build stepwise classifier for replay
option(BUILD_REPLAY_CLASSIFY_STEPWISE "Build replay stepwise classifier" ON)

# build stepwise regressor for replay
option(BUILD_REPLAY_PREDICT_STEPWISE "Build replay stepwise regressor" ON)

# build titanic example
option(BUILD_TITANIC "Build titanic ex" OFF)

# build multiclass pmlb driver
option(BUILD_MULTICLASS "Build multiclass inductive boost classifier" OFF)

# Profile
option(PROFILE_RESULTS "gprof enabled" OFF)

# Google profile
option(GPROFILE_RESULTS "google-prof enabled" OFF)

# OpenMP not required
option(USE_OPENMP "Use OpenMP" OFF)

# autodiff not required
option(USE_AUTODIFF "Use autodiff for gradient, hessian calculations" OFF)

# Eigen not required
option(USE_EIGEN "Use eigen, only needed for finding optimal t in DP calculation" OFF)

# Math optimizations
option(MATH_OPTS "Math optimizations" ON)

if (MATH_OPTS)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-math-errno")
endif()

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

# find_package(Filesystmem)
if (NOT Filesystem_FOUND)
  find_package(Boost COMPONENTS filesystem)
endif()

find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)

if (USE_OPENMP)
  find_package(OpenMP)
endif()

if (BUILD_CONTEXT_BUILDER OR BUILD_INC_CLASSIFY OR BUILD_INC_PREDICT)
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

include_directories("../include")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

################
# OBJECT FILES #
################
# object files
add_library(DP OBJECT DP.cpp)
add_library(analyticutils OBJECT analytic_utils.cpp)
add_library(utils OBJECT utils.cpp)
add_library(loss OBJECT loss.cpp)
add_library(pmlb OBJECT pmlb.cpp)
add_library(gradientboostclassifier OBJECT gradientboostclassifier.cpp)
add_library(gradientboostregressor OBJECT gradientboostregressor.cpp)
add_library(replay OBJECT replay.cpp)
add_library(score2 OBJECT score2.cpp)
add_library(classifier OBJECT classifier.cpp)
add_library(regressor OBJECT regressor.cpp)
add_library(compositeclassifier OBJECT compositeclassifier.cpp)
add_library(threadsafequeue OBJECT threadsafequeue.cpp)
add_library(contextmanager OBJECT contextmanager.cpp)
# add_library(negativefeedback OBJECT negativefeedback.cpp)
# add_library(gradientboostmulticlassifier OBJECT gradientboostmulticlassifier.cpp)

###########
# LINKAGE #
###########
target_link_libraries(DP PUBLIC utils score2
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)

target_link_libraries(loss PUBLIC $<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> 
        analyticutils
	${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES}
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)

target_link_libraries(replay PUBLIC utils threadsafequeue gradientboostclassifier
	${ARMADILLO_LIBRARIES}
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)

if (Filesystem_FOUND)
  target_compile_definitions(utils PUBLIC -DCXX_FILESYSTEM_HEADER=<filesystem.hpp>)
  target_compile_definitions(utils PUBLIC -DCXX_FILESYSTEM_NAMESPACE=std::filesystem)
  target_link_libraries(utils PUBLIC analyticutils loss std::filesystem 
	  $<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
elseif(Boost_FOUND)
  target_compile_definitions(utils PUBLIC -DCXX_FILESYSTEM_HEADER=<boost/filesystem.hpp>)
  target_compile_definitions(utils PUBLIC -DCXX_FILESYSTEM_NAMESPACE=boost::filesystem)
  target_link_libraries(utils PUBLIC loss Boost::filesystem
      	  $<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

target_link_libraries(compositeclassifier PUBLIC utils score2 classifier DP contextmanager
	${ARMADILLO_LIBRARIES}
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES}
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)	

target_link_libraries(classifier PUBLIC ${ARMADILLO_LIBRARIES}
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES}
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)	

target_link_libraries(regressor PUBLIC ${ARMADILLO_LIBRARIES}
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES}
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)	

target_link_libraries(threadsafequeue PUBLIC pthread 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)

target_link_libraries(gradientboostclassifier PUBLIC utils loss score2 DP classifier compositeclassifier
	${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)

# target_link_libraries(gradientboostmulticlassifier PUBLIC threadsafequeue 
# 	${ARMADILLO_LIBRARIES} pthread 
# 	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)


###############
# EXECUTABLES #
###############
# Armadillo example
if (BUILD_ARMA_EX)
  add_executable(arma_ex arma_ex.cpp)
  target_link_libraries(arma_ex ${ARMADILLO_LIBRARIES} 
	${BLAS_LIBRARIES} 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# Context creation
if (BUILD_CONTEXT_BUILDER)
  add_executable(createContext createContext.cpp)
  target_link_libraries(createContext LINK_PUBLIC ${Boost_LIBRARIES} utils loss
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# DP solver example	
if (BUILD_DP_SOLVER_EX)
  add_executable(DP_solver_ex DP_solver_ex.cpp)
  target_link_libraries(DP_solver_ex DP score2 pthread 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()
		  
# InductiveBoostClassifier driver on pmlb data
if (BUILD_SINGLECLASS)
  add_executable(single_class_pmlb_driver single_class_pmlb_driver.cpp)
  target_link_libraries(single_class_pmlb_driver loss utils
	gradientboostclassifier
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES} 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# InductiveBoostRegressor driver on pmlb data
if (BUILD_REGRESSOR)
  add_Executable(regressor_pmlb_driver regressor_pmlb_driver.cpp)
  target_link_libraries(regressor_pmlb_driver utils loss gradientboostregressor
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES}
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES} 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)	
endif()

# Incremental driver for sequential, serialized runs
if (BUILD_INC_CLASSIFY)
  add_executable(incremental_classify incremental_classify.cpp)
  target_link_libraries(incremental_classify LINK_PUBLIC ${Boost_LIBRARIES} 
	${Boost_SYSTEM_LIBRARY} utils 
	gradientboostclassifier 
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES} 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# Incremental predictor for sequential, serialized runs
if (BUILD_INC_PREDICT)
  add_executable(incremental_predict incremental_predict.cpp)
  target_link_libraries(incremental_predict LINK_PUBLIC 
	${Boost_LIBRARIES} utils 
	gradientboostclassifier 
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES} 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# Stepwise classifier for sequential, serialized runs
if (BUILD_STEPWISE_CLASSIFY)
  add_executable(stepwise_classify stepwise_classify.cpp)
  target_link_libraries(stepwise_classify LINK_PUBLIC ${Boost_LIBRARIES} loss utils 
	gradientboostclassifier replay
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> 
	${BLAS_LIBRARIES} $<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# Stepwise classifier for replay 
if (BUILD_REPLAY_CLASSIFY_STEPWISE)
  add_executable(replay_classify_stepwise replay_classify_stepwise.cpp)
  target_link_libraries(replay_classify_stepwise LINK_PUBLIC ${Boost_LIBRARIES} utils
	gradientboostclassifier replay
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> 
	${BLAS_LIBRARIES} $<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# Stepwise regressor for replay 
if (BUILD_REPLAY_PREDICT_STEPWISE)
  add_executable(replay_predict_stepwise replay_predict_stepwise.cpp)
  target_link_libraries(replay_predict_stepwise LINK_PUBLIC ${Boost_LIBRARIES} utils
	gradientboostregressor replay
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> 
	${BLAS_LIBRARIES} $<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# Stepwise predictor for sequential, serialized runs
if (BUILD_STEPWISE_PREDICT)
  add_executable(stepwise_predict stepwise_predict.cpp)
  target_link_libraries(stepwise_predict LINK_PUBLIC ${Boost_LIBRARIES} loss utils 
	gradientboostregressor replay
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> 
	${BLAS_LIBRARIES} $<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# OOS predictor for sequential, serialized runs
if (BUILD_OOS_PREDICTOR)
  add_executable(OOS_predict OOS_predict.cpp)
  target_link_libraries(OOS_predict LINK_PUBLIC ${Boost_LIBRARIES} loss utils 
	gradientboostregressor replay
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> 
	${BLAS_LIBRARIES} $<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# OOS classifier for sequential, serialized runs
if (BUILD_OOS_CLASSIFIER)
  add_executable(OOS_classify OOS_classify.cpp)
  target_link_libraries(OOS_classify LINK_PUBLIC ${Boost_LIBRARIES} loss utils 
	gradientboostclassifier replay
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> 
	${BLAS_LIBRARIES} $<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# Titanic dataset driver on pmlb data
if (BUILD_TITANIC)
  add_executable(titanic_driver titanic_driver.cpp)
  target_link_libraries(titanic_driver LINK_PUBLIC ${Boost_LIBRARIES} 
	${Boost_SYSTEM_LIBRARY} pmlb loss utils 
	gradientboostclassifier replay
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES} 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# Multiclass classification driver on pmlb data
if (BUILD_MULTICLASS)
  add_executable(multiple_class_pmlb_driver multiple_class_pmlb_driver.cpp)
  target_link_libraries(multiple_class_pmlb_driver pmlb loss utils gradientboostclassifier 
	gradientboostmulticlassifier 
	$<IF:$<BOOL:${USE_AUTODIFF}>,autodiff::autodiff,> ${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${USE_OPENMP}>,"${OpenMP_CXX_FLAGS}",> ${BLAS_LIBRARIES} 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

if (BUILD_TESTS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
  add_executable(gtest_all gtest_all.cpp)
  target_link_libraries(gtest_all LINK_PUBLIC ${Boost_SYSTEM_LIBRARY} utils replay DP 
	score2 gradientboostclassifier ${GTEST_LIBRARIES} 
	${ARMADILLO_LIBRARIES} pthread 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

if (BUILD_BENCHMARKS)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
  add_executable(benchmarks benchmarks.cpp)
  target_link_libraries(benchmarks benchmark::benchmark pthread utils
	score2 gradientboostclassifier 
	${ARMADILLO_LIBRARIES} 
	$<IF:$<BOOL:${GPROFILE_RESULTS}>,profiler,>)
endif()

# arma ex not required
MESSAGE(STATUS "Build armadillo ex:                          " ${BUILD_ARMA_EX})
MESSAGE(STATUS "Build context builder:                       " ${BUILD_CONTEXT_BUILDER})
MESSAGE(STATUS "Build DP solver ex:                          " ${BUILD_DP_SOLVER_EX})
MESSAGE(STATUS "Build google tests:                          " ${BUILD_TESTS})
MESSAGE(STATUS "Build google benchmarks:                     " ${BUILD_BENCHMARKS})
MESSAGE(STATUS "Build inductive boost classifier:            " ${BUILD_SINGLECLASS})
MESSAGE(STATUS "Build inductive boost regressor:             " ${BUILD_REGRESSOR})
MESSAGE(STATUS "Build incremental classifier:                " ${BUILD_INC_CLASSIFY})
MESSAGE(STATUS "Build incremental predictor:                 " ${BUILD_INC_PREDICT})
MESSAGE(STATUS "Build stepwise classifier:                   " ${BUILD_STEPWISE_CLASSIFY})
MESSAGE(STATUS "Build stepwise predictor:                    " ${BUILD_STEPWISE_PREDICT})
MESSAGE(STATUS "Build replay stepwise classifier:            " ${BUILD_REPLAY_CLASSIFY_STEPWISE})
MESSAGE(STATUS "Build replay stepwise regressor:             " ${BUILD_REPLAY_PREDICT_STEPWISE})
MESSAGE(STATUS "Build OOS classifier:                        " ${BUILD_OOS_CLASSIFIER})
MESSAGE(STATUS "Build OOS predictor:                         " ${BUILD_OOS_PREDICTOR})
MESSAGE(STATUS "Build titanic ex:                            " ${BUILD_TITANIC})
MESSAGE(STATUS "Build multiclass inductive boost classifier: " ${BUILD_MULTICLASS})
MESSAGE(STATUS "Math optimizations:                          " ${MATH_OPTS})
MESSAGE(STATUS "Gprof enabled:                               " ${PROFILE_RESULTS})
MESSAGE(STATUS "Google-pprof enabled:                        " ${GPROFILE_RESULTS})
MESSAGE(STATUS "Use OpenMP:                                  " ${USE_OPENMP})
MESSAGE(STATUS "Use autodiff lib:                            " ${USE_AUTODIFF})
MESSAGE(STATUS "Use eigen lib:                               " ${USE_EIGEN})

