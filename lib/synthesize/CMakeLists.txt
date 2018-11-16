project("synthesizer")
cmake_minimum_required(VERSION 2.8)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_definitions(-std=c++11)
add_definitions(-Wall)
add_definitions(-fPIC)
#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# supply compile-time directory to code
set(compileDirectory \"${PROJECT_SOURCE_DIR}\")
add_definitions(-DCOMPILE_DIR=${compileDirectory})

option(BUILD_POLY3 "build for poly3 camera model" on)
if(BUILD_POLY3)
  add_definitions(-DBUILD_POLY3)
endif()

find_package(CUDA REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

set(BOOST_INCLUDE_DIRS "/usr/local/include")
set(BOOST_LIBRARY_DIRS "/usr/local/lib/boost")
set(PYTHON_INCLUDE_DIRS "/usr/include/python2.7")

include_directories(${Pangolin_INCLUDE_DIRS}
                    ${EIGEN3_INCLUDE_DIR}
                    ${PROJECT_SOURCE_DIR}/include
                    ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
                    ${PYTHON_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${BOOST_INCLUDE_DIRS})

link_directories(${Pangolin_LIBRARY_DIRS}
                 ${BOOST_LIBRARY_DIRS})

link_libraries(${Pangolin_LIBRARIES}
               ${OpenCV_LIBS}
               boost_python
               boost_numpy
               assimp
               util)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11;--expt-relaxed-constexpr;-O3;-arch=sm_61;--expt-extended-lambda;--verbose;")

cuda_add_library(
  synthesizer
  SHARED
  src/backprojection.cu
  src/synthesize.cpp
  src/thread_rand.cpp
  src/backprojection.cpp
  src/camera.cpp
  src/cameraFactory.cpp
  src/poly3.cpp
)
