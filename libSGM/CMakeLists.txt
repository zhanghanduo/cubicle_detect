cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")

option(LIBSGM_SHARED        "Build a shared library" ON)
option(BUILD_OPENCV_WRAPPER "Make library compatible with cv::Mat and cv::cuda::GpuMat of OpenCV" ON)

project(libSGM VERSION 2.4.0)

if(BUILD_OPENCV_WRAPPER)
	find_package(OpenCV 3.4.0 REQUIRED core)
	include_directories(${OpenCV_INCLUDE_DIRS})
endif()

configure_file(${CMAKE_SOURCE_DIR}/include/libsgm_config.h.in
               ${CMAKE_SOURCE_DIR}/include/libsgm_config.h
)

add_subdirectory(src)
