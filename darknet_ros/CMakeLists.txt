cmake_minimum_required(VERSION 3.5)
project(cubicle_detect)

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


# Set c++11 cmake flags
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

set(CMAKE_BUILD_TYPE "Release")
#-DEIGEN_USE_MKL_ALL")

enable_language(C)
enable_language(CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../darknet/cmake/Modules/" ${CMAKE_MODULE_PATH})

#######################
# Print option status #
#######################

if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(ColourBold  "${Esc}[1m")
  set(Red         "${Esc}[31m")
  set(Green       "${Esc}[32m")
  set(Yellow      "${Esc}[33m")
  set(Blue        "${Esc}[34m")
  set(Magenta     "${Esc}[35m")
  set(Cyan        "${Esc}[36m")
  set(White       "${Esc}[37m")
  set(BoldRed     "${Esc}[1;31m")
  set(BoldGreen   "${Esc}[1;32m")
  set(BoldYellow  "${Esc}[1;33m")
  set(BoldBlue    "${Esc}[1;34m")
  set(BoldMagenta "${Esc}[1;35m")
  set(BoldCyan    "${Esc}[1;36m")
  set(BoldWhite   "${Esc}[1;37m")
endif()

# Define path of darknet folder here.
#find_path(DARKNET_PATH
#  NAMES "README.md"
#  HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../darknet/")
set(DARKNET_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../darknet/")
message("Darknet path dir = ${BoldGreen} ${DARKNET_PATH} ${ColourReset}")
add_definitions(-DDARKNET_FILE_PATH="${DARKNET_PATH}")

# Find CUDA
FIND_PACKAGE(CUDA)
if (CUDA_FOUND)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Version: ${BoldGreen} ${CUDA_VERSION_STRINGS} ${ColourReset}")
  message(STATUS "CUDA Libararies: ${BoldGreen} ${CUDA_LIBRARIES} ${ColourReset}")

  #Get CUDA compute capability
  set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/../darknet/scripts/cuda_script) # No suffix required
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/../darknet/scripts/check_cuda.cu)
  execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  execute_process(COMMAND ${OUTPUTFILE}
          RESULT_VARIABLE CUDA_RETURN_CODE
          OUTPUT_VARIABLE ARCH)

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    message( "CUDA Architecture: ${ARCH}")
    message( "CUDA Version: ${CUDA_VERSION_STRING}")
    message( "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
#    set(CUDA_NVCC_FLAGS "${ARCH}")
  else()
    message(WARNING ${ARCH})
  endif()

  if ("${CUDA_VERSION}" VERSION_LESS "10.0")
    set(
      CUDA_NVCC_FLAGS
      ${CUDA_NVCC_FLAGS};
      -O3
      -gencode arch=compute_50,code=[sm_50,compute_50]
      -gencode arch=compute_52,code=[sm_52,compute_52]
      -gencode arch=compute_61,code=[sm_61,compute_61]
      -gencode arch=compute_62,code=[sm_62,compute_62]  # GTX 1080 (Ti)
    )
  else()
  # Add SM 75 for CUDA versions >= 10.0
    set(CUDA_COMPUTE_MODEL_75 "75") #Compute capability for Tesla T4, RTX 2080
    set(CUDA_COMPUTE_MODEL_72 "72") #Compute capability for Jetson Xavier
    set(CUDA_COMPUTE_MODEL_70 "70") #Compute capability for Tesla V100
    set(CUDA_COMPUTE_MODEL_62 "62") #Compute capability for Jetson TX2
    set(CUDA_COMPUTE_MODEL_61 "61") #Compute capability for Tesla P40
    set(CUDA_COMPUTE_MODEL_60 "60") #Compute capability for Tesla P100
    set(CUDA_COMPUTE_MODEL_53 "53") #Compute capability for Jetson TX1
    set(CUDA_COMPUTE_MODEL_52 "52") #Compute capability for Tesla M40/M60

    set(default_cuda_compute_model ${CUDA_COMPUTE_MODEL_52})
    if(ENABLE_CUDA AND NOT CUDA_COMPUTE_MODEL)
      message(STATUS "Setting the compute model to ${default_cuda_compute_model} as none was specified.")
      set(CUDA_COMPUTE_MODEL "${default_cuda_compute_model}" CACHE
              STRING "Choose the best compute model supported by your GPU" FORCE)
      set_property(CACHE CUDA_COMPUTE_MODEL PROPERTY STRINGS ${CUDA_COMPUTE_MODEL_75} ${CUDA_COMPUTE_MODEL_72} ${CUDA_COMPUTE_MODEL_70} ${CUDA_COMPUTE_MODEL_62} ${CUDA_COMPUTE_MODEL_61} ${CUDA_COMPUTE_MODEL_60} ${CUDA_COMPUTE_MODEL_53} ${CUDA_COMPUTE_MODEL_52})
    endif()
    set(
            CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS};
            -O3
            -gencode arch=compute_52,code=[sm_52,compute_52]
            -gencode arch=compute_61,code=[sm_61,compute_61]
            -gencode arch=compute_62,code=[sm_62,compute_62]  # GTX 1080 (Ti)
            -gencode arch=compute_72,code=[sm_75,compute_72]  # Jetson Xavier
            -gencode arch=compute_75,code=[sm_75,compute_75]  # RTX 2080 (Ti)
    )
  endif()
  add_definitions(-DGPU)
else()
  list(APPEND LIBRARIES "m")
endif()

if(USE_INTEGRATED_LIBS)
  set(PThreads_windows_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/pthreads CACHE PATH "Path where pthreads for windows can be located")
  add_definitions(-D_TIMESPEC_DEFINED)
endif()
set(Stb_DIR ${CMAKE_CURRENT_LIST_DIR}/3rdparty/stb CACHE PATH "Path where Stb image library can be located")

set(CMAKE_DEBUG_POSTFIX d)
add_definitions(-DUSE_CMAKE_LIBS)

if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "-Wno-unused-result -Wno-write-strings ${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "-O0" "-Og" CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG})
  string(REGEX REPLACE "-O3" "-Ofast" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffp-contract=fast -mavx -mavx2 -msse3 -msse4.1 -msse4.2 -msse4a")
endif()

# Find CUDNN
function(find_cudnn)
  if (DEFINED ENV{CUDNN_ROOT} AND NOT DEFINED CUDNN_ROOT)  # use env variable if not defined
    set(CUDNN_ROOT $ENV{CUDNN_ROOT})
  elseif (DEFINED CUDA_TOOLKIT_ROOT_DIR AND NOT DEFINED CUDNN_ROOT)  # use env variable if not defined
    set(CUDNN_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
  endif()
  #  set(CUDNN_ROOT /usr/local/cuda CACHE PATH "CUDNN root path")
  find_path(CUDNN_INCLUDE_DIRS cudnn.h
          HINTS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/include
          DOC "CUDNN include path")
  find_library(CUDNN_LIBRARIES NAMES libcudnn.so cudnn.lib
          PATHS ${CUDNN_ROOT}
          ${CUDNN_ROOT}/lib
          ${CUDNN_ROOT}/lib64
          ${CUDNN_ROOT}/lib/x64
          DOC "CUDNN library path")
  if(CUDNN_INCLUDE_DIRS AND CUDNN_LIBRARIES)
    set(CUDNN_FOUND TRUE PARENT_SCOPE)
    message(STATUS "Found CUDNN (include: ${BoldGreen} ${CUDNN_INCLUDE_DIRS} ${ColourReset}, library: ${BoldGreen} ${CUDNN_LIBRARIES} ${ColourReset})")
    mark_as_advanced(CUDNN_INCLUDE_DIRS CUDNN_LIBRARIES)
  else()
    MESSAGE(STATUS "Failed to find CUDNN in path: ${BoldGreen} ${CUDNN_ROOT} (${BoldRed}Did you set CUDNN_ROOT properly? ${ColourReset})")
  endif()
endfunction()

# Find required packages
#find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenCV 3.3 REQUIRED)

find_package(catkin REQUIRED
        COMPONENTS
        cv_bridge
        roscpp
        rospy
        sensor_msgs
        geometry_msgs
        nodelet
        image_geometry
        std_msgs
        obstacle_msgs
        image_transport
        eigen_conversions
        message_generation
        tf
        )

# Enable OPENCV in darknet
add_definitions(-DOPENCV)
add_definitions(-O4 -g)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
message(STATUS "Eigen location: ${EIGEN3_INCLUDE_DIR}")

catkin_package(
  INCLUDE_DIRS
        include
  LIBRARIES
        ${PROJECT_NAME}_lib
#        ${DARKNET_PATH}/lib
  CATKIN_DEPENDS
        cv_bridge
        roscpp
        rospy
        std_msgs
        sensor_msgs
        geometry_msgs
        image_geometry
        message_filters
        obstacle_msgs
        image_transport
#  DEPENDS
#        Boost
)

include_directories(
        ${DARKNET_PATH}/src
        ${DARKNET_PATH}/include
        include
        ${OpenCV_INCLUDE_DIRS}
#        ${Boost_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

#if (CUDA_FOUND)

  link_directories(
    /usr/local/cuda/lib64
  )

  find_cudnn()
  if(CUDNN_FOUND)
      include_directories(SYSTEM ${CUDNN_INCLUDE_DIRS})
      list(APPEND CUDA_LIBRARIES ${CUDNN_LIBRARIES})
      message("${BoldGreen} -- Successfully include CUDNN flags ${ColourReset}")
      option(CUDNN "Enable CuDNN" ON)
      add_definitions(-DCUDNN)
      # If cuda arch is sm_75 (RTX 2080 or 2080 Ti), add definition of CUDNN_HALF to accelerate
      if( ${ARCH} STREQUAL "-arch=sm_75")
        add_definitions(-DCUDNN_HALF)
        message("${BoldGreen} Enabled CUDNN_HALF ${ColourReset}")
      endif()
  else()
      message("${BoldRed} -- CUDNN not found, some dependent functionalities will be disabled ${ColourReset}")
  endif()

  cuda_add_library(${PROJECT_NAME}_lib
          src/YoloObjectDetector.cu
          src/image_interface.c
          src/Blob.cu
          src/data.cpp
          #src/hog.cpp
          src/Hungarian.cpp
          src/Obstacles.cpp
          src/segengine.cpp
          src/stdafx.cpp
          src/structures.cpp
          src/functions.cpp
          src/utilsSeg.cpp
#          src/image_opencv.cpp
#          src/track_kalman.hpp
#          src/track.hpp

    ${DARKNET_PATH}/src/activation_layer.c        ${DARKNET_PATH}/src/im2col.c
    ${DARKNET_PATH}/src/activations.c             ${DARKNET_PATH}/src/image.c
    ${DARKNET_PATH}/src/avgpool_layer.c           ${DARKNET_PATH}/src/layer.c
    ${DARKNET_PATH}/src/batchnorm_layer.c         ${DARKNET_PATH}/src/list.c
    ${DARKNET_PATH}/src/blas.c                    ${DARKNET_PATH}/src/local_layer.c
    ${DARKNET_PATH}/src/box.c                     ${DARKNET_PATH}/src/lstm_layer.c
    ${DARKNET_PATH}/src/col2im.c                  ${DARKNET_PATH}/src/matrix.c
    ${DARKNET_PATH}/src/connected_layer.c         ${DARKNET_PATH}/src/maxpool_layer.c
    ${DARKNET_PATH}/src/convolutional_layer.c     ${DARKNET_PATH}/src/network.c
    ${DARKNET_PATH}/src/cost_layer.c              ${DARKNET_PATH}/src/normalization_layer.c
    ${DARKNET_PATH}/src/crnn_layer.c              ${DARKNET_PATH}/src/option_list.c
    ${DARKNET_PATH}/src/crop_layer.c              ${DARKNET_PATH}/src/parser.c
    ${DARKNET_PATH}/src/dark_cuda.c               ${DARKNET_PATH}/src/region_layer.c
    ${DARKNET_PATH}/src/data.c                    ${DARKNET_PATH}/src/reorg_layer.c
    ${DARKNET_PATH}/src/deconvolutional_layer.c   ${DARKNET_PATH}/src/rnn_layer.c
    ${DARKNET_PATH}/src/demo.c                    ${DARKNET_PATH}/src/route_layer.c
    ${DARKNET_PATH}/src/detection_layer.c         ${DARKNET_PATH}/src/shortcut_layer.c
    ${DARKNET_PATH}/src/dropout_layer.c           ${DARKNET_PATH}/src/softmax_layer.c
    ${DARKNET_PATH}/src/gemm.c                    ${DARKNET_PATH}/src/tree.c
    ${DARKNET_PATH}/src/gru_layer.c               ${DARKNET_PATH}/src/utils.c
    ${DARKNET_PATH}/src/upsample_layer.c          ${DARKNET_PATH}/src/conv_lstm_layer.c
    ${DARKNET_PATH}/src/reorg_old_layer.c         ${DARKNET_PATH}/src/image_opencv.cpp
    ${DARKNET_PATH}/src/yolo_layer.c              ${DARKNET_PATH}/src/http_stream.cpp

    ${DARKNET_PATH}/src/activation_kernels.cu     ${DARKNET_PATH}/src/crop_layer_kernels.cu
    ${DARKNET_PATH}/src/avgpool_layer_kernels.cu  ${DARKNET_PATH}/src/deconvolutional_kernels.cu
    ${DARKNET_PATH}/src/blas_kernels.cu           ${DARKNET_PATH}/src/dropout_layer_kernels.cu
    ${DARKNET_PATH}/src/col2im_kernels.cu         ${DARKNET_PATH}/src/im2col_kernels.cu
    ${DARKNET_PATH}/src/convolutional_kernels.cu  ${DARKNET_PATH}/src/maxpool_layer_kernels.cu
    ${DARKNET_PATH}/src/network_kernels.cu
    )

target_link_libraries(${PROJECT_NAME}_lib
        pthread
#        stdc++
        cuda
#        cudart
        cublas
        curand
#        cudnn
#        ${Boost_LIBRARIES}
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${CUDA_LIBRARIES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../libSGM/lib/libsgm.so
        )

cuda_add_executable(${PROJECT_NAME}
  src/yolo_object_detector_node.cpp
)

#endif()
target_link_libraries(${PROJECT_NAME}
  ${PROJECT_NAME}_lib
)

add_dependencies(${PROJECT_NAME}_lib
        obstacle_msgs_generate_messages_cpp
)

cuda_add_library(cubicle_nodelet src/object_detector_nodelet.cpp )
target_link_libraries(cubicle_nodelet ${PROJECT_NAME}_lib)


add_subdirectory(doc)

install(TARGETS ${PROJECT_NAME}_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)


