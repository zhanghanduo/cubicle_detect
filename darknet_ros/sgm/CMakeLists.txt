#    This file is part of sgm. (https://github.com/dhernandez0/sgm).
#
#    Copyright (c) 2016 Daniel Hernandez Juarez.

#    sgm is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    sgm is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 2.8)
project(stereo_sgm)
find_package( OpenCV REQUIRED )

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

# Find CUDA
FIND_PACKAGE(CUDA)
if (CUDA_FOUND)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Version: ${BoldGreen} ${CUDA_VERSION_STRINGS} ${ColourReset}")
  message(STATUS "CUDA Libararies: ${BoldGreen} ${CUDA_LIBRARIES} ${ColourReset}")
  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=[sm_61,compute_61]
    -gencode arch=compute_62,code=[sm_62,compute_62]
  )
  add_definitions(-DGPU)
else()
  list(APPEND LIBRARIES "m")
endif()

set(LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

CUDA_ADD_LIBRARY(stereo_sgm STATIC
    median_filter.cu hamming_cost.cu disparity_method.cu debug.cu costs.cu)

target_link_libraries( stereo_sgm ${OpenCV_LIBS} ${CUDA_LIBRARIES} cuda)
