cmake_minimum_required(VERSION 3.1)

find_package(CUDA REQUIRED)

#if(DEFINED AUTO_DETECT_ARCH AND "${AUTO_DETECT_ARCH}")
#	CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS "Auto")
#	set(CUDA_ARCH "${ARCH_FLAGS}" CACHE STRING "Value of the NVCC -arch option." FORCE)
#endif()
#message(STATUS "CUDA_ARCH: \"${CUDA_ARCH}\"")

include_directories(../include)

if (CMAKE_COMPILER_IS_GNUCXX)
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}
	-gencode arch=compute_50,code=[sm_50,compute_50]
	-gencode arch=compute_52,code=[sm_52,compute_52]
	-gencode arch=compute_61,code=[sm_61,compute_61]
	-gencode arch=compute_62,code=[sm_62,compute_62]")

file(GLOB STEREOSRCS "*.cu" "*.cpp")

SET(sgm_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib CACHE PATH "Target for the libraries")
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${sgm_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${sgm_LIBRARY_OUTPUT_DIRECTORY})

if(LIBSGM_SHARED)
	CUDA_ADD_LIBRARY(sgm stereo_sgm.cpp ${STEREOSRCS} SHARED)
	target_link_libraries(sgm ${CUDA_LIBRARIES})
	if(BUILD_OPENCV_WRAPPER)
		target_link_libraries(sgm ${OpenCV_LIBS})
	endif()
else()
	CUDA_ADD_LIBRARY(sgm stereo_sgm.cpp ${STEREOSRCS} STATIC)
endif()

install(
		TARGETS sgm
		ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/lib
		LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
		RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin
)

install(
		DIRECTORY ${CMAKE_SOURCE_DIR}/include
		DESTINATION ${CMAKE_INSTALL_PREFIX}
		FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(
		FILES ${CMAKE_SOURCE_DIR}/FindLibSGM.cmake
		DESTINATION ${CMAKE_INSTALL_PREFIX}
)
