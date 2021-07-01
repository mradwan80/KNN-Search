# Author: e1027470@student.tuwien.ac.at

cmake_minimum_required(VERSION 2.8)
project(knn)

#### detect operating system ####
#if(WIN32)
#  set(SYSTEM "Windows")
#  set(SYS "${SYSTEM}${ARCH}")
#elseif(UNIX)
#  set(SYSTEM "Unix")
#  set(SYS "${SYSTEM}${ARCH}")
#endif()
#message("System detected: ${SYS}")



find_package(CUDA REQUIRED)

#### use cuda ####
if(WIN32)
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-lineinfo;-Xptxas -v)
else()
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-lineinfo; -cudart=static; -Xptxas; -v)
endif()

#SET(CUDA_SEPARABLE_COMPILATION ON)


#### use glew ####
#find_package(GLEW)
#include_directories(${GLEW_INCLUDE_DIRS})

#### use glfw ####
#if(WIN32)
#	add_subdirectory(ext/glfw)
#	include_directories(ext/glfw/include)
#else()
#	find_package(PkgConfig REQUIRED)
#	pkg_search_module(GLFW REQUIRED glfw3)
#	include_directories(${GLFW_INCLUDE_DIRS})
#endif()

include_directories(glm)

CUDA_ADD_EXECUTABLE(
knn 
src/knn.cpp
src/knn.cu
src/knnWithSort.cpp
src/knnWithSort.cu
src/DDS.h
src/DDS.cpp
src/DDS.cu
src/main.cpp)

if(WIN32)
	target_link_libraries(knn)
elseif(UNIX)
	target_link_libraries(knn ${CUDA_CUDART_LIBRARY})
endif()

SET(GCC_OPTIMIZATION_FLAGS "-O2 -march=native")
add_definitions(${GCC_OPTIMIZATION_FLAGS})

