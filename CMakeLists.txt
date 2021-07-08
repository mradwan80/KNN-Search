cmake_minimum_required(VERSION 2.8)
project(knn LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 14)


#### use cuda ####
#if(WIN32)
#	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-lineinfo;-Xptxas -v)
#else()
#	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-lineinfo; -cudart=static; -Xptxas; -v)
#endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_50,code=sm_50 -lineinfo -cudart=static -Xptxas -v")

if(WIN32)
	#include_directories(glm $ENV{LIBS_DIRECTORY})	
	include_directories(glm)	
else()
	include_directories(glm $ENV{CUDA_INCLUDE_DIRECTORY})
	link_directories($ENV{CUDA_LIBRARY_DIRECTORY})
endif()

ADD_EXECUTABLE(
knn 
src/knn.cpp
src/knn.cu
src/knnWithSort.cpp
src/knnWithSort.cu
src/DDS.h
src/DDS.cpp
src/DDS.cu
src/main.cpp)
