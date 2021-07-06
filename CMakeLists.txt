cmake_minimum_required(VERSION 2.8)
project(knn LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 14)

set(CMAKE_CUDA_COMPILER $ENV{CUDA_COMPILER})

#find_package(CUDA REQUIRED)

#### use cuda ####
if(WIN32)
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-lineinfo;-Xptxas -v)
else()
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-lineinfo; -cudart=static; -Xptxas; -v)
endif()

#SET(CUDA_SEPARABLE_COMPILATION ON)


include_directories(glm $ENV{CUDA_INCLUDE_DIRECTORY})
link_directories($ENV{CUDA_LIBRARY_DIRECTORY})

#CUDA_ADD_EXECUTABLE(
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

#if(WIN32)
#	target_link_libraries(knn)
#elseif(UNIX)
#	target_link_libraries(knn ${CUDA_CUDART_LIBRARY})
#endif()

#SET(GCC_OPTIMIZATION_FLAGS "-O2 -march=native")
#add_definitions(${GCC_OPTIMIZATION_FLAGS})

