cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set( CMAKE_CUDA_HOST_COMPILER "$ENV{CU_CCBIN}" CACHE PATH "$ENV{CU_CCBIN}" )
set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -arch=sm_60" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}" )

project(Final_Project LANGUAGES C CXX CUDA)

#Jacobian on GPU
add_executable(jacobian jacobian.cu f_eval.cuh)
#Jacobian on CPU for comparison 
add_executable(jacobian_cpu jacobian_cpu.cu f_eval.cuh)
#Generates input files
add_executable(generator generator.cpp)

#target_link_libraries(jacobian m)
