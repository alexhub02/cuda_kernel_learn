cmake_minimum_required(VERSION 3.22)
project(cuda_kernel_learn VERSION 0.1.0 LANGUAGES CUDA CXX C)
find_package(CUDAToolkit)
add_subdirectory(1_cuda_reduce)
#set(CMAKE_CXX_STANDARD 17)

add_executable(cuda_kernel_learn main.cpp)
