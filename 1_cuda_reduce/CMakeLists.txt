add_executable(my_reduce_v0 my_reduce_v0_global_memory.cu)

target_link_libraries(my_reduce_v0 PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v0 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif ()