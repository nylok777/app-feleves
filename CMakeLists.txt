cmake_minimum_required(VERSION 4.2)
project(app_feleves CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(app_feleves main.cu
        types.cuh
        ldx.cuh
        numeric_funcs.cuh
        numeric.cuh
        types.cu
        numeric_funcs.cu
        ldx.cu)

set_target_properties(app_feleves PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
