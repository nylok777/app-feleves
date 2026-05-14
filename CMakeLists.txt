cmake_minimum_required(VERSION 4.2)
project(app_feleves LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(GTest REQUIRED)

add_library(app_feleves STATIC
        types.cuh
        ldx.cuh
        numeric_funcs.cuh
        solvers.cuh
        types.cu
        ldx.cu
        solver_funcs.cuh)

target_compile_options(app_feleves PUBLIC
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ptxas-options=-v>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
)

set_target_properties(app_feleves PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(app_feleves m)

add_executable(app_feleves_test test.cu)

target_link_libraries(app_feleves_test app_feleves GTest::gtest GTest::gtest_main)