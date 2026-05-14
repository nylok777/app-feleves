cmake_minimum_required(VERSION 4.2)
project(app_feleves LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(GTest REQUIRED)

add_executable(app_feleves main.cu
        types.cuh
        ldx.cuh
        numeric_funcs.cuh
        numeric.cuh
        types.cu
        ldx.cu)

target_compile_options(app_feleves PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ptxas-options=-v>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
)

set_target_properties(app_feleves PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(app_feleves m)

add_executable(app_feleves_test test.cu
        types.cuh
        numeric_funcs.cuh
        ldx.cuh
        types.cu
        numeric.cuh
        ldx.cu)

target_compile_options(app_feleves_test PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ptxas-options=-v>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
)

target_link_libraries(app_feleves_test GTest::gtest GTest::gtest_main)