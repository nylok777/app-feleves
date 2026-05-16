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
        pk_types.cu
        ldx.cu
        solver_funcs.cuh
        pk_types.cuh
        parareal.cuh)

target_compile_options(app_feleves PUBLIC
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--ptxas-options=-v>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-G>"
)

set_target_properties(app_feleves PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(app_feleves m)

add_executable(app_feleves_test test.cu)
add_executable(parareal_test parareal_test.cu)

set_target_properties(parareal_test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# add_compile_options(-G)
target_link_libraries(app_feleves_test app_feleves GTest::gtest GTest::gtest_main)
target_link_libraries(parareal_test app_feleves GTest::gtest GTest::gtest_main)