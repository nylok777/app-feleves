//
// Created by david on 5/15/26.
//
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "solver_funcs.cuh"

constexpr rosenbrock_coefficients<3> ROS3P_Coefficients
{
    Double3x3
    { 0, 0, 0,
      1, 0, 0,
      1, 0, 0 },
    Double3x3
    {
        0, 0, 0,
        -1.0156171083877702091975600115545, 0, 0,
        4.0759956452537699824805835358067, 9.2076794298330791242156818474003, 0
    },
    Double3x3::make_diagonal(0.43586652150845899941601945119356, 0.24291996454816804366592249683314, 2.1851380027664058511513169485832),
    Double3{1.0, 6.1697947043828245592553615689730, -0.4277225654321857332623837380651}
};

TEST(PararealTest, Launch)
{
    auto f = [] __device__ (const Double3& vy) {
        constexpr double sigma = 10.0;
        constexpr double rho = 28.0;
        constexpr double beta = 8.0 / 3.0;
        return Double3
        {
            sigma * (vy.y - vy.x),
            vy.x * (rho - vy.z) - vy.y,
            vy.x * vy.y - beta * vy.z
        };
    };
    auto jac = [] __device__ (const Double3& vy) {
        return Double3x3
        {
            -10.0, 10.0, 0,
            28.0 - vy.z, -1, -vy.x,
            vy.y, vy.x, -(8.0 / 3.0)
        };
    };
    constexpr double t_end = 86400;
    constexpr int device_max_threads = cudaDevAttrMaxThreadsPerMultiProcessor * cudaDevAttrMultiProcessorCount; // 81920;
    constexpr size_t block_size = 512;
    constexpr double step_size = t_end / device_max_threads;
    constexpr size_t num_blocks = device_max_threads / 512;
    auto G = [step_size, f, jac] __device__ (const Double3& y)
    {
        return linearly_implicit_euler_step(y, step_size, f, jac);
    };

    auto F = [step_size, f, jac] __device__ (const Double3&y)
    {
        return rosenbrock_method_step(y, step_size, ROS3P_Coefficients, f, jac);
    };
    
}