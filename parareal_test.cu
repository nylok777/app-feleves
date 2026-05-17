//
// Created by david on 5/15/26.
//
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "parareal.cuh"
#include "solver_funcs.cuh"

#define ROS3P_A_21 (1.0)
#define ROS3P_A_31 (1.0)
#define ROS3P_A_32 (0.0)
#define ROS3P_C_21 (-1.0156171083877702091975600115545)
#define ROS3P_C_31 (4.0759956452537699824805835358067)
#define ROS3P_C_32 (9.2076794298330791242156818474003)
#define ROS3P_GAMMA_DIAG (0.43586652150845899941601945119356)
#define ROS3P_b_1 (1.0)
#define ROS3P_b_2 (6.1697947043828245592553615689730)
#define ROS3P_b_3 (-0.4277225654321857332623837380651)

__host__ __device__ inline rosenbrock_coefficients<3> make_ros3p()
{
    return rosenbrock_coefficients
    {
        Double3x3{0, 0, 0,ROS3P_A_21, 0, 0,ROS3P_A_31,ROS3P_A_32, 0},
        Double3x3{0, 0, 0,ROS3P_C_21, 0, 0,ROS3P_C_31,ROS3P_C_32, 0},
        Double3x3::make_diagonal(ROS3P_GAMMA_DIAG, ROS3P_GAMMA_DIAG, ROS3P_GAMMA_DIAG),
        Double3{ROS3P_b_1, ROS3P_b_2, ROS3P_b_3}
    };
}

struct LinearDecaySystem {
    __device__ Double3 operator()(const Double3& y) const
    {
        return Double3{-1.0 * y[0], -0.5 * y[1], -0.25 * y[2]};
    }
};

struct LinearDecayJac {
    __device__ Double3x3 operator()(const Double3& /*y*/) const
    {
        return Double3x3::make_diagonal(-1.0, -0.5, -0.25);
    }
};

struct CoarseStep {
    __device__ Double3 operator()(const Double3& y) const
    {
        LinearDecaySystem f{};
        LinearDecayJac j{};
        return linearly_implicit_euler_step(y, h, f, j);
    }

    double h;
};

struct FineStep {
    __device__ Double3 operator()(const Double3& y) const
    {
        LinearDecaySystem f{};
        LinearDecayJac j{};
        const rosenbrock_coefficients<3> coeffs = make_ros3p();
        const double dh = h / static_cast<double>(n_substeps);
        Double3 yy = y;
        for (int i = 0; i < n_substeps; ++i) {
            yy = rosenbrock_method_step(yy, dh, coeffs, f, j);
        }
        return yy;
    }

    double h; // slice length
    int n_substeps; // number of fine sub-steps inside a slice
};

template<typename G>
__global__ void coarse_test_kernel(
    G coarse, Double3 y0, Double3* out_g)
{
    *out_g = coarse(y0);
}

template<typename F>
__global__ void fine_test_kernel(
    F fine, Double3 y0, Double3* out)
{
    *out = fine(y0);
}

class PararealTest : public testing::Test {
protected:
    static constexpr double T_END = 86400.0;
    static constexpr size_t SIZE = 168;
    static constexpr double H_SLICE = T_END / SIZE;
    static constexpr int SUBSTEPS = 32;
    static constexpr double ERRTOL = 1e-6;

    void SetUp()
    {
        int dev = 0;
        cudaGetDevice(&dev);
        int coop = 0;
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
        if (!coop) {
            GTEST_SKIP() << "Device does not support cooperative kernel launch.";
        }
    }
};

TEST_F(PararealTest, CoarseStep)
{
    Double3 y0{1.0, 2.0, 3.0};
    CoarseStep coarse{H_SLICE};

    Double3 host_g{};
    Double3* dev_g = nullptr;

    ASSERT_EQ(cudaMalloc(&dev_g, sizeof(Double3)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dev_g, &host_g, sizeof(Double3), cudaMemcpyHostToDevice), cudaSuccess);

    coarse_test_kernel<<<1,1>>>(coarse, y0, dev_g);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaMemcpy(&host_g, dev_g, sizeof(Double3), cudaMemcpyDeviceToHost);

    EXPECT_LT(host_g[0], y0[0]);

    cudaFree(dev_g);
}

TEST_F(PararealTest, FineStep)
{
    FineStep fine{H_SLICE, SUBSTEPS};
    Double3 y0{1.0, 2.0, 3.0};
    Double3* dev_f = nullptr;
    Double3 host_f{};

    ASSERT_EQ(cudaMalloc(&dev_f, sizeof(Double3)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dev_f, &host_f, sizeof(Double3), cudaMemcpyHostToDevice), cudaSuccess);

    fine_test_kernel<<<1,1>>>(fine, y0, dev_f);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    cudaMemcpy(&host_f, dev_f, sizeof(Double3), cudaMemcpyDeviceToHost);

    EXPECT_LT(host_f[0], y0[0]);

    const Double3 exact{
        y0[0] * std::exp(-1.00 * H_SLICE),
        y0[1] * std::exp(-0.50 * H_SLICE),
        y0[2] * std::exp(-0.25 * H_SLICE)
    };
    const double err_f =
        std::fabs(host_f[0] - exact[0]) + std::fabs(host_f[1] - exact[1]) + std::fabs(host_f[2] - exact[2]);
    EXPECT_LT(err_f, 1e-6);

    cudaFree(dev_f);
}

TEST_F(PararealTest, Launch)
{
    // ----------- Host-side allocation of inputs/outputs -----------
    Double3 y0{1.0, 2.0, 3.0};

    Double3* d_U_prev = nullptr;
    Double3* d_U = nullptr;
    Double3* d_U_next = nullptr;
    bool* d_conv = nullptr;

    ASSERT_EQ(cudaMalloc(&d_U_prev, sizeof(Double3) * SIZE), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_U, sizeof(Double3) * SIZE), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_U_next, sizeof(Double3) * SIZE), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_conv, sizeof(bool)), cudaSuccess);

    CoarseStep coarse{H_SLICE};
    FineStep fine{H_SLICE, SUBSTEPS};

    constexpr int BLOCK = 256;
    constexpr int GRID = (SIZE + BLOCK - 1) / BLOCK;

    void* kargs[] = {
        (void*)&y0,
        (void*)&coarse,
        (void*)&fine,
        (void*)&ERRTOL,
        (void*)&d_U_prev,
        (void*)&d_U,
        (void*)&d_U_next,
        (void*)&SIZE,
        (void*)&d_conv
    };

    auto kernel_ptr = reinterpret_cast<const void*>(
        &parareal_loop<CoarseStep, FineStep, 3>);

    cudaError_t launch_err = cudaLaunchCooperativeKernel(
        kernel_ptr,
        dim3(GRID),
        dim3(BLOCK),
        kargs);
    ASSERT_EQ(launch_err, cudaSuccess) << cudaGetErrorString(launch_err);

    cudaError_t sync_err = cudaDeviceSynchronize();
    ASSERT_EQ(sync_err, cudaSuccess) << cudaGetErrorString(sync_err);

    // ----------- Copy results back & validate -----------
    std::vector<Double3> host_U(SIZE);
    ASSERT_EQ(
        cudaMemcpy(host_U.data(), d_U_next, sizeof(Double3) * SIZE,
            cudaMemcpyDeviceToHost),
        cudaSuccess);

    // The analytical solution at every slice boundary t_i = (i+1)*H_SLICE
    // (parareal stores U[i] as the value at the end of the i-th slice).
    for (size_t i = 0; i < SIZE; ++i) {
        const double t = static_cast<double>(i + 1) * H_SLICE;
        const Double3 exact{
            y0[0] * std::exp(-1.00 * t),
            y0[1] * std::exp(-0.50 * t),
            y0[2] * std::exp(-0.25 * t)
        };
        // Parareal converges to the fine-propagator solution; we allow
        // a generous tolerance because the linearly-implicit Euler used
        // by the coarse propagator is only 1st-order accurate.
        const double tol = 5e-2;
        EXPECT_NEAR(host_U[i][0], exact[0], tol) << "slice " << i;
        EXPECT_NEAR(host_U[i][1], exact[1], tol) << "slice " << i;
        EXPECT_NEAR(host_U[i][2], exact[2], tol) << "slice " << i;
    }

    cudaFree(d_U_prev);
    cudaFree(d_U);
    cudaFree(d_U_next);
    cudaFree(d_conv);
}
