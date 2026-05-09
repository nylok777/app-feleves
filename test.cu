//
// Created by david on 4/26/26.
//

#include <gtest/gtest.h>
#include "numeric_funcs.cuh"
#include "numeric.cuh"

class NewtonRaphsonTest : public testing::Test
{
public:
    struct f
    {
        __host__ __device__ double operator()(double x) const { return cos(x) - pow(x, 3.0); }
    };
    struct df
    {
        __host__ __device__ double operator()(double x) const { return -sin(x) - 3.0 * pow(x, 2.0); }
    };

protected:
    NewtonRaphson<f, df> newton;

    NewtonRaphsonTest() : newton(f{}, df{}, 1e-7, 100) {}
};

template<typename N>
__global__ void test_kernel_newton(N newton, double x, maybe<double>* y)
{
    *y = newton(x);
}

TEST_F(NewtonRaphsonTest, Call)
{
    double x0 = 0.5;
    maybe<double>* dev_y = nullptr;
    cudaMalloc(&dev_y, sizeof(maybe<double>));
    test_kernel_newton<<<1, 1>>>(newton, x0, dev_y);
    maybe<double>* y = new maybe<double>;
    cudaMemcpy(y, dev_y, sizeof(maybe<double>), cudaMemcpyDeviceToHost);
    EXPECT_TRUE(y->has_value());
    std::cout << y->value_or(0.0) << '\n';
    delete y;
    cudaFree(dev_y);
}

struct LorenzSystem
{
    __host__ __device__ double3 operator()(const double3& vy) const
    {
        constexpr double sigma = 10.0;
        constexpr double rho = 28.0;
        constexpr double beta = 8.0 / 3.0;
        auto f1 = sigma * (vy.y - vy.x);
        auto f2 = vy.x * (rho - vy.z) - vy.y;
        auto f3 = vy.x * vy.y - beta * vy.z;
        return make_double3(f1, f2, f3);
    }
};

struct LorenzJac
{
    __host__ __device__ Double3x3 operator()(const double3& vy) const
    {
        double data[9]
        {
            -10.0, 10.0, 0,
            28.0 - vy.z, -1, -vy.x,
            vy.y, vy.x, -(8.0 / 3.0)
        };
        return Double3x3{data};
    }
};

class NewtonRaphsonSystemTest : public testing::TestWithParam<std::pair<double3, double3>>
{
protected:
    static constexpr double errtol = 1e-10;
    NewtonRaphsonSystem newton;

    NewtonRaphsonSystemTest() : newton(errtol, 1000) {}
};

template<typename NS, typename F, typename J>
__global__ void test_kernel_newton_system(NS newton, F f, J j, double3 x0, maybe<double3>* result)
{
    *result = newton(x0, f, j);
}

TEST_P(NewtonRaphsonSystemTest, FindRoot)
{
    auto [x0, root] = GetParam();
    maybe<double3>* dev_result = nullptr;
    cudaMalloc(&dev_result, sizeof(maybe<double3>));
    test_kernel_newton_system<<<1, 1>>>(newton, LorenzSystem{}, LorenzJac{}, x0, dev_result);
    auto* result = new maybe<double3>;
    cudaMemcpy(result, dev_result, sizeof(maybe<double3>), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result->has_value());

    constexpr double rtol = 1e-8;
    auto eps = root.x + root.y + root.z == 0.0 ? make_double3(errtol, errtol, errtol) : (rtol + errtol) * d3abs(result->value());
    EXPECT_LE(std::fabs(result->value().x - root.x), eps.x);
    EXPECT_LE(std::fabs(result->value().y - root.y), eps.y);
    EXPECT_LE(std::fabs(result->value().z - root.z), eps.z);

    std::cout << "x: " << result->value().x << "\ty: " << result->value().y << "\tz: " << result->value().z << '\n';

    delete result;
    cudaFree(dev_result);
}

INSTANTIATE_TEST_SUITE_P(LorenzRoots,
    NewtonRaphsonSystemTest,
    testing::Values(std::make_pair(make_double3(1.0, 1.0, 1.0), make_double3(0., 0., 0.)),
        std::make_pair(make_double3(10., 10., 30.), make_double3(8.48528137, 8.48528137, 27.0)),
        std::make_pair(make_double3(-10., -10., 30.), make_double3(-8.48528137, -8.48528137, 27.0))));

class BDFTest : public testing::Test
{
protected:
    static constexpr double atol = 1e-4;
    static constexpr double save_interval = 0.02;
    NewtonRaphsonSystem newton{atol, 3};
    BackwardEuler<LorenzSystem, LorenzJac, NewtonRaphsonSystem> backward_euler{atol, 1e-4, save_interval, LorenzSystem{}, LorenzJac{},
        newton};
    BDF2Step<LorenzSystem, LorenzJac, NewtonRaphsonSystem> bdf2_stepper{atol, 1e-4, LorenzSystem{}, LorenzJac{},
        NewtonRaphsonSystem{newton}};
};

template<typename S>
__global__ void test_kernel_bdf1_step(double3 x, step_result<1>* result, S solver)
{
    *result = solver.step(x);
}

TEST_F(BDFTest, Order1Step)
{
    auto x0 = make_double3(1., 1., 1.);
    step_result<1>* dev_result = nullptr;
    cudaMalloc(&dev_result, sizeof(step_result<1>));
    test_kernel_bdf1_step<<<1, 1>>>(x0, dev_result, backward_euler);
    step_result<1> result{};
    auto err = cudaMemcpy(&result, dev_result, sizeof(step_result<1>), cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
}

template<typename S>
__global__ void test_kernel_bdf(S solver, double3 x0, double t_end, double3* y_out, size_t size, solver_status* status)
{
    *status = solver(x0, t_end, y_out, size);
}

TEST_F(BDFTest, Order1)
{
    constexpr size_t size = 50'000;
    constexpr double t_end = 100;
    auto x0 = make_double3(1., 1., 1.);
    double3* dev_y_out = nullptr;
    cudaMalloc(&dev_y_out, sizeof(double3) * size);
    solver_status* dev_status = nullptr;
    cudaMalloc(&dev_status, sizeof(solver_status));
    test_kernel_bdf<<<1, 1>>>(backward_euler, x0, t_end, dev_y_out, size, dev_status);
    solver_status status{};
    cudaMemcpy(&status, dev_status, sizeof(solver_status), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(status.finished);
}