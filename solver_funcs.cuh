//
// Created by david on 5/14/26.
//

#ifndef APP_FELEVES_SOLVERS_CUH
#define APP_FELEVES_SOLVERS_CUH
#include <cooperative_groups.h>
#include "numeric_funcs.cuh"
#include "types.cuh"

namespace cg = cooperative_groups;

template<typename T, int N>
concept ODE_solver = std::invocable<T, DoubleND<N>, double, DoubleND<N>*, size_t> &&
    std::same_as<void, std::invoke_result_t<T, DoubleND<N>, double, DoubleND<N>*, size_t>>;

template<int N, typename SystemFn, typename JacobianFn>
__device__ maybe<DoubleND<N>> newton_raphson(
    DoubleND<N> y,
    const SystemFn& system_fn,
    const JacobianFn& jac_fn,
    double tolerance,
    int max_iter)
{
    for (int i = 0; i < max_iter; ++i) {
        DoubleND<N> fx = system_fn(y);
        DoubleMat<N> jx = jac_fn(y);
        auto fx_neg = -fx;
        auto dx = lu_solve<double,3>(jx, fx_neg);
        y += dx;
        if (l2norm(dx) < tolerance) {
            return y;
        }
    }
    return {};
}

template<typename F, typename Df>
__host__ __device__ maybe<double> newton_raphson(
    double y,
    const F& f,
    const Df& df,
    double tolerance,
    int max_iter)
{
    for (int i = 0; i < max_iter; ++i) {
        double next_y = y - (f(y) / df(y));

        if (fabs(next_y - y) < tolerance) return next_y;
        y = next_y;
    }
    return {};
}

template<typename F, int N>
__device__ DoubleND<N> forward_euler(const F& system, const DoubleND<N>& y, double h)
{
    return y + (h * system(y));
}

template<typename F, typename J, typename R, int N>
__device__ maybe<DoubleND<N>> bdf1_step(const DoubleND<N>& y, const double h, const F& f, const J& jac, const R& solver)
{
    auto nonlinear_eq = [h, &f, &y] __device__ (const DoubleND<N>& y_next)
    {
        return y_next - y - (h * f(y_next));
    };
    auto nonlinear_eq_jac = [h, &jac] __device__ (const DoubleND<N>& y_next)
    {
        return DoubleMat<N>::make_identity() - (h * jac(y_next));
    };
    auto y_pred = forward_euler(f, y, h);
    maybe<DoubleND<N>> y_opt = solver(y_pred, nonlinear_eq, nonlinear_eq_jac);
    return y_opt.or_else([](){ return maybe<DoubleND<N>>{}; });
}

template<typename F, typename J, typename R, int N>
__device__ DoubleND<N> bdf1_step_adaptive(
    DoubleND<N> y,
    double& t,
    double errtol,
    double& step_size,
    const F& sys_func,
    const J& jac_func,
    const R& solver)
{
    auto nonlinear_eq = [&step_size, &sys_func, &y] __device__ (const DoubleND<N>& y_next)
    {
        return y_next - y - (step_size * sys_func(y_next));
    };
    auto nonlinear_eq_jac = [&step_size, &jac_func] __device__ (const DoubleND<N>& y_next)
    {
        return DoubleMat<N>::make_identity() - (step_size * jac_func(y_next));
    };
    bool step_accepted = false;
    DoubleND<N> y_next;
    while (!step_accepted) {
        auto y_pred = forward_euler(sys_func, y, step_size);
        maybe<DoubleND<N>> y_opt = solver(y_pred, nonlinear_eq, nonlinear_eq_jac);
        if (!y_opt.has_value()) {
            step_size *= 0.5;
            continue;
        }
        y_next = std::move(y_opt.value());
        auto error = l2norm<N>(dNabs(y_next - y_pred)) / 2.0;
        if (error <= errtol) {
            t += step_size;
            step_accepted = true;
        }
        step_size *= (error == 0.0 ? 2.0 : fmax(0.1, fmin(5.0, 0.9 * sqrt(errtol / error))));
    }
    return y_next;
}

template<typename J, typename F, typename R, int N>
__device__ DoubleND<N> bdf2_step_adaptive(
    DoubleND<N> y,
    double& t,
    double& h,
    double errtol,
    DoubleND<N> y_prev,
    double& h_prev,
    const F& sys_func,
    const J& jac_func,
    const R& solver)
{
    DoubleND<N> y_next;
    bool step_accepted = false;

    // Safety factors for the adaptive step size controller
    constexpr double safety_factor = 0.9;
    constexpr double min_scale = 0.2;
    constexpr double max_scale = 2.0;

    while (!step_accepted) {
        // Ratio of current step to previous step
        const double rho = h / h_prev;

        // Variable step size BDF2 coefficients
        const double alpha1 = ((1.0 + rho) * (1.0 + rho)) / (1.0 + 2.0 * rho);
        const double alpha2 = -(rho * rho) / (1.0 + 2.0 * rho);
        const double beta = (1.0 + rho) / (1.0 + 2.0 * rho);

        // Predictor: Explicit linear extrapolation for initial Newton guess
        DoubleND<N> y_pred = (1.0 + rho) * y - rho * y_prev;

        // Lambda 1: The nonlinear equation G(y_next) = 0
        // G(y) = y - alpha1*y_n - alpha2*y_{n-1} - beta*h*f(y)
        auto G = [=] __device__ (DoubleND<N> yn)
        {
            return yn - (alpha1 * y) - (alpha2 * y_prev) - (beta * h) * sys_func(yn);
        };

        // Lambda 2: The derivative (Jacobian) of G
        // J_G(y) = I - beta * h * J_f(y)
        auto J_G = [=] __device__ (DoubleND<N> yn)
        {
            return Double3x3::make_identity() - (beta * h) * jac_func(yn);
        };

        // Solve the nonlinear system using the provided Newton-Raphson solver
        maybe<DoubleND<N>> y_next_maybe = solver(y_pred, G, J_G);
        if (!y_next_maybe.has_value()) {
            h *= 0.5;
            continue;
        }
        y_next = std::move(y_next_maybe.value());

        // Local Truncation Error (LTE) estimation via Predictor-Corrector difference
        const double error_norm = l2norm<N>(y_next - y_pred) / (1.0 + rho);

        if (error_norm <= errtol) {
            step_accepted = true;
            t += h;
            h_prev = h;
        }

        // Compute adaptive step size multiplier based on BDF2's O(h^3) local error
        double scale = safety_factor * pow(errtol / (error_norm + 1e-15), 1.0 / 3.0);

        // Clamp the scaling to prevent erratic jumps in step size
        scale = fmax(min_scale, fmin(scale, max_scale));

        h *= scale;
    }

    return y_next;
}

template<typename F, typename JacFn, int N>
__host__ __device__ DoubleND<N> linearly_implicit_euler_step(
    const DoubleND<N>& y,
    const double h,
    const F& f,
    const JacFn& jac)
{
    using matrix = DoubleMat<N>;
    using vector = DoubleND<N>;

    vector fy = f(y);
    matrix J = jac(y);

    auto A = matrix::make_identity() - h * J;
    auto b = h * fy;

    auto dy = lu_solve(A, b);
    return y + dy;
}

template<int Dim>
struct rosenbrock_coefficients {
    DoubleMat<Dim> A;
    DoubleMat<Dim> C;
    DoubleMat<Dim> gamma;
    DoubleND<Dim> m;
};

template<typename F, typename JacFn, int N, int S>
__device__ DoubleND<N> rosenbrock_method_step(
    const DoubleND<N>& y,
    const double h,
    const rosenbrock_coefficients<S>& coeffs,
    F&& system_fn,
    JacFn&& system_jac)
{
    using matrixN = DoubleMat<N>;

    using vectorN = DoubleND<N>;

    F f = std::forward<F>(system_fn);
    matrixN J = std::forward<JacFn>(system_jac)(y);
    matrixN I = matrixN::make_identity();
    vectorN u[S]{};

    for (int i = 0; i < S; ++i) {
        auto y_tmp = y;
        for (int j = 0; j < i; ++j) {
            y_tmp += coeffs.A(i, j) * u[j];
        }
        vectorN rhs = f(y_tmp);

        for (int j = 0; j < i; ++j) {
            rhs += (coeffs.C(i, j) / h) * u[j];
        }
        auto W = (1.0 / (h * coeffs.gamma(i, i))) * I - J;
        auto lu = lu_decomp<double,N>(W);
        u[i] = lu_solve<double,N>(lu, rhs);
    }

    auto y_next = y;
    for (int i = 0; i < S; ++i)
        y_next += coeffs.m[i] * u[i];
    return y_next;
}

namespace detail
{
template<typename G, int N>
__host__ __device__ int parareal_initial_solve(
    const DoubleND<N>& y0,
    const G& coarse_step,
    DoubleND<N>* U,
    const size_t size)
{
    using coarse_result = std::invoke_result_t<G, DoubleND<N>>;
    // szekvenciálisan kell ezt a részt GPU-n is !!
    auto y = y0;
    for (size_t i = 0; i < size; ++i) {
        if constexpr (std::same_as<coarse_result, maybe<DoubleND<N>>>) {
            auto y_maybe = coarse_step(y);
            if (!y_maybe.has_value()) return -2;
            y = y_maybe.value();
        }
        else {
            y = coarse_step(y);
        }
        U[i] = y;
    }
    return 0;
}

template<typename F, int N>
__device__ void parareal_fine_steps(
    const DoubleND<N>& y0,
    const F& fine_step,
    DoubleND<N>* U_pred,
    DoubleND<N>* U_f,
    const size_t size)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid == 0)
        U_f[tid] = fine_step(y0);
    else if (tid < size)
        U_f[tid] = fine_step(U_pred[tid]);

    __syncthreads();
}

template<int N, typename G>
__host__ __device__ int parareal_correction(
    const G& coarse_step,
    DoubleND<N>* U_g,
    DoubleND<N>* U_f,
    DoubleND<N>* U_next,
    const size_t size)
{
    using vector = DoubleND<N>;
    using coarse_result = std::invoke_result_t<G, vector>;

    double t = 0.0;
    vector u = U_f[0]; // first element
    U_next[0] = u;
    for (size_t i = 1; i < size; ++i) {
        if constexpr (std::same_as<coarse_result, maybe<vector>>) {
            maybe<vector> u_maybe = coarse_step(u) + U_f[i-1] - U_g[i-1];
            if (!u_maybe.has_value()) return -2;
            u = u_maybe.value();
        }
        else {
            u = coarse_step(u) + U_f[i-1] - U_g[i-1];
        }
        U_next[i] = u;
    }
    return 0;
}

template<int N>
__host__ __device__ void parareal_check_convergence(
    const DoubleND<N>* U_prev,
    const DoubleND<N>* U,
    const size_t size,
    const double errtol,
    bool* converged)
{
#ifdef __CUDA_ARCH__
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size && fabs(U[tid] - U_prev[tid]) > errtol)
        *converged = false;
#else
    for (size_t i = 0; i < size; ++i) {
        if (std::fabs(U[i] - U_prev[i]) > errtol) {
            *converged = false;
            return;
        }
    }
    *converged = true;
#endif

}

__device__ inline bool dev_converged = false;
__device__ inline int dev_parareal_exit_flag = 0;
}

template<typename G, typename F, int N>
__global__ void parareal_loop(
    const DoubleND<N>& y0,
    const G& coarse_step,
    const F& fine_step,
    const double errtol,
    DoubleND<N>* U_prev,
    DoubleND<N>* U,
    DoubleND<N>* U_next,
    const size_t size)
{
    const auto grid = cg::this_grid();

    const auto tx = threadIdx.x;
    const auto tid = blockDim.x * blockIdx.x + tx;

    if (tid == 0) {
        detail::parareal_initial_solve(y0, coarse_step, U_prev, size);
    }
    grid.sync();

    while (tid < size && !detail::dev_converged) {
        detail::parareal_fine_steps(y0, fine_step, U_prev, U, size);
        grid.sync();
        if (tid == 0) {
            detail::parareal_correction(coarse_step, U_prev, U, U_next, size);
        }
        grid.sync();
        detail::parareal_check_convergence(U, U_next, size, errtol, &detail::dev_converged);
        grid.sync();
        if (tid == 0 && !detail::dev_converged)
            U_prev = U_next;
        grid.sync();
    }
    grid.sync();
}

#endif //APP_FELEVES_SOLVERS_CUH
