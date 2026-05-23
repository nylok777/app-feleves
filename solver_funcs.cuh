//
// Created by david on 5/14/26.
//

#ifndef APP_FELEVES_SOLVERS_CUH
#define APP_FELEVES_SOLVERS_CUH
#include <cuda/std/tuple>
#include <cuda/std/numeric>

#include "numeric_funcs.cuh"
#include "types.cuh"

template<typename T, int N>
concept ODE_solver = std::invocable<T, DoubleVec<N>, double, DoubleVec<N>*, size_t> &&
    std::same_as<void, std::invoke_result_t<T, DoubleVec<N>, double, DoubleVec<N>*, size_t>>;

template<int N, typename SystemFn, typename JacobianFn>
__host__ __device__ maybe<DoubleVec<N>> newton_raphson(
    DoubleVec<N> y,
    const SystemFn& system_fn,
    const JacobianFn& jac_fn,
    double tolerance,
    int max_iter)
{
    for (int i = 0; i < max_iter; ++i) {
        DoubleVec<N> fx = system_fn(y);
        DoubleMat<N> jx = jac_fn(y);
        auto fx_neg = -fx;
        auto dx_maybe = lu_solve<double,N>(jx, fx_neg);
        if (!dx_maybe.has_value())
            dx_maybe = lu_solve<double,N>(jx + (1e-8 * DoubleMat<N>::make_identity()), fx_neg);
        if (!dx_maybe.has_value()) return {};
        y += dx_maybe.value();
        if (l2norm(dx_maybe.value()) < tolerance) {
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

template<typename F, int N, real_number Real>
__host__ __device__ Vector<Real,N> forward_euler(const F& f, const Vector<Real,N>& y, Real h)
{
    return y + (h * f(y));
}

template<typename F, int N>
__host__ __device__ DoubleVec<N> forward_euler(const F& system, const DoubleVec<N>& y, double h)
{
    return y + (h * system(y));
}

template<real_number Real, int N>
__host__ __device__ Vector<Real,N+1> constexpr make_bdf_coefficients() requires (N >= 2 && N < 7)
{
    using vector = Vector<Real,N+1>;

    auto generator = [] __host__ __device__ (Real numerators[N+1], Real denom)
    {
        vector vec;
        for (size_t i = 0; i < N + 1; ++i) {
            vec[i] = numerators[i] / denom;
        }
        return vec;
    };
    if constexpr (N == 2) {
        Real numerators[N+1] {-4.0, 1.0, 2.0};
        return generator(numerators, 3.0);
    }
    if constexpr (N == 3) {
        Real nums[N+1] {-18, 9, -2, 6};
        return generator(nums, 11);
    }
    if constexpr (N == 4) {
        Real nums[N+1] {-48, 36, -16, 3, 12};
        return generator(nums, 25);
    }
    if constexpr (N == 5) {
        Real nums[N+1] {-300, 300, -200, 75, -12, 60};
        return generator(nums, 137);
    }
    if constexpr (N == 6) {
        Real nums[N+1] {-360, 450, -400, 225, -72, 10, 60};
        return generator(nums, 147);
    }
}

template<typename F, typename J, typename R, real_number Real, int N, int O> requires (N >= 2 && N < 7)
__host__ __device__ maybe<Vector<Real,N>> bdf_step(
    const Vector<Real,N> Y[O],
    const Real h,
    const Vector<Real,N> y_pred,
    const F& f,
    const J& jac,
    const R& root)
{
    using vector = Vector<Real,N>;

    auto f_eq = [Y, h, f] __host__ __device__ (const vector& y_next)
    {
        const auto coeffs = make_bdf_coefficients<Real,O>();
        vector acc = y_next;
        for (int i = 0; i < O; ++i) {
            acc += Y[i] * coeffs[i];
        }
        return acc - ((h * coeffs[O]) * f(y_next));
    };
    auto jac_eq = [h, jac](const vector& y_next) {
        Real beta;
        {
            const auto tmp = make_bdf_coefficients<Real,O>();
            beta = tmp[O];
        }
        return Matrix<Real,N>::make_identity() - (beta * h) * jac(y_next);
    };
    return root(y_pred, f_eq, jac_eq);
}

template<typename F, typename J, typename R, real_number Real, int N, int O> requires (O >= 2 && O < 7)
__host__ __device__ maybe<Vector<Real,N>> bdf_step(
    const Vector<Real,N> Y[O],
    const Real h,
    const F& f,
    const J& jac,
    const R& root)
{
    Vector<Real,N> y_pred = forward_euler(f, Y[O-1], h);
    return bdf_step(Y, h, y_pred, f, jac, root);
}

template<typename F, typename J, typename R, real_number Real, int N>
__host__ __device__ maybe<Vector<Real,N>> bdf_step(
    const Vector<Real,N>& y,
    const Real h,
    const Vector<Real,N>& y_pred,
    const F& f,
    const J& jac,
    const R& root)
{
    using vector = Vector<Real,N>;

    auto f_eq = [f, h, y] __host__ __device__ (const vector& y_next)
    {
        return y_next - y - (h * f(y_next));
    };
    auto jac_eq = [h, jac] __host__ __device__ (const vector& y_next)
    {
        return DoubleMat<N>::make_identity() - (h * jac(y_next));
    };
    return root(y_pred, f_eq, jac_eq);
}

template<typename F, typename J, typename R, real_number Real, int N>
__host__ __device__ maybe<Vector<Real,N>> bdf_step(
    const Vector<Real,N>& y,
    const Real h,
    const F& f,
    const J& jac,
    const R& root)
{
    const auto y_pred = forward_euler(f, y, h);
    return bdf_step(y, h, y_pred, f, jac, root);
}

template<typename F, typename J, typename R, real_number Real, int N, int O> requires (O >= 1 && O < 7)
__host__ __device__ Vector<Real,N> bdf_step_adaptive(
    const Vector<Real,N> Y[O],
    Real h,
    const Real errtol,
    const F& f,
    const J& jac,
    const R& root,
    Real* h_accepted,
    Real* h_next)
{
    using limits = cuda::std::numeric_limits<Real>;
    using vector = Vector<Real,N>;

    constexpr Real safety_factor = 0.9;
    bool step_accepted = false;
    Real h_prev;
    vector y_next_out;

    while (!step_accepted) {
        auto y_pred = forward_euler(f, Y[O-1], h);
        maybe<vector> y_next_maybe;
        if constexpr (O > 1)
            y_next_maybe = bdf_step(Y, h, y_pred, f, jac, root);
        else
            y_next_maybe = bdf_step(Y[0], h, y_pred, f, jac, root);

        auto [y_res, h_prev_res, h_res, passed] =
            y_next_maybe.and_then([h, errtol, y_pred, safety_factor](const vector& y_next) {
                const auto error = l2norm(y_next - y_pred);
                return cuda::std::make_optional(cuda::std::make_tuple(
                    y_next,
                    h,
                    h * safety_factor * pow(errtol / error, Real{1} / Real{O+1}),
                    error <= errtol));
            })
            .or_else([h] {
                return cuda::std::make_optional(cuda::std::make_tuple(vector{limits::infinity()}, h, h * 0.5, false));
            })
            .value();

        y_next_out = std::move(y_res);
        h_prev = h_prev_res;
        h = h_res;
        step_accepted = passed;
    }
    *h_next = h;
    *h_accepted = h_prev;
    return y_next_out;
}

template<typename F, typename J, typename R, real_number Real, int N>
__host__ __device__ Vector<Real,N> bdf_step_adaptive(
    Vector<Real,N> y,
    Real h,
    const Real errtol,
    const F& f,
    const J& jac,
    const R& root,
    Real* h_accepted,
    Real* h_next)
{
    //Vector<Real,N> Y[1] {std::move(y)};
    return bdf_step_adaptive(&y, h, errtol, f, jac, root, h_accepted, h_next);
}

template<typename F, typename J, typename R, int N>
__host__ __device__ maybe<DoubleVec<N>> bdf1_step(const DoubleVec<N>& y, const double h, const F& f, const J& jac, const R& solver)
{
    auto f_eq = [h, f, y] __host__ __device__ (const DoubleVec<N>& y_next)
    {
        return y_next - y - (h * f(y_next));
    };
    auto jac_eq = [h, jac] __host__ __device__ (const DoubleVec<N>& y_next)
    {
        return DoubleMat<N>::make_identity() - (h * jac(y_next));
    };
    auto y_pred = forward_euler(f, y, h);
    return solver(y_pred, f_eq, jac_eq);
}

template<typename F, typename J, typename R, int N>
__host__ __device__ DoubleVec<N> bdf1_step_adaptive(
    const DoubleVec<N>& y,
    double errtol,
    double& step_size,
    const F& sys_func,
    const J& jac_func,
    const R& solver)
{
    auto f_eq = [step_size, sys_func, y] __host__ __device__ (const DoubleVec<N>& y_next)
    {
        return y_next - y - (step_size * sys_func(y_next));
    };
    auto jac_eq = [step_size, jac_func] __host__ __device__ (const DoubleVec<N>& y_next)
    {
        return DoubleMat<N>::make_identity() - (step_size * jac_func(y_next));
    };
    bool step_accepted = false;
    DoubleVec<N> y_next;
    while (!step_accepted) {
        auto y_pred = forward_euler(sys_func, y, step_size);
        maybe<DoubleVec<N>> y_opt = solver(y_pred, f_eq, jac_eq);
        if (!y_opt.has_value()) {
            step_size *= 0.5;
            continue;
        }
        y_next = std::move(y_opt.value());
        auto error = l2norm<N>(y_next - y_pred) / 2.0;
        if (error <= errtol) {
            step_accepted = true;
        }
        step_size *= (error == 0.0 ? 2.0 : fmax(0.1, fmin(5.0, 0.9 * sqrt(errtol / error))));
    }
    return y_next;
}

template<typename J, typename F, typename R, int N>
__host__ __device__ DoubleVec<N> bdf2_step_adaptive(
    const DoubleVec<N>& y,
    double& h,
    const double errtol,
    const DoubleVec<N>& y_prev,
    double& h_prev,
    const F& sys_func,
    const J& jac_func,
    const R& solver)
{
    DoubleVec<N> y_next;
    bool step_accepted = false;

    while (!step_accepted) {
        constexpr double max_scale = 2.0;
        constexpr double min_scale = 0.2;
        constexpr double safety_factor = 0.9;

        // Ratio of current step to previous step
        const double rho = h / h_prev;

        // Variable step size BDF2 coefficients
        const double alpha1 = ((1.0 + rho) * (1.0 + rho)) / (1.0 + 2.0 * rho);
        const double alpha2 = -(rho * rho) / (1.0 + 2.0 * rho);
        const double beta = (1.0 + rho) / (1.0 + 2.0 * rho);

        // Predictor: Explicit linear extrapolation for initial Newton guess
        DoubleVec<N> y_pred = (1.0 + rho) * y - rho * y_prev;

        // Lambda 1: The nonlinear equation G(y_next) = 0
        // G(y) = y - alpha1*y_n - alpha2*y_{n-1} - beta*h*f(y)
        auto G = [=] __host__ __device__ (const DoubleVec<N>& yn)
        {
            return yn - (alpha1 * y) - (alpha2 * y_prev) - (beta * h) * sys_func(yn);
        };

        // Lambda 2: The derivative (Jacobian) of G
        // J_G(y) = I - beta * h * J_f(y)
        auto J_G = [=] __host__ __device__ (const DoubleVec<N>& yn)
        {
            return Double3x3::make_identity() - (beta * h) * jac_func(yn);
        };

        // Solve the nonlinear system using the provided Newton-Raphson solver
        maybe<DoubleVec<N>> y_next_maybe = solver(y_pred, G, J_G);
        if (!y_next_maybe.has_value()) {
            h *= 0.5;
            continue;
        }
        y_next = std::move(y_next_maybe.value());

        // Local Truncation Error (LTE) estimation via Predictor-Corrector difference
        const double error_norm = l2norm(y_next - y_pred) / (1.0 + rho);

        if (error_norm <= errtol) {
            step_accepted = true;
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
__host__ __device__ DoubleVec<N> linearly_implicit_euler_step(
    const DoubleVec<N>& y,
    const double h,
    const F& f,
    const JacFn& jac)
{
    using matrix = DoubleMat<N>;
    using vector = DoubleVec<N>;

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
    DoubleVec<Dim> m;
};

template<typename F, typename JacFn, int N, int S>
__host__ __device__ DoubleVec<N> rosenbrock_method_step(
    const DoubleVec<N>& y,
    const double h,
    const rosenbrock_coefficients<S>& coeffs,
    F&& system_fn,
    JacFn&& system_jac)
{
    using matrixN = DoubleMat<N>;

    using vectorN = DoubleVec<N>;

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

#endif //APP_FELEVES_SOLVERS_CUH
