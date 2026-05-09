//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_NUMERIC_FUNCS_CUH
#define APP_FELEVES_NUMERIC_FUNCS_CUH
#include "types.cuh"

__host__ __device__ inline double3 operator*(const double3& a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline double3 operator*(double n, const double3& v)
{
    return v * n;
}

__host__ __device__ inline double3& operator+=(double3& lhs, const double3& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

__host__ __device__ inline double3 operator-(const double3& lhs, const double3& rhs)
{
    return make_double3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__host__ __device__ inline double3 d3abs(const double3& x)
{
    return make_double3(fabs(x.x), fabs(x.y), fabs(x.z));
}

__host__ __device__ inline double l2norm(const double3& vec)
{
    return sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__ inline double3 solve_linear_system(Double3x3 a, double* b)
{
    for (int i = 0; i < 3; ++i) {
        int max_row = i;
        for (int j = i + 1; j < 3; ++j) {
            if (abs(a(j, i)) > abs(a(max_row, i)))
                max_row = j;
        }
        for (int j = 0; j < 3; ++j) {
            auto tmp = a(i,j);
            a(i,j) = a(max_row,j);
            a(max_row,j) = tmp;
        }
        auto tmp = b[i];
        b[i] = b[max_row];
        b[max_row] = tmp;

        for (int j = i + 1; j < 3; ++j) {
            auto factor = a(j, i) / a(i, i);
            for (int k = i; k < 3; ++k) {
                a(j, k) -= factor * a(i, k);
            }
            b[j] -= factor * b[i];
        }
    }

    double x[3];
    for (double& i : x) i = 0.0;
    for (int i = 2; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < 3; ++j) {
            sum += a(i, j) * x[j];
        }
        x[i] = (b[i] - sum) / a(i, i);
    }
    return make_double3(x[0], x[1], x[2]);
}

template<typename SystemFn, typename JacobianFn>
__device__ maybe<double3> newton_raphson_system(
    double3 y,
    const SystemFn& system_fn,
    const JacobianFn& jac_fn,
    double tolerance,
    int max_iter)
{
    for (int i = 0; i < max_iter; ++i) {
        double3 fx = system_fn(y);
        Double3x3 jx = jac_fn(y);
        auto fx_neg = make_double3(-fx.x, -fx.y, -fx.z);
        double3 dx = solve_linear_system(jx, reinterpret_cast<double*>(&fx_neg));
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

template<typename F>
__device__ double3 forward_euler(const F& system, const double3& y, double h)
{
    return y + (h * system(y));
}

template<typename J, typename F, typename R>
__device__ double3 bdf1_step(
    double3 y,
    double& t,
    double errtol,
    double& step_size,
    const F& sys_func,
    const J& jac_func,
    const R& solver)
{
    auto nonlinear_eq = [&step_size, &sys_func, &y] __device__ (const double3& y_next) -> double3
    {
        return y_next - y + (step_size * sys_func(y_next));
    };
    auto nonlinear_eq_jac = [&step_size, &jac_func] __device__ (const double3& y_next) -> Double3x3
    {
        return Double3x3::make_identity() - (step_size * jac_func(y_next));
    };
    bool step_accepted = false;
    double3 y_next;
    while (!step_accepted) {
        auto y_pred = forward_euler(sys_func, y, step_size);
        maybe<double3> y_opt = solver(y_pred, nonlinear_eq, nonlinear_eq_jac);
        if (!y_opt.has_value()) {
            step_size *= 0.5;
            continue;
        }
        y_next = std::move(y_opt.value());
        auto error = l2norm(d3abs(y_next - y_pred)) / 2.0;
        if (error <= errtol) {
            t += step_size;
            step_accepted = true;
        }
        step_size *= (error == 0.0 ? 2.0 : fmax(0.1, fmin(5.0, 0.9 * sqrt(errtol / error))));
    }
    return y_next;
}

template<typename J, typename F, typename R>
__device__ double3 bdf2_step(
    double3 y,
    double& t,
    double& h,
    double errtol,
    double3 y_prev,
    double& h_prev,
    const F& sys_func,
    const J& jac_func,
    const R& solver)
{
    double3 y_next;
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
        double3 y_pred = (1.0 + rho) * y - rho * y_prev;

        // Lambda 1: The nonlinear equation G(y_next) = 0
        // G(y) = y - alpha1*y_n - alpha2*y_{n-1} - beta*h*f(y)
        auto G = [=] __device__ (double3 yn)
        {
            return yn - (alpha1 * y) - (alpha2 * y_prev) - (beta * h) * sys_func(yn);
        };

        // Lambda 2: The derivative (Jacobian) of G
        // J_G(y) = I - beta * h * J_f(y)
        auto J_G = [=] __device__ (double3 yn)
        {
            return Double3x3::make_identity() - (beta * h) * jac_func(yn);
        };

        // Solve the nonlinear system using the provided Newton-Raphson solver
        y_next = solver(y_pred, G, J_G);

        // Local Truncation Error (LTE) estimation via Predictor-Corrector difference
        const double error_norm = l2norm(y_next - y_pred) / (1.0 + rho);

        if (error_norm <= errtol) {
            step_accepted = true;
            t += h;
        }

        // Compute adaptive step size multiplier based on BDF2's O(h^3) local error
        double scale = safety_factor * pow(errtol / (error_norm + 1e-15), 1.0 / 3.0);

        // Clamp the scaling to prevent erratic jumps in step size
        scale = fmax(min_scale, fmin(scale, max_scale));

        if (step_accepted) {
            h_prev = h;
            h *= scale; // Scale up for the *next* step function call
        } else {
            h *= scale; // Scale down and retry the current step
        }
    }

    return y_next;
}
#endif //APP_FELEVES_NUMERIC_FUNCS_CUH
