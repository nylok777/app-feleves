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

__host__ __device__ inline double3& operator+=(double3& lhs, const double3& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

__host__ __device__ inline double3 d3abs(const double3& x)
{
    return make_double3(abs(x.x), abs(x.y), abs(x.z));
}

__host__ __device__ inline double l2norm(const double3& vec)
{
    double sum = vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
    return sqrt(sum);
}

__device__ double3 solve_linear_system(
    Mat3<double> a,
    double* b)
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
__device__ double3 newton_raphson_system(
    double3 y,
    const SystemFn& system_fn,
    const JacobianFn& jac_fn,
    double tolerance,
    int max_iter,
    bool* success)
{
    for (int i = 0; i < max_iter; ++i) {
        double3 fx = system_fn(y);
        Mat3<double> jx = jac_fn(y);
        auto fx_neg = make_double3(-fx.x, -fx.y, -fx.z);
        double3 dx = solve_linear_system(jx, reinterpret_cast<double*>(&fx_neg));
        y += dx;
        if (l2norm(dx) < tolerance) {
            *success = true;
            return y;
        }
    }
    *success = false;
    return make_double3(0.0, 0.0, 0.0);
}

template<typename F, typename Df>
__device__ double newton_raphson(
    double y,
    const F& f,
    const Df& df,
    double tolerance,
    int max_iter,
    bool* success)
{
    for (int i = 0; i < max_iter; ++i) {
        auto next_y = y - (f(y) / df(y));

        if (abs(next_y - y) < tolerance) return next_y;
        y = next_y;
    }
    *success = false;
    return 0.0
}

template<typename F>
__device__ double3 forward_euler(const F& system, const double3& y, double h)
{
    return y + (h * system(y));
}

template<typename Jf, typename F, typename R>
__device__ double3 bdf1_step(
    const double3& y,
    double errtol,
    double* step_size,
    const Jf& jac_func,
    const F& sys_func,
    const R& solver)
{
    bool converged = false;
    auto y_n = y;
    auto nonlinear_eq = [step_size, &sys_func, &y_n](const double3& y) -> double3
    {
        return y - y_n + (*step_size * sys_func(y));
    };
    auto nonlin_eq_jac = [step_size, &jac_func](const double3& y) -> Double3x3
    {
        Double3x3 j_f = jac_func(y);
        Double3x3 j_g;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                j_g(i, j) = (i == j ? 1.0 : 0.0) - (*step_size) * j_f(i, j);
            }
        }
        return j_g;
    }
    while (!converged) {
        auto y_pred = forward_euler(sys_func, y, *step_size);
        maybe<double3> y_opt = solver(y_pred, nonlinear_eq, nonlin_eq_jac);
        if (!y_opt.has_value()) {
            *step_size *= 0.5;
            continue;
        }
        auto y_curr = y_opt.value();
        auto error = l2norm(d3abs(y_curr - y_pred)) / 2.0;
    }
}
#endif //APP_FELEVES_NUMERIC_FUNCS_CUH
