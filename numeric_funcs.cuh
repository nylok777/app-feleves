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

__host__ __device__ inline double3 operator+(double3 lhs, const double3& rhs)
{
    return lhs += rhs;
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

template<number T, int N>
__device__ VectorND<T,N> forward_substitution(const Matrix<T,N>& L, const VectorND<T,N>& b,
    VectorND<T,N> y)
{
    for (int i = 0; i < N; ++i) {
        T sum = Matrix<T,N>::dot(L, y, i, i);
        y[i] = (b[i] - sum) / L(i,i);
    }
    return y;
}

template<number T, int N>
__device__ VectorND<T,N> forward_substitution(
    const Matrix<T,N>& L,
    const VectorND<T,N>& b,
    const VectorND<int,N>& pivot,
    VectorND<T,N> y)
{
    for (int i = 0; i < N; ++i) {
        T sum = Matrix<T,N>::dot(L, y, i, i);
        y[i] = (b[pivot[i]] - sum) / L(i, i);
    }
    return y;
}

template<number T, int N>
__device__ VectorND<T,N> backward_substitution(const Matrix<T,N>& U, const VectorND<T,N>& y, VectorND<T,N> x)
{
    for (int i = N - 1; i >= 0; --i) {
        T sum{0};
        if (int len = N - 1 - i; len > 0)
            sum = Matrix<T,N>::dot(U, x, i, len, i+1);
        x[i] = (y[i] - sum) / U(i,i);
    }
    return x;
}

template<number T, int N>
struct lu_matrices {
    Matrix<T,N> L;
    Matrix<T,N> U;
    VectorND<int,N> pivot;
};

template<number T, int N, typename vector = VectorND<T,N>>
__device__ vector lu_solve(const lu_matrices<T,N>& lu, const vector& b, vector x)
{
    auto y = forward_substitution(lu.L, b, lu.pivot, vector{});
    return backward_substitution(lu.U, y, x);
}

template<number T, int N, typename vector = VectorND<T,N>, typename matrix = Matrix<T,N>>
__device__ lu_matrices<T,N> lu_decomp(const matrix& A)
{
    using intvector = VectorND<int,N>;
    matrix L = matrix::make_identity();
    intvector p = intvector::make_sequence();
    matrix U{};
    vector u_col{};

    for (int i = 0; i < N; ++i) {
        T max_val{};
        int pivot_row = i;
        if (i > 0)
            u_col = U.col(i, i);
        for (int k = i; k < N; ++k) {
            T sum{0};
            if (i > 0)
                sum = matrix::dot(L, u_col, k, i);
            if (T val = fabs(A(p[k], i) - sum); val > max_val) {
                max_val = val;
                pivot_row = k;
            }
        }

        //TODO: check singular if can't avoid

        if (pivot_row != i) {
            cuda::std::swap(p[i], p[pivot_row]);
            for (int j = 0; j < i; ++j)
                cuda::std::swap(L(i, j), L(pivot_row, j));
        }

        for (int k = i; k < N; ++k) {
            u_col = U.col(k, i);
            T sum = matrix::dot(L, u_col, i, i);
            U(i, k) = A(p[i], k) - sum;
        }

        if (int k = i + 1; k < N) {
            u_col = U.col(i, i);
            for (; k < N; ++k) {
                T sum = matrix::dot(L, u_col, k, i);
                L(k, i) = (A(p[k], i) - sum) / U(i, i);
            }
        }
    }
    return {std::move(L), std::move(U), std::move(p)};
}

__device__ inline double3 solve_linear_system(Double3x3_Old a, const double3& b_v)
{
    double b[3] {b_v.x, b_v.y, b_v.z};
    for (int i = 0; i < 3; ++i) {
        int max_row = i;
        for (int j = i + 1; j < 3; ++j) {
            if (fabs(a(j, i)) > fabs(a(max_row, i)))
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
        Double3x3_Old jx = jac_fn(y);
        auto fx_neg = make_double3(-fx.x, -fx.y, -fx.z);
        double3 dx = solve_linear_system(jx, fx_neg);
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
__device__ double3 bdf1_step_adaptive(
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
        return y_next - y - (step_size * sys_func(y_next));
    };
    auto nonlinear_eq_jac = [&step_size, &jac_func] __device__ (const double3& y_next) -> Double3x3_Old
    {
        return Double3x3_Old::make_identity() - (step_size * jac_func(y_next));
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
__device__ double3 bdf2_step_adaptive(
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
            return Double3x3_Old::make_identity() - (beta * h) * jac_func(yn);
        };

        // Solve the nonlinear system using the provided Newton-Raphson solver
        maybe<double3> y_next_maybe = solver(y_pred, G, J_G);
        if (!y_next_maybe.has_value()) {
            h *= 0.5;
            continue;
        }
        y_next = std::move(y_next_maybe.value());

        // Local Truncation Error (LTE) estimation via Predictor-Corrector difference
        const double error_norm = l2norm(y_next - y_pred) / (1.0 + rho);

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
        u[i] = lu_solve<double,N>(lu, rhs, u[i]);
    }

    auto y_next = y;
    for (int i = 0; i < S; ++i)
        y_next += coeffs.m[i] * u[i];
    return y_next;
}

__device__ inline double3 linear_interpolation(const double3& y, const double3& y_next, double t, double t_next, double t_save)
{
    double theta = t_next == t ? 1.0 : (t_save - t) / (t_next - t);
    return y + ((y_next - y) * theta);
}

__device__ inline double3 quadratic_interpolation(
    const double3& y0, const double3& y1, const double3& y2,
    double t0, double t1, double t2,
    double t_out)
{
    // Calculate the Lagrange basis weights
    double w0 = ((t_out - t1) * (t_out - t2)) / ((t0 - t1) * (t0 - t2));
    double w1 = ((t_out - t0) * (t_out - t2)) / ((t1 - t0) * (t1 - t2));
    double w2 = ((t_out - t0) * (t_out - t1)) / ((t2 - t0) * (t2 - t1));

    // Combine
    return (y0 * w0) + (y1 * w1) + (y2 * w2);
}
#endif //APP_FELEVES_NUMERIC_FUNCS_CUH
