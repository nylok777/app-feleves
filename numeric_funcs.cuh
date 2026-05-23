//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_NUMERIC_FUNCS_CUH
#define APP_FELEVES_NUMERIC_FUNCS_CUH
#include <cuda_runtime.h>
#include "types.cuh"

template<signed_number T, int N> __host__ __device__ Vector<T,N> vabs(Vector<T,N> vec)
{
    for (int i = 0; i < N; ++i)
        if (vec[i] < T{0}) vec[i] *= T{-1};
    return vec;
}

template<real_number Real, int N> __host__ __device__ Real l2norm(const Vector<Real,N>& vec)
{
    Real sum{};
    for (int i = 0; i < N; ++i)
        sum += vec[i] * vec[i];
    return sqrt(sum);
}

template<int N> __host__ __device__ double l2norm(const DoubleVec<N>& vec)
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        sum += vec[i] * vec[i];
    return sqrt(sum);
}

template<signed_number T, int N>
__host__ __device__ Vector<T, N> forward_substitution(const Matrix<T, N>& L, const Vector<T, N>& b)
{
    Vector<T, N> y{};
    for (int i = 0; i < N; ++i) {
        T sum = Matrix<T, N>::dot(L, y, i, i);
        y[i] = (b[i] - sum) / L(i, i);
    }
    return y;
}

template<signed_number T, int N>
__host__ __device__ Vector<T, N> forward_substitution(
    const Matrix<T, N>& L,
    const Vector<T, N>& b,
    const Vector<int, N>& pivot)
{
    Vector<T, N> y{};
    for (int i = 0; i < N; ++i) {
        T sum = Matrix<T, N>::dot(L, y, i, i);
        y[i] = (b[pivot[i]] - sum) / L(i, i);
    }
    return y;
}

template<signed_number T, int N>
__host__ __device__ Vector<T, N> backward_substitution(const Matrix<T, N>& U, const Vector<T, N>& y)
{
    Vector<T, N> x{};
    for (int i = N - 1; i >= 0; --i) {
        T sum{0};
        if (int len = N - 1 - i; len > 0)
            sum = Matrix<T, N>::dot(U, x, i, len, i + 1);
        x[i] = (y[i] - sum) / U(i, i);
    }
    return x;
}

template<signed_number T, int N>
struct lu_result {
    Matrix<T, N> L;
    Matrix<T, N> U;
    Vector<int, N> pivot;
};

template<signed_number T, int N>
__host__ __device__ maybe<lu_result<T, N>> lu_decomp(const Matrix<T,N>& A)
{
    using vector = Vector<T,N>;
    using matrix = Matrix<T,N>;
    using int_vector = Vector<int, N>;

    matrix L = matrix::make_identity();
    int_vector p = int_vector::make_sequence();
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

        if (max_val < 1e-12)
            return {};

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
    return cuda::std::make_optional(lu_result<T,N>{std::move(L), std::move(U), std::move(p)});
}

template<signed_number T, int N>
__host__ __device__ maybe<Vector<T, N>> lu_solve(const Matrix<T, N>& A, const Vector<T, N>& b)
{
    return lu_decomp(A)
    .and_then([b](const auto& lu) {
        auto y = forward_substitution(lu.L, b, lu.pivot);
        return cuda::std::make_optional(backward_substitution(lu.U, y));
    })
    .or_else([]{ return maybe<Vector<T,N>>{}; });
}

template<int N>
__host__ __device__ constexpr DoubleVec<N> linear_interpolation(
    const DoubleVec<N>& y, const DoubleVec<N>& y_next, const double t, const double t_next, const double t_save)
{
    double theta = t_next == t ? 1.0 : (t_save - t) / (t_next - t);
    return y + ((y_next - y) * theta);
}

template<int N>
__host__ __device__ constexpr DoubleVec<N> quadratic_interpolation(
    const DoubleVec<N>& y0, const DoubleVec<N>& y1, const DoubleVec<N>& y2,
    const double t0, const double t1, const double t2,
    const double t_out)
{
    // Calculate the Lagrange basis weights
    double w0 = ((t_out - t1) * (t_out - t2)) / ((t0 - t1) * (t0 - t2));
    double w1 = ((t_out - t0) * (t_out - t2)) / ((t1 - t0) * (t1 - t2));
    double w2 = ((t_out - t0) * (t_out - t1)) / ((t2 - t0) * (t2 - t1));

    // Combine
    return (y0 * w0) + (y1 * w1) + (y2 * w2);
}
#endif //APP_FELEVES_NUMERIC_FUNCS_CUH
