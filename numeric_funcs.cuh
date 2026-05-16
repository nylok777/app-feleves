//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_NUMERIC_FUNCS_CUH
#define APP_FELEVES_NUMERIC_FUNCS_CUH
#include <cuda_runtime.h>
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

template<int N>
__host__ __device__ DoubleND<N> vabs(DoubleND<N> vec)
{
    for (int i = 0; i < N; ++i)
        if (vec[i] < 0.0) vec[i] *= -1.0;
    return vec;
}

template<int N>
__host__ __device__ double l2norm(const DoubleND<N>& vec)
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        sum += vec[i] * vec[i];
    return sqrt(sum);
}

template<number T, int N>
__device__ VectorND<T,N> forward_substitution(const Matrix<T,N>& L, const VectorND<T,N>& b)
{
    VectorND<T,N> y{};
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
    const VectorND<int,N>& pivot)
{
    VectorND<T,N> y{};
    for (int i = 0; i < N; ++i) {
        T sum = Matrix<T,N>::dot(L, y, i, i);
        y[i] = (b[pivot[i]] - sum) / L(i, i);
    }
    return y;
}

template<number T, int N>
__device__ VectorND<T,N> backward_substitution(const Matrix<T,N>& U, const VectorND<T,N>& y)
{
    VectorND<T,N> x{};
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

template<number T, int N, typename vector = VectorND<T,N>>
__device__ vector lu_solve(const lu_matrices<T,N>& lu, const vector& b)
{
    auto y = forward_substitution(lu.L, b, lu.pivot);
    return backward_substitution(lu.U, y);
}

template<signed_number T, int N>
__device__ VectorND<T,N> lu_solve(const Matrix<T,N>& A, const VectorND<T,N>& b)
{
    return lu_solve(lu_decomp<T,N>(A), b);
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
