//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_TYPES_CUH
#define APP_FELEVES_TYPES_CUH
#include <concepts>
#include <cuda/std/optional>

#define DEF_TENSOR_SCALAR_INPLACE_OP(type, op_fn, op) \
    __host__ __device__ type& op_fn(T n) { for (auto& x : m_data) x op n; return *this; }

#define DEF_TENSOR_SCALAR_OP(type, op_fn, inplace_op) \
    friend __host__ __device__ type op_fn(type tensor, T n) { return tensor inplace_op n; } \
    friend __host__ __device__ type op_fn(T n, type tensor) { return tensor inplace_op n; }

#define DEF_TENSOR_TENSOR_INPLACE_OP(type, dim, op_fn, op) \
    __host__ __device__ type& op_fn(const type& rhs) { for (int i = 0; i < dim; ++i) m_data[i] op rhs.m_data[i]; return *this; }

#define DEF_TENSOR_TENSOR_OP(type, op_fn, inplace_op) \
    friend __host__ __device__ type op_fn(type lhs, const type& rhs) { return lhs inplace_op rhs; }

#define DEF_TENSOR_CMP_OP(type, dim, op_fn, op) \
    friend __host__ __device__ bool op_fn(const type& a, const type& b) \
    { \
        for (int i = 0; i < dim; ++i) \
            if (!(a.m_data[i] op b.m_data[i])) return false; \
        return true; \
    }

template<typename T> using maybe = cuda::std::optional<T>;
template<typename T> concept number = std::integral<T> || std::floating_point<T>;
template<typename T> concept signed_number = std::signed_integral<T> || std::floating_point<T>;
template<typename T> concept real_number = std::floating_point<T>;

template<number T, int Dim>
class Vector {
    T m_data[Dim]{T{0}};

public:
    Vector() = default;
    __host__ __device__ explicit Vector(T x) : m_data(x) {}
    __host__ __device__ explicit Vector(T data[Dim]) : m_data() { memcpy(m_data, data, Dim * sizeof(T)); }
    template<number... U> __host__ __device__ explicit Vector(U...items) : m_data(items...) {}

    __host__ __device__ static Vector make_sequence(T start = T{0})
    {
        T x[Dim];
        for (int i = 0; i < Dim; ++i) {
            x[i] = T{start++};
        }
        return Vector{x};
    }

    __host__ __device__ T& operator[](int i) { return m_data[i]; }
    __host__ __device__ T operator[](int i) const { return m_data[i]; }
    __host__ __device__ static T dot(Vector a, const Vector& b, int len = Dim)
    {
        T sum{0};
        for (int i = 0; i < len; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    DEF_TENSOR_CMP_OP(Vector, Dim, operator<, <)
    DEF_TENSOR_CMP_OP(Vector, Dim, operator>, >)
    DEF_TENSOR_CMP_OP(Vector, Dim, operator==, ==)

    DEF_TENSOR_SCALAR_INPLACE_OP(Vector, operator+=, +=)
    DEF_TENSOR_SCALAR_INPLACE_OP(Vector, operator-=, -=)
    DEF_TENSOR_SCALAR_INPLACE_OP(Vector, operator*=, *=)
    DEF_TENSOR_SCALAR_INPLACE_OP(Vector, operator/=, /=)
    DEF_TENSOR_SCALAR_OP(Vector, operator+, +=)
    DEF_TENSOR_SCALAR_OP(Vector, operator-, -=)
    DEF_TENSOR_SCALAR_OP(Vector, operator*, *=)
    DEF_TENSOR_SCALAR_OP(Vector, operator/, /=)

    DEF_TENSOR_TENSOR_INPLACE_OP(Vector, Dim, operator+=, +=)
    DEF_TENSOR_TENSOR_INPLACE_OP(Vector, Dim, operator-=, -=)
    DEF_TENSOR_TENSOR_INPLACE_OP(Vector, Dim, operator*=, *=)
    DEF_TENSOR_TENSOR_INPLACE_OP(Vector, Dim, operator/=, /=)
    DEF_TENSOR_TENSOR_OP(Vector, operator+, +=)
    DEF_TENSOR_TENSOR_OP(Vector, operator-, -=)
    DEF_TENSOR_TENSOR_OP(Vector, operator*, *=)
    DEF_TENSOR_TENSOR_OP(Vector, operator/, /=)

    friend __host__ __device__ Vector operator-(Vector vec) { return vec *= T{-1}; }
};

template<number T, int Dim>
class Matrix {
    static constexpr int Size = Dim * Dim;
    T m_data[Size]{T{0}};

public:
    Matrix() = default;
    __host__ __device__ explicit Matrix(T data[Size]) : m_data() { memcpy(m_data, data, Size * sizeof(T)); }

    template<number... U>
    __host__ __device__ explicit Matrix(U...items) : m_data(items...) {}

    __host__ __device__ static Matrix make_identity()
    {
        T data[Size]{T{0}};
        for (size_t i = 0; i < Size; i += Dim + 1)
            data[i] = T{1};
        return Matrix{data};
    }

    template<number... Args> requires (sizeof...(Args) == Dim)
    __host__ __device__ static Matrix make_diagonal(Args...args)
    {
        T items[Dim] = {args...};
        Matrix matrix{};
        for (int i = 0; i < Dim; ++i) {
            matrix(i, i) = items[i];
        }
        return matrix;
    }

    __host__ __device__ T& operator()(int i, int j) { return m_data[(i * Dim) + j]; }
    __host__ __device__ T operator()(int i, int j) const { return m_data[(i * Dim) + j]; }
    __host__ __device__ Vector<T, Dim> col(int j, int num_rows = Dim) const
    {
        Vector<T, Dim> vec{};
        for (int i = 0; i < num_rows; ++i)
            vec[i] = (*this)(i, j);
        return vec;
    }

    __host__ __device__ static T dot(Matrix mat, const Vector<T, Dim>& vec, const int row, const int len = Dim, const int start_col = 0)
    {
        T sum{0};
        for (int i = start_col; i < len; ++i) {
            sum += mat(row, i) * vec[i];
        }
        return sum;
    }

    DEF_TENSOR_SCALAR_INPLACE_OP(Matrix, operator+=, +=)
    DEF_TENSOR_SCALAR_INPLACE_OP(Matrix, operator-=, -=)
    DEF_TENSOR_SCALAR_INPLACE_OP(Matrix, operator*=, *=)
    DEF_TENSOR_SCALAR_INPLACE_OP(Matrix, operator/=, /=)
    DEF_TENSOR_SCALAR_OP(Matrix, operator+, +=)
    DEF_TENSOR_SCALAR_OP(Matrix, operator-, -=)
    DEF_TENSOR_SCALAR_OP(Matrix, operator*, *=)
    DEF_TENSOR_SCALAR_OP(Matrix, operator/, /=)

    DEF_TENSOR_TENSOR_INPLACE_OP(Matrix, Size, operator+=, +=)
    DEF_TENSOR_TENSOR_INPLACE_OP(Matrix, Size, operator-=, -=)
    DEF_TENSOR_TENSOR_OP(Matrix, operator+, +=)
    DEF_TENSOR_TENSOR_OP(Matrix, operator-, -=)
};

template<int Dim> using DoubleMat = Matrix<double, Dim>;
template<int Dim> using DoubleVec = Vector<double, Dim>;
using Double3 = Vector<double, 3>;
using Double3x3 = DoubleMat<3>;

template<int N>
struct step_result {
    DoubleVec<N> y;
    double t;
};

struct solver_status {
    double t_last;
    size_t size;
    bool finished;
};

#endif //APP_FELEVES_TYPES_CUH
