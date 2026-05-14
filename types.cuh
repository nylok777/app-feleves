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

template<typename T>
using maybe = cuda::std::optional<T>;

template<typename T>
concept number = std::integral<T> || std::floating_point<T>;

template<typename T, int Dim>
    requires std::integral<T> || std::floating_point<T>
class VectorND {

    T m_data[Dim]{ T{0} };

public:
    VectorND() = default;
    explicit VectorND(T data[Dim]) : m_data() {memcpy(m_data, data, Dim * sizeof(T));}
    template<number... U> explicit VectorND(U... items) : m_data(items ...) {}
    __host__ __device__ static VectorND make_sequence(T start = T{0})
    {
        T x[Dim];
        for (int i = 0; i < Dim; ++i) {
            x[i] = T{start++};
        }
        return VectorND{x};
    }
    __host__ __device__ T& operator[](int i) { return m_data[i]; }
    __host__ __device__ T operator[](int i) const { return m_data[i]; }
    __host__ __device__ static T dot(VectorND a, const VectorND& b, int len = Dim)
    {
        T sum{0};
        for (int i = 0; i < len; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    DEF_TENSOR_CMP_OP(VectorND, Dim, operator<, <)
    DEF_TENSOR_CMP_OP(VectorND, Dim, operator>, >)
    DEF_TENSOR_CMP_OP(VectorND, Dim, operator==, ==)

    DEF_TENSOR_SCALAR_INPLACE_OP(VectorND, operator+=, +=)
    DEF_TENSOR_SCALAR_INPLACE_OP(VectorND, operator-=, -=)
    DEF_TENSOR_SCALAR_INPLACE_OP(VectorND, operator*=, *=)
    DEF_TENSOR_SCALAR_INPLACE_OP(VectorND, operator/=, /=)
    DEF_TENSOR_SCALAR_OP(VectorND, operator+, +=)
    DEF_TENSOR_SCALAR_OP(VectorND, operator-, -=)
    DEF_TENSOR_SCALAR_OP(VectorND, operator*, *=)
    DEF_TENSOR_SCALAR_OP(VectorND, operator/, /=)

    DEF_TENSOR_TENSOR_INPLACE_OP(VectorND, Dim, operator+=, +=)
    DEF_TENSOR_TENSOR_INPLACE_OP(VectorND, Dim, operator-=, -=)
    DEF_TENSOR_TENSOR_INPLACE_OP(VectorND, Dim, operator*=, *=)
    DEF_TENSOR_TENSOR_INPLACE_OP(VectorND, Dim, operator/=, /=)
    DEF_TENSOR_TENSOR_OP(VectorND, operator+, +=)
    DEF_TENSOR_TENSOR_OP(VectorND, operator-, -=)
    DEF_TENSOR_TENSOR_OP(VectorND, operator*, *=)
    DEF_TENSOR_TENSOR_OP(VectorND, operator/, /=)

};

template<typename T, int Dim>
    requires std::integral<T> || std::floating_point<T>
class Matrix {
    static constexpr int Size = Dim * Dim;
public:
    Matrix() = default;
    explicit Matrix(T data[Size]) : m_data() {memcpy(m_data, data, Size * sizeof(T)); }
    __host__ __device__ T& operator()(int i, int j) { return m_data[(i * Dim) + j]; }
    __host__ __device__ T operator()(int i, int j) const { return m_data[(i * Dim) + j]; }
    __host__ __device__ VectorND<T,Dim> col(int j, int num_rows = Dim) const
    {
        VectorND<T,Dim> vec{};
        for (int i = 0; i < num_rows; ++i)
            vec[i] = (*this)(i, j);
        return vec;
    }
    __host__ __device__ static Matrix make_identity()
    {
        T data[Size]{ T{0} };
        for (size_t i = 0; i < Size; i += Dim + 1)
            data[i] = T{1};
        return Matrix{data};
    }
    __host__ __device__ static T dot(Matrix mat, const VectorND<T,Dim>& vec, const int row, const int len = Dim, const int start_col = 0)
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

private:
    T m_data[Size]{ T{0} };
};

template<int Dim> using DoubleMat = Matrix<double, Dim>;
template<int Dim> using DoubleND = VectorND<double, Dim>;

using Double3 = VectorND<double, 3>;
using Double3x3 = DoubleMat<3>;

template<typename T>
    requires std::integral<T> || std::floating_point<T>
class Mat3 {
public:
    Mat3() = default;
    __device__ explicit Mat3(const T* data) : m_data() { for (size_t i = 0; i < 9; ++i) m_data[i] = data[i]; }
    __device__ explicit Mat3(T data[9]) : m_data() { memcpy(m_data, data, 9 * sizeof(T)); }
    __device__ T& operator()(int i, int j) { return m_data[(i * 3) + j]; }
    __device__ T operator()(int i, int j) const { return m_data[(i * 3) + j]; }

    __device__ static Mat3 make_identity()
    {
        T data[9]{ T{0} };
        for (size_t i = 0; i < 9; i += 4)
            data[i] = T{1};
        return Mat3{data};
    }

    friend __device__ Mat3 operator*(const Mat3& mat, T n)
    {
        Mat3 out = mat;
        for (auto& item : out.m_data)
            item *= n;
        return out;
    }

    friend __device__ Mat3 operator*(T n, const Mat3& mat) { return mat * n; }

    friend __device__ Mat3 operator-(Mat3 lhs, const Mat3& rhs)
    {
        for (size_t i = 0; i < 9; ++i) {
            lhs.m_data[i] -= rhs.m_data[i];
        }
        return lhs;
    }

private:
    T m_data[9];
};

struct step_result {
    double3 y;
    double t;
};

struct solver_status {
    double t_last;
    size_t size;
    bool finished;
};

class ProDrugPkParams {
public:
    class ActiveDrug {
    public:
        __host__ __device__ static maybe<ActiveDrug> make(double pb, double hl, double tmax);
        __host__ __device__ double protein_binding() const { return m_pb; }
        __host__ __device__ double half_life() const { return m_t_half; }
        __host__ __device__ double t_max() const { return m_t_max; }
        __host__ __device__ double elim_rate_c() const { return m_elim; }
        __host__ __device__ double form_rate_c() const { return m_form; }

    private:
        __host__ __device__ ActiveDrug (
        double pb,
        double hl,
        double tmax,
        double elim_rate,
        double form_rate
        );
        double m_pb;
        double m_t_half;
        double m_t_max;
        double m_elim;
        double m_form;
    };

    __host__ __device__ static maybe<ProDrugPkParams> make(double bioavail, double hl, double tmax, ActiveDrug active);
    __host__ __device__ const ActiveDrug& parent_drug() const { return m_parent; }
    __host__ __device__ double bioavail() const { return m_bio; }
    __host__ __device__ double half_life() const { return m_t_half; }
    __host__ __device__ double t_max() const { return m_t_max; }
    __host__ __device__ double elim_rate_c() const { return m_elim; }
    __host__ __device__ double abs_rate_c() const { return m_abs; }

private:
    __host__ __device__ ProDrugPkParams (
    double bioavail,
    double hl,
    double tmax,
    double elim_rate,
    double abs_rate, ActiveDrug active
    );
    ActiveDrug m_parent;
    double m_bio;
    double m_t_half;
    double m_t_max;
    double m_elim;
    double m_abs;
};

struct MichaelisMentenParams {
    double v_max;
    double km;
};

using Double3x3_Old = Mat3<double>;

#endif //APP_FELEVES_TYPES_CUH
