//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_TYPES_CUH
#define APP_FELEVES_TYPES_CUH
#include <cuda/std/optional>

template<typename T>
using maybe = cuda::std::optional<T>;

template<typename T>
    requires std::integral<T> || std::floating_point<T>
class Mat3
{
public:
    Mat3() = default;
    __device__ explicit Mat3(const T* data) : m_data() { for (size_t i = 0; i < 9; ++i ) m_data[i] = data[i]; }
    __device__ explicit Mat3(T data[9]) : m_data() { memcpy(m_data, data, 9 * sizeof(T)); }
    __device__ T& operator()(int i, int j) { return m_data[(i * 3) + j]; }
    __device__ T operator()(int i, int j) const { return m_data[(i * 3) + j]; }

    __device__ static Mat3 make_identity()
    {
        T data[9];
        for (size_t i = 0; i < 9; ++i)
            data[i] == (i % 4 == 0 ? T{1} : T{0});
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

template<int Order>
struct step_result
{
    double3 y[Order];
    double t;
};

template<>
struct step_result<1>
{
    double3 y;
    double t;
};

template<>
struct step_result<2>
{
    double3 y_n;
    double3 y;
    double t;
};

struct solver_status
{
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
        ActiveDrug(double pb, double hl, double tmax, double elim_rate, double form_rate);
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
    ProDrugPkParams(double bioavail, double hl, double tmax, double elim_rate, double abs_rate, ActiveDrug active);
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

using Double3x3 = Mat3<double>;

#endif //APP_FELEVES_TYPES_CUH
