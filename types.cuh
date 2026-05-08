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
    __device__ T& operator()(int i, int j) { return m_data[(i * 3) + j]; }
    __device__ T operator()(int i, int j) const { return m_data[(i * 3) + j]; }

private:
    T m_data[9];
};

class ProDrugPkParams {
public:
    class ActiveDrug {
    public:
        ActiveDrug(double pb, double hl, double tmax);
        __device__ double protein_binding() const { return m_pb; }
        __device__ double half_life() const { return m_t_half; }
        __device__ double t_max() const { return m_t_max; }
        __device__ double elim_rate_c() const { return m_elim; }
        __device__ double form_rate_c() const { return m_form; }

    private:
        double m_pb;
        double m_t_half;
        double m_t_max;
        double m_elim;
        double m_form;
    };
    ProDrugPkParams(double bioavail, double hl, double tmax, ActiveDrug active);
    __device__ const ActiveDrug& parent_drug() const { return m_parent; }
    __device__ double bioavail() const { return m_bio; }
    __device__ double half_life() const { return m_t_half; }
    __device__ double t_max() const { return m_t_max; }
    __device__ double elim_rate_c() const { return m_elim; }
    __device__ double abs_rate_c() const { return m_abs; }

private:
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
