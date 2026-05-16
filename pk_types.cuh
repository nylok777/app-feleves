//
// Created by david on 5/16/26.
//

#ifndef APP_FELEVES_PK_TYPES_CUH
#define APP_FELEVES_PK_TYPES_CUH
#include <cuda/std/optional>

template<typename T> using maybe = cuda::std::optional<T>;

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
        __host__ __device__ ActiveDrug(
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
    __host__ __device__ ProDrugPkParams(
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
#endif //APP_FELEVES_PK_TYPES_CUH
