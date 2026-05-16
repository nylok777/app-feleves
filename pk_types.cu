//
// Created by david on 4/26/26.
//
#include <cuda_runtime.h>

#include "pk_types.cuh"
#include "solver_funcs.cuh"

__host__ __device__ static double find_ka_func(double ka, double elim_rate, double tmax)
{
    return log(ka) - log(elim_rate) - tmax * (ka - elim_rate);
}

__host__ __device__ static double find_ka_deriv(double ka, double tmax)
{
    return (1.0 / ka) - tmax;
}

__host__ __device__ ProDrugPkParams::ProDrugPkParams(double bioavail, double hl, double tmax, double elim_rate, double abs_rate,
    ActiveDrug active)
    : m_parent(active), m_bio(bioavail), m_t_half(hl), m_t_max(tmax), m_elim(elim_rate), m_abs(abs_rate)
{
}

__host__ __device__ maybe<ProDrugPkParams> ProDrugPkParams::make(double bioavail, double hl, double tmax, ActiveDrug active)
{
    auto elim_rate = log(2.0) / hl;
    auto f = [elim_rate, tmax] __host__ __device__ (const double ka)
    {
        return find_ka_func(ka, elim_rate, tmax);
    };
    auto df = [tmax] __host__ __device__ (const double ka)
    {
        return find_ka_deriv(ka, tmax);
    };
    auto abs_rate = newton_raphson((1 / tmax) + elim_rate, f, df, 1e-7, 100);
    if (!abs_rate.has_value()) return {};
    return ProDrugPkParams{bioavail, hl, tmax, elim_rate, abs_rate.value(), std::move(active)};
}

__host__ __device__ maybe<ProDrugPkParams::ActiveDrug> ProDrugPkParams::ActiveDrug::make(double pb, double hl, double tmax)
{
    auto elim_rate = log(2.0) / hl;
    auto f = [elim_rate, tmax] __host__ __device__ (const double ka)
    {
        return find_ka_func(ka, elim_rate, tmax);
    };
    auto df = [tmax] __host__ __device__ (const double ka)
    {
        return find_ka_deriv(ka, tmax);
    };
    auto form_rate = newton_raphson((1 / tmax) + elim_rate, f, df, 1e-7, 100);
    if (!form_rate.has_value()) return {};
    return ActiveDrug{pb, hl, tmax, elim_rate, form_rate.value()};
}

__host__ __device__ ProDrugPkParams::ActiveDrug::ActiveDrug(double pb, double hl, double tmax, double elim_rate, double form_rate)
    : m_pb(pb), m_t_half(hl), m_t_max(tmax), m_elim(elim_rate), m_form(form_rate)
{
}
