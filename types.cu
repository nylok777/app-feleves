//
// Created by david on 4/26/26.
//
#include <cuda_runtime.h>

#include "types.cuh"
#include "numeric_funcs.cuh"

ProDrugPkParams::ProDrugPkParams(double bioavail, double hl, double tmax, double elim_rate, double abs_rate, ActiveDrug active)
    : m_parent(active), m_bio(bioavail), m_t_half(hl), m_t_max(tmax), m_elim(elim_rate), m_abs(abs_rate)
{
}

__host__ __device__ maybe<ProDrugPkParams> ProDrugPkParams::make(double bioavail, double hl, double tmax, ActiveDrug active)
{
    auto elim_rate = log(2) / hl;
    auto f = [=](const double ka)
    {
        return log(ka) - log(elim_rate) - tmax * (ka - elim_rate);
    };
    auto df = [=](const double ka)
    {
        return (1 / ka) - tmax;
    };
    auto abs_rate = newton_raphson((1 / tmax) + elim_rate, f, df, 1e-7, 100);
    if (!abs_rate.has_value()) return {};
    return ProDrugPkParams{bioavail, hl, tmax, elim_rate, abs_rate.value(), std::move(active)};
}

__host__ __device__ maybe<ProDrugPkParams::ActiveDrug> ProDrugPkParams::ActiveDrug::make(double pb, double hl, double tmax)
{
    auto elim_rate = log(2) / hl;
    auto f = [=](const double ka)
    {
        return log(ka) - log(elim_rate) - tmax * (ka - elim_rate);
    };
    auto df = [=](const double ka)
    {
        return (1 / ka) - tmax;
    };
    auto form_rate = newton_raphson((1 / tmax) + elim_rate, f, df, 1e-7, 100);
    if (!form_rate.has_value()) return {};
    return ActiveDrug{pb, hl, tmax, elim_rate, form_rate.value()};
}

ProDrugPkParams::ActiveDrug::ActiveDrug(double pb, double hl, double tmax, double elim_rate, double form_rate)
    : m_pb(pb), m_t_half(hl), m_t_max(tmax), m_elim(elim_rate), m_form(form_rate)
{
}
