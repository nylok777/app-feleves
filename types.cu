//
// Created by david on 4/26/26.
//
#include <cuda_runtime.h>

#include "types.cuh"
#include "numeric_funcs.cuh"

ProDrugPkParams::ProDrugPkParams(double bioavail, double hl, double tmax, ActiveDrug active)
    : m_parent(active), m_bio(bioavail), m_t_half(hl), m_t_max(tmax), m_elim(log(2) / hl)
{
    auto f = [this](const double ka)
    {
        return log(ka) - log(m_elim) - m_t_max * (ka - m_elim);
    };
    auto df = [this](const double ka)
    {
        return (1 / ka) - m_t_max;
    };
    m_abs = newton_raphson((1 / m_t_max) + m_elim, f, df, 1e-7, 100);
}

ProDrugPkParams::ActiveDrug::ActiveDrug(double pb, double hl, double tmax)
    : m_pb(pb), m_t_half(hl), m_t_max(tmax), m_elim(log(2) / hl)
{
    auto f = [this](const double ka)
    {
        return log(ka) - log(m_elim) - m_t_max * (ka - m_elim);
    };
    auto df = [this](const double ka)
    {
        return (1 / ka) - m_t_max;
    };
    m_form = newton_raphson((1 / m_t_max) + m_elim, f, df, 1e-7, 100);
}
