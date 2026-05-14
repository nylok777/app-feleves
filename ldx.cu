//
// Created by david on 5/9/26.
//

#include "ldx.cuh"

Double3 ldx::ldx_model(const Double3& y, const ProDrugPkParams& params, const MichaelisMentenParams& mm_params)
{
    const auto& active_params = params.parent_drug();
    auto v = (mm_params.v_max * y[1]) / (mm_params.km + y[1]);
    auto prodrug_rate = params.abs_rate_c() * y[0];
    auto active_rate = y[1] * active_params.form_rate_c() * (v - params.elim_rate_c());

    auto d_CGIp_dt = -prodrug_rate;
    auto d_Cp_dt = prodrug_rate - active_rate;
    auto d_Cd_dt = active_rate - active_params.elim_rate_c() * y[2];
    return Double3(d_CGIp_dt, d_Cp_dt, d_Cd_dt);
}

Double3x3 ldx::jacobian(const Double3& y, const ProDrugPkParams& params, const MichaelisMentenParams& mm_params)
{
    auto v = (mm_params.v_max * y[1]) / (mm_params.km + y[1]);
    const auto& d_params = params.parent_drug();
    auto d_rate = d_params.form_rate_c() * (v - params.elim_rate_c());
    return Double3x3
        {
            -params.abs_rate_c(), 0, 0,
            params.abs_rate_c(), -d_rate, 0,
            0, d_rate, -d_params.elim_rate_c()
        };
}

Double3 ldx::LdxModel::operator()(const Double3& y) const
{
    return ldx_model(y, pk_params, mm_params);
}

Double3x3 ldx::LdxModel::jac(const Double3& y) const
{
    return jacobian(y, pk_params, mm_params);
}
