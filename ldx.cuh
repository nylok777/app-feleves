//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_LDX_CUH
#define APP_FELEVES_LDX_CUH
#include "types.cuh"

namespace ldx
{

__device__ Double3 ldx_model(
    const Double3& y,
    const ProDrugPkParams& params,
    const MichaelisMentenParams& mm_params
);

__device__ Double3x3 jacobian(
    const Double3& y,
    const ProDrugPkParams& params,
    const MichaelisMentenParams& mm_params
);

struct LdxModel {
    __device__ Double3 operator()(const Double3& y) const;
    __device__ Double3x3 jac(const Double3& y) const;

    ProDrugPkParams pk_params;
    MichaelisMentenParams mm_params;
};

}

#endif //APP_FELEVES_LDX_CUH
