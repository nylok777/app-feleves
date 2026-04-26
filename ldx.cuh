//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_LDX_CUH
#define APP_FELEVES_LDX_CUH
#include "types.cuh"

namespace ldx
{

__device__ double3 ldx_model(
    const double3& y,
    const ProDrugPkParams& params,
    const MichaelisMentenParams& mm_params
);

__device__ Mat3<double> jacobian(
    const double3& y,
    const ProDrugPkParams& params,
    const MichaelisMentenParams& mm_params
);

struct LdxModel {
    __device__ double3 operator()(const double3& y);

    ProDrugPkParams pk_params;
    MichaelisMentenParams mm_params;
};

}

#endif //APP_FELEVES_LDX_CUH
