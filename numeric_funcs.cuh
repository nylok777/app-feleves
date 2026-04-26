//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_NUMERIC_FUNCS_CUH
#define APP_FELEVES_NUMERIC_FUNCS_CUH
template<typename SystemFn, typename JacobianFn>
__device__ double3 newton_raphson_system(
    double3 y,
    const SystemFn& system_fn,
    const JacobianFn& jac_fn,
    double tolerance,
    int max_iter
);

template<typename F, typename Df>
__device__ double newton_raphson(
    double y,
    const F& f,
    const Df& df,
    double tolerance,
    int max_iter)
{
    
}

template<typename R>
__device__ double3 bdf1_step(
    const double3& y,
    double t_next,
    double errtol,
    double* step_size,
    const R& root_finder
);
#endif //APP_FELEVES_NUMERIC_FUNCS_CUH
