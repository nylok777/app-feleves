//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_NUMERIC_CUH
#define APP_FELEVES_NUMERIC_CUH
#include "numeric_funcs.cuh"

template<typename SystemFn, typename JacobianFn>
class NewtonRaphsonSystem {
public:
    NewtonRaphsonSystem(
        const SystemFn& sys_fn,
        const JacobianFn& jac_fn,
        double errtol,
        int max_iter
    );
    __device__ double3 operator()(double3 y);
};

template<typename F, typename Df>
class NewtonRaphson {
public:
    NewtonRaphson(
        const F& f,
        const Df& df,
        double tolerance,
        int max_iter
    );
    __device__ double operator()(double y);
};

template<typename R>
class BDF2Step {
public:
    BDF2Step(
        R&& solver,
        double errtol,
        double step_size
    );
    __device__ double3 operator()(
        const double3& y,
        const double3& y_prev,
        double t_next
    );

private:
    double m_errtol;
    mutable double m_step_size;
    R m_root_solver;
};

template<typename R>
class BackwardEuler {
public:
    BackwardEuler(
        R&& solver,
        double errtol,
        double step_size
    );
    __device__ double3 operator()(
        const double3& y,
        double t_next
    );

private:
    R m_root_solver;
    double m_errtol;
    mutable double m_step_size;
};

template<typename G, typename F>
__device__ void parareal(
    double t_start,
    double t_end,
    double* out,
    const G& coarse_solver,
    const F& fine_solver
);

#endif //APP_FELEVES_NUMERIC_CUH
