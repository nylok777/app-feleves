//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_NUMERIC_CUH
#define APP_FELEVES_NUMERIC_CUH
#include <cooperative_groups.h>
#include <cuda/std/tuple>
#include "numeric_funcs.cuh"

namespace cg = cooperative_groups;

class NewtonRaphsonSystem
{
public:
    NewtonRaphsonSystem(double errtol, int max_iter)
        : m_errtol(errtol),
          m_iters(max_iter) {}

    template<typename SystemFn, typename JacobianFn>
    __device__ maybe<double3> operator()(double3 y, const SystemFn& sys_fn, const JacobianFn& jac_fn) const
    {
        return newton_raphson_system(std::move(y), sys_fn, jac_fn, m_errtol, m_iters);
    }

private:
    double m_errtol;
    int m_iters;
};

template<typename F, typename Df>
class NewtonRaphson
{
public:
    NewtonRaphson(
        const F& f,
        const Df& df,
        double tolerance,
        int max_iter)
        : m_fn(f),
          m_derivfn(df),
          m_errtol(tolerance),
          m_iters(max_iter) {}

    __device__ maybe<double> operator()(double y)
    {
        return newton_raphson(y, m_fn, m_derivfn, m_errtol, m_iters);
    }

private:
    F m_fn;
    Df m_derivfn;
    double m_errtol;
    int m_iters;
};

template<typename F, typename J, typename R>
class BDF2Step
{
public:
    BDF2Step(double errtol, double step_size, F&& system_fn, J&& jacobian_fn, R&& root_solver)
        : m_system_fn(std::move(system_fn)),
          m_jacobian_fn(std::move(jacobian_fn)),
          m_root_solver(std::move(root_solver)),
          m_errtol(errtol),
          m_prev_step_size(step_size),
          m_step_size(step_size) {}

    __device__ step_result operator()(const double3& y0, double t0, double& t1, double3& y1)
    {
        t1 = t0;
        y1 = bdf1_step_adaptive(y0, t1, m_errtol, m_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        m_prev_step_size = t1 - t0;
        auto t2 = t1;
        auto y_next = bdf2_step_adaptive(y1, t2, m_step_size, m_errtol, y0, m_prev_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        return {std::move(y_next), t2};
    }

    __device__ step_result operator()(const double3& y_prev, const double3& y, double t)
    {
        auto t_next = t;
        auto y_next = bdf2_step_adaptive(y, t_next, m_step_size, m_errtol, y_prev, m_prev_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        return {std::move(y_next), t_next};
    }

private:
    F m_system_fn;
    J m_jacobian_fn;
    R m_root_solver;
    double m_errtol;
    mutable double m_prev_step_size;
    mutable double m_step_size;
};

template<typename F, typename J, typename R>
class BackwardEulerStepAdaptive
{
public:
    BackwardEulerStepAdaptive(double errtol, double step_size, F system_fn, J jacobian_fn, R root_solver)
        : m_system_fn(std::move(system_fn)),
          m_jacobian_fn(std::move(jacobian_fn)),
          m_root_solver(std::move(root_solver)),
          m_errtol(errtol),
          m_step_size(step_size) {}

    __device__ step_result operator()(const double3& y, double t) const
    {
        auto res = bdf1_step_adaptive(y, t, m_errtol, m_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        return {std::move(res), t};
    }

    __device__ double step_size() const { return m_step_size; }
    __device__ void set_step_size(double s) const { m_step_size = s; }

private:
    F m_system_fn;
    J m_jacobian_fn;
    R m_root_solver;
    double m_errtol;
    mutable double m_step_size;
};

template<typename F, typename J, typename R>
class BackwardEuler
{
public:
    BackwardEuler(double errtol, double step_size, double output_interval, F system_fn, J jacobian_fn, R root_solver, double t0 = 0.0)
        : m_stepper(errtol, step_size, std::move(system_fn), std::move(jacobian_fn), std::move(root_solver)),
          m_out_interval(output_interval), m_t(t0) {}

    __device__ solver_status operator()(
        const double3& y0, double t0, double h, double t_end, double3* y_out, double* t_out, size_t size, size_t start_step = 0)
    {
        m_t = t0;
        m_stepper.set_step_size(h);
        return (*this)(y0, t_end, y_out, t_out, size, start_step);
    }

    __device__ solver_status operator()(
        const double3& y0, double t_end, double3* y_out, size_t size, size_t start_step = 0)
    {
        size_t save_idx = start_step;
        double t_save = m_out_interval;
        y_out[save_idx] = y0;
        auto y_curr = y0;
        while (m_t < t_end && save_idx < size) {
            auto [y_next, t] = m_stepper(y_curr, m_t);
            while (t >= t_save && save_idx < size) {
                double theta = t == m_t ? 1.0 : (t_save - m_t) / (t - m_t);
                y_out[++save_idx] = y_curr + ((y_next - y_curr) * theta);
                t_save += m_out_interval;
            }
            y_curr = y_next;
            m_t = t;
        }
        return {m_t, save_idx, m_t >= t_end};
    }

private:
    BackwardEulerStepAdaptive<F, J, R> m_stepper;
    double m_out_interval;
    mutable double m_t;
};

#endif //APP_FELEVES_NUMERIC_CUH
