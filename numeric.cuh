//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_NUMERIC_CUH
#define APP_FELEVES_NUMERIC_CUH
#include <cooperative_groups.h>
#include "numeric_funcs.cuh"

namespace cg = cooperative_groups;

template<>
class NewtonRaphsonSystem
{
public:
    NewtonRaphsonSystem(double errtol, int max_iter)
        : m_errtol(errtol),
          m_iters(max_iter) {}

    template<typename SystemFn, typename JacobianFn>
    __device__ maybe<double3> operator()(double3 y, const SystemFn& sys_fn, const JacobianFn& jac_fn)
    {
        return newton_raphson_system(std::move(y), sys_fn, jac_fn, m_errtol, m_iters);
    }

private:
    double m_errtol;
    int m_iters;
};

template<typename SystemFn, typename JacobianFn>
class NewtonRaphsonSystem
{
public:
    NewtonRaphsonSystem(
        const SystemFn& sys_fn,
        const JacobianFn& jac_fn,
        double errtol,
        int max_iter)
        : m_system_fn(sys_fn),
          m_jacobian_fn(jac_fn),
          m_errtol(errtol),
          m_iters(max_iter) {}

    __device__ maybe<double3> operator()(double3 y)
    {
        return newton_raphson_system(std::move(y), m_system_fn, m_jacobian_fn, m_errtol, m_iters);
    }

private:
    SystemFn m_system_fn;
    JacobianFn m_jacobian_fn;
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

    __device__ step_result<2> operator()(const double3& y0, double t0)
    {
        auto t = t0;
        auto y = bdf1_step(y0, t, m_errtol, m_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        auto y_next = bdf2_step(y, t, m_step_size, m_errtol, y0, m_prev_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        return {std::move(y), std::move(y_next), t};
    }

    __device__ step_result<2> operator()(const double3& y_prev, const double3& y, double t)
    {
        auto y_next = bdf2_step(y, t, m_step_size, m_errtol, y_prev, m_prev_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        return {y, std::move(y_next), t};
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
class BackwardEulerStep
{
public:
    BackwardEulerStep(double errtol, double step_size, F system_fn, J jacobian_fn, R root_solver)
        : m_system_fn(std::move(system_fn)),
          m_jacobian_fn(std::move(jacobian_fn)),
          m_root_solver(std::move(root_solver)),
          m_errtol(errtol),
          m_step_size(step_size) {}

    __device__ step_result<1> operator()(const double3& y, double t) const
    {
        auto res = bdf1_step(y, t, m_errtol, m_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
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
    BackwardEuler(double errtol, double step_size, F system_fn, J jacobian_fn, R root_solver, double t0 = 0.0)
        : m_stepper(errtol, step_size, std::move(system_fn), std::move(jacobian_fn), std::move(root_solver)),
          m_t(t0) {}

    __device__ solver_status operator()(
        const double3& y0, double t0, double h, double t_end, double3* y_out, double* t_out, size_t size, size_t start_step = 0)
    {
        m_t = t0;
        m_stepper.set_step_size(h);
        return (*this)(y0, t_end, y_out, t_out, size, start_step);
    }

    __device__ solver_status operator()(
        const double3& y0, double t_end, double3* y_out, double* t_out, size_t size, size_t start_step = 0)
    {
        size_t step = start_step;
        y_out[step] = y0;
        t_out[step++] = m_t;
        while (m_t < t_end && step < size) {
            if (m_t + m_stepper.step_size())
                m_stepper.set_step_size(t_end - m_t);

            auto res = m_stepper(y_out[step - 1], m_t);
            m_t = res.t;
            t_out[step] = m_t;
            y_out[step++] = res.y;
        }
        return {m_t, step, m_t >= t_end};
    }

    __device__ step_result<1> step(const double3& y)
    {
        auto res = m_stepper(y, m_t);
        m_t = res.t;
        return res;
    }

private:
    BackwardEulerStep<F, J, R> m_stepper;
    mutable double m_t;
};

template<typename G, typename F>
__device__ void parareal(
    const double3& y0,
    double t_start,
    double t_end,
    double3* y_out,
    double* t_out,
    double3* f_values,
    double3* g_values,
    size_t size,
    double errtol,
    int max_iters,
    const G& coarse_solver,
    const F& fine_stepper,
    bool* converged)
{
    cg::grid_group group = cg::this_grid();

    auto tid = group.thread_rank();
    auto group_size = group.size();
    if (tid == 0) {
        *converged = false;
        coarse_solver(y0, t_start, t_end, y_out, t_out, size);
    }

    group.sync();

    for (int k = 0; k < max_iters; ++k) {
        if (*converged) break;

        for (int i = tid; i < size - 1; i += group_size) {
            if (i == 0) {
                f_values[i] = fine_stepper(y_out[i], t_out[i]).y;
                g_values[i] = coarse_solver.step(y_out[i], t_out[i]).y;
            }
            else {
                f_values[i] = fine_stepper(y_out[i - 1], y_out[i], t_out[i]).y;
                g_values[i] = coarse_solver.step(y_out[i], t_out[i]).y;
            }
        }
        group.sync();

        if (tid == 0) {
            double max_error = 0.0;

            for (size_t i = 0; i < size - 1; ++i) {
                auto y_prev = y_out[i + 1];
                auto [g_new_y, g_new_t] = coarse_solver.step(y_out[i], t_out[i]);
                y_out[i + 1] = g_new_y + f_values[i] - g_values[i];
                auto error = d3abs(y_out[i + 1] - y_prev);
                max_error = fmax(max_error, fmax(error.x, fmax(error.y, error.z)));
            }
            *converged = max_error < errtol;
        }
    }
    group.sync();
}

#endif //APP_FELEVES_NUMERIC_CUH
