//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_NUMERIC_CUH
#define APP_FELEVES_NUMERIC_CUH
#include <cooperative_groups.h>
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
        y1 = bdf1_step(y0, t1, m_errtol, m_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        auto t2 = t1;
        auto y_next = bdf2_step(y1, t2, m_step_size, m_errtol, y0, m_prev_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        return {std::move(y_next), t2};
    }

    __device__ step_result operator()(const double3& y_prev, const double3& y, double t)
    {
        auto t_next = t;
        auto y_next = bdf2_step(y, t_next, m_step_size, m_errtol, y_prev, m_prev_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
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
class BDF2
{
public:
    BDF2(F sys_fn, J jac_fn, R root_solver, double errtol, double step_size, double out_interval, double t0 = 0.0)
        :
        m_stepper(errtol, step_size, std::move(sys_fn), std::move(jac_fn), std::move(root_solver)),
        m_t(t0),
        m_t_prev(t0),
        m_out_interval(out_interval) {}

    __device__ solver_status operator()(const double3& y0, double t_end, double3* y_out, size_t size, size_t start_idx = 0)
    {
        size_t save_idx = start_idx;

        step(y0, y_out, save_idx, size);

        while (m_t < t_end && save_idx < size) {
            step(y_out, save_idx, size);
        }
        return {m_t, save_idx, m_t >= t_end};
    }

    __device__ void step(const double3& y0, double3* y_out, size_t& idx, size_t size)
    {
        y_out[idx] = y0;
        auto [y_next, t_next] = m_stepper(y0, m_t_prev, m_t, m_y);

        while (t_next >= m_t_save && idx < size) {
            y_out[++idx] = linear_interpolation(m_y, y_next, m_t, t_next, m_t_save);
            m_t_save += m_out_interval;
        }
        m_t_prev = m_t;
        m_t = t_next;
        m_y_prev = m_y;
        m_y = y_next;
    }

    __device__ void step(double3* y_out, size_t& idx, size_t size)
    {
        auto [y_next, t_next] = m_stepper(m_y_prev, m_y, m_t);

        while (t_next >= m_t_save && idx < size) {
            y_out[++idx] = quadratic_interpolation(m_y_prev, m_y, y_next, m_t_prev, m_t, t_next, m_t_save);
            m_t_save += m_out_interval;
        }
        m_y_prev = m_y;
        m_y = y_next;
        m_t_prev = m_t;
        m_t = t_next;
    }

private:
    BDF2Step<F, J, R> m_stepper;
    mutable double3 m_y_prev{};
    mutable double3 m_y{};
    mutable double m_t;
    mutable double m_t_prev;
    double m_out_interval;
    mutable double m_t_save = m_out_interval;
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

    __device__ step_result operator()(const double3& y, double t) const
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
    BackwardEuler(double errtol, double step_size, double output_interval, F system_fn, J jacobian_fn, R root_solver, double t0 = 0.0)
        : m_stepper(errtol, step_size, std::move(system_fn), std::move(jacobian_fn), std::move(root_solver)),
          m_out_interval(output_interval), m_t(t0) {}

    __device__ solver_status operator()(
        const double3& y0, double t_end, double3* y_out, size_t size, size_t start_idx = 0)
    {
        size_t save_idx = start_idx;
        y_out[save_idx] = y0;
        m_y = y0;
        while (m_t < t_end && save_idx < size) {
            step(y_out, save_idx, size);
        }
        return {m_t, save_idx, m_t >= t_end};
    }

    __device__ void step(const double3& y0, double3* y_out, size_t& idx, size_t size)
    {
        m_y = y0;
        y_out[idx] = y0;
        step(y_out, idx, size);
    }

    __device__ void step(double3* y_out, size_t& idx, size_t size)
    {
        auto [y_next, t_next] = m_stepper(m_y, m_t);
        while (t_next >= m_t_save && idx < size) {
            y_out[++idx] = linear_interpolation(m_y, y_next, m_t, t_next, m_t_save);
            m_t_save += m_out_interval;
        }
        m_y = y_next;
        m_t = t_next;
    }

private:
    BackwardEulerStep<F, J, R> m_stepper;
    mutable double3 m_y{};
    double m_out_interval;
    mutable double m_t;
    mutable double m_t_save = m_out_interval;
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
