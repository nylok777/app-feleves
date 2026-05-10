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
    BackwardEulerStep<F, J, R> m_stepper;
    double m_out_interval;
    mutable double m_t;
};

template<typename F, typename J, typename R>
class Parareal
{
public:
    Parareal(BackwardEulerStep<F,J,R> coarse_stepper, BDF2Step<F,J,R> fine_stepper, double errtol, double t_save, int max_iters)
        :
        m_coarse_stepper(std::move(coarse_stepper)),
        m_fine_stepper(std::move(fine_stepper)),
        m_errtol(errtol),
        m_t_save(t_save),
        m_max_iters(max_iters) {}

    __device__ void operator()(const double3& y0, const double t0, const double tf, double3* y_out, double3* f_values, double3* g_values,
        bool* converged)
    {
        cg::grid_group group = cg::this_grid();
        const auto tid = group.thread_rank();
        const auto group_size = group.size();

        double t_curr;
        double3 y_curr;

        if (tid == 0) {
            *converged = false;
            size_t idx = 0;
            double t_out = m_t_save;
            t_curr = t0;
            y_curr = y0;
            y_out[idx] = y0;
            while (t_curr < tf && idx < group_size) {
                auto [y_next, t_next] = m_coarse_stepper(y_curr, t_curr);
                while (t_out < t_next && idx < group_size) {
                    y_out[++idx] = linear_interpolation(y_curr, y_next, t_curr, t_next, t_out);
                    t_out += m_t_save;
                }
                y_curr = y_next;
                t_curr = t_next;
            }
        }

        group.sync();

        double t_prev = 0.0;
        double3 y_prev;
        if (tid > 0) {
            t_prev = m_t_save * static_cast<double>(tid - 1);
            y_prev = y_out[tid - 1];
        }
        t_curr = m_t_save * static_cast<double>(tid);
        y_curr = y_out[tid];

        for (int k = 0; !*converged && k < m_max_iters; ++k) {
            if (tid < group_size) {
                double3 y_n = y_curr, y = y_curr;
                double t_n = t_curr, t = t_curr;

                auto res = m_fine_stepper(y_curr, t_curr, t, y);
                double3 y_next = std::move(res.y);
                double t_next = res.t;

                while (t < t_curr + m_t_save) {
                    if (t_next <= t) {
                        f_values[tid] = y_next;
                        break;
                    }

                    // FIX 2: Check if this is the first step of the slice (t_n == t_curr)
                    if (t_n == t_curr && t_next >= t_curr + m_t_save) {
                        f_values[tid] = linear_interpolation(y, y_next, t, t_next, t_curr + m_t_save);
                        break;
                    }
                    if (t_next >= t_curr + m_t_save) {
                        f_values[tid] = quadratic_interpolation(y_n, y, y_next, t_n, t, t_next, t_curr + m_t_save);
                        break;
                    }

                    res = m_fine_stepper(y, y_next, t_next);
                    y_n = y;
                    y = y_next;
                    y_next = std::move(res.y);
                    t_n = t;
                    t = t_next;
                    t_next = res.t;
                }
            }

            group.sync();

            if (tid == 0) {
                double max_error = 0.0;

                for (size_t i = 0; i < group_size; ++i) {
                    auto y_old = y_out[i + 1];
                    double t_slice = static_cast<double>(i) * m_t_save;
                    double t_cg = t_slice;
                    double3 y_cg = y_out[i];
                    double3 g_new_y;

                    while (t_cg < t_slice + m_t_save) {
                        auto [y_next, t_next] = m_coarse_stepper(y_cg, t_cg);
                        if (t_next <= t_cg) {
                            g_new_y = y_next;
                            break;
                        }
                        if (t_next >= t_slice + m_t_save) {
                            g_new_y = linear_interpolation(y_cg, y_next, t_cg, t_next, t_slice + m_t_save);
                            break;
                        }
                        y_cg = y_next;
                        t_cg = t_next;
                    }

                    y_out[i + 1] = g_new_y + f_values[i] - g_values[i];
                    g_values[i] = g_new_y;
                    auto error = d3abs(y_out[i + 1] - y_old);
                    max_error = fmax(max_error, fmax(error.x, fmax(error.y, error.z)));
                }
                *converged = max_error < m_errtol;
            }

            group.sync();

            if (tid > 0 && tid < group_size)
                y_prev = y_out[tid - 1];
            y_curr = y_out[tid];
        }
        group.sync();
    }

private:
    BackwardEulerStep<F,J,R> m_coarse_stepper;
    BDF2Step<F,J,R> m_fine_stepper;
    double m_errtol;
    double m_t_save;
    int m_max_iters;
};

#endif //APP_FELEVES_NUMERIC_CUH
