//
// Created by david on 4/26/26.
//

#ifndef APP_FELEVES_NUMERIC_CUH
#define APP_FELEVES_NUMERIC_CUH
#include <cooperative_groups.h>
#include "solver_funcs.cuh"

namespace cg = cooperative_groups;

template<int N>
class NewtonRaphson {
public:
    NewtonRaphson(double errtol, int max_iter)
        : m_errtol(errtol),
          m_iters(max_iter) {}

    template<typename SystemFn, typename JacobianFn>
    __device__ maybe<DoubleND<N>> operator()(DoubleND<N> y, const SystemFn& sys_fn, const JacobianFn& jac_fn) const
    {
        return newton_raphson(std::move(y), sys_fn, jac_fn, m_errtol, m_iters);
    }

private:
    double m_errtol;
    int m_iters;
};

template<>
class NewtonRaphson<1> {
public:
    NewtonRaphson(
        double tolerance,
        int max_iter)
        : m_errtol(tolerance),
          m_iters(max_iter) {}

    template<typename F, typename Df>
    __device__ maybe<double> operator()(double y, F&& f, Df&& df)
    {
        return newton_raphson(y, std::forward<F>(f), std::forward<Df>(df), m_errtol, m_iters);
    }

private:
    double m_errtol;
    int m_iters;
};

template<typename F, typename J, typename R, int N>
class BDF2Step {
public:
    BDF2Step(double errtol, double step_size, F&& system_fn, J&& jacobian_fn, R&& root_solver)
        : m_system_fn(std::move(system_fn)),
          m_jacobian_fn(std::move(jacobian_fn)),
          m_root_solver(std::move(root_solver)),
          m_errtol(errtol),
          m_prev_step_size(step_size),
          m_step_size(step_size) {}

    __device__ step_result<N> operator()(const DoubleND<N>& y0, double t0, double& t1, DoubleND<N>& y1)
    {
        t1 = t0;
        y1 = bdf1_step_adaptive(y0, t1, m_errtol, m_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        m_prev_step_size = t1 - t0;
        auto t2 = t1;
        auto y_next = bdf2_step_adaptive(y1, t2, m_step_size, m_errtol, y0, m_prev_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        return {std::move(y_next), t2};
    }

    __device__ step_result<N> operator()(const DoubleND<N>& y_prev, const DoubleND<N>& y, double t)
    {
        auto t_next = t;
        auto y_next = bdf2_step_adaptive(
            y,
            t_next,
            m_step_size,
            m_errtol,
            y_prev,
            m_prev_step_size,
            m_system_fn,
            m_jacobian_fn,
            m_root_solver);
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

template<typename F, typename J, typename R, int N>
class BackwardEulerStep {
public:
    BackwardEulerStep(double h, F system_fn, J jacobian_fn, R root_solver)
        :
        m_sys_fn(std::move(system_fn)),
        m_jac_fn(std::move(jacobian_fn)),
        m_solver_fn(std::move(root_solver)),
        m_step_size(h)
    {}

    __device__ maybe<DoubleND<N>> operator()(const DoubleND<N>& y) const
    {
        return bdf1_step(y, m_step_size, m_sys_fn, m_jac_fn, m_solver_fn);
    }
    __device__ double step_size() const { return m_step_size; }
    __device__ void set_step_size(double h) { m_step_size = h; }

private:
    F m_sys_fn;
    J m_jac_fn;
    R m_solver_fn;
    double m_step_size;
};

template<typename F, typename J, typename R, int N>
class BackwardEulerStepAdaptive {
public:
    BackwardEulerStepAdaptive(double errtol, double step_size, F system_fn, J jacobian_fn, R root_solver)
        : m_system_fn(std::move(system_fn)),
          m_jacobian_fn(std::move(jacobian_fn)),
          m_root_solver(std::move(root_solver)),
          m_errtol(errtol),
          m_step_size(step_size) {}

    __device__ step_result<N> operator()(const DoubleND<N>& y, double t) const
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

template<typename F, typename J, typename R, int N>
class BackwardEuler {
public:
    BackwardEuler(double errtol, double step_size, double output_interval, F system_fn, J jacobian_fn, R root_solver, double t0 = 0.0)
        : m_stepper(errtol, step_size, std::move(system_fn), std::move(jacobian_fn), std::move(root_solver)),
          m_out_interval(output_interval),
          m_t(t0) {}

    __device__ solver_status operator()(
        const DoubleND<N>& y0, double t0, double h, double t_end, DoubleND<N>* y_out, double* t_out, size_t size, size_t start_step = 0)
    {
        m_t = t0;
        m_stepper.set_step_size(h);
        return (*this)(y0, t_end, y_out, t_out, size, start_step);
    }

    __device__ solver_status operator()(
        const DoubleND<N>& y0, double t_end, DoubleND<N>* y_out, size_t size, size_t start_step = 0)
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
    BackwardEulerStepAdaptive<F, J, R, N> m_stepper;
    double m_out_interval;
    mutable double m_t;
};

template<typename F, typename J, int S>
class RosenbrockStep {
public:
    RosenbrockStep(double step_size, double errtol, rosenbrock_coefficients<S> coeffs, F sys_fn, J jac)
        :
        m_coefficients(std::move(coeffs)),
        m_sys_fn(std::move(sys_fn)),
        m_jac(std::move(jac)),
        m_h(step_size),
        m_errtol(errtol)
    {}

    template<int N>
    __host__ __device__ DoubleND<N> operator()(const DoubleND<N>& y) const
    {
        return rosenbrock_method_step(y, m_h, m_coefficients, m_sys_fn, m_jac);
    }

    __host__ __device__ double step_size() const { return m_h; }
    __host__ __device__ void set_step_size(double h) { m_h = h; }

private:
    rosenbrock_coefficients<S> m_coefficients;
    F m_sys_fn;
    J m_jac;
    double m_h;
    double m_errtol;
};

#endif //APP_FELEVES_NUMERIC_CUH
