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
    __host__ __device__ NewtonRaphson(double errtol, int max_iter)
        : m_errtol(errtol),
          m_iters(max_iter) {}

    template<typename SystemFn, typename JacobianFn>
    __host__ __device__ maybe<DoubleVec<N>> operator()(DoubleVec<N> y, const SystemFn& sys_fn, const JacobianFn& jac_fn) const
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
    __host__ __device__ NewtonRaphson(double tolerance, int max_iter)
        : m_errtol(tolerance),
          m_iters(max_iter) {}

    template<typename F, typename Df>
    __host__ __device__ maybe<double> operator()(double y, F&& f, Df&& df)
    {
        return newton_raphson(y, std::forward<F>(f), std::forward<Df>(df), m_errtol, m_iters);
    }

private:
    double m_errtol;
    int m_iters;
};

template<typename F, typename J, typename R>
class BDFStepBase {
public:
    __host__ __device__ BDFStepBase(F system_fn, J jacobian_fn, R root_solver)
        : m_sys_fn(std::move(system_fn)),
          m_jac_fn(std::move(jacobian_fn)),
          m_solver_fn(std::move(root_solver))
    {}

protected:
    __host__ __device__ auto system() -> F& { return m_sys_fn; }
    __host__ __device__ auto system() const -> const F& { return m_sys_fn; }

    __host__ __device__ auto jacobian() -> J& { return m_jac_fn; }
    __host__ __device__ auto jacobian() const -> const J& { return m_jac_fn; }

    __host__ __device__ auto root_finder() -> R& { return m_solver_fn; }
    __host__ __device__ auto root_finder() const -> const R& { return m_solver_fn; }

private:
    F m_sys_fn;
    J m_jac_fn;
    R m_solver_fn;
};

template<typename F, typename J, typename R, real_number Real>
class BDFFixedStepBase : public BDFStepBase<F,J,R> {
    using Base = BDFStepBase<F,J,R>;
public:
    __host__ __device__ BDFFixedStepBase(Real step_size, F f, J jac, R root_finder)
        : Base(f, jac, root_finder), m_step_size(step_size)
    {}

    __host__ __device__ Real step_size() const { return m_step_size; }
    __host__ __device__ void set_step_size(Real h) { m_step_size = h; }

private:
    Real m_step_size;
};

template<typename F, typename J, typename R, real_number Real, int N>
class BDFStep : public BDFFixedStepBase<F,J,R,Real> {
    using Base = BDFFixedStepBase<F,J,R,Real>;
public:
    __host__ __device__ BDFStep(Real step_size, F system_fn, J jacobian_fn, R root_solver)
        : Base(step_size, system_fn, jacobian_fn, root_solver) {}

    __host__ __device__ maybe<Vector<Real,N>> operator()(const Vector<Real,N>& y) const
    {
        return bdf_step(y, Base::step_size(), Base::system(), Base::jacobian(), Base::root_finder());
    }
};

template<typename F, typename J, typename R, real_number Real, int N, int O> requires (O >= 2 && O < 7)
class BDFStep : public BDFFixedStepBase<F,J,R,Real> {
    using Base = BDFFixedStepBase<F,J,R,Real>;
    using vector = Vector<Real,N>;
public:
    __host__ __device__ BDFStep(Real step_size, F system_fn, J jacobian_fn, R root_solver)
        : Base(step_size, system_fn, jacobian_fn, root_solver) {}

    __host__ __device__
    __host__ __device__ maybe<Vector<Real,N>> operator()(const Vector<Real,N>& y0, Vector<Real,N> Y_prev_states_out[O])
    {
        auto* Y = Y_prev_states_out;
        bool success = false;

        Y[0] = y0;
        success = bdf_step(y0, Base::step_size(), Base::system(), Base::jacobian(), Base::root_finder())
            .and_then([Y](const vector& y_next) {
                Y[1] = y_next;
                return cuda::std::make_optional(true);
            })
            .or_else([] { return cuda::std::make_optional(false); })
            .value();

        if (!success) return cuda::std::nullopt;

        for (int i = 2; i < O; ++i) {
            success = bdf_step<F,J,R,Real,N,i>(Y, Base::step_size(), Base::system(), Base::jacobian(), Base::root_finder())
                .and_then([Y, i](const vector& y_next) {
                    Y[i] = y_next;
                    return cuda::std::make_optional(true);
                })
                .or_else([] { return cuda::std::make_optional(false); })
                .value();
            if (!success) return cuda::std::nullopt;
        }

        return bdf_step(Y, Base::step_size(), Base::system(), Base::jacobian(), Base::root_finder());
    }
};

template<typename F, typename J, typename R, real_number Real, int N, int O> requires (O >= 2 && O < 7)
class BDFAdaptiveStep : public BDFStepBase<F,J,R> {
    using Base = BDFStepBase<F,J,R>;
public:
    __host__ __device__ BDFAdaptiveStep(Real errtol, Real step_size, Real t0, F f, J jac, R root_finder)
        :
        Base(f, jac, root_finder), m_errtol(errtol), m_step_size(step_size), m_time(t0) {}

    __host__ __device__ Vector<Real,N> operator()(const Vector<Real,N> Y[O]) const
    {
        auto h = m_step_size;
        auto y_next = bdf_step_adaptive<>(Y, m_step_size, m_errtol, Base::system(), Base::jacobian(), Base::root_finder(), &h,
            &m_step_size);
        m_time += h;
        return y_next;
    }

    __host__ __device__ Vector<Real,N> operator()(const Vector<Real,N>& y0, Vector<Real,N> Y_prev_states_out[O])
    {
        Vector<Real,N>* Y = Y_prev_states_out;
        auto h = m_step_size;
        Y[0] = y0;
        Y[1] = bdf_step_adaptive(
            y0, m_step_size, m_errtol, Base::system(), Base::jacobian(), Base::root_finder(), &h, &m_step_size);

        for (int i = 2; i < O; ++i) {
            Y[i] = bdf_step_adaptive<F,J,R,Real,N,i>(
                Y, m_step_size, m_errtol, Base::system(), Base::jacobian(), Base::root_finder(), &h, &m_step_size);
            m_time += h;
        }
        auto y = bdf_step_adaptive(Y, m_step_size, m_errtol, Base::system(), Base::jacobian(), Base::root_finder(), &h, &m_step_size);
        m_time += h;
        return y;
    }

    __host__ __device__ Real step_size() const { return m_step_size; }
    __host__ __device__ Real time() const { return m_time; }
    __host__ __device__ Real errtol() const { return m_errtol; }

    __host__ __device__ void set_step_size(Real h) { m_step_size = h; }
    __host__ __device__ void set_errtol(Real tol) { m_errtol = tol; }

    __host__ __device__ void reset(Real step_size, Real t0) const
    {
        m_step_size = step_size;
        m_time = t0;
    }

private:
    Real m_errtol;
    mutable Real m_step_size;
    mutable Real m_time;
};

template<typename F, typename J, typename R, int N>
class BackwardEulerStep : public BDFStepBase<F,J,R> {
    using Base = BDFStepBase<F,J,R>;
public:
    __host__ __device__ BackwardEulerStep(double step_size, F system_fn, J jacobian_fn, R root_solver)
        : Base(system_fn, jacobian_fn, root_solver), m_step_size(step_size) {}

    __host__ __device__ maybe<DoubleVec<N>> operator()(const DoubleVec<N>& y) const
    {
        return bdf1_step(y, Base::step_size(), Base::system(), Base::jacobian(), Base::root_finder());
    }

    __host__ __device__ double step_size() const { return m_step_size; }
    __host__ __device__ void set_step_size(double h) { m_step_size = h; }

private:
    double m_step_size;
};

template<typename F, typename J, typename R, int N>
class BackwardEulerStepAdaptive {
public:
    BackwardEulerStepAdaptive(double errtol, double step_size, double t0, F system_fn, J jacobian_fn, R root_solver)
        : m_system_fn(std::move(system_fn)),
          m_jacobian_fn(std::move(jacobian_fn)),
          m_root_solver(std::move(root_solver)),
          m_errtol(errtol),
          m_step_size(step_size) {}

    __device__ step_result<N> operator()(const DoubleVec<N>& y, double t) const
    {
        auto res = bdf1_step_adaptive(y, m_errtol, m_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        return {std::move(res), t};
    }

    __device__ double step_size() const { return m_step_size; }
    __device__ void set_step_size(double h) const { m_step_size = h; }

private:
    F m_system_fn;
    J m_jacobian_fn;
    R m_root_solver;
    double m_errtol;
    mutable double m_step_size;
    mutable double m_time;
};

template<typename F, typename J, typename R, int N>
class BackwardEulerAdaptive {
public:
    BackwardEulerAdaptive(double errtol, double step_size, double output_interval, F system_fn, J jacobian_fn, R root_solver, double t0 = 0.0)
        : m_stepper(errtol, step_size, std::move(system_fn), std::move(jacobian_fn), std::move(root_solver)),
          m_out_interval(output_interval),
          m_t(t0) {}

    __device__ solver_status operator()(
        const DoubleVec<N>& y0, double t0, double h, double t_end, DoubleVec<N>* y_out, double* t_out, size_t size, size_t start_step = 0)
    {
        m_t = t0;
        m_stepper.set_step_size(h);
        return (*this)(y0, t_end, y_out, t_out, size, start_step);
    }

    __device__ solver_status operator()(
        const DoubleVec<N>& y0, double t_end, DoubleVec<N>* y_out, size_t size, size_t start_step = 0)
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

template<typename F, typename J, typename R, int N>
class BDF2StepAdaptive {
public:
    __host__ __device__ BDF2StepAdaptive(double errtol, double step_size, double t0, F&& system_fn, J&& jacobian_fn, R&& root_solver)
        :
        m_system_fn(std::move(system_fn)),
        m_jacobian_fn(std::move(jacobian_fn)),
        m_root_solver(std::move(root_solver)),
        m_errtol(errtol),
        m_step_size_prev(step_size),
        m_step_size(step_size),
        m_time(t0) {}

    __host__ __device__ step_result<N> operator()(const DoubleVec<N>& y0, double* t1, DoubleVec<N>* y1) const
    {
        *y1 = bdf1_step_adaptive(y0, m_errtol, m_step_size, m_system_fn, m_jacobian_fn, m_root_solver);
        double t_n = m_time + m_step_size;
        auto y_next = bdf2_step_adaptive(*y1, m_step_size, m_errtol, y0, m_step_size_prev, m_system_fn, m_jacobian_fn, m_root_solver);
        *t1 = t_n;
        m_time = t_n + m_step_size_prev;
        return {std::move(y_next), m_time};
    }

    __host__ __device__ step_result<N> operator()(const DoubleVec<N>& y_prev, const DoubleVec<N>& y) const
    {
        auto y_next = bdf2_step_adaptive(
            y,
            m_step_size,
            m_errtol,
            y_prev,
            m_step_size_prev,
            m_system_fn,
            m_jacobian_fn,
            m_root_solver);
        return {std::move(y_next), (m_time += m_step_size_prev)};
    }

    __host__ __device__ double time() const { return m_time; }
    __host__ __device__ double step_size() const { return m_step_size; }
    // ReSharper disable CppMemberFunctionMayBeConst
    __host__ __device__ void set_step_size(const double h) { m_step_size = h; }
    __host__ __device__ void reset(const double step_size, const double t0 = 0.0)
    {
        m_time = 0;
        m_step_size = step_size;
        m_step_size_prev = step_size;
    }
    // ReSharper restore CppMemberFunctionMayBeConst

private:
    F m_system_fn;
    J m_jacobian_fn;
    R m_root_solver;
    double m_errtol;
    mutable double m_step_size_prev;
    mutable double m_step_size;
    mutable double m_time;
};

template<typename F, typename J, int S>
class RosenbrockStep {
public:
    RosenbrockStep(double step_size, double errtol, rosenbrock_coefficients<S> coeffs, F sys_fn, J jac)
        : m_coefficients(std::move(coeffs)),
          m_sys_fn(std::move(sys_fn)),
          m_jac(std::move(jac)),
          m_h(step_size),
          m_errtol(errtol) {}

    template<int N>
    __host__ __device__ DoubleVec<N> operator()(const DoubleVec<N>& y) const
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
