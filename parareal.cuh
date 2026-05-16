//
// Created by david on 5/16/26.
//

#ifndef APP_FELEVES_PARAREAL_CUH
#define APP_FELEVES_PARAREAL_CUH
#include <cooperative_groups.h>

#include "numeric_funcs.cuh"
#include "types.cuh"

namespace cg = cooperative_groups;

namespace detail
{
template<typename G, int N>
__host__ __device__ int parareal_initial_solve(
    const DoubleND<N>& y0,
    const G& coarse_step,
    DoubleND<N>* U,
    const size_t size)
{
    using coarse_result = std::invoke_result_t<G, DoubleND<N>>;
    // szekvenciálisan kell ezt a részt GPU-n is !!
    auto y = y0;
    for (size_t i = 0; i < size; ++i) {
        if constexpr (std::same_as<coarse_result, maybe<DoubleND<N>>>) {
            auto y_maybe = coarse_step(y);
            if (!y_maybe.has_value()) return -2;
            y = y_maybe.value();
        }
        else {
            y = coarse_step(y);
        }
        U[i] = y;
    }
    return 0;
}

template<typename F, int N>
__device__ void parareal_fine_steps(
    const DoubleND<N>& y0,
    const F& fine_step,
    DoubleND<N>* U_pred,
    DoubleND<N>* U_f,
    const size_t size)
{
    const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid == 0)
        U_f[tid] = fine_step(y0);
    else if (tid < size)
        U_f[tid] = fine_step(U_pred[tid - 1]);
}

template<int N, typename G>
__host__ __device__ int parareal_correction(
    const G& coarse_step,
    DoubleND<N>* U_g,
    DoubleND<N>* U_f,
    DoubleND<N>* U_next,
    const size_t size)
{
    using vector = DoubleND<N>;
    using coarse_result = std::invoke_result_t<G, vector>;

    vector u = U_f[0]; // first element
    U_next[0] = u;
    for (size_t i = 1; i < size; ++i) {
        vector G_u;
        vector G_u_prev;
        if constexpr (std::same_as<coarse_result, maybe<vector>>) {
            auto G_u_maybe = coarse_step(u);
            auto G_u_prev_maybe = coarse_step(U_g[i - 1]);
            if (!G_u_maybe.has_value() || !G_u_prev_maybe.has_value()) return -2;
            G_u = G_u_maybe.value();
            G_u_prev = G_u_prev_maybe.value();
        }
        else {
            G_u = coarse_step(u);
            G_u_prev = coarse_step(U_g[i - 1]);
        }

        u = G_u + U_f[i] - G_u_prev;
        U_next[i] = u;
    }
    return 0;
}

template<int N>
__host__ __device__ void parareal_check_convergence(
    const DoubleND<N>* U_prev,
    const DoubleND<N>* U,
    const size_t size,
    const double errtol,
    bool* converged)
{
#ifdef __CUDA_ARCH__
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        auto err = l2norm(U[tid] - U_prev[tid]);
        // printf("%f\n", err);
        if (err > errtol) *converged = false;
    }
#else
    for (size_t i = 0; i < size; ++i) {
        if (l2norm(U[i] - U_prev[i]) > errtol) {
            *converged = false;
            return;
        }
    }
    *converged = true;
#endif

}

__device__ inline int dev_parareal_exit_flag = 0;
}

template<typename G, typename F, int N>
__global__ void parareal_loop(
    const DoubleND<N> y0,
    G coarse_step,
    F fine_step,
    const double errtol,
    DoubleND<N>* U_prev,
    DoubleND<N>* U,
    DoubleND<N>* U_next,
    const size_t size,
    bool* converged)
{
    const auto grid = cg::this_grid();

    const auto tx = threadIdx.x;
    const auto tid = blockDim.x * blockIdx.x + tx;

    if (tid == 0) {
        *converged = false;
        // printf("%s", "at initial solve");
        detail::parareal_initial_solve(y0, coarse_step, U_prev, size);
        // printf("%s", "after initial solve\n");
    }
    grid.sync();

    while (!*converged) {
        grid.sync();
        if (tid == 0) *converged = true;
        grid.sync();
        if (tid < size) {
            // if (tid == 0) printf("%s", "at fine steps");
            detail::parareal_fine_steps(y0, fine_step, U_prev, U, size);
        }
        grid.sync();
        if (tid == 0) {
            // printf("%s", "at correction");
            detail::parareal_correction(coarse_step, U_prev, U, U_next, size);
            // printf("%s", "after correction\n");
        }
        grid.sync();
        detail::parareal_check_convergence(U_prev, U_next, size, errtol, converged);
        grid.sync();
        if (!*converged) {
            auto* tmp = U_prev;
            U_prev = U_next;
            U_next = tmp;
        }
        grid.sync();
    }
}
#endif //APP_FELEVES_PARAREAL_CUH
