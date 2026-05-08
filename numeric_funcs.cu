//
// Created by david on 5/8/26.
//

#include "numeric_funcs.cuh"

__device__ double3 solve_linear_system(Double3x3 a, double* b)
{
    for (int i = 0; i < 3; ++i) {
        int max_row = i;
        for (int j = i + 1; j < 3; ++j) {
            if (abs(a(j, i)) > abs(a(max_row, i)))
                max_row = j;
        }
        for (int j = 0; j < 3; ++j) {
            auto tmp = a(i,j);
            a(i,j) = a(max_row,j);
            a(max_row,j) = tmp;
        }
        auto tmp = b[i];
        b[i] = b[max_row];
        b[max_row] = tmp;

        for (int j = i + 1; j < 3; ++j) {
            auto factor = a(j, i) / a(i, i);
            for (int k = i; k < 3; ++k) {
                a(j, k) -= factor * a(i, k);
            }
            b[j] -= factor * b[i];
        }
    }

    double x[3];
    for (double& i : x) i = 0.0;
    for (int i = 2; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < 3; ++j) {
            sum += a(i, j) * x[j];
        }
        x[i] = (b[i] - sum) / a(i, i);
    }
    return make_double3(x[0], x[1], x[2]);
}
