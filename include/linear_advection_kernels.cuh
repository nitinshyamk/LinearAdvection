#ifndef LINEAR_ADVECTION_KERNELS_CUH
#define LINEAR_ADVECTION_KERNELS_CUH

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <float.h>

__global__ void euler_step_state_krnl(double* prevstate, double* state, size_t M, double adv_coeff, double diff_coeff);

__device__ double compute_individual_step_krnl(double left, double self, double right, double adv_coeff, double diff_coeff);

__global__ void initialize_state_krnl(double* state, size_t M, double adv_coeff, double diff_coeff);

__global__ void compute_analytical_solution_krnl(double* state, size_t M, double pi, double A, double k, double D, double t, double h, double v);

__global__ void compute_err_krnl(double* state, double* solution, double* errv, size_t M);

#endif LINEAR_ADVECTION_KERNELS_CUH