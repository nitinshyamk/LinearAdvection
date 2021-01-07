// linear_advection_kernels.cu: Kernelized Euler Step
//
#include "../include/linear_advection_kernels.cuh"

__global__ void euler_step_state_krnl(double* prevstate, double* state, size_t M, double adv_coeff, double diff_coeff)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double* fo = prevstate, * f = state;
	
	if (i == 0)
	{
		f[0] = fo[M - 1];
		return;
	}

	if (i == M - 1)
	{
		// boundary conditions seem weird? is this right?
		f[M - 1] = compute_individual_step_krnl(fo[M - 2], fo[M - 1], fo[1], adv_coeff, diff_coeff);
		return;
	}

	if (i < M)
	{
		f[i] = compute_individual_step_krnl(fo[i - 1], fo[i], fo[i + 1], adv_coeff, diff_coeff);
	}

}

__device__ double compute_individual_step_krnl(double left, double self, double right, double adv_coeff, double diff_coeff)
{
	return self - adv_coeff * (right - left) + diff_coeff * (left + right - 2 * self);
}

__global__ void initialize_state_krnl(double* state, size_t M, double A, double kh)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < M)
	{
		state[i] = A * sinpi(2 * kh * (i - 1));
	}
}

__global__ void compute_analytical_solution_krnl(double* state, size_t M, double pi, double A, double k, double D, double t, double h, double v)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < M)
	{
		state[i] = A * exp(-4 * pi * pi * k * k * D * t) * sinpi(2 * k * (h * (i - 1) - v * t));
	}
}

__global__ void compute_err_krnl(double* state, double* solution, double* err, size_t M)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < M)
	{
		double e = state[i] - solution[i];
		err[i] = e * e;
	}
}
