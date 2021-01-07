#ifndef LINEAR_ADVECTION_CUH
#define LINEAR_ADVECTION_CUH

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "../include/cudpp.h"
#include "../include/cuda_gpu_matrix.h"
#include "../include/cuda_gpu_vector.h"
#include "../include/linear_algebra.cuh"
#include "../include/utilities.h"

#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

#include "params.h"
#include "cudpp_library.h"
#include "recorders.h"
#include "util.h"
#include "linear_advection_kernels.cuh"

class linear_advection
{
public:

	/// <summary>
	/// Solves the linear advection problem
	/// </summary>
	/// <param name="r"></param>
	/// <param name="p"></param>
	/// <param name="verbose"></param>
	void solve(const params& p, std::unique_ptr<recorder>& r, bool verbose = true)
	{
		// configure variables
		double h = p.h();
		double adv_coeff = p.V * p.dt / (2 * h);
		double diff_coeff = p.D * p.dt / (h * h);
		cuda_gpu_vector f(p.N), f_prev(p.N), a(p.N), errv(p.N);

		// run checks and initialize state
		validate_grid_stability(p, verbose);
		initialize_state(f, p);

		// step solution through time
		double time = 0, err = 0;
		for (size_t t = 0; t < p.T; ++t)
		{
			r->record(f, time, err);
			using std::swap;
			swap(f, f_prev);

			euler_step(f_prev, f, adv_coeff, diff_coeff);
			compute_analytical_solution(a, p, time, h);
			err = compute_err(f, a, errv);
			time += p.dt;
		}
	}

	/// <summary>
	/// Performs a few computations to verify stability of the system. Prints results.
	/// Throws an exception if dt is too small.
	/// </summary>
	/// <param name="solver_params"></param>
	/// <param name="advection_params"></param>
	void validate_grid_stability(const params& p, bool should_print = false) const
	{
		double h = p.h(),
			courant_number = p.V * p.dt / h,
			diffusion_number = p.D * p.dt / (h * h),
			dt_max = std::min(h / p.V, 0.5 * h * h / p.D);

		if (should_print)
		{
			std::cout << "Courant Number: " << courant_number << std::endl;
			std::cout << "Diffusion Number: " << diffusion_number << std::endl;
			std::cout << "dt max: " << dt_max << " (dt = " << p.dt << ")" << std::endl;
		}

		if (p.dt >= dt_max)
			throw std::out_of_range("dt exceeds maximum allowed dt");
	}

	/// <summary>
	/// Computes the next state given the previous using an Euler step. 
	/// Executed on GPU
	/// </summary>
	/// <param name="prevstate"></param>
	/// <param name="nextstate"></param>
	/// <param name="adv_coeff"></param>
	/// <param name="diff_coeff"></param>
	void euler_step(
		const cuda_gpu_vector& prevstate,
		cuda_gpu_vector& nextstate,
		double adv_coeff,
		double diff_coeff)
	{
		size_t M = prevstate.M();
		size_t blockdim = 1 << 5;
		size_t griddim = (M + blockdim - 1) / blockdim;
		euler_step_state_krnl KERNEL_ARGS2(griddim, blockdim) (prevstate.c_ptr(), nextstate.c_ptr(), M, adv_coeff, diff_coeff);
	}

	/// <summary>
	/// Computes the initial state for the boundary problem.
	/// Executed on GPU
	/// </summary>
	/// <param name="state"></param>
	/// <param name="adv_coeff"></param>
	/// <param name="diff_coeff"></param>
	void initialize_state(
		cuda_gpu_vector& state,
		const params& p)
	{
		size_t M = state.M();
		size_t blockdim = 1 << 5;
		size_t griddim = (M + blockdim - 1) / blockdim;
		initialize_state_krnl KERNEL_ARGS2(griddim, blockdim)(state.c_ptr(), M, p.A, p.k * p.h());
	}

	/// <summary>
	/// Computes an analytical solution for the Linear Advection Diffusion problem
	/// Executed on GPU
	/// </summary>
	/// <param name="state"></param>
	/// <param name="p"></param>
	/// <param name="t"></param>
	/// <param name="h"></param>
	void compute_analytical_solution(cuda_gpu_vector& state, const params& p, double t, double h)
	{
		const double pi = 3.14159265358979323846;
		size_t M = state.M();
		size_t blockdim = 1 << 5;
		size_t griddim = (M + blockdim - 1) / blockdim;
		compute_analytical_solution_krnl KERNEL_ARGS2(griddim, blockdim) (state.c_ptr(), M, pi, p.A, p.k, p.D, t, p.h(), p.V);
	}

	/// <summary>
	/// Computes the error for a single timestep.
	/// Executed on GPU with help from CUDPP
	/// </summary>
	/// <param name="state"></param>
	/// <param name="solution"></param>
	/// <param name="errv"></param>
	/// <returns></returns>
	double compute_err(const cuda_gpu_vector& state, const cuda_gpu_vector& solution, const cuda_gpu_vector& errv)
	{
		size_t M = state.M();
		size_t blockdim = 1 << 5;
		size_t griddim = (M + blockdim - 1) / blockdim;
		
		compute_err_krnl KERNEL_ARGS2(griddim, blockdim) (state.c_ptr(), solution.c_ptr(), errv.c_ptr(), M);

		double err = reduce_sum(_cudpp_lib.get(), errv);
		return 0;
	}

private:
	cudpp_library _cudpp_lib;
	
	// Check stability conditions (Advection specific) - x 

	// Initialize the state (L 66-68), store new state 
	// Iterate over T timesteps 
		// update analytical solution

		// update middle points, log error contribution (EulerSolve)
		// update boundary conditions, log error contribution (EulerSolve)
		
		// adding up error contributions can be done in sync (cooperative groups)

		// store new state

};

#endif LINEAR_ADVECTION_CUH