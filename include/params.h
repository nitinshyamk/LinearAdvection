// LinearAdvection.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>

/// <summary>
/// Parameters for finite element solver
/// </summary>
struct params
{
	/// <summary>
	/// (N)umber of grid points
	/// </summary>
	long N;

	/// <summary>
	/// Number of (T)ime steps
	/// </summary>
	long T;

	/// <summary>
	/// Time step width (in sec)
	/// </summary>
	double dt;

	/// <summary>
	/// Domain length (m)
	/// </summary>
	double L;

	/// <summary>
	/// Velocity (m / s)
	/// </summary>
	double V;

	/// <summary>
	/// Diffusion coefficient (m^2 / s)
	/// </summary>
	double D;

	/// <summary>
	/// Amplitude of initial solution (m)
	/// </summary>
	double A;

	/// <summary>
	/// Wave number (1 / m)
	/// </summary>
	double k;

	inline double h() const { return L / ((double)N - 1.0); }
};

