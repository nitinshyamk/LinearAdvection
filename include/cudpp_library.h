#ifndef LINEAR_ADVECTION_CUDPP_LIBRARY
#define LINEAR_ADVECTION_CUDPP_LIBRARY

#include "../include/cudpp.h"

/// <summary>
/// Thin wrapper around cudpp library
/// </summary>
struct cudpp_library
{
	cudpp_library()
	{
		cudppCreate(&_cudpp_lib);
	}

	~cudpp_library()
	{
		cudppDestroy(_cudpp_lib);
	}

	CUDPPHandle get()
	{
		return _cudpp_lib;
	}



private:
	CUDPPHandle _cudpp_lib;
};

#endif LINEAR_ADVECTION_CUDPP_LIBRARY