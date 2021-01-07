#include "../include/cuda_host_matrix.h";
#include "../include/linear_advection.cuh";
#include "cuda_runtime.h"
#include "../include/params.h"

int main(int argc, char* argv[])
{
    // EXTRACT BASIC INFORMATION AND VALIDATION
    if (argc < 1) throw std::invalid_argument("Must specify recording method");
    string rt(argv[0]);
    std::transform(rt.begin(), rt.end(), rt.begin(), ::tolower);
    if (rt == "file" && argc < 2)
        throw std::invalid_argument("Must specify a file directory if using file recording");
    size_t startind = (rt == "file") ? 2 : 1;

    // OBTAIN PARAMETERS
    params p{ 21, 100, 0.05, 2.0, 1.0, 0.05, 0.5, 1 }; // default parameters, feel free to change
    for (size_t ind = startind; ind < argc; ++ind)
    {
        parse_param_and_update(p, argv[ind]);
    }

    // SETUP RECORDING
    std::unique_ptr<recorder> r;
    if (rt == "console") r = std::unique_ptr<console_recorder>(new console_recorder(p.N));
    else if (rt == "file") r = std::unique_ptr<file_recorder>(new file_recorder(argv[1], p.N));

    std::cout << p.A << std::endl;
    std::cout << p.D << std::endl;
    std::cout << p.dt << std::endl;
    std::cout << p.h() << std::endl;
    std::cout << p.k << std::endl;
    std::cout << p.L << std::endl;
    std::cout << p.N << std::endl;
    std::cout << p.T << std::endl;
    std::cout << p.V << std::endl;

    /*linear_advection solver;
    solver.solve(p, r);*/

    return 0;
}
