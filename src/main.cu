#include "../include/cuda_host_matrix.h";
#include "../include/linear_advection.cuh";
#include "cuda_runtime.h"
#include "../include/params.h"

int main(int argc, char* argv[])
{
    // EXTRACT BASIC INFORMATION AND VALIDATION
    if (argc < 2) error_to_user("Must specify recording method");
    string rt(argv[1]);
    std::transform(rt.begin(), rt.end(), rt.begin(), ::tolower);

    if (rt == "file" && argc < 3) error_to_user("Must specify a file directory if using file recording");
    size_t startind = (rt == "file") ? 3 : 2;

    // OBTAIN PARAMETERS
    params p{ 21, 100, 0.05, 2.0, 1.0, 0.05, 0.5, 1 }; // default parameters, feel free to change
    for (size_t ind = startind; ind < argc; ++ind)
    {
        parse_param_and_update(p, argv[ind]);
    }

    // SETUP RECORDING
    std::unique_ptr<recorder> r;
    if (rt == "console") r = std::unique_ptr<console_recorder>(new console_recorder(p.N));
    else if (rt == "file") r = std::unique_ptr<file_recorder>(new file_recorder(argv[2], p.N));
    else error_to_user("invalid record method selected");

    auto space = " - ";
    std::cout << "N:" << p.N << space;
    std::cout << "T:" << p.T << space;
    std::cout << "dt:" << p.dt << space;
    std::cout << "L:" << p.L << space;
    std::cout << "h:" << p.h() << space;
    std::cout << "A:" << p.A << space;
    std::cout << "D:" << p.D << space;
    std::cout << "k:" << p.k << space;
    std::cout << "V:" << p.V << std::endl;

    linear_advection solver;
    solver.solve(p, r);

    return 0;
}
