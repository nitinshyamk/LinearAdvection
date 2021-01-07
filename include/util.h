#ifndef LINEAR_ADVECTION_UTIL
#define LINEAR_ADVECTION_UTIL

#include <algorithm>
#include <iostream>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include "../include/cudpp.h"
#include "../include/cuda_gpu_matrix.h"
#include "params.h"

using string = std::string;

double reduce_sum(CUDPPHandle cudpp_lib, const cuda_gpu_matrix& v);
void parse_param_and_update(params& p, string param);
std::pair<string, string> split_parse_param(string param);
params get_default_params();
void error_to_user(const char* str);


#endif LINEAR_ADVECTION_UTIL