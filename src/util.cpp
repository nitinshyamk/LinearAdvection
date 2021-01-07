#include "../include/util.h"

double reduce_sum(CUDPPHandle cudpp_lib, const cuda_gpu_matrix& m)
{
	CUDPPConfiguration config;
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_DOUBLE;
	config.algorithm = CUDPP_REDUCE;

	CUDPPHandle plan;
	CUDPPResult res = cudppPlan(cudpp_lib, &plan, config, m.M() * m.N(), 1, 0);
	
	if (res != CUDPP_SUCCESS) throw std::runtime_error("unable to configure cudppPlan");

	std::shared_ptr<double> ans = allocate_on_device<double>(sizeof(double));
	res = cudppReduce(plan, ans.get(), m.c_ptr(), m.M() * m.N());
	if (res != CUDPP_SUCCESS) throw std::runtime_error("unable to run cudppPlan");

	return *ans;
}

void parse_param_and_update(params& p, string param)
{
	string name, value;
	std::tie(name, value) = split_parse_param(param);
	std::transform(name.begin(), name.end(), name.begin(), ::tolower);

	if (name == "dt") { p.dt = std::stod(value); }
	else if (name[0] == 'n') { p.N = std::stol(value); }
	else if (name[0] == 't') { p.T = std::stol(value); }
	else if (name[0] == 'l') { p.L = std::stod(value); }
	else if (name[0] == 'v') { p.V = std::stod(value); }
	else if (name[0] == 'd') { p.D = std::stod(value); }
	else if (name[0] == 'a') { p.A = std::stod(value); }
	else if (name[0] == 'k') { p.k = std::stod(value); }
	else error_to_user("invalid parameter specification");
}

std::pair<string, string> split_parse_param(string param)
{
	size_t ind = param.find('=');
	if (ind == string::npos) error_to_user("invalid param specification");
	string name = param.substr(0, ind);
	if (param.length() - ind - 1 < 1) error_to_user("invalid param specification");
	string value = param.substr(ind + 1, param.length() - ind - 1);
	return std::make_pair(name, value);
}

params get_default_params()
{
	return params{ 21, 100, 0.05, 2.0, 1.0, 0.05, 0.5, 1 };
}

void error_to_user(const char* str)
{
	std::cout << "ERROR: " << str << std::endl;
	throw std::invalid_argument(str);
}
