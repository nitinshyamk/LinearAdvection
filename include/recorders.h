#include <iostream>
#include <fstream>
#include <string>
#include "../include/cuda_gpu_matrix.h"
#include "../include/cuda_host_matrix.h"

struct recorder
{
	virtual void record(const cuda_gpu_vector& state, double time, double err) = 0;
};


struct file_recorder : public recorder
{
	file_recorder(std::string fname, size_t statedim) : 
		filename(fname), writefile(fname), buffer(statedim, 1) 
	{
		std::cout << fname << std::endl;
	}

	void set_file(std::string fname)
	{
		writefile.close();
		writefile = std::ofstream(fname);
	}

	~file_recorder()
	{
		writefile.close();
	}

	virtual void record(const cuda_gpu_vector& state, double time, double err)
	{
		buffer.copy_to_host_memory(state);

		writefile << time << ", ";
		for (size_t i = 0; i < buffer.M(); ++i)
		{
			writefile << buffer[i][0] << ", ";
		}
		writefile << err << std::endl;
	}

private:
	std::string filename;
	std::ofstream writefile;
	cuda_host_matrix buffer;
};

struct console_recorder : public recorder
{
	console_recorder(size_t statedim) : buffer(statedim, 1) {}

	virtual void record(const cuda_gpu_vector& state, double time, double err)
	{
		buffer.copy_to_host_memory(state);
		std::cout << time << ", " << err << std::endl;
	}

private:
	cuda_host_matrix buffer;
};