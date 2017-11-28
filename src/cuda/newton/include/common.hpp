#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <fstream>
#include <string>

#include <chrono>

#include <assert.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

/////////////////NBODY///////////////
#define G 1  // gravitational constant
#define sqr(x) ((x) * (x))
#define THREADS_PER_BLOCK 256
#define EPS2 \
	0.001  // small number to avoid division by zero as |r1-r2| approaches 0
#define dist 0.5

#define KERNEL __global__
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE HOST DEVICE
#define DBG_MSG std::cout<<"DEBUG FILE:"<<__FILE__<< ";LINE:"<<__LINE__ <<std::endl;


struct Params {
       public:
	HOST_DEVICE Params()
	    : sim_N{10U},
	      sim_DIM{3U},
	      temp{1.0f},
	      sim_dt{0.025f},
	      display_dt{25},
	      NUM_BLOCKS{1U},
	      NUM_THREADS{THREADS_PER_BLOCK} {
		;
	}
	void print_params() {
		std::cout << "Params: (" << sim_N << "," << sim_DIM << ","
			  << temp << ","
			  << "sim_dt"
			  << "," << display_dt << ")" << std::endl;
	}
	HOST_DEVICE Params(Params const& other) = default;
	HOST_DEVICE Params& operator=(Params const& other) = default;

	// disable move semantics
	HOST_DEVICE Params(Params&& other) = delete;
	HOST_DEVICE Params& operator=(Params&& other) = delete;
	HOST_DEVICE ~Params() { ; }

	size_t sim_N;
	size_t sim_DIM;

	float temp;

	float sim_dt;       // in sec
	size_t display_dt;  // in usec

	size_t NUM_BLOCKS;
	size_t NUM_THREADS;
};

/*User input handling functions*/
int init(int argc, char** argv, Params&);

int find_option(int argc, char** argv, const std::string& option);
int read_int(int argc, char** argv, const std::string& option,
	     int default_value);
float read_float(int argc, char** argv, const std::string& option,
		 float default_value);

#endif
