#ifndef _COMMON_H_
#define _COMMON_H_

#include <iostream>
#include <fstream>
#include <string>

#include <chrono>

#include <assert.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

// runtime API for cuda and openGL interop
#include <cuda_gl_interop.h>


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

struct Params {
       public:
	HOST_DEVICE Params()
	    : sim_N{10U},
	      sim_DIM{3U},
	      temp{1.0f},
	      sim_dt{.025f},
	      display_dt{25.0f},
	      NUM_BLOCKS{1U},
	      NUM_THREADS{THREADS_PER_BLOCK} {
		;
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

	float sim_dt;      // in sec
	float display_dt;  // in usec

	size_t NUM_BLOCKS;
	size_t NUM_THREADS;
};

/*User input handling functions*/
int init(int argc, char** argv, Params&);

int find_option(int argc, const char** argv, const std::string& option);
int read_int(int argc, const char** argv, const std::string& option,
	     int default_value);
float read_float(int argc, const char** argv, const std::string& option,
		 float default_value);
int save_to_file(int argc, char** argv, const char* option);

#endif