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
// small number to avoid division by zero as |r1-r2| approaches 0
#define EPS2 0.001
#define dist 0.5
///////////////CUDASTUFF/////////////
#define KERNEL __global__
#define HOST __host__
#define DEVICE __device__
#define HOST_DEVICE HOST DEVICE
#define DBG_MSG                                                       \
	std::cout << "DEBUG MSG:" << __FILE__ << ";LINE:" << __LINE__ \
		  << std::endl;

void cuda_check(cudaError_t code, const char* file, int line);

// custom implementation of custom_unique_ptr -> on device no stl available
template <typename T>
class custom_unique_ptr {
       public:
	using pointer_T = T*;

	HOST_DEVICE custom_unique_ptr(pointer_T resource)  //  normal ctor
	{
		ptr_ = resource;
	}

	HOST_DEVICE custom_unique_ptr(custom_unique_ptr<T>&& other)  // move
								     // ctor
	{
		ptr_ = other.ptr_;
		other.ptr_ = nullptr;
	}

	HOST_DEVICE ~custom_unique_ptr()  // dtor
	{
		delete ptr_;
	}

	HOST_DEVICE custom_unique_ptr() : ptr_(nullptr) {};

	// cannot have copy ctor
	HOST_DEVICE custom_unique_ptr(custom_unique_ptr<T> const& other) =
	    delete;

	// cannot have assignement operator!
	HOST_DEVICE custom_unique_ptr& operator=(
	    custom_unique_ptr<T> const& other) = delete;

	// move assignment operator?
	HOST_DEVICE custom_unique_ptr& operator=(custom_unique_ptr<T>&& other) {
		delete ptr_;
		ptr_ = other.ptr_;
		other.ptr_ = nullptr;
		return *this;
	}
	HOST_DEVICE pointer_T get() { return ptr_; }

	HOST_DEVICE void reset(pointer_T resource) {
		delete ptr_;
		ptr_ = resource;
	}
	HOST_DEVICE T& operator*() { return *ptr_; }
	// could be abused! do not use with non-array data. no guarantee that
	// this will be exception safe!
	HOST_DEVICE T& operator[](size_t idx) { return ptr_[idx]; }
	HOST_DEVICE pointer_T operator&() { return ptr_; }
	HOST_DEVICE pointer_T operator->() { return ptr_; }

       private:
	pointer_T ptr_;
};

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
			  << temp << "," << sim_dt << "," << display_dt << ")"
			  << std::endl;
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
