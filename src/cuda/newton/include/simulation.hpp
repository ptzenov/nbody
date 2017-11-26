#ifndef _SIMULAITON_HPP_
#define _SIMULAITON_HPP_

#include "common.hpp"

#include <cuda.h>
#include <cuda_gl_interop.h>

#define KERNEL __global__
#define DEVICE __device__
#define HOST __host__
#define HOST_DEVICE HOST DEVICE

// custom implementation of unique_ptr -> on device no stl available
template <typename T>
class unique_ptr {
       public:
	using pointer = T*;

	HOST_DEVICE unique_ptr(pointer resource)  //  normal ctor
	{
		ptr_ = resource;
	}

	HOST_DEVICE unique_ptr(unique_ptr<T>&& other)  // move ctor
	{
		ptr_ = other.ptr_;
		other.ptr_ = nullptr;
	}

	HOST_DEVICE ~unique_ptr()  // dtor
	{
		delete ptr_;
	}

	HOST_DEVICE unique_ptr() : ptr_(nullptr) {};

	// cannot have copy ctor
	HOST_DEVICE unique_ptr(unique_ptr<T> const& other) = delete;

	// cannot have assignement operator!
	HOST_DEVICE unique_ptr& operator=(unique_ptr<T> const& other) = delete;

	// move assignment operator?
	HOST_DEVICE unique_ptr& operator=(unique_ptr<T>&& other) {
		delete ptr_;
		ptr_ = other.ptr_;
		other.ptr_ = nullptr;
		return *this;
	}
	HOST_DEVICE pointer get();
	HOST_DEVICE void reset(pointer resource) {
		delete ptr_;
		ptr_ = resource;
	}

	HOST_DEVICE T& operator*() { return *ptr_; }

	HOST_DEVICE pointer operator&() { return ptr_; }
	HOST_DEVICE pointer operator->() { return ptr_; }

       private:
	T* ptr_;
};

// NBody Simulator (on device)
class Simulator {
       private:
	Params sim_params_d;

	unique_ptr<float> position_data;
	unique_ptr<float> velocity_data;
	unique_ptr<float> force_data;
	unique_ptr<float> mass_data;
	unique_ptr<float> radius_data;
	unique_ptr<float> rgb_data;

	DEVICE void computeForces(size_t i);
	DEVICE void applyForce(size_t i, size_t j);
	DEVICE void updateX(size_t i);
	DEVICE void updateV(size_t i, float* Fold);

       public:
	DEVICE Simulator(Params params)
	    : sim_params_d{params},
	      position_data{
		  new float[sim_params_d.sim_N * sim_params_d.sim_DIM]},
	      velocity_data{
		  new float[sim_params_d.sim_N * sim_params_d.sim_DIM]},
	      force_data{new float[sim_params_d.sim_N * sim_params_d.sim_DIM]},
	      mass_data{new float[sim_params_d.sim_N]},
	      radius_data{new float[sim_params_d.sim_N]},
	      rgb_data{
		  new float[3 * sim_params_d.sim_N * sim_params_d.sim_DIM]} {
		;
	}

	DEVICE void init_data();
	DEVICE void step();
};

// this is the Kernel to be launched! it takes a simulator (on device)
KERNEL void init_simulator(Simulator* simulator);
KERNEL void launch_kernel(Simulator const* simulator);

/*** CUDA-OpenGL interface ***/
void fillVBOCuda();

#endif
