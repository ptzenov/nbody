#ifndef _SIMULAITON_HPP_
#define _SIMULAITON_HPP_

#include "common.hpp"

#include <cuda_gl_interop.h>
#include <curand_kernel.h>
// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop
#include <vector_types.h>

#define KERNEL // __global__
#define DEVICE // __device__ 
#define HOST // __host__
#define HOST_DEVICE HOST DEVICE

// custom implementation of unique_ptr -> on device no stl available
template <typename T>
class unique_ptr
{
public:
        using pointer =  T*;
        HOST_DEVICE unique_ptr(pointer resource)  //  normal ctor
        {
                ptr_ = resource;
        }

        HOST_DEVICE unique_ptr(unique_ptr<T>&& other)   // move ctor
        {
                ptr_ = other.ptr_;
                other.ptr_ = nullptr;
        }
        HOST_DEVICE	~unique_ptr()  // dtor
        {
                delete ptr_;
        }

        HOST_DEVICE	unique_ptr(): ptr_(nullptr) {};
        // cannot have copy ctor
        unique_ptr(unique_ptr<T> const & other) = delete;
        unique_ptr& operator=(unique_ptr<T> const & other)  = delete;

        // move assignment operator?
        HOST_DEVICE	unique_ptr& operator=(unique_ptr<T> && other)
        {
                delete ptr_;
                ptr_ = other.ptr_;
                other.ptr_ = nullptr;
                return *this;
        }
	HOST_DEVICE pointer get();
        HOST_DEVICE	void reset(pointer resource)
        {
                delete ptr_;
                ptr_ = resource;
        }

        HOST_DEVICE	T& operator*()
        {
                return *ptr_;
        }

        HOST_DEVICE	pointer operator&()
        {
                return ptr_;
        }
        HOST_DEVICE	pointer operator->()
        {
                return ptr_;
        }
private:
        T* ptr_;
};

// Kernel simulator
class Simulator
{
private:

	Params sim_params_d;

	unique_ptr<float> position_data;
        unique_ptr<float> velocity_data;
	unique_ptr<float> force_data;
	unique_ptr<float> mass_data;
	unique_ptr<float> radius_data;
        unique_ptr<float> rgb_data;
        
	DEVICE Simulator(size_t N, size_t DIM,float dt, float heat):
		sim_params_d{N,DIM, dt,heat},
                position_data{new float[N*DIM]}, velocity_data {new float[N*DIM]},
		force_data{new float[N*DIM]}, mass_data{new float[N]},
	       	radius_data {new float[N]}, rgb_data {new float[3*N*DIM]}
        {
	;	
        }
/*	// default and copy ctors 
	DEVICE Simulator(): Simulator(10U,3U,0.01f,1.f) {;}
	
	DEVICE Simulator(Simulator const & other):
			Simulator( other.sim_params_d.sim_N, other.sim_params_d.sim_DIM,
					other.var_params_d.dt,	other.var_params_d.temp)
	{;}

	DEVICE ~Simulator(){;}
*/

        DEVICE void computeForces(size_t i);
        DEVICE void applyForce(size_t i, size_t j);
        DEVICE void updateX(size_t i);
        DEVICE void updateV(size_t i, float* Fold);

public:
        HOST_DEVICE static Simulator* get_singleton(size_t N = 10,size_t DIM = 3, float dt = 0.01, float heat = 1)
        {
        // initialized on first run
		static Simulator* instance = new Simulator{N,DIM,dt,heat};
		return instance;
	} 
	
        DEVICE void init_data();
        KERNEL void simulate();
};

void step(int value); // the function to be called after GL plots results -> this launches all Kernels

/*** CUDA-OpenGL interface ***/
void fillVBOCuda();

#endif





