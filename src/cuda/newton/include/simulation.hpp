#ifndef _SIMULAITON_HPP_
#define _SIMULAITON_HPP_

#include "common.hpp"

// custom implementation of custom_unique_ptr -> on device no stl available
template <typename T>
class custom_unique_ptr {
       public:
	using pointer_T = T*;

	HOST_DEVICE custom_unique_ptr(pointer_T resource)  //  normal ctor
	{
		ptr_ = resource;
	}

	HOST_DEVICE custom_unique_ptr(custom_unique_ptr<T>&& other)  // move ctor
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
	HOST_DEVICE custom_unique_ptr(custom_unique_ptr<T> const& other) = delete;

	// cannot have assignement operator!
	HOST_DEVICE custom_unique_ptr& operator=(custom_unique_ptr<T> const& other) = delete;

	// move assignment operator?
	HOST_DEVICE custom_unique_ptr& operator=(custom_unique_ptr<T>&& other) {
		delete ptr_;
		ptr_ = other.ptr_;
		other.ptr_ = nullptr;
		return *this;
	}
	
	HOST_DEVICE pointer_T get(){ 
		return ptr_;
	}

	HOST_DEVICE void reset(pointer_T resource) {
		delete ptr_;
		ptr_ = resource;
	}

	HOST_DEVICE T& operator*() { return *ptr_; }

	HOST_DEVICE pointer_T operator&() { return ptr_; }
	HOST_DEVICE pointer_T operator->() { return ptr_; }

       private:
	pointer_T ptr_;
};


// NBody Simulator (on device)
class Simulator {
       public:
	Params params_d;

       private:
	size_t mode;
	size_t seed;

	custom_unique_ptr<float> position_data;
	custom_unique_ptr<float> velocity_data;
	custom_unique_ptr<float> force_data;
	custom_unique_ptr<float> mass_data;

	DEVICE void computeForces(size_t i,float *);
	DEVICE void applyForce(size_t i, size_t j);
	DEVICE void updateX(size_t i);
	DEVICE void updateV(size_t i, float *);

       public:
	DEVICE Simulator(Params params, size_t in_seed)
	    : params_d{params},
	      mode{0},
	      seed(in_seed),
	      position_data{new float[params_d.sim_N * params_d.sim_DIM]},
	      velocity_data{new float[params_d.sim_N * params_d.sim_DIM]},
	      force_data{new float[params_d.sim_N * params_d.sim_DIM]},
	      mass_data{new float[params_d.sim_N]} {
		;
	}
	
	DEVICE ~Simulator() { ; }
	DEVICE void init_data();
	DEVICE void make_step();
};

KERNEL void init_simulator(Simulator*, Params, size_t);
KERNEL void free_simulator(Simulator*);

#endif
