#ifndef _SIMULAITON_HPP_
#define _SIMULAITON_HPP_

#include "common.hpp"

// NBody simulator class. Contains the data and the simulation logics

class Simulator {
       public:
	Params params_d;

       private:
	size_t mode;
	size_t seed;

	float* position_data; // this is the shared object between openGL and CUDA
	custom_unique_ptr<float> velocity_data; // velocity 
	custom_unique_ptr<float> force_data; // forces 
	custom_unique_ptr<float> mass_data; // masses

	DEVICE void computeForces(size_t i, float*); // thread i computes forces interacting on particle i
	DEVICE void applyForce(size_t i, size_t j); // thread i applies newton force between particles i and j
	DEVICE void updateX(size_t i); // update position of particle i
	DEVICE void updateV(size_t i, float*); // update velocity of particle i

       public:
	HOST_DEVICE void set_position_data(float* pos_data) {
		position_data = pos_data;
	}
	
	DEVICE Simulator(Params params, size_t in_seed); // seed for the cuda random number generator 	
	DEVICE ~Simulator() { ; } // destructor -> everything is on the stack so nothing to destroy!
	
	DEVICE void init_data(); // randomizes the positions of all particles
	DEVICE void make_step(); // performs the updates for each particle
};

// kernel functions to setup memory for the simulator and also to free mem
KERNEL void init_simulator(Simulator*, Params, size_t); 
KERNEL void init_simulation_data(Simulator *simulator);


KERNEL void free_simulator(Simulator*);

#endif
