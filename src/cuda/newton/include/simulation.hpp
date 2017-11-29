#ifndef _SIMULAITON_HPP_
#define _SIMULAITON_HPP_

#include "common.hpp"

// NBody Simulator (on device)
class Simulator {
       public:
	Params params_d;

       private:
	size_t mode;
	size_t seed;

	float* position_data;
	custom_unique_ptr<float> velocity_data;
	custom_unique_ptr<float> force_data;
	custom_unique_ptr<float> mass_data;

	DEVICE void computeForces(size_t i, float*);
	DEVICE void applyForce(size_t i, size_t j);
	DEVICE void updateX(size_t i);
	DEVICE void updateV(size_t i, float*);

       public:
	HOST_DEVICE void set_position_data(float* pos_data) {
		position_data = pos_data;
	}
	
	DEVICE Simulator(Params params, size_t in_seed);	
	DEVICE ~Simulator() { ; }
	
	DEVICE void init_data();
	DEVICE void make_step();
};

KERNEL void init_simulator(Simulator*, Params, size_t);
KERNEL void free_simulator(Simulator*);

#endif
