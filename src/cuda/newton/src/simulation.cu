#include "simulation.hpp"
#include "common.hpp"

#include <algorithm>

// cuda random numbers
#include <curand.h>
#include <curand_kernel.h>

// to be launched from one thread!
KERNEL void init_simulator(Simulator *simulator, Params params, size_t seed) {
	simulator = new Simulator(params, seed);
	simulator->init_data();
}

KERNEL void free_simulator(Simulator *simulator) { delete simulator; }

DEVICE void Simulator::make_step() {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < params_d.sim_N) {
		if (mode == 0) updateX(i);

		if (mode == 1) {
			float Fold[3];
			computeForces(i, Fold);
			updateV(i, Fold);
		}
	}
	if (i == 0) mode = (mode + 1) % 2;
	// in either case wait for all threads to finish!
	__syncthreads();
}

DEVICE void Simulator::updateX(size_t i) {

	float f = params_d.sim_dt * params_d.sim_dt / (2 * mass_data.get()[i]);
	int idx = i * params_d.sim_DIM;
	for (size_t d = 0; d < params_d.sim_DIM; d++)
		position_data.get()[idx + d] +=
		    params_d.sim_dt * velocity_data.get()[idx + d] +
		    f * force_data.get()[idx + d];
}

DEVICE void Simulator::computeForces(size_t i, float Fold[3]) {

	// each thread computes all the interactions of particle i with all
	// other particles!
	int idx = i * params_d.sim_DIM;
	for (int d = 0; d < params_d.sim_DIM; d++) {
		Fold[d] = force_data.get()[idx + d];
		force_data.get()[idx + d] = 0;
	}
	for (int j = 0; j < params_d.sim_N; j++)
		if (i != j) applyForce(i, j);
}

DEVICE void Simulator::applyForce(size_t i, size_t j) {
	float k_i = G * mass_data.get()[i];
	float r = 0;
	float k = k_i * mass_data.get()[j];
	int idx_i = i * params_d.sim_DIM;
	int idx_j = j * params_d.sim_DIM;
	for (size_t d = 0; d < params_d.sim_DIM; d++)
		r += sqr(position_data.get()[idx_j + d] -
			 position_data.get()[idx_i + d]);  // radius length

	r += EPS2;
	float f = k / (sqrt(r) * r);
	// update the force on particle i with the force exerted by particle j
	for (int d = 0; d < params_d.sim_DIM; d++)
		force_data.get()[idx_i + d] +=
		    f * (position_data.get()[idx_j + d] -
			 position_data.get()[idx_i + d]);
}

DEVICE void Simulator::updateV(size_t i, float Fold[3]) {
	float f = params_d.sim_dt / (2 * mass_data.get()[i]);
	size_t idx = i * params_d.sim_DIM;
	for (int d = 0; d < params_d.sim_DIM; d++) {
		velocity_data.get()[idx + d] +=
		    f * (force_data.get()[idx + d] + Fold[d]);
		velocity_data.get()[idx + d] *= sqrt(params_d.temp);
	}
}

// performed on host !!
DEVICE void Simulator::init_data() {

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int i_d = i * params_d.sim_DIM;

	curandState_t state;
	curand_init(seed, i, 0, &state);

	for (int d = 0; d < params_d.sim_DIM; d++) {
		position_data.get()[i_d + d] =
		    (float(curand_uniform(&state)) - 0.5) * 20;
		velocity_data.get()[i_d + d] = 0;
		force_data.get()[i_d + d] = 0;
	}

	mass_data.get()[i] = 10;  // make the mass proportional to the volume
}
