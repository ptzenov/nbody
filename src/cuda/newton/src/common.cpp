#include "common.hpp"
#include <cstdlib>

void cuda_check_impl(cudaError_t code, const char* file, int line) {
	if (code != cudaSuccess) {
		std::cout << "Cuda call on: " << file << " line " << line
			  << " failed. Cuda error message:\""
			  << cudaGetErrorString(code) << "\"" << std::endl;
		cudaDeviceReset();
		assert(false);
	}
}

int find_option(int argc, char** argv, const std::string& option) {
	for (int i = 1; i < argc; i++) {
		if (std::string { argv[i] }.compare(option) == 0)
			return i;
	}
	return -1;
}

int read_int(int argc, char** argv, const std::string& option,
	     int default_value) {

	int i = find_option(argc, argv, option);
	if (i > 0 && i < argc - 1) return std::atoi(argv[i + 1]);
	return default_value;
}

float read_float(int argc, char** argv, const std::string& option,
		 float default_value) {

	int i = find_option(argc, argv, option);
	if (i > 0 && i < argc - 1) return std::atof(argv[i + 1]);
	return default_value;
}

int init(int argc, char** argv, Params& sim_params) {
	const std::string argv_s{*argv};

	if (find_option(argc, argv, "-h") > 0 ||
	    find_option(argc, argv, "--help") > 0) {

		std::cout << "Hello to our N-Body simulation program. While "
			     "running the program, please provide an"
			  << " appropriate input" << std::endl;
		std::cout << "type -h or --help for this menu." << std::endl;
		std::cout << "type -n <uint> to specify number of "
			     "bodies/particles to simulate (by default n=10). "
			  << std::endl;
		std::cout << "type -dim <uint> to specify the euclidean "
			     "dimension you wish to simulate in (by default "
			     "dim=3)" << std::endl;
		std::cout << "type -sim_dt <float> to specity the time step of "
			     "the simulation (in sec) (by default sim_dt = "
			     "0.025 sec = 25 usec )" << std::endl;
		std::cout << "type -display_dt <uint> to specity the time step "
			     "of the (in usec)  (by default display_dt = 25 "
			     "usec)" << std::endl;
		std::cout << "type -temp <float> to specity the temperature "
			     "factor of the simulation  (by default temp=1)"
			  << std::endl;
		return 1;
	}

	sim_params.sim_N = read_int(argc, argv, "-n", 10U);
	sim_params.sim_DIM = read_int(argc, argv, "-dim", 3U);
	assert(sim_params.sim_N >= 3);
	assert(sim_params.sim_DIM >= 1 && sim_params.sim_DIM <= 3);

	// set the variable parameters to default;
	sim_params.sim_dt = read_float(argc, argv, "-sim_dt", 0.025f);
	sim_params.display_dt = read_int(argc, argv, "-display_dt", 25U);

	sim_params.temp = read_float(argc, argv, "-temp", 1.0f);

	// how many blocks, each with 1024 threads do we need to accomodate 1
	// body per thread  thread!
	sim_params.NUM_BLOCKS =
	    (sim_params.sim_N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	sim_params.NUM_THREADS =
	    (sim_params.sim_N + sim_params.NUM_BLOCKS - 1) /
	    sim_params.NUM_BLOCKS;
	return 0;
}
