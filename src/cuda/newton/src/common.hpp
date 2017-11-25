#ifndef _COMMON_H_
#define _COMMON_H_

#include <vector>

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <assert.h>

#include <cuda.h>


/////////////////NBODY///////////////
#define milisec 25
#define G 1 // gravitational constant
#define sqr(x) ((x)*(x))
#define size 3 // size of the potential box 
#define THREADS_PER_BLOCK 256
#define EPS2 0.001 // small number to avoid division by zero as |r1-r2| approaches 0
#define dist 0.5

#define HOST_DEVICE //__host__ __device__

struct Params
{
public:
	
        HOST_DEVICE Params(): sim_N{10U}, sim_DIM {3U}, 
		    temp{1.0f}, dt{.01f},
		    NUM_BLOCKS{1U}, NUM_THREADS{THREADS_PER_BLOCK}
        {;}
        
	size_t sim_N;
        size_t sim_DIM;

	float temp;
        float dt;

        size_t NUM_BLOCKS;
        size_t NUM_THREADS;
};

/*User input handling functions*/
int init(int argc, char** argv, Params&);

int find_option(int argc, const char** argv, const std::string & option);
int read_int(int argc, const char** argv, const std::string& option,int default_value);
int save_to_file(int argc, char** argv, const char* option);


#endif





