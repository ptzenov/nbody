/*
 * graphics.h
 *
 *  Created on: 10 May 2013
 *      Author: kenny
 */

#ifndef _COMMON_H_
#define _COMMON_H_
// OpenGL 
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>


// cuda
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>
#include <vector>

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <time.h>
#include <assert.h>

/////////////////NBODY///////////////

#define milisec 25
#define G 1 // gravitational constant
#define sqr(x) ((x)*(x))
#define THREADS_PER_BLOCK 1024
#define size 3 // size of the potential box 
// not used !! 

#define EPS2 0.001
#define dist 0.5



struct particle_t{

	float p[3];
	float v[3];
	float force[3];
	float m;
	float r,g,b;
	float radius;
	
};

struct ConstParams{
	
	int N_sim;
	int D_sim;
	int buff_stride;
	
	};

struct VarParams{
	
	float heateffect_sim;
	float dt_sim;
	
	};

/***************************SIMULATION.cpp*********************************/


/*User input handling functions*/
int init(int argc, char** argv);
int find_option(int argc, char** argv, const char* option);
int read_int(int argc, char** argv, const char* option,int default_value);
int saveToFile(int argc, char** argv, const char* option);

/*Application Logic's Functions*/

//Host Functions!! 

void initData(particle_t* Particles);

// Kernel Functions
__global__ void launch_kernel(particle_t* Particles,float* databuffer, int KernelMode);

__global__ void setSimParams(ConstParams* cparams_d);
__global__ void setVarParams(VarParams* params_d);   

//Device Functions
__device__ void computeForces(particle_t* Particles, int i);
__device__ void applyForce( particle_t * particle_i, particle_t* neighbor_j);
__device__ void updateX(particle_t* i);              
__device__ void updateV(particle_t* i, float* Fold); 
//__device__ void updateV(particle_t* i, float3 Fold); 


/******************CUDA OpenGL interface************************/

void createVBO();
void fillVBOCuda();


/*****************GRAPHICS.cpp**************************/

//Initializes 3D rendering
void initGraphics();
void update(int value);
//Draws the 3D scene
void drawScene() ;
void keyboardNavigator(int key, int x, int y);
void handleKeypress(unsigned char key, int x, int y);
//Called when the window is resized
void handleResize(int w, int h);

#endif /* common_H_ */



	
	



















