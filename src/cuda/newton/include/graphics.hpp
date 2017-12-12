#ifndef _GRAPHICS_HPP_
#define _GRAPHICS_HPP_

// OpenGL
#include <GL/glew.h>

#include <GL/glut.h>
#include <GL/gl.h>

// runtime API for cuda and openGL interop
#include <cuda_gl_interop.h>

#include <memory>
#include "common.hpp"
#include "simulation.hpp"


KERNEL void launch_simulation_kernel(Simulator* simulator, float* position_data,
				     int val);
KERNEL void init_simulator(Simulator*, Params, size_t);
KERNEL void free_simulator(Simulator*);
/***
 * Main renderer class that encapsulates all the rendering subroutines. current desing is
 * tightly coupled to OpenGL. 
 ***/
class Renderer {

       public:
	Renderer(Params, Simulator* in_simulator);

	// initializes the scene	
	void init_graphics();

	// Draws the 3D scene
	void draw_scene();
	// keyboard navigation. 
	void keyboard_navigator(int key, int x, int y);
	void handle_keypress(unsigned char key, int x, int y);

	void handle_resize(int w, int h);
	void make_step(int val);
       
       private:
	std::unique_ptr<float[]> radius_data;
	std::unique_ptr<float[]> rgb_data;

	Params sim_params;
	Simulator* simulator; // a handle to the simulator object -> this is where all the cuda action 
	// takes place .
	
	int disp_0;
	int simulation_started;
	
	// cuda-openGL interop-related data
	GLuint buffer_ID;
	GLuint buffer_size;
	cudaGraphicsResource_t buffer_resource;
	float* position_data; // this is the buffer (array) which is shared between openGL and CUDA. 
	bool resource_mapped;

	// GL stuff for the camera viewport control
	GLfloat angle_x; // rotation angle for x-direction
	GLfloat angle_y; // rotation angle for y-direction
	
	// translation vector 
	GLfloat z_translate; 
	GLfloat x_translate; 
	GLfloat y_translate;
};

#endif
