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

class Renderer {

       public:
	Renderer(Params, Simulator* in_simulator);
	// initializes the scene	
	void init_graphics();
	// Draws the 3D scene
	void draw_scene();
	void keyboard_navigator(int key, int x, int y);
	void handle_keypress(unsigned char key, int x, int y);

	// Called when the window is resized
	void handle_resize(int w, int h);
	void make_step(int val) {
		// do i need to do that every time??
		if (!resource_mapped) {
			cudaGraphicsMapResources(1, &buffer_resource, 0);
			size_t size = static_cast<size_t>(buffer_size);
			cudaGraphicsResourceGetMappedPointer(
			    (void**)&position_data, &size, buffer_resource);
		}
		launch_simulation_kernel
			<< <sim_params.NUM_BLOCKS, sim_params.NUM_THREADS>>>
		    (simulator, position_data, val);
		if (!resource_mapped)
			cudaGraphicsUnmapResources(1, &buffer_resource, 0);
		else
			resource_mapped = true;
	}


       private:
	std::unique_ptr<float[]> radius_data;
	std::unique_ptr<float[]> rgb_data;

	Params sim_params;
	Simulator* simulator;

	int disp_0;
	int simulation_started;

	// cuda-openGL interop
	GLuint buffer_ID;
	GLuint buffer_size;
	cudaGraphicsResource_t buffer_resource;
	float* position_data;
	bool resource_mapped;

	// GL stuff
	GLfloat angle_x;
	GLfloat angle_y;

	GLfloat z_translate;
	GLfloat x_translate;
	GLfloat y_translate;
};

#endif
