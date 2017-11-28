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

KERNEL void launch_simulation_kernel(Simulator* simulator, int val);

KERNEL void init_simulator(Simulator*, Params, size_t);
KERNEL void free_simulator(Simulator*);


class Renderer {

       public:
	static	Renderer* get_singleton(Params, Simulator*);
	// Initializes 3D rendering
	void init_graphics();
	// Draws the 3D scene
	void draw_scene();
	void keyboard_navigator(int key, int x, int y);
	void handle_keypress(unsigned char key, int x, int y);

	// Called when the window is resized
	void handle_resize(int w, int h);
	void make_step(int val){
		launch_simulation_kernel<<<sim_params.NUM_BLOCKS, sim_params.NUM_THREADS>>>(simulator, val);
	}

       private:
	Renderer(Params, Simulator* in_simulator);
	
	std::unique_ptr<float []> radius_data; 
	std::unique_ptr<float []> rgb_data;


	Params sim_params;
	Simulator* simulator;

	int disp_0;
	int simulation_started;

	// GL stuff
	GLfloat angle_x;
	GLfloat angle_y;

	GLfloat z_translate;
	GLfloat x_translate;
	GLfloat y_translate;
};

#endif
