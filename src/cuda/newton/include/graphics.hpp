#ifndef _GRAPHICS_HPP_
#define _GRAPHICS_HPP_

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/gl.h>

#include <memory>
#include "common.hpp"
#include "simulation.hpp"

class Renderer {

       public:
	Renderer(Params, Simulator * in_simulator);

	// Initializes 3D rendering
	void init_graphics();
	// Draws the 3D scene
	void draw_scene();
	void keyboard_navigator(int key, int x, int y);
	void handle_keypress(unsigned char key, int x, int y);

	// Called when the window is resized
	void handle_resize(int w, int h);
	void update(int);
	
       private:
	Params sim_params;
	Simulator* sim_ptr;

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
