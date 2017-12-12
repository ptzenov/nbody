#include <time.h>

#include "common.hpp"
#include "simulation.hpp"
#include "graphics.hpp"

Renderer* renderer;
Params sim_params{};

size_t milisec = 25;

void draw_scene() {
	// delegate to renderer
	if (renderer != nullptr) renderer->draw_scene();
}

void handle_resize(int w, int h) {
	// delegate to renderer
	if (renderer != nullptr) renderer->handle_resize(w, h);
}

void handle_keypress(unsigned char key, int x, int y) {
	// delegate to renderer
	if (renderer != nullptr) renderer->handle_keypress(key, x, y);
}

void keyboard_navigator(int key, int x, int y) {
	// delegate to renderer
	if (renderer != nullptr) renderer->keyboard_navigator(key, x, y);
}

void update(int val) {
	if (renderer != nullptr) renderer->make_step(val);
	glutPostRedisplay();
	glutTimerFunc(sim_params.display_dt, update, 0);
}

int main(int argc, char** argv) {

	// create default params set
	// parse from user input if any
	assert(init(argc, argv, sim_params) == 0);

	// initialize OpenGL
	DBG_MSG("Initializing GLUT graphics");
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(400, 400);
	glutCreateWindow("Nbody Simulation");

	DBG_MSG("Initializing CUDA context.");
	cuda_check(cudaSetDevice(0));
	cuda_check(cudaGLSetGLDevice(0)); 
	cudaDeviceSynchronize();

	// Set handler functions for drawing, keypresses, and window resizes
	glutDisplayFunc(draw_scene);
	glutKeyboardFunc(handle_keypress);
	glutSpecialFunc(keyboard_navigator);
	glutReshapeFunc(handle_resize);
	glewInit();

	Simulator* simulator;
	// init the simulator on device
	init_simulator << <1, 1>>> (simulator, sim_params, time(NULL));
	cuda_check(cudaDeviceSynchronize());
	DBG_MSG("Simulator initializaiton completed");

	// init renderer and assigne simulator to it!
	DBG_MSG("Initializing renderer");
	renderer = new Renderer(sim_params, simulator);
	renderer->init_graphics();

	init_simulation_data << <1, 1>>> (simulator);
	cuda_check(cudaDeviceSynchronize());
	
	DBG_MSG("Starting glut loop");
	glutMainLoop();
	free_simulator << <1, 1>>> (simulator);
	cudaDeviceReset();
	return 0;
}
