#include "common.hpp"
#include "simulation.hpp"

#include "graphics.hpp"

#include <time.h>

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

	// create cuda context
	std::cout << "Initializing Cuda Context" << std::endl;
	cudaSetDevice(0);
	cudaGLSetGLDevice(0);

	Simulator* simulator;
	// init the simulator on device
	init_simulator<<<1, 1>>> (simulator, sim_params, time(NULL));
	std::cout << "Simulator initialization completed! " << std::endl;
	std::cout << "Cuda Initialization done!" << std::endl;
	// initialize OpenGL
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(400, 400);  // Set the window size
	// Create the window
	glutCreateWindow("Nbody Simulation");

	// Set handler functions for drawing, keypresses, and window resizes
	glutDisplayFunc(draw_scene);
	glutKeyboardFunc(handle_keypress);
	glutSpecialFunc(keyboard_navigator);
	glutReshapeFunc(handle_resize);
	glewInit();


	// init renderer and assigne simulator to it!
	std::cout << "Initializing Graphics" << std::endl;
	std::cout << "fetching renderer! " << std::endl;
	renderer = new Renderer(sim_params, simulator);
	std::cout << "renderer->init_graphics" << std::endl;
	renderer->init_graphics();

	// Set handler functions for drawing, keypresses, and window resizes
	glutMainLoop();
	free_simulator << <1, 1>>> (simulator);
	return 0;
}
