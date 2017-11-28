#include "common.hpp"
#include "simulation.hpp"


#include "graphics.hpp"

#include <time.h>

Renderer* renderer;
Params sim_params{}; 

void draw_scene()
{
	// delegate to renderer
	if(renderer != nullptr)
		renderer->draw_scene();
}

void handle_resize(int w, int h)
{
	// delegate to renderer
	if(renderer != nullptr)
		renderer->handle_resize(w,h);
}

void handle_keypress(unsigned char key, int x, int y)
{
	// delegate to renderer
	if(renderer != nullptr)
		renderer->handle_keypress(key,x, y); 
}

void keyboard_navigator(int key, int x, int y)
{
	// delegate to renderer
	if(renderer != nullptr)
		renderer->keyboard_navigator(key,x, y); 
}

void update(int val)
{
	if(renderer!=nullptr)
		renderer->make_step(val);
}


int main(int argc, char** argv)
{
	// create default params set 
	// parse from user input if any
	assert(init(argc,argv,sim_params) == 0);

	Simulator * simulator;
        // init the simulator on device 	
	init_simulator<<<1,1>>>(simulator,sim_params,time(NULL)); 

	// initialize OpenGL
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(400, 400); //Set the window size
        //Create the window
        glutCreateWindow("Nbody Simulation");
	
	// init renderer and assigne simulator to it!
	renderer = Renderer::get_singleton(sim_params,simulator);
	std::cout<<"Initializing Graphics"<<std::endl;
	renderer->init_graphics();

	//Set handler functions for drawing, keypresses, and window resizes
	glutDisplayFunc(draw_scene);
        glutKeyboardFunc(handle_keypress);
        glutSpecialFunc(keyboard_navigator);
        glutReshapeFunc(handle_resize);
	glewInit();

	// create cuda context
	std::cout<<"Initializing Cuda Context"<<std::endl;
        cudaSetDevice(0);
        cudaGLSetGLDevice(0);
	
	// now allocate openGL buffer memeory 
	// create buffer object
/*	
	GLuint gl_buffer_ID;
        glGenBuffers(1, &gl_buffer_ID);
     // make this buffer the current array buffer
     	glBindBuffer(GL_ARRAY_BUFFER, gl_buffer_ID);
	// allocate memory for the array buffer  
        glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
        // register buffer 
	checkCudaErrors(cudaGLRegisterBufferObject( dataBufferID));
*/	
	
	glutMainLoop();
	free_simulator<<<1,1>>>(simulator); 
	return 0; 
}


