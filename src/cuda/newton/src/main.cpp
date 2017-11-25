#include "common.hpp"
#include "graphics.hpp"

Renderer* renderer;

void draw_scene(){
	if(renderer != nullptr)
		renderer->draw_scene();
}

void handle_resize(int w, int h){
	if(renderer != nullptr)
		renderer->handle_resize(w,h);
}

void handle_keypress(unsigned char key, int x, int y)
{
	if(renderer != nullptr)
		renderer->handle_keypress(key,x, y); 
}

void keyboard_navigator(int key, int x, int y)
{
	if(renderer != nullptr)
		renderer->keyboard_navigator(key,x, y); 
}


int main(int argc, char** argv)
{
        // set program params

	Params sim_params{}; 
	assert(init(argc,argv,sim_params) == 0);

		
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(400, 400); //Set the window size
        //Create the window
        glutCreateWindow("Nbody Simulation");
	std::cout<<"Initializing Graphics"<<std::endl;
	//Set handler functions for drawing, keypresses, and window resizes
	renderer = new Renderer();

        glutDisplayFunc(draw_scene);
        glutKeyboardFunc(handle_keypress);
        glutSpecialFunc(keyboard_navigator);
        glutReshapeFunc(handle_resize);
        glewInit();
        
	glutMainLoop();
        delete renderer;
	return 0; //This line is never reached
}


