#include <iostream>
#include <stdlib.h> //Needed for "exit" function
#include "common.hpp"

using namespace std;

particle_t* Particles;
int N;
int DIM;

/////////////////NBODY///////////////

int init(int argc, char** argv){
	
	
	if(find_option(argc,argv,"-h") >0 || find_option(argc,argv,"--help")>0)
	{

		std::cout<<"Hello to our N-Body simulation program. While running the program, please provide an"<<
		" appropriate input"<<std::endl;
		std::cout<<"type -h or --help for this menu."<<std::endl;
		std::cout<<"type -n <int> to specify number of bodies/particles to simulate (by default n=10). "<<std::endl;
		std::cout<<"type -dim <int> to specify the euclidean dimension you wish to simulate in (by default dim=3)"<<std::endl;
		return 1;

	}
	
	N = read_int(argc, argv, "-n", 10);
	DIM = read_int(argc,argv,"-dim",3);
	
	if(N < 3)
	{
		cout<<"N should be a number greater than 2"<<endl;
		cout<<"setting N to 10"<<endl;		
		N = 10;	
	}

	if(DIM < 1 || DIM >3)
	{
		cout<<"dim must have the values 1, 2 or 3."<<endl;
		cout<<"setting default dimension to 3 "<<endl;		
		DIM = 3;	
	}
	return 0;
	
	}


int main(int argc, char** argv) {

	if (init(argc, argv) == 1 )
		return 0;

	int N = 2000;
	int DIM =3;
	Particles= new particle_t[N];

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(400, 400); //Set the window size
	//Create the window
	glutCreateWindow("Nbody Simulation");
	
	/*initialize particle data and compute initial forces*/
	initData(Particles,DIM,N);
	 /*initialize graphics drawing functions*/
	initGraphics(N,DIM,Particles); //Initialize rendering

	//Set handler functions for drawing, keypresses, and window resizes
	glutDisplayFunc(drawScene);
	glutKeyboardFunc(handleKeypress);
	glutSpecialFunc(keyboardNavigator);
	glutReshapeFunc(handleResize);
	glutMainLoop(); //Start the main loop.  glutMainLoop doesn't return.
	freeData(Particles);
	return 0; //This line is never reached
	
}








