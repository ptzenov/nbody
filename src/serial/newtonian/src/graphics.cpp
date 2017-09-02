/*
 * graphics.cpp
 */


#include <time.h>
#include <assert.h>
#include <stdio.h>
#include "common.hpp"

/////////////////NBODY///////////////

#define dist 0.5

int N_particles;
int DIMENSION;
particle_t* particle_array;

int current_width = 400;
int current_height = 400;
int disp_0 = 1;
int simulation_started = 0;

// particle parameters


float* r;
float* g;
float* b;

GLfloat angle_x = 0;
GLfloat angle_y = 0;
GLfloat z_translate =-20.0f;
GLfloat x_translate =0.0f;
GLfloat y_translate = 0.0f;


void keyboardNavigator(int key, int x, int y){


	switch (key){

		case GLUT_KEY_DOWN : {
					     angle_x +=5;
					     if(angle_x>360)
						     angle_x-=360;
					     break;
				     }

		case GLUT_KEY_UP : {
					   angle_x -=5;
					   if(angle_x<-360)
						   angle_x+=360;
					   break;
				   }

		case GLUT_KEY_LEFT : {

					     angle_y -=5;
					     if(angle_y<-360)
						     angle_y+=360;
					     break;

				     }

		case GLUT_KEY_RIGHT :{
					     angle_y +=5;
					     if(angle_y>360)
						     angle_y-=360;
					     break;
				     }
		case GLUT_KEY_F1:{
					 if(simulation_started == 0)
						 glutTimerFunc(milisec,update,0);
				 }

		default : {
				  // do nothing
			  }

	}

	glutPostRedisplay();
}

void handleKeypress(unsigned char key, int x, int y){

	if(key == '+')
		z_translate+=dist;
	if(key == '-')
		z_translate-=dist;

	if(key == 'a')
		x_translate+=dist;
	if(key == 'd')
		x_translate-=dist;

	if(key == 's')
		y_translate+=dist;
	if(key == 'w')
		y_translate-=dist;

	if(key== 'h')
		heateffect+=0.05;
	if(key=='c'){	
		heateffect-=0.05;
		if(heateffect<0)
			heateffect = 0;
	}
	if(key == 'i')
		dt=-1*dt;
	if(key == '0')
		disp_0 = -1*disp_0;



	glutPostRedisplay();


}




//Initializes 3D rendering
void initGraphics(int N, int DIM, particle_t* pp) {
	//Makes 3D drawing work when something is in front of something else
	particle_array =pp;
	DIMENSION = DIM;
	N_particles = N;

	//OpenGL Stuff
	glEnable(GL_DEPTH_TEST);
	glEnable( GL_TEXTURE_2D );
	glDepthFunc(GL_LEQUAL);
	glClearColor(0.0f,0.0f,0.0f,1.0f);

}



//Called when the window is resized
void handleResize(int w, int h) {

	current_width = w;
	current_height = h;
	//Tell OpenGL how to convert from coordinates to pixel values
	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION); //Switch to setting the camera perspective

	//Set the camera perspective
	glLoadIdentity(); //Reset the camera
	gluPerspective(45.0,                  //The camera angle
			(double)w / (double)h, //The width-to-height ratio
			1.0,                   //The near z clipping coordinate
			2000.0);                //The far z clipping coordinate
}




void update(int value){

	move(particle_array);
	glutTimerFunc(milisec,update,0);
	//glutPostRedisplay();


}


//Draws the 3D scene
void drawScene() {//Clear information from last draw

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearDepth(1);
	glClearColor (0.0,0.0,0.0,1.0);

	glMatrixMode(GL_MODELVIEW); //Switch to the drawing perspective
	glLoadIdentity(); //Reset the drawing perspective

	glTranslatef(x_translate,y_translate,z_translate);
	glRotatef(angle_x,1,0,0.0f);
	glRotatef(angle_y,0,1,0.0f);

	if(disp_0 == 1){
		for(int i=0;i<N_particles;i++){
			glPushMatrix();
			//glColor3f(particle_array[i].r,particle_array[i].g,particle_array[i].b);
			glColor3f(particle_array[i].r,particle_array[i].g,particle_array[i].b);
			if(DIMENSION == 2) {
				glTranslatef((float)particle_array[i].p[0],(float)particle_array[i].p[1],-5.0f);
			}else{
				glTranslatef((float)particle_array[i].p[0],(float)particle_array[i].p[1],(float)particle_array[i].p[2]);
			}


			//glutSolidSphere(particle_array[i].radius,15,15);
			glutSolidSphere(particle_array[i].radius,15,15);

			glPopMatrix();
		}
	}else{
		glPushMatrix();
		//glColor3f(particle_array[i].r,particle_array[i].g,particle_array[i].b);
		glColor3f(particle_array[0].r,particle_array[0].g,particle_array[0].b);
		if(DIMENSION == 2) {
			glTranslatef((float)particle_array[0].p[0],(float)particle_array[0].p[1],-5.0f);
		}else{
			glTranslatef((float)particle_array[0].p[0],(float)particle_array[0].p[1],(float)particle_array[0].p[2]);
		}


		glutSolidSphere(particle_array[0].radius,15,15);
		glPopMatrix();

	}





	glutSwapBuffers(); //Send the 3D scene to the screen
}

