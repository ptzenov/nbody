/*
 * graphics.h
 *
 *  Created on: 10 May 2013
 *      Author: kenny
 */

#ifndef _COMMON_H_
#define _COMMON_H_
#include <GL/glut.h>
#include <GL/gl.h>
#include <math.h>
#include <stdlib.h>


/////////////////NBODY///////////////

#define milisec 25
#define G 1
#define sqr(x) ((x)*(x))
#define size 3
#define cutoff  (2*radius_all)
#define min_r   (cutoff/100)

extern double heateffect;
extern double T;
extern double Ekin;
extern double dt;
extern double radius_all;

struct particle_t{
	double* p;
	double* v;
	double* force;
	double m;
	float r,g,b;
	float radius;
	~particle_t();
};






/***************************SIMULATION.cpp*********************************/


/*User input handling functions*/

int find_option(int argc, char** argv, const char* option);
int read_int(int argc, char** argv, const char* option,int default_value);
int saveToFile(int argc, char** argv, const char* option);



/**/
void initData(particle_t* Particles, int DIM, int N);
void freeData(particle_t* Particles);




/*Application Logic's Functions*/

void computeForces(particle_t*P);

void updateX(particle_t* P);
void updateV(particle_t* P, double* Fold);

void applyForce( particle_t * particle, particle_t* neighbor);
void move(particle_t* Particles);
/*In case the user wants to read initial configuraiton from a file*/
//int init_particles(particle* Particles, int N, char* filename, int DIM);



/*****************GRAPHICS.cpp**************************/
void idle(void);
void drawScene();
void update(int value);
void handleResize(int w, int h);
void initGraphics(int, int , particle_t*);
void keyboardNavigator(int key, int x, int y);
void handleKeypress(unsigned char key, int x, int y);


#endif /* common_H_ */
