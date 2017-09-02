
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include "common.hpp"

/*Main object to hold particles*/

int DIM_SIMULATION;
int N_SIMULATION;
double heateffect = 1;
double T;
double Ekin;
double dt = 0.01;
#define EPS2 0.001

double radius_all = 0.1;

particle_t::~particle_t(){
	delete[] p;
	delete[] v;
	delete[] force;
}



void computeForces(particle_t* Particles){

	for(int i=0;i<N_SIMULATION;i++){
		// delete previously computed fore
		for(int d=0;d<DIM_SIMULATION;d++)
			Particles[i].force[d] = 0;

		double k_i = G*Particles[i].m;
		for(int j=0;j<N_SIMULATION;j++){
			if(i!=j){
				double r=0;
				double k = k_i*Particles[j].m;

				for(int d=0;d<DIM_SIMULATION;d++){
					r+=sqr(Particles[j].p[d]-Particles[i].p[d]); // radius length computation
				}
				r+= EPS2;

				double f =k/(sqrt(r)*r);
				for(int d=0;d<DIM_SIMULATION;d++)
					Particles[i].force[d]+=f*(Particles[j].p[d]-Particles[i].p[d]);
			}// if i != j
		} // for j
	}// for i
}// computeForces


void move(particle_t* Particles){

	double Fold[N_SIMULATION][DIM_SIMULATION];

	for(int i=0;i<N_SIMULATION;i++)
		updateX(Particles+i);

	glutPostRedisplay();

	for(int i=0;i<N_SIMULATION;i++)
		for(int d=0;d<DIM_SIMULATION;d++)
			Fold[i][d]=Particles[i].force[d];

	computeForces(Particles);
	for(int i=0;i<N_SIMULATION;i++)
		updateV(Particles+i,Fold[i]);

}



void initData(particle_t* Particles, int DIM, int N){


	DIM_SIMULATION = DIM;
	N_SIMULATION = N;

	srand48(time(NULL));

	// initialize 'DARK MATTER PARTICLE' - black hole!!!
	Particles[0].p =(double*)malloc(DIM_SIMULATION*sizeof(double));
	Particles[0].v =(double*)malloc(DIM_SIMULATION*sizeof(double));
	Particles[0].force =(double*)malloc(DIM_SIMULATION*sizeof(double));

	for(int d=0;d<DIM_SIMULATION;d++){
		Particles[0].p[d] = 0;//(drand48()-0.5)*size;
		Particles[0].v[d] = 0;
		Particles[0].force[d] = 0;
	}
	Particles[0].radius =  radius_all*10 ; // (drand48()+0.5)/10; // radius between 0.05 and 0.15
	Particles[0].r = 1;
	Particles[0].g = 1;
	Particles[0].b = 0;
	Particles[0].m = 1; // large mass "the sun"

	double rand;
	for(int i=1;i<N_SIMULATION;i++){
		// allocate storage for the position velocity and force vectors, respectively.
		Particles[i].p =(double*)malloc(DIM_SIMULATION*sizeof(double));
		Particles[i].v =(double*)malloc(DIM_SIMULATION*sizeof(double));
		Particles[i].force =(double*)malloc(DIM_SIMULATION*sizeof(double));

		for(int d=0;d<DIM_SIMULATION;d++){
			rand = 	(drand48()-0.5)*size; 
			Particles[i].p[d] = rand >= 0 ? rand+1 : rand-1;
			Particles[i].v[d] =(drand48()-0.5);
			Particles[i].force[d] = 0;
		}
		Particles[i].radius =  radius_all ; // (drand48()+0.5)/10; // radius between 0.05 and 0.15
		Particles[i].r = drand48();
		Particles[i].g = drand48();
		Particles[i].b = drand48();
		Particles[i].m = 3e-6; // small mass! 
	}

	computeForces(Particles);


	//DIM_SIMULATION = DIM;
	//N_SIMULATION  = N;
	///*SUN*/
	//Particles[0].p = new double[DIM_SIMULATION];
	//Particles[0].v = new double[DIM_SIMULATION];
	//Particles[0].force = new double[DIM_SIMULATION];


	//Particles[0].p[0] = 0.;
	//Particles[0].p[1] = 0.;
	//Particles[0].p[2] = 0.;

	//Particles[0].v[0] = 0.;
	//Particles[0].v[1] = 0.;
	//Particles[0].v[2] = 0.;

	//Particles[0].m = 1;

	//Particles[0].r = 0.5f;
	//Particles[0].g = .0f;
	//Particles[0].b = 0.0f;
	//Particles[0].radius = (float)radius_all;


	///*EARTH*/
	//Particles[1].p = new double[DIM_SIMULATION];
	//Particles[1].v = new double[DIM_SIMULATION];
	//Particles[1].force = new double[DIM_SIMULATION];

	//Particles[1].p[0] = 0.;
	//Particles[1].p[1] = 1.;
	//Particles[1].p[2] = 0.;
	//Particles[1].v[0] = -1.;
	//Particles[1].v[1] = 0.;
	//Particles[1].v[2] = 0.;
	//Particles[1].m = 3.0e-6;
	//Particles[1].r = 0.0f;
	//Particles[1].g = 0.0f;
	//Particles[1].b = 1.0f;
	//Particles[1].radius = (float)radius_all/3;



	///*JUPITER*/
	//Particles[2].p = new double[DIM_SIMULATION];
	//Particles[2].v = new double[DIM_SIMULATION];
	//Particles[2].force = new double[DIM_SIMULATION];

	//Particles[2].p[0] = 0.;
	//Particles[2].p[1] = 5.36;
	//Particles[2].p[2] = 0.;
	//Particles[2].v[0] = -0.425;
	//Particles[2].v[1] = 0.;
	//Particles[2].v[2] = 0.;
	//Particles[2].m = 9.55e-4;
	//Particles[2].r = 0.4f;
	//Particles[2].g = .0f;
	//Particles[2].b = 1.0f;
	//Particles[2].radius = (float)radius_all/2;

	///*Halley Commet*/
	//Particles[3].p = new double[DIM_SIMULATION];
	//Particles[3].v = new double[DIM_SIMULATION];
	//Particles[3].force = new double[DIM_SIMULATION];

	//Particles[3].p[0] = 34.75;
	//Particles[3].p[1] = 0.;
	//Particles[3].p[2] = 0.;
	//Particles[3].v[0] = 0.;
	//Particles[3].v[1] = 0.0296;
	//Particles[3].v[2] = 0.;
	//Particles[3].m = 1.e-14;

	//Particles[3].r = 1.0f;
	//Particles[3].g = 0.0f;
	//Particles[3].b = 0.0f;

	//Particles[3].radius = (float)radius_all/2;
	//computeForces(Particles);

}

void freeData(particle_t* Particles){
	delete[] Particles;
}

int find_option(int argc, char** argv, const char* option){

	for(int i=1;i<argc;i++){

		if(strcmp(argv[i],option) == 0){
			return i;
		}

	}
	return -1;

}

int read_int(int argc, char** argv, const char* option, int default_value){
	
	int i = find_option(argc,argv, option);
	if(i>0 && i<argc-1)
		return atoi( argv[i+1] );
	return default_value;

}

void updateX(particle_t* i){
	float f = dt*dt/(2*i->m);
	for(int d=0;d<DIM_SIMULATION;d++)
		i->p[d]+=dt*i->v[d]+f*i->force[d];
}

void updateV(particle_t* i, double* Fold){
	
	float f = dt/(2*i->m);
	for(int d=0;d<DIM_SIMULATION;d++)
		i->v[d]+=f*(i->force[d]+Fold[d]);
}


