
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include "common.hpp"
using namespace std;
/////////////////NBODY///////////////

/*Main object to hold particles*/
double heateffect = 1;
double T;
double Ekin;
double dt = 0.001;
int N_SIMULATION;
int DIM_SIMULATION;
double radius_all = 0.01;
double sigma = 2*radius_all+2*radius_all/10.0;
double sigma_six = sigma*sigma*sigma*sigma*sigma*sigma;

particle_t::~particle_t(){

	delete[] p;
	delete[] v;
	delete[] force;

	}



void applyForce( particle_t* i, particle_t* j)
{

	double *r_vec = new double[DIM_SIMULATION];
	double r2 =0;
	for(int d=0;d<DIM_SIMULATION;d++){
		
		r_vec[d] = i->p[d]-j->p[d];
		r2 +=r_vec[d]*r_vec[d]; 
	}
	double radius_sum = i->radius+j->radius;
	
	if(r2 < radius_sum*(1.1)*radius_sum*(1.1)) 
	{	
		double u1, u2;
		for(int d=0;d<DIM_SIMULATION;d++){
		u1 = i->v[d];
		u2 = j->v[d];
		i->v[d]=(u1*(i->m-j->m)+2*j->m*u2)/(i->m+j->m);
		j->v[d]=(u2*(j->m-i->m)+2*i->m*u1)/(i->m+j->m);;
		}
		cout<<endl;
	}
	
	delete[] r_vec;



}



void computeForces(particle_t* Particles){

	for(int i=0;i<N_SIMULATION;i++){
		// set the current acceleration to 0;
		for(int d=0;d<DIM_SIMULATION;d++)
				Particles[i].force[d]=0;
		for(int j=i+1;j<N_SIMULATION;j++){
			//if(i!=j)
			applyForce(Particles+i,Particles+j);
		}
		


	}

}// computeForces


void move(particle_t* Particles){

	
	Ekin = 0;
	for(int i=0;i<N_SIMULATION;i++){

	// simple integrator!
	//kinetic energy of ith particle
	double E_i =0;	
	for(int d=0;d<DIM_SIMULATION;d++){
		//Particles[i].v[d]+=Particles[i].force[d]*dt/Particles[i].m;
		// apply the heating effect - speed up or slow down
		Particles[i].v[d]*=sqrt(heateffect);
		Particles[i].p[d]+=Particles[i].v[d]*dt;
		E_i+=Particles[i].v[d]*Particles[i].v[d];	
	}
	//calculate total kinetic energy as to compute the temperature! 
	Ekin+=0.5*Particles[i].m*E_i;

	for(int d=0;d<DIM_SIMULATION;d++){
		if((Particles[i].p[d] - Particles[i].radius)< -size/2 || (Particles[i].p[d] + Particles[i].radius) > size/2){
			Particles[i].p[d] = ((Particles[i].p[d] - Particles[i].radius) <-size/2) ? 
				(-size/2+(Particles[i].radius)) : size/2-(Particles[i].radius) ;
			Particles[i].v[d] = -(Particles[i].v[d]); // change the direction of the speed!
		}
	}
	}
	heateffect =1;
	computeForces(Particles);
	
} // move!



void initData(particle_t* Particles, int DIM, int N){
	DIM_SIMULATION = DIM;
	N_SIMULATION = N;



	
	srand48(time(NULL));
	// make a bigger particle!!
	Particles[0].p =(double*)malloc(DIM_SIMULATION*sizeof(double));
	Particles[0].v =(double*)malloc(DIM_SIMULATION*sizeof(double));
	Particles[0].force =(double*)malloc(DIM_SIMULATION*sizeof(double));
	
	for(int d=0;d<DIM_SIMULATION;d++){
			Particles[0].p[d] = (drand48()-0.5)*size;//(drand48()-0.5)*size;
			Particles[0].v[d] =(drand48()-0.5)*10;
			Particles[0].force[d] = 0;
     	}
		Particles[0].radius =  10*radius_all ; // (drand48()+0.5)/10; // radius between 0.05 and 0.15
		Particles[0].r =1;
		Particles[0].g = 0;
		Particles[0].b = 0;
		Particles[0].m = 10; // make the mass proportional to the volume	
		
	for(int i=1;i<N_SIMULATION;i++){
		// allocate storage for the position velocity and force vectors, respectively.
		Particles[i].p =(double*)malloc(DIM_SIMULATION*sizeof(double));
		Particles[i].v =(double*)malloc(DIM_SIMULATION*sizeof(double));
		Particles[i].force =(double*)malloc(DIM_SIMULATION*sizeof(double));

		for(int d=0;d<DIM_SIMULATION;d++){
			Particles[i].p[d] = (drand48()-0.5)*size;//(drand48()-0.5)*size;
			Particles[i].v[d] =(drand48()-0.5)*10;
			Particles[i].force[d] = 0;
     	}
		Particles[i].radius =  radius_all ; // (drand48()+0.5)/10; // radius between 0.05 and 0.15
		Particles[i].r =0;
		Particles[i].g = 0;
		Particles[i].b = 1;
		Particles[i].m = 1; // make the mass proportional to the volume
	}

	computeForces(Particles);
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
	//pass
}

void updateV(particle_t* i, double* Fold){
	//pass
}
