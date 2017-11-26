#include "simulation.hpp"
#include "common.hpp"
#include <algorithm>

// to be launched from one thread! 
KERNEL void init_simulator(Simulator * simulator, Params params)
{
	simulator = Simulator::get_singleton(params);
}

KERNEL void launch_kernel(Simulator const * simulator)
{
;
}	

void fillVBOCuda()
{
        // cout<<"Filling Buffer"<<endl;
        cudaGraphicsMapResources(1, resources, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&dataPtr, &num_bytes,resources[0]);
        //fill the Graphics Resource with particle position Data!
        launch_kernel<<<NUM_BLOCKS,NUM_THREADS>>>(particles_d,dataPtr,1);
        // unmap buffer object
        cudaGraphicsUnmapResources(1, resources, 0);
}


// cuda Kernel - assumes all particle data is on device memory!
KERNEL void Simulator::simulate()
{
        int i = blockIdx.x*blockDim.x+threadIdx.x;

        if(i < sim_params_d.sim_N)
        {
                //if KernelMode = 1 then Update X
                updateX(i);
                
                float Fold[3];
		computeForces(i,Fold);
                updateV(i,Fold);
        }
        // in either case wait for all threads to finish!
        __syncthreads();
}

DEVICE  void Simulator::updateX(int i)
{

        float f = var_params_d.dt*var_params_d.dt/(2*mass_data.get()[i]);
        int idx = i*sim_params_d.sim_DIM;
        for(size_t d=0; d<sim_params_d.sim_DIM; d++)
                position_data[idx+d] += var_params_d.dt*velocity_data.get()[idx+d]+
                                        f*force_data.get()[idx+d];
}



DEVICE void Simulator::computeForces(int i,float Fold[3])
{

        // each thread computes all the interactions of particle i with all other particles!
        int idx = i*sim_params_d.sim_DIM;
        for(int d=0; d<sim_params_d.sim_DIM; d++)
        {
                Fold[d] = force_data.get()[idx+d];
                force_data.get()[idx+d] = 0;
        }
        for(int j = 0; j< sim_params_d.sim_N; j++)
                if(i!=j)
                        applyForce(i,j);
}

DEVICE void Simulator::applyForce(size_t i,size_t j)
{
        float k_i = G*mass_data.get()[i];
        float r=0;
        float k = k_i*mass_data.get()[j];
        int idx_i = i*sim_params_d.sim_DIM;
        int idx_j = j*sim_params_d.sim_DIM;
        for(size_t d=0; d<sim_params_d.sim_DIM; d++)
                r+=sqr(position_data.get()[idx_j+d]-position_data.get()[idx_i+d]); // radius length computation

        r+= EPS2;
        float f =k/(sqrt(r)*r); // shoul dbe available readily on the DEVICE
        // update the force on particle i with the force exerted by particle j
        for(int d=0; d<sim_params_d.sim_DIM; d++)
                force_data.get()[idx_i+d]+=f*(position_data.get()[idx_j+d]-position_data.get()[idx_i+d]);
}


DEVICE void Simulator::updateV(size_t i,float Fold[3])
{
        float f = var_params_d.dt/(2*particle_i.m);
	size_t idx = i*sim_params_d.sim_DIM;
        for(int d=0; d<sim_params_d.sim_DIM; d++)
        {
                velocity_data.get()[idx+d] +=f*(force_data.get()[idx+d]+Fold[d]);
                velocity_data.get()[idx+d] *=sqrt(var_params_d.temp);
        }
}

// performed on host !!
DEVICE void Simulator::init_data()
{
        srand48(time(NULL));
        int i = threadIdx.x+blockIdx.x*blockDim.x;
        int i_d = i*sim_params_d.sim_DIM;

        // initialize 'DARK MATTER PARTICLE' - black hole!!!
        for(int d=0; d<DIM; d++)
        {
                position_data.get()[i_d+d] = (drand48()-0.5)*20;
                velocity_data.get()[i_d+d] = 0;
                force_data.get()[i_d+d] = 0;
        }

        radius_data.get()[i] =  radius_all/10 ; // (drand48()+0.5)/10; // radius between 0.05 and 0.15
        rgb_data.get()[i*3] = 0;
        rgb_data.get()[i*3+1] = 0;
        rgb_data.get()[i*3+2] = 0;
        mass_data.get()[i] = 10; // make the mass proportional to the volume
}


//Initializes 3D rendering
void update(int value)
{
        cudaGraphicsMapResources(1, resources, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&dataPtr, &num_bytes,resources[0]);
        // calculate particles 
        launch_kernel<<<cuda_params.NUM_BLOCKS,cuda_params.NUM_THREADS>>>();
        // unmap buffer object
        cudaGraphicsUnmapResources(1, resources, 0);
        glutPostRedisplay();
        glutTimerFunc(milisec,update,0);
}

