#include "common_newton.hpp"

using namespace std;

GLuint	dataBufferID;
particle_t* Particles_d;
particle_t* Particles_h;
float* heatEff_device;
float* dt_device;

cudaGraphicsResource *resources[1];

//simulation parameters
int N;
int DIM;
int k;
int bufferStride;
int NUM_BLOCKS;
int NUM_THREADS;
int buffer_size;

float radius_all = 0.1;
float* dataPtr;
int disp_0 = 1;
int simulation_started = 0;

VarParams vpar_h;
VarParams* vpar_d;
//GL n Cuda stuff

GLfloat angle_x = 0;
GLfloat angle_y = 0;
GLfloat z_translate =-20.0f;
GLfloat x_translate =0.0f;
GLfloat y_translate = 0.0f;



// simulation parameters - on Device
__device__ int bufferStride_d;
__device__ int N_d;
__device__ int DIM_d;

// variable sim parameters on device!
__device__ float heateffect_d;
__device__ float dt_d;


int init(int argc, char** argv)
{

        if(find_option(argc,argv,"-h") >0 || find_option(argc,argv,"--help")>0)
        {

                std::cout<<"Hello to our N-Body simulation program. While running the program, please provide an"<<
                         " appropriate input"<<std::endl;
                std::cout<<"type -h or --help for this menu."<<std::endl;
                std::cout<<"type -n <int> to specify number of bodies/particles to simulate (by default n=10). "<<std::endl;
                std::cout<<"type -big <int> to specify the number of bigger( heavier ) particles (default k = 0)."<<std::endl;
                std::cout<<"type -dim <int> to specify the euclidean dimension you wish to simulate in (by default dim=3)"<<std::endl;
                return 1;

        }

        N = read_int(argc, argv, "-n", 10);
        DIM = read_int(argc,argv,"-dim",3);
        k = read_int(argc,argv,"-big",1);
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

        if( k < 0 || k > N)
        {
                cout<<" The program parameter -big should be an integer between 1 and -n"<<endl;
                cout<<"setting -big to  its default value 0"<<endl;
                k = 0;
        }


        //set the variable parameters to default;
        vpar_h.dt_sim = 0.01;
        vpar_h.heateffect_sim = 1;


        // how many blocks, each with 1024 threads do we need to accomodate 1 body per thread  thread!

        bufferStride = DIM+4;
        buffer_size = bufferStride*N*sizeof(float);

        NUM_BLOCKS =(N+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
        NUM_THREADS = (N+NUM_BLOCKS-1)/NUM_BLOCKS;

        cout<< "NUM_BLOCKS = "<<NUM_BLOCKS <<", THREADS_PER_BLOCK = "<<NUM_THREADS<<", Num Particles = "<<N<<", DIM = "<<DIM<<endl;
        return 0;

}
/////////////////NBODY///////////////

int main(int argc, char** argv)
{

        // set program params
        if (init(argc,argv) == 1 )
                return 1;

        Particles_h = (particle_t*)malloc(N*sizeof(particle_t));
        cout<<"Initializing GLUT Context"<<endl;
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
        glutInitWindowSize(400, 400); //Set the window size
        //Create the window
        glutCreateWindow("Nbody Simulation");

        /*initialize graphics drawing functions
         * setup the GLViewPort & coordinate system
         * Generate GL Buffers*/
        cout<<"Initializing Graphics"<<endl;
        initGraphics(); //Initialize rendering

        //Set handler functions for drawing, keypresses, and window resizes
        glutDisplayFunc(drawScene);
        glutKeyboardFunc(handleKeypress);
        glutSpecialFunc(keyboardNavigator);
        glutReshapeFunc(handleResize);

        glewInit();
        //initialze Cuda Context and register the GL Buffer dataBufferID!
        cout<<"Initializing Cuda Context"<<endl;
        cudaSetDevice(0);
        cudaGLSetGLDevice(0);

        cout<<"Initializing Particle Data"<<endl;
        /*initialize particle data, allocate space on host*/
        initData(Particles_h);

        cout<<"Allocating Memory for Particles_d On Device"<<endl;
        // allocate memory on device for Particles_d and dataBuffer_d
        checkCudaErrors(cudaMalloc((void**)&Particles_d,N*sizeof(particle_t)));

        cout<<"Copying Memory Particles Data to device memory"<<endl;
        //copy particles DATA from Host to Memory
        checkCudaErrors(cudaMemcpy(Particles_d,Particles_h,N*sizeof(particle_t),cudaMemcpyHostToDevice));

	cout<<"Generating Buffers"<<endl;
        // generate GL Buffers to be shared with CUDA! - register GL Buffers with CUDA! allocate memory!
        createVBO();
        fillVBOCuda();

        cout<<"Initialization DONE! Handing over control to OpenGL"<<endl;
        glutMainLoop();
        return 0; //This line is never reached

}

void createVBO()
{

        // create buffer object
        glGenBuffers(1, &dataBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, dataBufferID);
        glBufferData(GL_ARRAY_BUFFER, buffer_size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(resources, dataBufferID, cudaGraphicsMapFlagsNone));

}



void fillVBOCuda()
{
        // cout<<"Filling Buffer"<<endl;
        cudaGraphicsMapResources(1, resources, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&dataPtr, &num_bytes,resources[0]);
        //fill the Graphics Resource with particle position Data!
        launch_kernel<<<NUM_BLOCKS,NUM_THREADS>>>(Particles_d,dataPtr,1);
        // unmap buffer object
        cudaGraphicsUnmapResources(1, resources, 0);
        //cudaMemcpy(Particles_h,Particles_d,N*sizeof(particle_t),cudaMemcpyDeviceToHost);
}


__global__ void setSimParams(ConstParams* cparams_d)
{

        N_d = cparams_d->N_sim;
        DIM_d = cparams_d->D_sim;
        bufferStride_d = cparams_d->buff_stride;
}

__global__ void setVarParams(VarParams* params_d)
{

        heateffect_d = params_d->heateffect_sim;
        dt_d = params_d->dt_sim;
}


// cuda Kernel - assumes all particle data is on device memory!
__global__ void launch_kernel(particle_t* Particles,GLfloat* dataBuffer_d, int KernelMode)
{
        int i = blockIdx.x*blockDim.x+threadIdx.x;

        if(i < N_d)
        {
                //if KernelMode = 1 then Update X
                if(KernelMode == 1)
                {
                        updateX(Particles+i);
                        ////and than update dataBuffer (for OpenGL drawScene function) !
                        for(int d=0; d<DIM_d; d++)
                                dataBuffer_d[i*bufferStride_d+d] = Particles[i].p[d]; // update the new coordinate positions in the data buffer!
                        // fill in also the RGB data and the radius. In general THIS IS NOT NECESSARY!! NEED TO PERFORM ONCE! REFACTOR!!!
                        dataBuffer_d[i*bufferStride_d+DIM_d] =Particles[i].r;
                        dataBuffer_d[i*bufferStride_d+DIM_d+1] =Particles[i].g;
                        dataBuffer_d[i*bufferStride_d+DIM_d+2] =Particles[i].b;
                        dataBuffer_d[i*bufferStride_d+DIM_d+3] =Particles[i].radius;
                }
                else
                {
                        // if KernelMode = 2 then Update Y
                        float* Fold = new float[DIM_d];
                        for(int d=0; d<DIM_d; d++)
                                Fold[d]=Particles[i].force[d];

                        //of course in parallel :)
                        computeForces(Particles,i);
                        updateV(Particles+i,Fold);
                        heateffect_d = 1;
                        delete [] Fold;
                }
        }
        // in either case wait for all threads to finish!
        __syncthreads();
}




__device__ void computeForces(particle_t* Particles,int i)
{

        // each thread computes all the interactions of particle i with all other particles!
        for(int d=0; d<DIM_d; d++)
                Particles[i].force[d] = 0;
        for(int j = 0; j< N_d; j++)
                if(i!=j)
                        applyForce(Particles+i,Particles+j);

}// computeForces

// device function. Assumes particle_i and neighbor_j are on device memory
__device__ void applyForce( particle_t * particle_i, particle_t* neighbor_j)
{
        
	float k_i = G*particle_i[0].m;
        float r=0;
        float k = k_i*neighbor_j[0].m;
        
	for(int d=0; d<DIM_d; d++)
                r+=sqr(neighbor_j[0].p[d]-particle_i[0].p[d]); // radius length computation
        
	r+= EPS2;
        float f =k/(sqrt(r)*r);
        // update the force on particle i with the force exerted by particle j
        for(int d=0; d<DIM_d; d++)
                particle_i[0].force[d]+=f*(neighbor_j[0].p[d]-particle_i[0].p[d]);
}




__device__  void updateX(particle_t* particle_i)
{

        float f = dt_d*dt_d/(2*particle_i->m);
        for(int d=0; d<DIM_d; d++)
                particle_i[0].p[d]+=dt_d*particle_i[0].v[d]+f*particle_i[0].force[d];
}



__device__ void updateV(particle_t* particle_i, float* Fold)
{
        float f = dt_d/(2*particle_i[0].m);
        for(int d=0; d<DIM_d; d++)
        {
                particle_i[0].v[d]+=f*(particle_i[0].force[d]+Fold[d]);
                particle_i[0].v[d]*=sqrt(heateffect_d);
        }
}

// performed on host !!
void initData(particle_t* Particles)
{
        srand48(time(NULL));
        for ( int i = 0; i<k; i++)
        {

                // initialize 'DARK MATTER PARTICLE' - black hole!!!
                for(int d=0; d<DIM; d++)
                {
                        Particles[i].p[d] = (drand48()-0.5)*20;
                        Particles[i].v[d] = 0;
                        Particles[i].force[d] = 0;
                }
                Particles[i].radius =  radius_all/10 ; // (drand48()+0.5)/10; // radius between 0.05 and 0.15
                Particles[i].r = 0;
                Particles[i].g = 0;
                Particles[i].b = 0;
                Particles[i].m = 100; // make the mass proportional to the volume
        }

        float rand;
        for(int i=k; i < N; i++)
        {
                // allocate storage for the position velocity and force vectors, respectively.
                for(int d=0; d<DIM; d++)
                {
                        rand = 	(drand48()-0.5)*size;
                        Particles[i].p[d] = rand >= 0 ? rand+1 : rand-1;
                        Particles[i].v[d] =(drand48()-0.5)*5;
                        Particles[i].force[d] = 0;
                }
                Particles[i].radius =  radius_all ; // (drand48()+0.5)/10; // radius between 0.05 and 0.15
                Particles[i].r =drand48();
                Particles[i].g = drand48();
                Particles[i].b = drand48();
                Particles[i].m = 1; // make the mass proportional to the volume
        }

        ConstParams* par_d;
        ConstParams par_h;
        par_h.N_sim = N;
        par_h.D_sim = DIM;
        par_h.buff_stride = bufferStride;

        cudaMalloc((void**)&par_d,sizeof(ConstParams));
        cudaMemcpy(par_d,&par_h,sizeof(ConstParams),cudaMemcpyHostToDevice);
        setSimParams<<<NUM_BLOCKS,NUM_THREADS>>>(par_d);

        cudaMalloc((void**)&vpar_d,sizeof(VarParams));
        cudaMemcpy(vpar_d,&vpar_h,sizeof(VarParams),cudaMemcpyHostToDevice);
        setVarParams<<<NUM_BLOCKS,NUM_THREADS>>>(vpar_d);

}

void freeData()
{
        delete[] Particles_h;
        cudaFree(Particles_d);
}

int find_option(int argc, char** argv, const char* option)
{

        for(int i=1; i<argc; i++)
        {
                if(strcmp(argv[i],option) == 0)
                {
                        return i;
                }

        }
        return -1;
}

int read_int(int argc, char** argv, const char* option, int default_value)
{

        int i = find_option(argc,argv, option);
        if(i>0 && i<argc-1)
                return atoi( argv[i+1] );

        return default_value;

}

///////////////////////////////graphics code/////////////////////////////////////



//Initializes 3D rendering
void initGraphics()
{
        
	//Makes 3D drawing work when something is in front of something else
        //OpenGL initialization stuff
        glEnable(GL_DEPTH_TEST);
        glEnable( GL_TEXTURE_2D );
        glDepthFunc(GL_LEQUAL);
        glClearColor(0.0f,0.0f,0.0f,1.0f);

        int w = 400;
        int h = 400;
        // setup the view port and coord system
        glViewport(0, 0, w,h);
        glMatrixMode(GL_PROJECTION); //Switch to setting the camera perspective

        //Set the camera perspective
        glLoadIdentity(); //Reset the camera
        gluPerspective(45.0,                  //The camera angle
                       (float)w / (float)h, //The width-to-height ratio
                       1.0,                   //The near z clipping coordinate
                       2000.0);                //The far z clipping coordinate


}

int mode = 1;
void update(int value)
{
        // cout<<"Filling Buffer"<<endl;
        cudaGraphicsMapResources(1, resources, 0);
        size_t num_bytes;
        cudaGraphicsResourceGetMappedPointer((void **)&dataPtr, &num_bytes,resources[0]);
        //fill the Graphics Resource with particle position Data!
        launch_kernel<<<NUM_BLOCKS,NUM_THREADS>>>(Particles_d,dataPtr,mode);
        // unmap buffer object
        cudaGraphicsUnmapResources(1, resources, 0);
        mode = (mode==1) ? 2 : 1;
        glutPostRedisplay();
        glutTimerFunc(milisec,update,0);
        
	//cout<<"AFTER CoPY"<<" copy buffer size:"<<sizeof(particle_t)<<endl;
        //cudaMemcpy(Particles_h,Particles_d,N*sizeof(particle_t),cudaMemcpyDeviceToHost);
        //for(int i = 0;i<N;i++){
        //cout<<"r,g,b = "<< Particles_h[i].r<<", "<<Particles_h[i].g<<", "<<Particles_h[i].b<<endl<<endl;
        //}
}




//Draws the 3D scene
void drawScene()  //Clear information from last draw
{
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearDepth(1);
        glClearColor (0.0,0.0,0.0,1.0);

        glMatrixMode(GL_MODELVIEW); //Switch to the drawing perspective
        glLoadIdentity(); //Reset the drawing perspective
        
	glTranslatef(x_translate,y_translate,z_translate);
        glRotatef(angle_x,1,0,0.0f);
        glRotatef(angle_y,0,1,0.0f);
        
	glBindBuffer(GL_ARRAY_BUFFER,dataBufferID);
        // map the data stored in the OpenGL buffer to our databuffer!
        GLfloat* data = (GLfloat*)glMapBuffer(GL_ARRAY_BUFFER,GL_READ_WRITE);

        // if display alO!
        if(disp_0 == 1)
        {

                for(int i=0; i<N*bufferStride; i+=bufferStride)
                {
                        glPushMatrix();
                        glColor3f(*(data+i+DIM),*(data+i+DIM+1),*(data+i+DIM+2));
                        if(DIM == 1)
                        {
                                glTranslatef(*(data+i),0.0,-5.0f);
                        }
                        else
                        {
                                if(DIM==2)
                                {
                                        glTranslatef(*(data+i),*(data+i+1),-5.0f);
                                }
                                else
                                {
                                        glTranslatef(*(data+i),*(data+i+1),*(data+i+2));
                                }
                        }

                        glutSolidSphere(*(data+i+DIM+3),15,15);
                        glPopMatrix();
                }// for loop
        }
        else
        {
                for(int i=0; i<k*bufferStride; i+=bufferStride)
                {

                        glPushMatrix();
                        glColor3f(*(data+i+DIM),*(data+i+DIM+1),*(data+i+DIM+2));
                        if(DIM == 1)
                        {
                                glTranslatef(*(data+i),0.0,-5.0f);
                        }
                        else
                        {
                                if(DIM==2)
                                {
                                        glTranslatef(*(data+i),*(data+i+1),-5.0f);
                                }
                                else
                                {
                                        glTranslatef(*(data+i),*(data+i+1),*(data+i+2));
                                }
                        }

                        glutSolidSphere(*(data+i+DIM+3),15,15);
                        glPopMatrix();

                }
        }
        
	// finish drawing the scene and now unmap and unbind dataBuffer!!!
        glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER,0);
        glutSwapBuffers(); //Send the 3D scene to the screen
}

void keyboardNavigator(int key, int x, int y)
{

        switch (key)
        {
        case GLUT_KEY_DOWN :
        {

                angle_x +=5;
                if(angle_x>360)
                        angle_x-=360;
                break;
        }

        case GLUT_KEY_UP :
        {
                angle_x -=5;
                if(angle_x<-360)
                        angle_x+=360;
                break;
        }

        case GLUT_KEY_LEFT :
        {
                angle_y -=5;
                if(angle_y<-360)
                        angle_y+=360;
                break;
        }

        case GLUT_KEY_RIGHT :
        {
                angle_y +=5;
                if(angle_y>360)
                        angle_y-=360;
                break;
        }
        case GLUT_KEY_F1:
        {
                if(simulation_started == 0)
                        glutTimerFunc(milisec,update,0);
        }

        default :
        {
                // do nothing
        }
        }

        glutPostRedisplay();
}

void handleKeypress(unsigned char key, int x, int y)
{

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
        {
                vpar_h.heateffect_sim+=0.05;
                cudaMemcpy(vpar_d,&vpar_h,sizeof(VarParams),cudaMemcpyHostToDevice);
                setVarParams<<<NUM_BLOCKS,NUM_THREADS>>>(vpar_d);

        }
        if(key=='c')
        {
                vpar_h.heateffect_sim -=0.05;
                if(vpar_h.heateffect_sim <0)
                        vpar_h.heateffect_sim = 0;
                cudaMemcpy(vpar_d,&vpar_h,sizeof(VarParams),cudaMemcpyHostToDevice);
                setVarParams<<<NUM_BLOCKS,NUM_THREADS>>>(vpar_d);
        }
        if(key == 'i')
        {
                vpar_h.dt_sim=-1*vpar_h.dt_sim;
                cudaMemcpy(vpar_d,&vpar_h,sizeof(VarParams),cudaMemcpyHostToDevice);
                setVarParams<<<NUM_BLOCKS,NUM_THREADS>>>(vpar_d);
        }
        if(key == '0')
                disp_0 = -1*disp_0;

        //cuda set the parameters


        glutPostRedisplay();


}





//Called when the window is resized
void handleResize(int w, int h)
{

        //Tell OpenGL how to convert from coordinates to pixel values
        glViewport(0, 0, w, h);

        glMatrixMode(GL_PROJECTION); //Switch to setting the camera perspective

        //Set the camera perspective
        glLoadIdentity(); //Reset the camera
        gluPerspective(45.0,                  //The camera angle
                       (float)w / (float)h, //The width-to-height ratio
                       1.0,                   //The near z clipping coordinate
                       200.0);                //The far z clipping coordinate
}










