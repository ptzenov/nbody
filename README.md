NBODY simulations implementation on CPU and an NVIDIA-GPU (using CUDA)

Two distinct simulation scenarios
1) newton - a full nxn simulator for gravitational interaction between n distinct massive objects
2) browanian - brownian motion simulator - the motion of a single large massive particle is determined by the random collisions of the latter with a multitude of small massive "molecules" ( e.g. pollen and molecules of water)

Visualization is implemented with OpenGL. To compile and run the serial version you will need to install a 
version of freeglut3 via the command (in linux)
 
sudo apt-get install freeglut3-dev

as well as the glew library via 

sudo apt-get install libglew-dev

############# SERIAL VERSION #####################

navigate to src/serial -> chosing newton or brownian and run 

make // to compile 
make clean // to clean 

./nbody --help to query the initial arguments 
./nbody to run default 

############# CUDA VERSION #####################

currently the CUDA version is under re-construction. to be able to compile the cuda version navigate to
src/cuda/newtonian. Compilation here works with cmake according to the conventional way:

mkdir build

cd build 

cmake .. 

make

############# During simulation #####################

press "F1" to start the simulaiton 
press "w-a-s-d" to move the camera viewpoint
press the "arrow keys" to rotate the camera viewpoint  
press "h" to heat up the simulation
press "c" to cool down the simulation
press "i" to reverse the direction of time 

Enjoy and modify the code at will! 
