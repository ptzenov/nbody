#include "common.hpp"
#include "graphics.hpp"


extern int update(int);

GLuint dataBufferID;

Renderer::Renderer(Params sp, Simulator* simtor) {
	disp_0 = 1;
	simulation_started = 0;
	sim_ptr = simtor;
	// GL stuff
	angle_x = 0;
	angle_y = 0;
	z_translate = -20.0f;
	x_translate = 0.0f;
	y_translate = 0.0f;
}

// Initializes 3D rendering
void Renderer::update(int value) {
	// calculate particles
	launch_kernel<< sim_params.NUM_BLOCKS, sim_params.NUM_THREADS>>(sim_ptr);
	// unmap buffer object
	glutPostRedisplay();
	glutTimerFunc(milisec, update, 0);
}
void Renderer::init_graphics() {
	// Makes 3D drawing work when something is in front of something else
	// OpenGL initialization stuff
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glDepthFunc(GL_LEQUAL);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	int w = 400;
	int h = 400;
	// setup the view port and coord system
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);  // Switch to setting the camera
				      // perspective

	// Set the camera perspective
	glLoadIdentity();  // Reset the camera
	gluPerspective(45.0,  // The camera angle
		       (float)w / (float)h,  // The width-to-height ratio
		       1.0,  // The near z clipping coordinate
		       2000.0);  // The far z clipping coordinate
}

// Draws the 3D scene
void Renderer::draw_scene()  // Clear information from last draw
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearDepth(1);
	glClearColor(0.0, 0.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);  // Switch to the drawing perspective
	glLoadIdentity();  // Reset the drawing perspective

	glTranslatef(x_translate, y_translate, z_translate);
	glRotatef(angle_x, 1, 0, 0.0f);
	glRotatef(angle_y, 0, 1, 0.0f);

	// map the data stored in the OpenGL buffer to our databuffer!
	GLfloat* data = nullptr;
	
	// if display alO!
	if (disp_0 == 1) {
		// loop over all particles and drow then as spheres
		for (size_t i = 0; i < sim_params.sim_N * sim_params.sim_DIM;i+=sim_params.sim_DIM) {
			
			glPushMatrix();
			glColor3f(1,1,1); // contain colors here 
			for( size_t d = 0;d<sim_params.sim_DIM;d++) // contian position here
					glTranslatef(*(data + i), 0.0, -5.0f);
			GLdouble radius = 1.0;	
			glutSolidSphere(radius,15, 15);
			glPopMatrix();
		}  // for loop
	}
	glutSwapBuffers();  // Send the 3D scene to the screen
}

void Renderer::keyboard_navigator(int key, int x, int y) {
	switch (key) {
		case GLUT_KEY_DOWN: {
			angle_x += 5;
			if (angle_x > 360) angle_x -= 360;
			break;
		}

		case GLUT_KEY_UP: {
			angle_x -= 5;
			if (angle_x < -360) angle_x += 360;
			break;
		}

		case GLUT_KEY_LEFT: {
			angle_y -= 5;
			if (angle_y < -360) angle_y += 360;
			break;
		}

		case GLUT_KEY_RIGHT: {
			angle_y += 5;
			if (angle_y > 360) angle_y -= 360;
			break;
		}
		case GLUT_KEY_F1: {
			if (simulation_started == 0)
				glutTimerFunc(milisec, &simulate, 0);
			break;
		}

		default: {
			// do nothing
		}
	}

	glutPostRedisplay();
}

void Renderer::handle_keypress(unsigned char key, int x, int y) {
	switch (key) {
		case '+':
			z_translate += dist;
			break;
		case '-':
			z_translate -= dist;
			break;
		case 'a':
			x_translate += dist;
			break;
		case 'd':
			x_translate -= dist;
			break;
		case 's':
			y_translate += dist;
			break;
		case 'w':
			y_translate -= dist;
			break;
		case 'h':
			var_params.heateffect_sim += 0.05;
			// cudaMemcpy(var_params_d,&vpar_h,sizeof(VarParams),cudaMemcpyHostToDevice);
			// setVarParams<<<NUM_BLOCKS,NUM_THREADS>>>(vpar_d);
			break;
		case 'c':
			var_params.heateffect_sim -= 0.05;
			// if(vpar_h.heateffect_sim <0)
			//         vpar_h.heateffect_sim = 0;
			//  cudaMemcpy(vpar_d,&vpar_h,sizeof(VarParams),cudaMemcpyHostToDevice);
			//  setVarParams<<<NUM_BLOCKS,NUM_THREADS>>>(vpar_d);
			break;
		case 'i':
			var_params.dt_sim = -var_params.dt_sim;
			// cudaMemcpy(vpar_d,&vpar_h,sizeof(VarParams),cudaMemcpyHostToDevice);
			// setVarParams<<<NUM_BLOCKS,NUM_THREADS>>>(vpar_d);
			break;

		case '0':
			disp_0 = -1 * disp_0;
			break;
		default: { ; }
	}
	glutPostRedisplay();
}

// Called when the window is resized
void Renderer::handle_resize(int w, int h) {
	// Tell OpenGL how to convert from coordinates to pixel values
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);  // Switch to setting the camera
				      // perspective
	// Set the camera perspective
	glLoadIdentity();  // Reset the camera
	gluPerspective(45.0,  // The camera angle
		       (float)w / (float)h,  // The width-to-height ratio
		       1.0,  // The near z clipping coordinate
		       200.0);  // The far z clipping coordinate
}
