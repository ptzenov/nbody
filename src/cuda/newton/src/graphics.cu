#include "graphics.hpp"
#include "iostream"

KERNEL void launch_simulation_kernel(Simulator* sim, float* position_data,
				     int val) {
	sim->set_position_data(position_data);
	sim->make_step();
}
extern void update(int);

Renderer::Renderer(Params sp, Simulator* simulator_ptr)
    : sim_params(sp),
      radius_data{new float[sp.sim_N]},
      rgb_data{new float[3 * sp.sim_N]} {
	sim_params.print_params();	
	size_t i_d = 0;
	for (size_t i = 0; i < sim_params.sim_N;
	     i++, i_d = i * sim_params.sim_DIM) {
		radius_data[i] = 1.;
		rgb_data[i_d] = 1;
		rgb_data[i_d + 1] = 1;
		rgb_data[i_d + 2] = 1;
	}

	simulator = simulator_ptr;
	std::cout<<"Constructor" << std::endl;
	disp_0 = 1;
	simulation_started = 0;

	angle_x = 0;
	angle_y = 0;
	z_translate = -20.0f;
	x_translate = 0.0f;
	y_translate = 0.0f;
	/** is the cuda resource mapped to the OPENGL buffer??
	*
	*/
	resource_mapped = false;
	buffer_ID = 1;
	// size of opengl/cuda buffer (in bytes)
	buffer_size =
	    (GLuint)sim_params.sim_N * sim_params.sim_DIM * sizeof(float);
	glGenBuffers(1, &buffer_ID);
	DBG_MSG;
	glBindBuffer(GL_ARRAY_BUFFER, buffer_ID);
	DBG_MSG;
	glBufferData(GL_ARRAY_BUFFER, buffer_size, 0,
		     GL_DYNAMIC_DRAW);  // allocate memory for the buffer
	DBG_MSG;
	glBindBuffer(GL_ARRAY_BUFFER, 0);  // unbind the buffer
	DBG_MSG;
	cudaGraphicsGLRegisterBuffer(
	    &buffer_resource, buffer_ID,
	    cudaGraphicsRegisterFlagsNone);  // register the gl buffer with cuda
      }

// Initializes 3D rendering
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
	glLoadIdentity();		     // Reset the camera
	gluPerspective(45.0,		     // The camera angle
		       (float)w / (float)h,  // The width-to-height ratio
		       1.0,		     // The near z clipping coordinate
		       2000.0);		     // The far z clipping coordinate
}

// Draws the 3D scene
void Renderer::draw_scene()  // Clear information from last draw
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearDepth(1);
	glClearColor(0.0, 0.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);  // Switch to the drawing perspective
	glLoadIdentity();	    // Reset the drawing perspective

	glTranslatef(x_translate, y_translate, z_translate);
	glRotatef(angle_x, 1, 0, 0.0f);
	glRotatef(angle_y, 0, 1, 0.0f);

	// if display alO!
	if (disp_0 == 1) {
		// map the data stored in the OpenGL buffer to our databuffer!
		glBindBuffer(GL_ARRAY_BUFFER, buffer_ID);
		// map the data stored in the OpenGL buffer to our databuffer!
		GLfloat* data =
		    (GLfloat*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

		// loop over all particles and drow then as spheres
		size_t i_d = 0;
		size_t i_c = 0;
		for (size_t i = 0; i < sim_params.sim_N; i++) {
			glPushMatrix();

			i_c = i * 3;
			i_d = i * sim_params.sim_DIM;

			glColor3f(rgb_data[i_c], rgb_data[i_c + 1],
				  rgb_data[i_c + 2]);
			switch (sim_params.sim_DIM) {
				case 1: {
					glTranslatef(data[i_d], 0.0, -5.0f);
					break;
				}
				case 2: {
					glTranslatef(data[i_d], data[i_d + 1],
						     -5.0f);
					break;
				}
				case 3: {
					glTranslatef(data[i_d], data[i_d + 1],
						     data[i_d + 2]);
					break;
				}
				default: { ; }
			}
			GLdouble radius = radius_data[i];
			glutSolidSphere(radius, 15, 15);
			glPopMatrix();
		}				   // for loop
		glBindBuffer(GL_ARRAY_BUFFER, 0);  // unmap the data buffer
	}
	glutSwapBuffers();  // Send the 3D scene to the screen
}

/**
 * Keyboard control routines
 */
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
				glutTimerFunc(sim_params.display_dt, update, 0);
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
			sim_params.temp += 0.05;
			cudaMemcpy(&sim_params, &simulator->params_d,
				   sizeof(Params), cudaMemcpyHostToDevice);
			break;
		case 'c':
			sim_params.temp -= 0.05;
			cudaMemcpy(&sim_params, &simulator->params_d,
				   sizeof(Params), cudaMemcpyHostToDevice);

			break;
		case 'i':
			sim_params.sim_dt = -sim_params.sim_dt;
			cudaMemcpy(&sim_params, &simulator->params_d,
				   sizeof(Params), cudaMemcpyHostToDevice);
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
	glLoadIdentity();		     // Reset the camera
	gluPerspective(45.0,		     // The camera angle
		       (float)w / (float)h,  // The width-to-height ratio
		       1.0,		     // The near z clipping coordinate
		       200.0);		     // The far z clipping coordinate
}
