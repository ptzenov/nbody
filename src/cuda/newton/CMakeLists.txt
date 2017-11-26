cmake_minimum_required(VERSION 3.0) 
project(NBODY)



set(CMAKE_CXX "nvcc")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "--std=c++11")

#Bring the headers, such as Student.h into the project
include_directories( "include"	)
link_directories( "lib"	)

message ("CXX IS ${CMAKE_CXX}")

# find some packages! 
find_package(OpenMP)
if (OPENMP_FOUND)
	message("OpenMP FOUND") 
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

find_package(OpenGL)
if (OPENGL_FOUND)
	message("OPENGL FOUND")
	include_directories(${OPENGL_INDLUCE_DIRS})
endif(OPENGL_FOUND)

find_package (GLUT)
if (GLUT_FOUND)
	message("GLUT FOUND!")
	include_directories(${GLUT_INCLUDE_DIRS})
endif (GLUT_FOUND)

find_package(CUDA)
if(CUDA_FOUND)
	message("CUDA FOUND!")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CUDA_CXX_FLAGS}")
	include_directories(
		${INCLUDE_DIRECTORIES}
		${CUDA_INCLUDE_DIRS}
		)
else(CUDA_FOUND)
	message("CUDA NOT FOUND")
endif(CUDA_FOUND)

message( "Include Dirs are:")
get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
	  message(STATUS "dir='${dir}'")
endforeach()

file(GLOB SOURCES "src/*.cpp")
add_executable(nbody ${SOURCES})

if (OPENGL_FOUND)
	target_link_libraries (nbody ${OPENGL_LIBRARIES})
endif(OPENGL_FOUND)

if (GLUT_FOUND)
	target_link_libraries (nbody ${GLUT_LIBRARIES})
endif (GLUT_FOUND)


if (CUDA_FOUND)
	target_link_libraries(nbody ${CUDA_LIBRARIES})
endif(CUDA_FOUND)
