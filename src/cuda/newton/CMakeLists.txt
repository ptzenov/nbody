cmake_minimum_required(VERSION 3.8) 

project(NBODY LANGUAGES CXX CUDA) # already fidns and sets the correct CXX and cuda compiler

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "--std=c++11")
set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-arch=sm_30")

#Bring the headers, such as Student.h into the project
include_directories( "include"	)
link_directories( "lib"	)

# find some packages! 
find_package(OpenMP)
if (OPENMP_FOUND)
	message("OpenMP FOUND") 
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

find_package(OpenGL REQUIRED)
if (OPENGL_FOUND)
	message("OPENGL FOUND")
	include_directories(${OPENGL_INDLUCE_DIRS})
endif(OPENGL_FOUND)

find_package (GLUT REQUIRED)
if (GLUT_FOUND)
	message("GLUT FOUND!")
	include_directories(${GLUT_INCLUDE_DIRS})
endif (GLUT_FOUND)

find_package (GLEW REQUIRED)
if (GLEW_FOUND)
	message("GLEW FOUND!")
	include_directories(${GLEW_INCLUDE_DIRS})
endif (GLEW_FOUND)


find_package(CUDA REQUIRED)
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

# recover all source files
file(GLOB SOURCES "src/*.cpp" "src/*.cu")
message( "Sources are: ") 
foreach(src ${SOURCES} )
		message( "Src = ${src}")
endforeach()

add_executable(nbody ${SOURCES})

set_target_properties(nbody PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_features(nbody PUBLIC cxx_std_11)

if (CUDA_FOUND)
	target_link_libraries(nbody ${CUDA_LIBRARIES})
endif(CUDA_FOUND)

if (OPENGL_FOUND)
	target_link_libraries (nbody ${OPENGL_LIBRARIES})
endif(OPENGL_FOUND)

if (GLUT_FOUND)
	target_link_libraries (nbody ${GLUT_LIBRARIES})
endif (GLUT_FOUND)

if (GLEW_FOUND)
	target_link_libraries (nbody ${GLEW_LIBRARIES})
endif (GLEW_FOUND)
