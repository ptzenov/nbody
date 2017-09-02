BINARY = nbody
BUILDDIR = build
SOURCEDIR = src

CXX = g++
CXXFLAGS = -Wall -g -o3 -lglut -lGLU -lGL 
RM = rm -f
SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)
OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDDIR)/%.o, $(SOURCES))

.PHONY: all clean

all: $(BINARY)

$(BINARY): $(OBJECTS)
	$(CXX) $(OBJECTS) $(CXXFLAGS) -o $(BINARY)

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp
	$(CXX)  $(SOURCETREE)  -c $< -o $@ 

clean:
	$(RM) $(OBJECTS)
