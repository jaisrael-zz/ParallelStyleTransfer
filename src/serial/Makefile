#Compile and link flags
CXX = g++
CFLAGS = -Wall -g
#Compilation
CXXFLAGS += `pkg-config opencv --cflags`

#Linking
LDFLAGS += -lm `pkg-config opencv --libs`

#Name your target executables here
all = stylexfer	

all: $(all)

# Program dependencies (.o files will be compiled by implicit rules)
ImageAlignment.o: ImageAnalogy.h

main.o: main.cpp ImageAnalogy.h
#main.o: main.cpp LaplacianBlender.h SeamMaker.h
	$(CXX) $(CXXFLAGS) -c main.cpp

stylexfer: main.o ImageAnalogy.o
	$(CXX) $(CXXFLAGS)  -o $@ main.o ImageAnalogy.o $(LDFLAGS)
#blend: main.o LaplacianBlender.o SeamMaker.o
#	$(CXX) $(CXXFLAGS)  -o $@ main.o LaplacianBlender.o SeamMaker.o $(LDFLAGS)

clean:
	rm -f $(all) *.o
