# environment
SM := 35

GCC := g++
NVCC := nvcc

# Remove function
RM = rm -f
 
# Specify opencv Installation
#opencvLocation = /usr/local/opencv
opencvLIB= -L/shared/apps/opencv-3.0.0-beta/INSTALL/lib
opencvINC= -I/shared/apps/opencv-3.0.0-beta/INSTALL/include

# Compiler flags:
# -g    debugging information
# -Wall turns on most compiler warnings
GENCODE_FLAGS := -gencode arch=compute_$(SM),code=sm_$(SM)
LIB_FLAGS := -lcudadevrt -lcudart

NVCCFLAGS := -O3
GccFLAGS = -fopenmp -O3 

OPENCV_LINK =  -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_photo -lopencv_video -lopencv_imgcodecs 

all: build

debug: GccFLAGS += -DDEBUG -g -Wall
debug: NVCCFLAGS += -g -G
debug: all

# The build target executable:
TARGET = edge_detector

build: $(TARGET)

$(TARGET): dlink.o main.o $(TARGET).o
	$(NVCC) $(NVCCFLAGS) $(opencvLIB) $(opencvINC) $^ -o $@ $(GENCODE_FLAGS) $(OPENCV_LINK) -link 

dlink.o: $(TARGET).o 
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(GENCODE_FLAGS) -dlink

main.o: main.cpp
	$(GCC) $(GccFLAGS) $(opencvLIB) $(opencvINC) -c $< -o $@
	
$(TARGET).o: $(TARGET).cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@ $(GENCODE_FLAGS) 
	
clean:
	$(RM) $(TARGET) *.o *.o *.tar* *.core* *out*.jpg 

