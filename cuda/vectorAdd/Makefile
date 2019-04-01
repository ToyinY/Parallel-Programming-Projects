all: vector_add.cu
	nvcc vector_add.cu -arch=sm_35 -O3 -o vector_add

clean: 
	rm -rf vector_add
