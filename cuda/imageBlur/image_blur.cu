#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TILE_SIZE 16

unsigned char *d_input;
unsigned char *d_output;
float *d_filter;

inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

__global__ void gaussianBlur(unsigned char *input,
							 unsigned char *output,
							 unsigned int rows,
							 unsigned int cols,
							 float *filter,
							 int filter_width) {
	
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x > cols || y > rows)
		return;
	int index = y * cols + x;

	// Blur algorthm using weighted average (recommended)
	float result = 0.0;
	for (int r = -filter_width / 2; r < filter_width / 2; r++) {
		for (int c = -filter_width / 2; c < filter_width / 2; c++) {
			int cur_row = r + y; 
			int cur_col = c + x;
			//if pixel is not at the edge of the image
			if ((cur_row > -1) && (cur_row < rows) &&
				(cur_col > -1) && (cur_col < cols)) { 
				int filter_id = (r + filter_width / 2) * filter_width + (c + filter_width / 2);	
				result += input[cur_row * cols + cur_col] * filter[filter_id]; 
			}
		}
	}
	output[index] = result;
}

void imageBlur (unsigned char* h_input,
			    unsigned char* h_output,
				unsigned int rows,
			    unsigned int cols,
				float* h_filter,
				int filter_width) {

	// block and grid size
	int gridX = 1 + ((cols - 1) / TILE_SIZE);
	int gridY = 1 + ((rows - 1) / TILE_SIZE);
	dim3 dimGrid(gridX, gridY);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	// allocate memory and copy to GPU
	int size = rows * cols;
	checkCuda(cudaMalloc((void**)&d_input, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&d_output, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&d_filter, filter_width * filter_width * sizeof(float)));
	checkCuda(cudaMemset(d_output, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemcpy(d_input, h_input, size * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_filter, h_filter, filter_width * filter_width * sizeof(float), cudaMemcpyHostToDevice));
		
	//kernel call
	gaussianBlur<<<dimGrid, dimBlock>>>(d_input, d_output, rows, cols, d_filter, filter_width);

	//copy output to host
	checkCuda(cudaMemcpy(h_output, d_output, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// free memory
	checkCuda(cudaFree(d_input));
	checkCuda(cudaFree(d_output));
}
