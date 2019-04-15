#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define TILE_SIZE 16

unsigned char *d_input;
unsigned char *d_output_blur;
unsigned char *d_output_sobel;
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
	if (x >= cols || y >= rows)
		return;
	int index = y * cols + x;
	
	// blur algorithm
	/*float result = 0.f;
	int filter_r, filter_c;		
	for (filter_r = -filter_width / 2; filter_r <= filter_width / 2; filter_r++) {
		for(filter_c = -filter_width / 2; filter_c <= filter_width / 2; filter_c++) {
			int r = y + filter_r;
			int c = x + filter_c;
			if (((r >= 0) && (r < rows)) && 
				((c >= 0) && (c < cols))) {
				int image_r = min(max(x + filter_r, 0), rows - 1);
				int image_c = min(max(x + filter_c, 0), cols - 1);
				float image_value = input[r * cols + c];
				float filter_value = filter[(r + filter_width / 2) * filter_width + (c + filter_width / 2)];
				result += image_value * filter_value;
			}
		}
	}*/

	float result = 0.0f;
	for (int fx = 0; x < filter_width; fx++) {
		for (int fy = 0; fy < filter_width; fy++) {
			int imagex = x + fx - filter_width / 2;
			int imagey = y + fy - filter_width / 2;
			imagex = min(max(imagex, 0), cols - 1);
			imagey = min(max(imagey, 0), rows - 1);
			result += (filter[fy * filter_width + fx] * input[imagey * cols + imagex]);
		}
	}
	printf("pixel value: %f ", result);	
	output[index] = result;
}

__global__ void sobelFilter(unsigned char *input,
						   unsigned char *output,
						   unsigned int rows,
						   unsigned int cols) {
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x >= cols || y >= rows)
		return;
	int index = y * cols + x;

	float dx, dy;
	if (x > 0 && y > 0 && x < cols - 1 && y < rows - 1) {
		dx = (-1 * input[(y - 1) * cols + (x - 1)]) + (-2 * input[y * 
				cols + (x - 1)]) + (-1 * input[(y + 1) * cols + (x - 1)]) + 
				(input[(y - 1) * cols + (x + 1)]) + (2 * input[y * cols + 
				(x + 1)]) + (input[(y + 1) * cols + (x + 1)]);
        dy = (input[(y - 1) * cols + (x - 1)]) + (2 * input[(y - 1) * cols + x])
				 + (input[(y - 1) * cols + (x + 1)]) + (-1 * input[(y + 1) * 
				cols + (x - 1)]) + (-2 * input[(y + 1) * cols + x]) + (-1 * 
				input[(y + 1) * cols + (x + 1)]);
        float result = sqrt((dx * dx) + (dy * dy));
		output[index] = result;
	}
}

void edgeDetector (unsigned char* h_input,
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

	// allocate memory
	int size = rows * cols;
	checkCuda(cudaMalloc((void**)&d_input, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&d_output_blur, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&d_output_sobel, size * sizeof(unsigned char)));

	// copy to GPU
	checkCuda(cudaMemset(d_output_blur, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_output_sobel, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemcpy(d_input, h_input, size * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_filter, h_filter, filter_width * filter_width * sizeof(float), cudaMemcpyHostToDevice));
	printf("rows: %d cols: %d gridx: %d gridy: %d\n", rows, cols, gridX, gridY);
	
	//*** Canney Edge Detector ***//

	// 1) Noise Reduction with Gaussian Blur
	//gaussianBlur<<<dimGrid, dimBlock>>>(d_input, d_output_blur, rows, cols, d_filter, filter_width);

	// 2) Gradient Calulation with Sobel Filter
	sobelFilter<<<dimGrid, dimBlock>>>(/*d_output_blur*/d_input, d_output_sobel, rows, cols);

	// 3) Non-Maximum Suppression


	// 4) Hysteresis


	//copy final output to host
	checkCuda(cudaMemcpy(h_output, d_output_sobel, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// free memory
	checkCuda(cudaFree(d_input));
	checkCuda(cudaFree(d_output_blur));
	checkCuda(cudaFree(d_output_sobel));
}
