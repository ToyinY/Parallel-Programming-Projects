#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TILE_SIZE 16

// buffers for gpu processing
unsigned char *d_input;
unsigned char *d_output_blur;
unsigned char *d_output_sobel;
unsigned char *d_output_nms;
unsigned char *d_output_thresh;
unsigned char *d_output;
double *d_edge_direction;
float *d_edge_magnitude;
float *d_sobel_mask_x;
float *d_sobel_mask_y;
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
	
	// Blur algorthm using weighted average
	float result = 0.0;
	for (int r = -filter_width / 2; r < filter_width / 2; r++) {
		for (int c = -filter_width / 2; c < filter_width / 2; c++) {
			int cur_row = r + y; 
			int cur_col = c + x;
			
			if ((cur_row > -1) && (cur_row < rows) &&
				(cur_col > -1) && (cur_col < cols)) { 
				int filter_id = (r + filter_width / 2) * filter_width + (c + filter_width / 2);	
				result += input[cur_row * cols + cur_col] * filter[filter_id]; 
			}
		}
	}
	output[index] = result;
}

__global__ void medianFilter(unsigned char *input,
			     unsigned char *output,
			     unsigned int rows,
			     unsigned int cols){
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x >= cols - 1 || y >= rows - 1 || x == 0 || y == 0)
		return;
	int index = y * cols + x;

	// Array of pixels to sort
	float pixels[9];
	pixels[0] = input[y * cols + (x + 1)];
	pixels[1] = input[y * cols + (x - 1)];
	pixels[2] = input[(y + 1) * cols + (x - 1)];
	pixels[3] = input[(y - 1) * cols + (x + 1)];
	pixels[4] = input[(y + 1) * cols + x];
	pixels[5] = input[(y - 1) * cols + x];
	pixels[6] = input[(y - 1) * cols + (x - 1)];
	pixels[7] = input[(y + 1) * cols + (x + 1)];
	pixels[8] = input[index];

	// sort pixel values with insertion sort
	int i, j;
	float temp;
	for (i = 0; i < 9; i++) {
		temp = pixels[i];
		for (j = i - 1; j >= 0 && temp < pixels[j]; j--) {
			pixels[j + 1] = pixels[j];
		}
		pixels[j + 1] = temp;
	}	

	// Assign output pixel to median pixel value
	output[index] = pixels[4];
}

__global__ void sobelFilter(unsigned char *input, 
			    unsigned char *output, 
			    float *edge_magnitude,
			    double *edge_direction,
			    float *sobel_mask_x,
			    float *sobel_mask_y,
			    unsigned int rows,
			    unsigned int cols){
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x >= cols || y >= rows)
		return;
	int index = y * cols + x;

	// magnitude calculation
	float value_x = 0, value_y = 0, angle = 0;
	for (int k = 0; k < 3; k++) {
		for (int l = 0; l < 3; l++) {
			value_x += sobel_mask_x[l * 3 + k] * input[((y + 1) + (1 - l)) * cols + ((x + 1) + (1 - k))];
			value_y += sobel_mask_y[l * 3 + k] * input[((y + 1) + (1 - l)) * cols + ((x + 1) + (1 - k))];
		}
	}
	edge_magnitude[index] = sqrt(value_x * value_x + value_y * value_y);
	//if (x == 0) printf("edge_magnitude: %f\n", edge_magnitude[index]);
	output[index] = edge_magnitude[index];
	
	// angle direction calculation
	if ((value_x != 0) || (value_y != 0)) {
		angle = atan2(value_y, value_x) * 180.0 / 3.14159;
	} else {
		angle = 0.0;
	}
	if (((angle > -22.5) && (angle <= 22.5)) ||//0
	    ((angle > 157.5) && (angle <= 202.5))||//180
		((angle > -202.5) && (angle <= -157.7))) {//-180
		edge_direction[index] = 0.0;
	} else if (((angle > 22.5) && (angle <= 67.5)) ||//45
	          ((angle > -157.5) && (angle <= -112.5))) {//-135
		edge_direction[index] = 45.0;
	} else if (((angle > 67.5) && (angle <= 112.5)) ||//90
	           ((angle > -112.5) && (angle <= -67.5))) {//-90
		edge_direction[index] = 90.0;
	} else if (((angle > 112.5) && (angle <= 157.5)) ||//135
	           ((angle > -67.5) && (angle <= -22.5))) {//-45
		edge_direction[index] = 135.0;
	}
}

__global__ void nonMaxSuppression(unsigned char *input,
				  unsigned char *output,
				  double *edge_direction,
				  unsigned int rows,
				  unsigned int cols) {
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x >= cols - 1 || y >= rows - 1 || x == 0 || y == 0)
		return;
	int index = y * cols + x;
	
	float pixel_1 = 255, pixel_2 = 255;
	
	if (edge_direction[index] == 0) {
		pixel_1 = input[y * cols + (x + 1)];
		pixel_2 = input[y * cols + (x - 1)];			
	} else if (edge_direction[index] == 45) {
		pixel_1 = input[(y + 1) * cols + (x - 1)];
		pixel_2 = input[(y - 1) * cols + (x + 1)];
	} else if (edge_direction[index] == 90) {
		pixel_1 = input[(y + 1) * cols + x];
		pixel_2 = input[(y - 1) * cols + x];
	} else if (edge_direction[index] == 135) {
		pixel_1 = input[(y - 1) * cols + (x - 1)];
		pixel_2 = input[(y + 1) * cols + (x + 1)];
	}
	
	if ((input[index] >= pixel_1) && (input[index] >= pixel_2)) {
		output[index] = input[index];
	} else {
		output[index] = 0;
	}
}

__global__ void doubleThreshold(unsigned char *input,
				unsigned char *output,
				unsigned int rows,
				unsigned int cols) {
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x >= cols || y >= rows)
		return;
	int index = y * cols + x;

	float high = 70, low = 20, weak = 50, strong = 255;
	
	if (input[index] >= high) { //strong
		output[index] = strong;
	} else if (input[index] <= high && input[index] >= low) { //weak
		output[index] = weak;
	} else { //non-relevent
		output[index] = 0;
	}
}

__global__ void histeresis(unsigned char *input, 
			  unsigned char *output,
			  unsigned int rows,
			  unsigned int cols) {
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x >= cols - 1 || y >= rows - 1 || x == 0 || y == 0)
		return;
	int index = y * cols + x;
	
	float weak = 50, strong = 255;
	
	if (input[index] == weak) { // if current pixel is weak
		if ((input[y * cols + (x + 1)] == strong) || 
			(input[y * cols + (x - 1)] == strong) || 
			(input[(y + 1) * cols + (x - 1)] == strong) || 
			(input[(y - 1) * cols + (x + 1)] == strong) ||
			(input[(y + 1) * cols + x] == strong) || 
			(input[(y - 1) * cols + x] == strong) ||
			(input[(y - 1) * cols + (x - 1)] == strong) || 
			(input[(y + 1) * cols + (x + 1)] == strong)) // if a surrounding pixel is strong
			output[index] = strong; // set the weak pixel to strong
		else {
			output[index] = 0; // otherwise, set the pixel to be non-relevant
		}
	}
}

void edgeDetector (unsigned char* h_input,
		   unsigned char* h_output,
		   unsigned int rows,
		   unsigned int cols,
		   float* h_filter,
		   int filter_width,
		   float *h_sobel_mask_x,
		   float *h_sobel_mask_y) {

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
	checkCuda(cudaMalloc((void**)&d_output_nms, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&d_output_thresh, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&d_edge_direction, size * sizeof(double)));
	checkCuda(cudaMalloc((void**)&d_edge_magnitude, size * sizeof(float)));
	checkCuda(cudaMalloc((void**)&d_sobel_mask_x, filter_width * sizeof(double)));
	checkCuda(cudaMalloc((void**)&d_sobel_mask_y, filter_width * sizeof(double)));
	checkCuda(cudaMalloc((void**)&d_filter, filter_width * filter_width * sizeof(float)));
	checkCuda(cudaMalloc((void**)&d_output, size * sizeof(unsigned char)));

	checkCuda(cudaMemset(d_output, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_output_blur, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_output_nms, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_edge_direction, 0, size * sizeof(double)));
	checkCuda(cudaMemset(d_output_sobel, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_output_thresh, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_edge_magnitude, 0, size * sizeof(float)));

	// copy to GPU with streams
	cudaStream_t s1, s2, s3, s4;
	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);
	cudaStreamCreate(&s3);
	cudaStreamCreate(&s4);

	checkCuda(cudaMemcpyAsync(d_input, h_input, size * sizeof(unsigned char), cudaMemcpyHostToDevice, s1));
	checkCuda(cudaMemcpyAsync(d_sobel_mask_x, h_sobel_mask_x, filter_width * sizeof(float), cudaMemcpyHostToDevice, s2));	
	checkCuda(cudaMemcpyAsync(d_sobel_mask_y, h_sobel_mask_y, filter_width * sizeof(float), cudaMemcpyHostToDevice, s2));
	checkCuda(cudaMemcpyAsync(d_filter, h_filter, filter_width * filter_width * sizeof(float), cudaMemcpyHostToDevice, s2));
	cudaDeviceSynchronize();

	// destroy streams
	cudaStreamDestroy(s1);
	cudaStreamDestroy(s2);
	cudaStreamDestroy(s3);
	cudaStreamDestroy(s4);

	printf("The image has: rows= %d cols= %d\nKernels grid dimensions: gridx=" 
			"%d gridy= %d\nKernels block dimensions: blockx= %d blocky= %d\n", 
			rows, cols, gridX, gridY, TILE_SIZE, TILE_SIZE);
	
	//*** Canny Edge Detector Algorithm ***//

	// 1) Noise Reduction with Gaussian Blur
	gaussianBlur<<<dimGrid, dimBlock>>>(d_input, d_output_blur, rows, cols, d_filter, filter_width);
	cudaDeviceSynchronize();

	// Alternative noise reduction algorithm for ultra sound images
	//medianFilter<<<dimGrid, dimBlock>>>(d_input, d_output, rows, cols);

	// 2) Gradient Calulation with Sobel Filter
	sobelFilter<<<dimGrid, dimBlock>>>(d_output_blur, d_output_sobel, d_edge_magnitude, d_edge_direction, d_sobel_mask_x, d_sobel_mask_y, rows, cols);
	cudaDeviceSynchronize();

	// 3) Non-Maximum Suppression
	nonMaxSuppression<<<dimGrid, dimBlock>>>(d_output_sobel, d_output_nms, d_edge_direction, rows, cols);
	cudaDeviceSynchronize();

	// 4) Double threshold
	doubleThreshold<<<dimGrid, dimBlock>>>(d_output_nms, d_output_thresh, rows, cols);
	cudaDeviceSynchronize();

	// 5) Hysteresis (only call if needed)
	/*histeresis<<<dimGrid, dimBlock, s1>>>(d_output_thresh, d_output, rows, cols);
	cudaDeviceSynchronize();*/

	//*** Edge Detection Finished ***//

	// copy final output image to host (use correct output)
	checkCuda(cudaMemcpy(h_output, d_output_thresh, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// free memory
	checkCuda(cudaFree(d_input));
	checkCuda(cudaFree(d_output_blur));
	checkCuda(cudaFree(d_output_nms));
	checkCuda(cudaFree(d_output_sobel));
	checkCuda(cudaFree(d_output_thresh));
	checkCuda(cudaFree(d_output));
	checkCuda(cudaFree(d_edge_magnitude));
	checkCuda(cudaFree(d_edge_direction));
	checkCuda(cudaFree(d_sobel_mask_x));
	checkCuda(cudaFree(d_sobel_mask_y));
}
