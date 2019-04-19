#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define TILE_SIZE 16

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
	if (x >= cols || y >= rows)
		return;
	int index = y * cols + x;
	
	float result = 0.0f;
	for (int fx = 0; fx < filter_width; fx++) {
		for (int fy = 0; fy < filter_width; fy++) {
			int imagex = x + fx - filter_width / 2;
			int imagey = y + fy - filter_width / 2;
			imagex = min(max(imagex, 0), cols - 1);
			imagey = min(max(imagey, 0), rows - 1);
			//printf("filter width: %d fx: %d fy %d\n", filter_width, fx, fy);
			//printf("imagex = %d imagey = %d\n", imagex, imagey);
			printf("x = %d y = %d\n", fy * filter_width + fx, imagey * cols + imagex);
			//float image_value = (float)input[imagey * cols + imagex];
			//float filter_value = (float)filter[fy * filter_width + fx];
			//printf("result: %f\n", image_value * filter_value);
			//result += image_value * filter_value;		
		}
	}
	printf("pixel value: %f ", result);	
	output[index] = (unsigned char)result;
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
	float value_x = 0.0, value_y = 0.0, angle = 0.0;
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
	if ((value_x != 0.0) || (value_y != 0.0)) {
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

__global__ void nonMaxSuppression(unsigned char *output,
								  unsigned char *input,
								  double *edge_direction,
								  unsigned int rows,
						   		  unsigned int cols) {
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x >= cols - 1 || y >= rows - 1 || x < 1 || y < 1)
		return;
	int index = y * cols + x;
	
	float pixel_1 = 255.0f, pixel_2 = 255.0f;
	
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

__global__ void doubleThreshold(unsigned char *output,
								unsigned char *input,
								unsigned int rows,
								unsigned int cols) {
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x >= cols || y >= rows)
		return;
	int index = y * cols + x;

	float high = 80, low = 30, weak = 25, strong = 255;
	//if (x == 0) printf("input: %f\n", input[index]);
	if (input[index] >= high) { //strong
		output[index] = strong;
	} else if (input[index] <= high && input[index] >= low) { //weak
		output[index] = weak;
	} else { //non-relevent
		output[index] = 0;
	}
	//printf("output: %f\n", output[index]);
}

__global__ void histeresis(unsigned char *input, 
						   unsigned char *output,
						   unsigned int rows,
						   unsigned int cols) {
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x >= cols - 1 || y >= rows - 1 || x < 1 || y < 1)
		return;
	int index = y * cols + x;

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
	checkCuda(cudaMalloc((void**)&d_output_sobel, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&d_output_thresh, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&d_edge_direction, size * sizeof(double)));
	checkCuda(cudaMalloc((void**)&d_edge_magnitude, size * sizeof(float)));
	checkCuda(cudaMalloc((void**)&d_sobel_mask_x, filter_width * sizeof(double)));
	checkCuda(cudaMalloc((void**)&d_sobel_mask_y, filter_width * sizeof(double)));
	checkCuda(cudaMalloc((void**)&d_output, size * sizeof(unsigned char)));

	// copy to GPU
	checkCuda(cudaMemset(d_output_blur, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_output_nms, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_edge_direction, 0, size * sizeof(double)));
	checkCuda(cudaMemset(d_output_sobel, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_output_thresh, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_edge_magnitude, 0, size * sizeof(float)));
	checkCuda(cudaMemcpy(d_sobel_mask_x, h_sobel_mask_x, filter_width * sizeof(float), cudaMemcpyHostToDevice));	
	checkCuda(cudaMemcpy(d_sobel_mask_y, h_sobel_mask_y, filter_width * sizeof(float), cudaMemcpyHostToDevice));
	checkCuda(cudaMemset(d_output, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemcpy(d_input, h_input, size * sizeof(unsigned char), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_filter, h_filter, filter_width * filter_width * sizeof(float), cudaMemcpyHostToDevice));
	printf("rows: %d cols: %d gridx: %d gridy: %d\n", rows, cols, gridX, gridY);
	
	//*** Canney Edge Detector ***//

	// 1) Noise Reduction with Gaussian Blur (not working now)
	//gaussianBlur<<<dimGrid, dimBlock>>>(d_input, d_output_blur, rows, cols, d_filter, filter_width);

	// 2) Gradient Calulation with Sobel Filter
	sobelFilter<<<dimGrid, dimBlock>>>(/*d_output_blur*/d_input, d_output_sobel, d_edge_magnitude, d_edge_direction, d_sobel_mask_x, d_sobel_mask_y, rows, cols);

	// 3) Non-Maximum Suppression
	nonMaxSuppression<<<dimGrid, dimBlock>>>(d_output_nms, d_output_sobel, d_edge_direction, rows, cols);

	// 4) Double threshold
	doubleThreshold<<<dimGrid, dimBlock>>>(d_output_thresh, d_output_nms, rows, cols);

	// 5) Hysteresis
	histeresis<<<dimGrid, dimBlock>>>(d_output, d_output_thresh, rows, cols);

	//copy final output to host
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
