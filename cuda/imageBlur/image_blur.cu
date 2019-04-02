#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

unsigned char *input, 
unsigned char *output,
float filter;

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}


int imageBlur (unsigned char *input,
			   unsigned char *output,
			   unsigned int rows,
			   unsigned int cols) {

	// Allocate memory
	int size = rows * cols;
	checkCuda(cudaMalloc((void**)&input_image, size * sizeof(uchar4)));
	checkCuda(cudaMalloc((void**)&output_image, size * sizeof(uchar4)));
	checkCuda(cudaMemset(output, 0 , size * sizeof(unsigned char)));
	
	// block and grid size
	int block_width = 16; // tile
	const dim3 block_size(block_width, block_width);	
	int blocks_x = (cols + block_width - 1) / block_width;
	int blocks_y = (rows + block_width - 1) / block_width;
	const dim3 grid_size(blocks_x, blocks_y);	

	// copy to GPU
	checkCuda(cudaMemcpy(input_image, input, size * sizeof(uchar4), cusaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	// gaussian blur on channel
	filter_width = ;
	gaussianBlur <<<grid_size, block_size, filter_width * filter_width * sizeof(float)>>>(red, red_blurred, rows, cols, d_filer, filter_width);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// free memory

}
