#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

unsigned char *red;
unsigned char *green;
unsigned char *blue;
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

__global__ void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= numCols || py >= numRows) {
      return;
  }
  int i = py * numCols + px;
  redChannel[i] = inputImageRGBA[i].x;
  greenChannel[i] = inputImageRGBA[i].y;
  blueChannel[i] = inputImageRGBA[i].z;
}

__global__ void combineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  //make sure we don't try and access memory outside the image
  //by having any threads mapped there return early
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

__global__ void gaussianBlur(const unsigned char* const input,
							  unsigned char* const output, 
							  int rows, int cols,
							  const float* const filter, const int filter_width)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= cols || y >= rows)
		return;
	
	float c = 0.0f;
	for (int fx = 0; x < filter_width; x++) {
		for (int fy = 0; fy < filter_width; y++) {
			int imagex = x + fx - filter_width / 2;
			int imagey = y + fy - filter_width / 2;
			imagex = min(max(imagex, 0), cols - 1);
			imagey = min(max(imagey, 0), rows - 1);
			c += (filter[fy * filter_width + fx] * input[imagey * cols + imagex]);
		}
	}	
	output[y * cols + x] = c;
}

void imageBlur (const uchar4* const input,
			    uchar4* output,
				unsigned char* red_blurred,
                unsigned char* green_blurred,
                unsigned char* blue_blurred,
			    unsigned int rows,
			    unsigned int cols,
				const float* const h_filter,
				int filter_width) {

	// block and grid size
	int block_width = 16; // tile
	const dim3 block_size(block_width, block_width);	
	int blocks_x = (cols + block_width - 1) / block_width;
	int blocks_y = (rows + block_width - 1) / block_width;
	const dim3 grid_size(blocks_x, blocks_y);	

	// allocate memory and copy to GPU
	checkCuda(cudaMalloc(&red, sizeof(unsigned char) * rows * cols));
	checkCuda(cudaMalloc(&green, sizeof(unsigned char) * rows * cols));
	checkCuda(cudaMalloc(&blue, sizeof(unsigned char) * rows * cols));
	checkCuda(cudaMalloc(&d_filter, sizeof(float) * filter_width * filter_width));
	checkCuda(cudaMemcpy(d_filter, h_filter, sizeof(float) * filter_width * filter_width, cudaMemcpyHostToDevice));

	// separate channels
	separateChannels<<<grid_size, block_size>>>(input, rows, cols, red, green, blue);

	//call gaussian blur kernel on each channel
	gaussianBlur <<<grid_size, block_size>>>(red, red_blurred, rows, cols, d_filter, filter_width);
	gaussianBlur <<<grid_size, block_size>>>(green, green_blurred, rows, cols, d_filter, filter_width);
	gaussianBlur <<<grid_size, block_size>>>(blue, blue_blurred, rows, cols, d_filter, filter_width);
	cudaDeviceSynchronize();

	//combine channels
	combineChannels<<<grid_size, block_size>>>(red_blurred, 
											   green_blurred,
											   blue_blurred,
											   output,
											   rows,
											   cols);
	cudaDeviceSynchronize();

	// free memory
	cudaFree(red);
	cudaFree(green);	
	cudaFree(blue);
	cudaFree(d_filter);
}
