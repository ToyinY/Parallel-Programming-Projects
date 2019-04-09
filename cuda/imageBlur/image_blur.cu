#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TILE_SIZE 16

unsigned char *d_input;
unsigned char *d_output;

inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

/*__global__ void separateChannels(const uchar4* const inputImageRGBA,
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
*/

__global__ void gaussianBlur(unsigned char *input,
							 unsigned char *output,
							 unsigned int rows,
							 unsigned int cols) {
	
	int x = blockIdx.x * TILE_SIZE + threadIdx.x;
	int y = blockIdx.y * TILE_SIZE + threadIdx.y;
	if (x > cols || y > rows)
		return;
	int index = x * rows + y;
	
	// test, copy input to ouput
	output[index] = input[index];
}

void imageBlur (unsigned char* h_input,
			    unsigned char* h_output,
				unsigned int rows,
			    unsigned int cols) {

	// block and grid size
	int gridX = 1 + ((cols - 1) / TILE_SIZE);
	int gridY = 1 + ((rows - 1) / TILE_SIZE);
	dim3 dimGrid(gridX, gridY);
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);

	// allocate memory and copy to GPU
	int size = rows * cols;
	checkCuda(cudaMalloc((void**)&d_input, size * sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&d_output, size * sizeof(unsigned char)));
	checkCuda(cudaMemset(d_output, 0, size * sizeof(unsigned char)));
	checkCuda(cudaMemcpy(d_input, h_input, size * sizeof(unsigned char), cudaMemcpyHostToDevice));

	//kernel call
	gaussianBlur<<<dimGrid, dimBlock>>>(d_input, d_output, rows, cols);

	//copy output to host
	checkCuda(cudaMemcpy(h_output, d_output, size * sizeof(unsigned char), cudaMemcpyDeviceToHost));

	// free memory
	checkCuda(cudaFree(d_input));
	checkCuda(cudaFree(d_output));
}
