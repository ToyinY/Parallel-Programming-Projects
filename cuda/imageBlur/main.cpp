#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdint.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;

cv::Mat inputRGBA;
cv::Mat outputRGBA;

inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}

extern void imageBlur(const uchar4* const h_input,
					  const uchar4* const d_input,
					  uchar4* d_ouput,
					  unsigned int rows,
					  unsigned int cols, 
					  unsigned char *red_blurred,
					  unsigned char *green_blurred,
					  unsigned char *blue_blurred,
					  const float* const h_filter,
					  const int filter_width);

int main( int argc, const char** argv ) {
        
    // Check inputs
    /*if (argc != 3){
        cout << "Incorrect number of inputs" << endl;
        cout << argv[0] << " <input file> <output file name>" << endl;
        return -1;
    } */       
        
    // Read input image from argument
    Mat input_image = imread("input.jpg", IMREAD_COLOR); //argv[1]
    if (input_image.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    } 
    unsigned int rows = input_image.rows;
    unsigned int cols = input_image.cols;

	cv::cvtColor(input_image, inputRGBA, CV_BGR2RGBA);
	outputRGBA.create(rows, cols, CV_8UC4);
	*h_input = (uchar4 *)inputRGBA.ptr<unsigned char>(0);
	*h_output = (uchar4 *)outputRGBA.ptr<unsigned char>(0);	

	float *h_filter;
	int filter_width;

	//allocate memory and copy
	uchar4 d_input, d_output;
	unsigned char *red_blurred, *green_blurred, *blue_blurred;
	cudaMalloc(d_input, sizeof(uchar4) * rows * cols);
	cudaMalloc(d_output, sizeof(uchar4) * rows * cols);
	cudaMemset(*d_output, 0, rows * cols * sizeof(uchar4));
	cudaMemcpy(*d_input, *h_input, sizeof(uchar4) * rows * cols, cudaMemcpyHostToDevice);
	cudaMalloc(red_blurred, sizeof(unsigned char) * rows * cols);
	cudaMalloc(green_blurred, sizeof(unsigned char) * rows * cols);
	cudaMalloc(blue_blurred, sizeof(unsigned char) * rows * cols);
	cudaMemset(red_blurred, 0, sizeof(unsigned char) * rows * cols);
	cudaMemset(green_blurred, 0, sizeof(unsigned char) * rows * cols);
	cudaMemset(blue_blurred, 0, sizeof(unsigned char) * rows * cols);

	// create filter
	const int blurKernelWidth = 9;
  	const float blurKernelSigma = 2.;
	filter_width = blurKernelWidth;
	float* h_filter = (float*)malloc(filter_width * filter_width * sizeof(float));
	checkCuda(cudaMalloc(&h_filter, filter_width * filter_width * sizeof(float)));
	*h_filter__ = *h_filter;
	float filterSum = 0.f; //for normalization
	int r, c;
	for (r = -filter_width / 2; r <= filter_width / 2; ++r) {
    	for (c = -filter_width/2; c <= filter_width / 2; ++c) {
      		float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      		(*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      		filterSum += filterValue;
    	}
  	}
  	float normalizationFactor = 1.f / filterSum;
  	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    	for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      		(*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = normalizationFactor;
    	}
  	}	

	// call external function
	imageBlur(h_input, d_input, d_output, rows, cols, red_blurred, green_blurred, blue_blurred, h_filter, filter_width);

	// Write to image
    imwrite ("output.jpg", h_output);

    return 0;
}
