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

// Timer function
double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

// gpu function
extern void edgeDetector (unsigned char *h_input,
	                  	  unsigned char *h_output,
       			      	  unsigned int rows,
       			      	  unsigned int cols,
					  	  float *h_filter,
					  	  int filter_width,
            			  float *h_sobel_mask_x,
                    float *h_sobel_mask_y);

int main( int argc, const char** argv ) {
        
    // Read input image 
    Mat input_image = imread("input.jpg", IMREAD_GRAYSCALE);
    if (input_image.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    } 
    unsigned int rows = input_image.rows;
    unsigned int cols = input_image.cols;

	Mat output_image = Mat::zeros(rows, cols, CV_8U);

	// create filter for gaussian blur
	const int blurKernelWidth = 9;
  	const float blurKernelSigma = 2.;
  	int filter_width = blurKernelWidth;

  	//create and fill the filter to convolve with
    float *h_filter;
  	h_filter = (float*) malloc(filter_width * filter_width * sizeof(float));
  	float filterSum = 0.f; 
  	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    	for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      		float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      		h_filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
     	filterSum += filterValue;
    	}
  	}
  	float normalizationFactor = 1.f / filterSum;
  	for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    	for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      	h_filter[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    	}
  	}

  	// init and create sobel masks
  	float *h_sobel_mask_x, *h_sobel_mask_y;
  	h_sobel_mask_x = (float*)malloc(filter_width * sizeof(float));
  	h_sobel_mask_y = (float*)malloc(filter_width * sizeof(float));
  	h_sobel_mask_x[0] = 1.0; h_sobel_mask_x[1] = 0.0; h_sobel_mask_x[2] = -1.0;
  	h_sobel_mask_x[3] = 2.0; h_sobel_mask_x[4] = 0.0; h_sobel_mask_x[5] = -2.0;
  	h_sobel_mask_x[6] = 1.0; h_sobel_mask_x[7] = 0.0; h_sobel_mask_x[8] = -1.0;
  	h_sobel_mask_y[0] = -1.0; h_sobel_mask_y[1] = -2.0; h_sobel_mask_y[2] = -1.0;
  	h_sobel_mask_y[3] =  0.0; h_sobel_mask_y[4] =  0.0; h_sobel_mask_y[5] =  0.0;
  	h_sobel_mask_y[6] =  1.0; h_sobel_mask_y[7] =  2.0; h_sobel_mask_y[8] =  1.0;

	// call and time gpu function
	double start = CLOCK();
	edgeDetector((unsigned char *)input_image.data,
			     (unsigned char *)output_image.data,
			     rows, cols, h_filter, filter_width,
               	 h_sobel_mask_x, h_sobel_mask_y);
	double end = CLOCK();
	cout << "GPU execution time: " << end - start << "ms" << endl;

	// write final image
	imwrite ("Output-Images/output.jpg", output_image);

    return 0;
}
