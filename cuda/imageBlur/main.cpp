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

float *h_filter;

// Timer function
double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

// gpu function
extern void imageBlur(unsigned char *h_input,
	                  unsigned char *h_output,
       			      unsigned int rows,
       			      unsigned int cols,
					  float *h_filter,
					  int filter_width);

int main( int argc, const char** argv ) {
        
    // Read input image 
    Mat input_image = imread("input.jpg", IMREAD_COLOR);
    if (input_image.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    } 
    unsigned int rows = input_image.rows;
    unsigned int cols = input_image.cols;

	// separate image to three channels
	vector<Mat> rgb_channels(3);
	split(input_image, rgb_channels);
	Mat input_red, input_blue, input_green;
	input_red = rgb_channels[2];
	input_green = rgb_channels[1];
	input_blue = rgb_channels[0];
	
    // output image channels
    Mat output_red = Mat::zeros(rows, cols, CV_8U);
	Mat output_green = Mat::zeros(rows, cols, CV_8U);
	Mat output_blue = Mat::zeros(rows, cols, CV_8U);
	Mat output_image = Mat::zeros(rows, cols, CV_8U);

	// create filter for gaussian blur
	const int blurKernelWidth = 9;
  	const float blurKernelSigma = 2.;
  	int filter_width = blurKernelWidth;

  	//create and fill the filter to convolve with
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

	// call and time gpu function
	double start = CLOCK();
	imageBlur((unsigned char *)input_red.data,
			  (unsigned char *)output_red.data,
			   rows, cols, h_filter, filter_width);
	imageBlur((unsigned char *)input_green.data,
			  (unsigned char *)output_green.data,
			   rows, cols, h_filter, filter_width);
	imageBlur((unsigned char *)input_blue.data,
			  (unsigned char *)output_blue.data,
			   rows, cols, h_filter, filter_width);
	double end = CLOCK();
	cout << "GPU execution time: " << end - start << "ms" << endl;

	// Merge and write image
   	vector<Mat> out_channels;
	out_channels.push_back(output_blue);
	out_channels.push_back(output_green);
	out_channels.push_back(output_red);
	merge(out_channels, output_image);
	imwrite ("output.jpg", output_image);

    return 0;
}
