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

extern void imageBlur(unsigned char *input,
	                  unsigned char *output,
       			      unsigned int rows,
       			      unsigned int cols);

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

    // output image
    Mat output_image = Mat::zeros(rows, cols, CV_8U);

	double start = CLOCK();

	// call external function
	imageBlur((unsigned char *)input_image.data,
			  (unsigned char *)output_image.data,
			   rows, cols);

	double end = CLOCK();
	cout << "GPU execution time: " << end - start << "ms" << endl;

	// Write to image
    imwrite ("output.jpg", output_image);

    return 0;
}
