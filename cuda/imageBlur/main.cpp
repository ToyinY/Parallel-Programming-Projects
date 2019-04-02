#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

// CUDA kernel
//extern "C" __global__ void imageBlur();

extern void imageBlur() {};

int main( int argc, const char** argv ) {
    // arg 1: Input image 
    // arg 2: output file name
        
    // Check inputs
    if (argc != 3){
        cout << "Incorrect number of inputs" << endl;
        cout << argv[0] << " <input file> <output file name>" << endl;
        return -1;
    }        
        
    // Read input image from argument
    Mat input_image = imread(argv[1], IMREAD_COLOR);
    if (input_image.empty()){
        cout << "Image cannot be loaded..!!" << endl;
        return -1;
    } 
    unsigned int height = input_image.rows;
    unsigned int  width = input_image.cols;

	// Separate Color Channels      
	Mat image_channels[3];
	split(input_image, image_channels);

	// blur each channel in GPU
	

	// combine channels
	Mat output_image = Mat::zeros(height, width, CV_8U);


	// Write to image
    cout << "writing output image " << argv[2] << endl;
    imwrite (argv[2], output_image);

    return 0;
}
