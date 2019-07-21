#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

int main (int argc, char* argv[])
{
    try
    {
        cv::Mat src_host = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
        cv::cuda::GpuMat dst, src;
        src.upload(src_host);
        cv::cuda::threshold(src, dst, 128.0, 255.0, cv::THRESH_BINARY);
        cv::Mat result_host(dst);
        cv::imwrite("Result", result_host);
        cv::waitKey();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }

	/*try
    {
        cv::Mat src_host = cv::imread("input.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat dst;

        cv::threshold(src_host, dst, 128.0, 255.0, cv::THRESH_BINARY);

        cv::imwrite("result.jpg", dst);
        cv::waitKey();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }*/

    return 0;
}
