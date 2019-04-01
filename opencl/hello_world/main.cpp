#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#include <iostream>
#include <CL/cl.hpp>

int main() {

	//get platforms
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size()==0) {
		std::cout << "No platforms found.\n";
		exit(1);
	}
	cl::Platform default_platform=platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() <<"\n";

	// get devices
	std::vector<cl::Device> devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	if (devices.size() == 0){
		std::cout << "No devices.\n";
		exit(1);
	}
	cl::Device default_device=devices[0];
	std::cout<< "using device: " << default_device.getInfo<CL_DEVICE_NAME>()<<"\n";

}
