# Parallel Canny Image Edge Detector

## The Parallel Edge Detection Problem
### Task: Creates a fast parallel program that detects the edges of objects on an image.
### Methodology:
  - Write the program in CUDA C++ using a NVIDIA GPU and CUDA Toolkit on Northeastern's Discovery Cluster
    - Information about the Discovery Cluster can be found here: www.rc.northeastern.edu
  - Implement the steps of the Canny Edge Detector Algorithm in parallel
  - USe the OpenCV library to read and write image data

## Canny Edge Detector Algorithm
1. Noise Reduction with Gaussian Blur
   - Weighted average value of the surrounding pixels in grey scale
2. Gradient Calculation with Sobel Filter
   - Detects edge intensity and direction
3. Non-Maximum Suppression
   - Find picel with the maximum value in edge directions
4. Double Threshold
   - Identifies string, weak, and non relevant pixels
5. Hysteresis
   - Transforms weak pixels to strong or irrelevant pixels

## Instructions on runnning program on the discvoery cluster
1. Create an account on the clusster and ssh in.
2. In the login node you can create an compile programs, but you cannot run programs. To get to a GPU node, use   command:
```
srun --pty --nodes 1 --job-name=interactive --partition=gpu --gres=gpu:1 /bin/bash
```
To specify the time of your reservation, add the ```--time``` flag. For example this reservation is for 1 hour:
```
srun --pty --nodes 1 --job-name=interactive --partition=gpu --gres=gpu:1 --time=1:00:00 /bin/bash
```
After a few moments you will be connected to a node.

3. If you have not done so already, clown the github repository and run ```make``` in the edgeDetector directory (Parallel-Programming-Projects/cuda/edgeDetector).
Note: You can edit the ```main.cpp``` file to specify input image and output image name. (Work in progress: have users choose their image when call program from the command line)

4. run command 
```./edge_detector```

5. The program will print the GPU execution time and write the output image to the Output-Images folder. It will also print kernel information. An example output is:
```
The image has: rows= 600 cols= 419
Kernels grid dimensions: gridx=27 gridy= 38
Kernels block dimensions: blockx= 16 blocky= 16
GPU execution time: 3.08297ms
```

## Image Results after each stage of the algorithm
### Input Image
![Alt text](Input-Images/input_rdj.jpg)

### Image after Gaussian Blur
![Alt text](Output-Images/output_blur_average.jpg)

### Image after sobel filter
![Alt text](Output-Images/output_sobel_with_avg_blur.jpg)

### Image afer Non-Maximum 
![Alt text](Output-Images/output_nms_with_avg_blur.jpg)

### Image after Double Threshold
![Alt text](Output-Images/output_thresh_with_avg_blur.jpg)

### Final Image
![Alt text](Output-Images/output_rdj.jpg)

