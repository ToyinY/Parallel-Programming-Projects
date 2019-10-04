# Image Blur
## Perform a gaussian blur on an image.
## Instructions on runnning program on the discovery cluster
1. Create an account on the cluster and ssh in.
2. In the login node you can create and compile programs, but you cannot run programs. To get to a GPU node, use command:
```
srun --pty --nodes 1 --job-name=interactive --partition=gpu --gres=gpu:1 /bin/bash
```
To specify the time of your reservation, add the ```--time``` flag. For example this reservation is for 1 hour:
```
srun --pty --nodes 1 --job-name=interactive --partition=gpu --gres=gpu:1 --time=1:00:00 /bin/bash
```
After a few moments you will be connected to a node.

3. If you have not done so already, clown the repository and run ```make``` in the imageBlur directory (Parallel-Programming-Projects/cuda/imageBlur).

4. run command 
```./image_blur```

5. The program write the output image in the current directory.
