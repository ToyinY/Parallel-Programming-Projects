To run on the discovery cluster, first build on you login node:
`make`
Then enter the following command to get to a node:
`srun --pty --nodes 1 --job-name=interactive --partition=gpu --gres=gpu:1 /bin/bash`
Run the program:
`./vector_add`
