#include <stdio.h>
#define N 1000

__global__ void vector_add(float *out, float *a, float *b, int n) {
	for (int i = 0; i < n; i++) {
		out[i] = a[i] + b[i];
	}
}

int main(){
    float *d_a, *d_b, *d_c;
	float *h_a, *h_b, *h_c;

	h_a = (float*)malloc(N * sizeof(float));
	h_b = (float*)malloc(N * sizeof(float));
	h_c = (float*)malloc(N * sizeof(float)); 

	cudaMalloc(&d_a, N * sizeof(float));
	cudaMalloc(&d_b, N * sizeof(float));
	cudaMalloc(&d_c, N * sizeof(float));

    // Initialize array
    for(int i = 0; i < N; i++){
        h_a[i] = 1.0f; h_b[i] = 2.0f;
    }

	// copy to host
	cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, N * sizeof(float), cudaMemcpyHostToDevice);

	int blocks, grids;
	blocks = 32;
	grids = (float)ceil((float)N / blocks);
	vector_add<<<grids, blocks>>>(d_c, d_a, d_b, N);
	cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

	int i;
	for (i = 0; i < N; i++) {
		printf("%f ", h_c[i]);
	}
	printf("\n");

	// free
	cudaFree(d_a);
	cudaFree(d_b);	
	cudaFree(d_c);
	cudaFree(h_a);
	cudaFree(h_b);
	cudaFree(h_c);

	return 0;
}

