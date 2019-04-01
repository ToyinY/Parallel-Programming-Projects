#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/times.h>

#define M1_ROWS 10000
#define M1_COLS 10000
#define M2_COLS 10000
#define NUM_THREADS 2

int chunk;

// matrices
int a[M1_ROWS][M1_COLS]; // first matrix (n*m)
int b[M1_COLS][M2_COLS]; // second matrix (m*l)
int c[M1_ROWS][M2_COLS]; // result (n*l)

void *matrixMult(void *arg) {
	int tid = (int)arg;
	int i, j, k;

	// Divide matrix rows by number of threads	
	int lo = M1_ROWS / NUM_THREADS * tid;
	int hi = M1_ROWS / NUM_THREADS * (tid + 1) - 1;
	chunk = M1_ROWS / NUM_THREADS;
	if (tid + 1 == NUM_THREADS && hi < M1_ROWS - 1)
		hi = M1_ROWS - 1;	
	
	// have each thread work on a subsection of result matrix
	printf("Thread %d is computing rows %d to %d of resulting matrix.\n", tid, lo, hi);
	for (i = lo; i <= hi; i++)
		for (j = 0; j < M2_COLS; j++)
			for (k = 0; k < M1_COLS; k++)
				c[i][j] += a[i][k] * b[k][j];
}

int main () {

	int i, j;
	struct timeval tvalBefore, tvalAfter;

	//initialize matrices
	srand(time(0));
	for (i = 0; i < M1_ROWS; i++)
		for (j = 0; j < M2_COLS; j++)
			a[i][j] = 1;//rand() % 10;

	for (i = 0; i < M1_COLS; i++)
		for (j = 0; j < M2_COLS; j++)
			b[i][j] = 1;//rand() % 10;

	for (i = 0; i < M1_ROWS; i++)
		for (j = 0; j < M2_COLS; j++)
			c[i][j] = 0;

	// launch threads for matrix multiply, parallelize by row
	pthread_t threads[NUM_THREADS];
	
	//start timer
	gettimeofday(&tvalBefore, NULL);
	
	// launch threads
	int rc;
	for (i = 0; i < NUM_THREADS; i++) {
		rc = pthread_create(&threads[i], NULL, matrixMult, (void *)i);
		if (rc) {
			printf("ERROR; return code from pthread_create() is %d\n", rc);
			exit(1);
		}			
	}

	// join threads
	for (i = 0; i < NUM_THREADS; i++) {
		rc = pthread_join(threads[i], NULL);
		if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(1);
        }	
	}

	//end timer
	gettimeofday(&tvalAfter, NULL);

	// print resulting matrix
	/*printf("Multiply: a[%d][%d] * b[%d][%d] = c[%d][%d]\nResult:\n", 
			M1_ROWS, M1_COLS, M1_COLS, M2_COLS, M1_ROWS, M2_COLS);
	for (i = 0; i < M1_ROWS; i++) {
		for (j = 0; j < M2_COLS; j++) {
			printf("%d ", c[i][j]);
		}
		printf("\n");
	}*/

	// print run time results
	printf("Multiply: a[%d][%d] * b[%d][%d] = c[%d][%d] chunk = %d\n",
			M1_ROWS, M1_COLS, M1_COLS, M2_COLS, M1_ROWS, M2_COLS, chunk);
	printf("Number of threads: %d Total time: %ld "
			"microsecs\n", NUM_THREADS, ((tvalAfter.tv_sec - 
			tvalBefore.tv_sec)*1000000L + tvalAfter.tv_usec) - 
			tvalBefore.tv_usec);

	return 0;
}

