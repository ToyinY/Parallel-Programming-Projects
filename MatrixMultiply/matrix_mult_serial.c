#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>

#define M1_ROWS 10000
#define M1_COLS 10000
#define M2_COLS 10000

int main () {

	int a[M1_ROWS][M1_COLS];
	int b[M1_COLS][M2_COLS];
 	int c[M1_ROWS][M2_COLS];
	int i, j, k;	
	struct timeval tvalBefore, tvalAfter;

	srand(time(NULL));

	// initialize buffers
	for (i = 0; i < M1_ROWS; i++)
		for (j = 0; j < M1_COLS; j++)
			a[i][j] = 1;//rand() % 10;

	for (i = 0; i < M1_COLS; i++)
		for (j = 0; j < M2_COLS; j++)
			b[i][j] = 1;//rand() % 10;

	// results
	for (i = 0; i < M1_ROWS; i++)
		for (j = 0; j < M2_COLS; j++)
			c[i][j] = 0;

	int num_procs, tid;
	// start timer
	gettimeofday(&tvalBefore, NULL);

	// multiply
	for (i = 0; i < M1_ROWS; i++)
		for (j = 0; j < M2_COLS; j++)
			for (k = 0; k < M1_COLS; k++)
				c[i][j] += a[i][k] * b[k][j];

	// end timer
	gettimeofday(&tvalAfter, NULL);

	//print result
	/*printf("Multiply: a[%d][%d] * b[%d][%d] = c[%d][%d]\nResult:\n", 
			M1_ROWS, M1_COLS, M1_COLS, M2_COLS, M1_ROWS, M2_COLS);
	for (i = 0; i < M1_ROWS; i++) {
		for (j = 0; j < M2_COLS; j++) {
			printf("%d ", c[i][j]);
		}
		printf("\n");
	}*/
	
	//print run time results
	printf("Multiply: a[%d][%d] * b[%d][%d] = c[%d][%d]\n", 
			M1_ROWS, M1_COLS, M1_COLS, M2_COLS, M1_ROWS, M2_COLS);
	printf("Total time: %ld microsecs\n", ((tvalAfter.tv_sec - 
			tvalBefore.tv_sec)*1000000L + tvalAfter.tv_usec) - 
			tvalBefore.tv_usec);

	return 0;
}

