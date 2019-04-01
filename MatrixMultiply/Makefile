all: omp pthread serial

omp: matrix_mult_omp.c
	gcc -fopenmp matrix_mult_omp.c -o matrix_mult_omp

pthread: matrix_mult_pth.c
	 gcc matrix_mult_pth.c -o matrix_mult_pth -lpthread

serial: matrix_mult_serial.c
	gcc matrix_mult_serial.c -o matrix_mult_serial

clean:
	rm -rf matrix_mult_omp
	rm -rf matrix_mult_pth
	rm -rf matrix_mult_serial

