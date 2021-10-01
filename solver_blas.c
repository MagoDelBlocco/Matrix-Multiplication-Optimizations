/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include <cblas.h>
#include <string.h>
#include "utils.h"

/**
 * BLAS implementation of the following equation:
 * C = A * B * B' + A' * A
 * where A is an upper triangular matrix.
 */
double* my_solver(int N, double *A, double *B) {
	double *C = (double *)malloc(N * N * sizeof(double));
	memcpy(C, A, N * N * sizeof(double));
	double *aux = (double *)malloc(N * N * sizeof(double));
	memcpy(aux, B, N * N * sizeof(double));

	// C = A' * A
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
				N, N, 1.0, A, N, C, N);

	// B = A * B
	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
				N, N, 1.0, A, N, B, N);

	// C = A * B' + C
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N,
				N, N, 1.0, B, N, aux, N, 1.0, C, N);

	free(aux);

	return C;
}
