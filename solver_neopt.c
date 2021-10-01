/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"

/**
 * Unoptimized matrix multiplication using the naive algorithm. The
 * expression calculated:
 * A * B * B' + A' * A
 * where A is an upper triangular matrix.
 *
 * All multiplications are done in the i-j-k standard order.
 */
double* my_solver(int N, double *A, double* B) {
	double *C = (double *)calloc(N * N, sizeof(double));
	double *D = (double *)calloc(N * N, sizeof(double));

	/** A' * A
	 * Since A' * A is always a symmetric matrix, regardless of the
	 * properties of A, we can exploit that and iterate only the upper
	 * part of the resulting matrix, and assign the associated elements
	 * of the lower part as we compute the upper part.
	 */
	for (int i = 0; i < N; ++i) {
		for (int j = i; j < N; ++j) {
			for (int k = 0; k <= i; ++k) {
				C[j * N + i] = C[i * N + j] += A[k * N + i] * A[k * N + j];
			}
		}
	}

	// A * B
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = i; k < N; ++k) {
				D[i * N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}

	// (A * B) * B'
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			for (int k = 0; k < N; ++k) {
				C[i * N + j] += D[i * N + k] * B[j * N + k];
			}
		}
	}

	free(D);

	return C;
}
