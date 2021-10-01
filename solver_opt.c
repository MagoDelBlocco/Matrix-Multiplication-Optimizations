/*
 * Tema 2 ASC
 * 2021 Spring
 */
#include "utils.h"

// D[i][j] = A[i][k] * B[k][j]
#define D_ij__pluseq__A_ik__x__B_kj(OFFSET) \
						"movapd " OFFSET "(%2), 		%%xmm1\n\t" \
						"mulpd 			%%xmm0, 		%%xmm1\n\t" \
						"addpd "  OFFSET "(%0), 		%%xmm1\n\t" \
						"movapd 		%%xmm1, " OFFSET "(%0)\n\t"

// C[i][j] += D[i][k] * B[j][k]
#define C_ij__pluseq__D_ik__x__B_jk(OFFSET) \
						"movapd " OFFSET "(%1), 		%%xmm1\n\t" \
						"movapd " OFFSET "(%2), 		%%xmm2\n\t" \
						"mulpd 			%%xmm1, 		%%xmm2\n\t" \
						"addpd 			%%xmm2, 		%%xmm0\n\t"

#define RANDOMIZED 0
#define SEQUENTIAL 3

#define prefetch_read(addr) 		__builtin_prefetch(addr, 0, SEQUENTIAL);
#define prefetch_write(addr, local) __builtin_prefetch(addr, 1, local);

#define likely(x)       			__builtin_expect((x),1)
#define unlikely(x)     			__builtin_expect((x),0)

#define lea(base, i, j) (&base[i * N + j])

#define TILE 8

/**
 * Performs a highly optimized matrix multiplication algorithm, based on the
 * expression
 * A * B * B' + A' * A
 * where A is an upper triangular matrix.
 */
double* my_solver(
	register int N,
	register double *__restrict__ A,
	register double *__restrict__ B) {
	double *__restrict__ C = (double *)calloc(N * N, sizeof(double));
	double *__restrict__ D = (double *)calloc(N * N, sizeof(double));
	register double *__restrict__ pa;
	register double *__restrict__ pb;
	register double *__restrict__ pc;
	register double *__restrict__ origC;
	register double *__restrict__ origD;
	register int i, j, k;
	// OBS: loop tiling doesn't improve performance
	// obs: memorie nu e aliniata pentru AVX, primesc General Protection Fault cand incerc

	i ^= i;
	do { // for i = 0..N
		origC = lea(C, i, i);
		origD = lea(D, i, 0);

		k ^= k;
		do { // for k = 0..N
			if (k <= i) { // A' * A (~10% load)
				pa = lea(A, k, i);
				pb = pc = origC;
				register double constant = *pa;

				prefetch_read(pa);
				prefetch_write(pb, SEQUENTIAL);
				prefetch_write(pc, RANDOMIZED);
				j = i;
				do { // for j = i..N
					// C[j][i] = C[i][j] += A[k][i] * A[k][j]
					*pc = *pb += constant * *pa;
					++pb;
					++pa;
					pc += N;
					prefetch_write(pc, 0);
				} while (likely(++j < N)); // j = i : N
			}

			if (k >= i) { // A * B (~35% load)
				pb = lea(B, k, 0);
				pa = origD;
				register double constant __asm__("xmm0");
				// loads A[i][k] in HIGH(constant) and LOW(constant)
				__asm__ (
					"movddup (%1), %%xmm0"
					: "=X" (constant)
					: "rm" (lea(A, i, k))
				);

				prefetch_read(pb);
				prefetch_write(pa, SEQUENTIAL);
				j = N;
				do { // for j = 0..N
					// D[i][j] += A[i][k] * B[k][j]
					// translates to: *pa += constant * *pb
					__asm__ (
						D_ij__pluseq__A_ik__x__B_kj("0x0")
						D_ij__pluseq__A_ik__x__B_kj("0x10")
						D_ij__pluseq__A_ik__x__B_kj("0x20")
						D_ij__pluseq__A_ik__x__B_kj("0x30")

						: "+rm" (pa)
						: "X" (constant), "rm" (pb)
						: "xmm1", "memory"
					);

					pa += TILE;
					pb += TILE;
					prefetch_read(pb);
					prefetch_write(pa, SEQUENTIAL);
				} while (j -= TILE);
			}
		} while (likely(++k < N));
	} while (likely(++i < N));

	// D * B' + C (~55% load)
	pc = lea(C, 0, 0);
	i ^= i;

	do { // for i = 0..N
		origD = lea(D, i, 0);

		j ^= j;
		do { // for j = 0..N
			pb = lea(B, j, 0);
			pa = origD;
			register double sum __asm__("xmm0");
			__asm__ (
				"pxor %%xmm0, %%xmm0"
				: "+X" (sum)
				: "X" (sum)
			);

			k = N;
			prefetch_read(pa);
			prefetch_read(pb);
			do { // for k = 0..N
				// translates to: sum += *pa * *pb;
				__asm__ (
					C_ij__pluseq__D_ik__x__B_jk("0x0")
					C_ij__pluseq__D_ik__x__B_jk("0x10")
					C_ij__pluseq__D_ik__x__B_jk("0x20")
					C_ij__pluseq__D_ik__x__B_jk("0x30")

					: "=X" (sum)
					: "rm" (pa), "rm" (pb)
					: "xmm1", "xmm2", "memory"
				);

				pa += TILE;
				pb += TILE;
				prefetch_read(pa);
				prefetch_read(pb);
			} while (likely(k -= TILE));

			prefetch_write(pc, 0);
			// Adds the two sums encoded in `sum`, then performs *pc += LOW(sum)
			__asm__ (
				"haddpd %%xmm0, %%xmm0\n\t"
				"addsd (%1),	%%xmm0\n\t"
				"movsd %%xmm0,    (%1)\n\t"
				: "+X" (sum)
				: "rm" (pc)
			);

			++pc;
		} while (++j < N);
	} while (++i < N);

	free(D);

	return C;
}
