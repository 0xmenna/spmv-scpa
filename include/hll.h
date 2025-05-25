#ifndef SPARSE_HLL_H
#define SPARSE_HLL_H

#include <stdbool.h>
#include <stdlib.h>

#include "csr.h"
#include "utils.h"

#define HACK_SIZE 32

/** ELLPACK block */
typedef struct {
      int M, N, NZ;
      int max_NZ;
      int *JA;
      double *AS;
} ellpack_block;

static inline void init_ellpack_block(ellpack_block *blk, int M, int N, int NZ,
                                      int max_NZ) {
      blk->M = M;
      blk->N = N;
      blk->NZ = NZ;
      blk->max_NZ = max_NZ;
      blk->JA = NULL;
      blk->AS = NULL;
}

/** HLL matrix */
typedef struct {
      char name[MAX_NAME];
      int M, N, NZ;
      int hack_size;
      int num_blocks;
      ellpack_block *blocks;
} sparse_hll;

static inline void init_hll(sparse_hll *H, const char *name, int M, int N,
                            int NZ, int num_blocks) {
      snprintf(H->name, sizeof(H->name), "%s", name);
      H->M = M;
      H->N = N;
      H->NZ = NZ;
      H->hack_size = HACK_SIZE;
      H->num_blocks = num_blocks;
      H->blocks = NULL;
}

/**
 * Convert a CSR matrix into BlockELLPACK.
 * @return NULL on error.
 */
sparse_hll *csr_to_hll(const sparse_csr *A, bool is_col_major);

/** Free a BlockELLPACK */
void hll_free(sparse_hll *H);

int bench_hll_serial(const sparse_hll *H, const double *x, bench *out);

int bench_hll_omp(const sparse_hll *H, const double *x, bench_omp *out);

int bench_hll_cuda_threads_row_major(const sparse_hll *H, const double *x,
                                     bench_cuda *out);
int bench_hll_cuda_threads_col_major(const sparse_hll *H, const double *x,
                                     bench_cuda *out);
int bench_hll_cuda_warp_block(const sparse_hll *H, const double *x,
                              bench_cuda *out);
int bench_hll_cuda_halfwarp_row(const sparse_hll *H, const double *x,
                                bench_cuda *out);

#endif /* SPARSE_HLL_H */
