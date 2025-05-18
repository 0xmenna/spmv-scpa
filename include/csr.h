#ifndef SPARSE_CSR_H
#define SPARSE_CSR_H

#include "utils.h"

/** Compressed Sparse Row matrix */
typedef struct sparse_matrix_csr {
      char name[MAX_NAME];
      int M, N, NZ;
      int *IRP;
      int *JA;
      double *AS;
} sparse_csr;

static inline void init_csr(sparse_csr *A, const char *name, int M, int N,
                            int NZ, int *IRP, int *JA, double *AS) {
      snprintf(A->name, sizeof(A->name), "%s", name);
      A->M = M;
      A->N = N;
      A->NZ = NZ;
      A->IRP = IRP;
      A->JA = JA;
      A->AS = AS;
}

/**
 * Loads a sparse matrix in CSR format from a Matrix Market file.
 */
sparse_csr *io_load_csr(const char *path);

/** Free a SparseCSR */
void csr_free(sparse_csr *A);

int bench_csr_serial(const sparse_csr *A, bench *out);

int bench_csr_omp_guided(const sparse_csr *A, bench_omp *out);
int bench_csr_omp_nnz_balancing(const sparse_csr *A, bench_omp *out);

int bench_csr_cuda_thread_row(const sparse_csr *A, bench_cuda *out);
int bench_csr_cuda_warp_row(const sparse_csr *A, bench_cuda *out);
int bench_csr_cuda_warp_row_ldg(const sparse_csr *A, bench_cuda *out);
int bench_csr_cuda_block_row(const sparse_csr *A, bench_cuda *out);
int bench_csr_cuda_warp_row_text(const sparse_csr *A, bench_cuda *out);

#endif /* SPARSE_CSR_H */
