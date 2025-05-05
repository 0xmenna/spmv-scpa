#ifndef SPARSE_CSR_H
#define SPARSE_CSR_H

#include "utils.h"

#define MAX_NAME 32

typedef struct {
      int col;
      double val;
} Entry;

/** Compressed Sparse Row matrix */
typedef struct {
      char name[MAX_NAME];
      int rows;
      int cols;
      int nnz;
      int *row_off;
      Entry *entries;
} SparseCSR;

static inline void init_csr(SparseCSR *A, const char *name, int rows, int cols,
                            int nnz, int *row_off, Entry *entries) {
      snprintf(A->name, sizeof(A->name), "%s", name);
      A->rows = rows;
      A->cols = cols;
      A->nnz = nnz;
      A->row_off = row_off;
      A->entries = entries;
}

/**
 * Loads a sparse matrix in CSR format from a Matrix Market file.
 */
SparseCSR *io_load_csr(const char *path);

/** Free a SparseCSR */
void csr_free(SparseCSR *A);

Bench bench_csr_serial(const SparseCSR *A);

#endif /* SPARSE_CSR_H */
