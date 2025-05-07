#ifndef SPARSE_HLL_H
#define SPARSE_HLL_H

#include "csr.h"
#include "utils.h"
#include <stdlib.h>

#define HACK_SIZE 32

/** Blocked-ELLPACK (“HLL”) block */
typedef struct {
      int block_rows;
      int max_per_row;
      int *col_idx;
      double *block_vals;
} ellpack_block;

/** Full Blocked-ELLPACK matrix */
typedef struct {
      char name[MAX_NAME];
      int nnz;
      int rows, cols;
      int num_blocks;
      ellpack_block *blocks;
} BlockELLPACK;

static inline void init_hll(BlockELLPACK *H, const char *name, int nnz,
                            int rows, int cols, int num_blocks) {
      snprintf(H->name, sizeof(H->name), "%s", name);
      H->nnz = nnz;
      H->rows = rows;
      H->cols = cols;
      H->num_blocks = num_blocks;
      H->blocks = NULL;
}

/**
 * Convert a CSR matrix into BlockELLPACK.
 * @return NULL on error.
 */
BlockELLPACK *csr_to_hll(const sparse_csr *A);

/** Free a BlockELLPACK */
void hll_free(BlockELLPACK *H);

int bench_hll_serial(const BlockELLPACK *H, bench *out);

#endif /* SPARSE_HLL_H */
