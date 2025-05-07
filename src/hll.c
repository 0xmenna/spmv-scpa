// hll.c

#include <stdlib.h>
#include <errno.h>

#include "../include/csr.h"
#include "../include/err.h"
#include "../include/hll.h"
#include "../include/vector.h"

/**
 * Convert a CSR matrix into BlockELLPACK with given HACK_SIZE.
 * @return NULL on error.
 */
BlockELLPACK *csr_to_hll(const sparse_csr *A) {
      int M = A->M;
      int blocks = (M + HACK_SIZE - 1) / HACK_SIZE;

      // 1) Count nonzeros per row and total nnz
      int *nnz_per = malloc(M * sizeof(int));
      if (!nnz_per)
            return NULL;

      // 2) Allocate and init top-level structure
      BlockELLPACK *H = malloc(sizeof(*H));
      if (!H) {
            free(nnz_per);
            return NULL;
      }

      init_hll(H, A->name, A->NZ, A->M, A->N, blocks);

      H->blocks = calloc(blocks, sizeof(*H->blocks));
      if (!H->blocks) {
            free(nnz_per);
            free(H);
            return NULL;
      }

      // 3) For each block, compute max per row and populate
      for (int b = 0; b < blocks; b++) {
            int r0 = b * HACK_SIZE;
            int r1 = (r0 + HACK_SIZE < M ? r0 + HACK_SIZE : M);
            int br = r1 - r0;

            // find maximum nnz in this block
            int maxr = 0;
            for (int i = r0; i < r1; i++) {
                  if (nnz_per[i] > maxr)
                        maxr = nnz_per[i];
            }

            ellpack_block *blk = &H->blocks[b];
            blk->block_rows = br;
            blk->max_per_row = maxr;

            // allocate storage
            blk->col_idx = malloc(br * maxr * sizeof(int));
            blk->block_vals = malloc(br * maxr * sizeof(double));
            if (!blk->col_idx || !blk->block_vals) {
                  // cleanup
                  for (int j = 0; j <= b; j++) {
                        free(H->blocks[j].col_idx);
                        free(H->blocks[j].block_vals);
                  }
                  free(H->blocks);
                  free(H);
                  free(nnz_per);
                  return NULL;
            }

            // init empty
            for (int i = 0; i < br * maxr; i++) {
                  blk->col_idx[i] = -1;
                  blk->block_vals[i] = 0.0;
            }

            // fill from CSR
            for (int i = r0; i < r1; i++) {
                  int off = (i - r0) * maxr;
                  int start = A->IRP[i];
                  int end = A->IRP[i + 1];
                  int p = 0;
                  for (int k = start; k < end; k++, p++) {
                        blk->col_idx[off + p] = A->JA[k];
                        blk->block_vals[off + p] = A->AS[k];
                  }
            }
      }

      free(nnz_per);
      return H;
}

void hll_free(BlockELLPACK *H) {
      if (!H)
            return;
      for (int b = 0; b < H->num_blocks; b++) {
            free(H->blocks[b].col_idx);
            free(H->blocks[b].block_vals);
      }
      free(H->blocks);
      free(H);
}

static void inline hll_spmv_serial(const BlockELLPACK *H, const double *x,
                                   double *y) {
      for (int b = 0; b < H->num_blocks; b++) {
            const ellpack_block *blk = &H->blocks[b];
            int br = blk->block_rows;
            int mz = blk->max_per_row;
            int base = b * HACK_SIZE;

            for (int i = 0; i < br; i++) {
                  double sum = 0.0;
                  int off = i * mz;
                  for (int p = 0; p < mz; p++) {
                        int c = blk->col_idx[off + p];
                        if (c < 0)
                              break;
                        sum += blk->block_vals[off + p] * x[c];
                  }
                  y[base + i] = sum;
            }
      }
}

// Run HLL SpMV serial benchmark
int bench_hll_serial(const BlockELLPACK *H, bench *out) {
      vec x = vec_create(H->cols);
      if (!x.data)
            return -ENOMEM;

      vec_fill(&x, 1.0);

      vec y = vec_create(H->rows);
      if (!y.data) {
            vec_put(&x);
            return -ENOMEM;
      }
      
      double start = now();
      hll_spmv_serial(H, x.data, y.data);
      double end = now();

      vec_put(&x);
      vec_put(&y);

      out->duration_ms = end - start;
      out->gflops = compute_gflops(end - start, H->nnz);

      return 0;
}
