// hll.c

#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

#include "csr.h"
#include "cuda_hll.h"
#include "err.h"
#include "hll.h"
#include "vector.h"

// The number of threads for OpenMP
static int num_threads = 2;

/**
 * Convert a CSR matrix into a HLL matrix.
 * @return NULL on error.
 */
sparse_hll *csr_to_hll(const sparse_csr *A, bool is_col_major) {
      int M = A->M;
      int N = A->N;
      int blocks = (M + HACK_SIZE - 1) / HACK_SIZE;

      sparse_hll *H = malloc(sizeof(*H));
      if (!H) {
            return ERR_PTR(-ENOMEM);
      }

      // Initialize the HLL header
      init_hll(H, A->name, A->M, A->N, A->NZ, blocks);

      H->blocks = aligned_malloc(blocks * sizeof(ellpack_block));
      if (!H->blocks) {
            free(H);
            return ERR_PTR(-ENOMEM);
      }

      for (int b = 0; b < blocks; b++) {
            int row_start = b * HACK_SIZE;
            int row_end =
                (row_start + HACK_SIZE < M ? row_start + HACK_SIZE : M);
            int blk_rows = row_end - row_start;

            // Find the max # of nonzeros in any row in this block, and the
            // total NZ in the block
            int max_nnz = 0;
            int total_nnz = 0;
            for (int i = row_start; i < row_end; i++) {
                  int r_nz = A->IRP[i + 1] - A->IRP[i];
                  total_nnz += r_nz;
                  if (r_nz > max_nnz)
                        max_nnz = r_nz;
            }

            ellpack_block *blk = &H->blocks[b];
            init_ellpack_block(blk, blk_rows, N, total_nnz, max_nnz);

            int total_len = blk_rows * max_nnz;

            blk->JA = aligned_malloc(sizeof(int) * total_len);
            blk->AS = aligned_malloc(sizeof(double) * total_len);
            if (!blk->JA || !blk->AS) {
                  // Clean up previously allocated memory
                  for (int bb = 0; bb <= b; bb++) {
                        free(H->blocks[bb].JA);
                        free(H->blocks[bb].AS);
                  }
                  free(H->blocks);
                  free(H);
                  return ERR_PTR(-ENOMEM);
            }

            for (int i = 0; i < total_len; i++) {
                  blk->JA[i] = -1;
                  blk->AS[i] = 0.0;
            }

            for (int i = 0; i < blk_rows; i++) {
                  int row = row_start + i;
                  int start = A->IRP[row];
                  int row_nz = A->IRP[row + 1] - start;
                  for (int j = 0; j < row_nz; j++) {
                        if (is_col_major) {
                              blk->JA[j * blk_rows + i] = A->JA[start + j];
                              blk->AS[j * blk_rows + i] = A->AS[start + j];
                        } else {
                              blk->JA[i * max_nnz + j] = A->JA[start + j];
                              blk->AS[i * max_nnz + j] = A->AS[start + j];
                        }
                  }
            }
      }

      return H;
}

void hll_free(sparse_hll *H) {
      if (!H)
            return;
      for (int i = 0; i < H->num_blocks; i++) {
            free(H->blocks[i].JA);
            free(H->blocks[i].AS);
      }
      free(H->blocks);
      free(H);
}

static int compute_benchmark_hll(const sparse_hll *H, bench *out,
                                 double (*spmv_f)(const sparse_hll *H,
                                                  const double *x, double *y)) {
      vec x = vec_create(H->N);
      if (!x.data)
            return -ENOMEM;

      vec_fill(&x, 1.0);

      vec y = vec_create(H->M);
      if (!y.data) {
            vec_put(&x);
            return -ENOMEM;
      }

      double duration = spmv_f(H, x.data, y.data);

      vec_put(&x);

      out->duration_ms = duration;
      out->gflops = compute_gflops(duration, H->NZ);
      out->data = y;

      return 0;
}

static inline double hll_spmv_serial(const sparse_hll *H, const double *x,
                                     double *y) {
      double start = now();
      for (int b = 0; b < H->num_blocks; b++) {
            const ellpack_block *blk = &H->blocks[b];
            int M = blk->M;
            int max_NZ = blk->max_NZ;
            int base = b * HACK_SIZE;

            for (int i = 0; i < M; i++) {
                  double sum = 0.0;
                  int off = i * max_NZ;
                  for (int j = 0; j < max_NZ; j++) {
                        int c = blk->JA[off + j];
                        if (c != -1)
                              sum += blk->AS[off + j] * x[c];
                  }
                  y[base + i] = sum;
            }
      }
      double end = now();

      return end - start;
}

static inline double _hll_spmv_serial_col_major(const sparse_hll *H,
                                                const double *x, double *y) {

      double start = now();
      for (int b = 0; b < H->num_blocks; b++) {
            const ellpack_block *blk = &H->blocks[b];
            int M = blk->M;
            int max_NZ = blk->max_NZ;
            int base = b * HACK_SIZE;

            for (int i = 0; i < M; i++) {
                  double sum = 0.0;
                  for (int j = 0; j < max_NZ; j++) {
                        int c = blk->JA[j * M + i];
                        if (c != -1)
                              sum += blk->AS[j * M + i] * x[c];
                  }
                  y[base + i] = sum;
            }
      }
      double end = now();

      return end - start;
}

static inline double hll_spmv_omp(const sparse_hll *restrict H,
                                  const double *restrict x,
                                  double *restrict y) {

      assert(num_threads <= omp_get_max_threads());

      double start = omp_get_wtime();

// Each thread is assigned to an ellpack block execution.
#pragma omp parallel for schedule(guided) num_threads(num_threads)
      for (int b = 0; b < H->num_blocks; b++) {
            const ellpack_block blk = H->blocks[b];
            int M = blk.M;
            int max_NZ = blk.max_NZ;
            int base = b * HACK_SIZE;

            for (int i = 0; i < M; i++) {
                  double sum = 0.0;
                  int off = i * max_NZ;
                  for (int j = 0; j < max_NZ; j++) {
                        int c = blk.JA[off + j];
                        if (c != -1)
                              sum += blk.AS[off + j] * x[c];
                  }
                  y[base + i] = sum;
            }
      }

      double end = omp_get_wtime();

      return (end - start) * 1e3;
}

// Run HLL SpMV serial benchmark
inline int bench_hll_serial(const sparse_hll *H, bench *out) {
      return compute_benchmark_hll(H, out, hll_spmv_serial);
}

inline int bench_hll_omp(const sparse_hll *H, bench_omp *out) {
      num_threads = out->num_threads;

      return compute_benchmark_hll(H, &out->bench, hll_spmv_omp);
}

inline int bench_hll_cuda_threads_row_major(const sparse_hll *H,
                                            bench_cuda *out) {
      set_hll_warps_per_block(out->warps_per_block);

      return compute_benchmark_hll(H, &out->bench,
                                   hll_spmv_cuda_threads_row_major);
}

inline int bench_hll_cuda_threads_col_major(const sparse_hll *H,
                                            bench_cuda *out) {
      set_hll_warps_per_block(out->warps_per_block);

      return compute_benchmark_hll(H, &out->bench,
                                   hll_spmv_cuda_threads_col_major);
}

inline int bench_hll_cuda_warp_block(const sparse_hll *H, bench_cuda *out) {
      set_hll_warps_per_block(out->warps_per_block);

      return compute_benchmark_hll(H, &out->bench, hll_spmv_cuda_warp_block);
}

inline int bench_hll_cuda_halfwarp_row(const sparse_hll *H, bench_cuda *out) {
      set_hll_warps_per_block(out->warps_per_block);

      return compute_benchmark_hll(H, &out->bench, hll_spmv_cuda_halfwarp_row);
}
