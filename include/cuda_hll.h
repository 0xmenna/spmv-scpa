#ifndef CUDA_HLL_H
#define CUDA_HLL_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hll.h"

void set_hll_warps_per_block(int wppb);

double hll_spmv_cuda_threads_row_major(const sparse_hll *H, const double *x,
                                       double *y, void *_unused);

double hll_spmv_cuda_threads_col_major(const sparse_hll *H, const double *x,
                                       double *y, void *_unused);

double hll_spmv_cuda_warp_block(const sparse_hll *H, const double *x, double *y,
                                void *_unused);

double hll_spmv_cuda_halfwarp_row(const sparse_hll *H, const double *x,
                                  double *y, void *_unused);

#ifdef __cplusplus
}
#endif

#endif