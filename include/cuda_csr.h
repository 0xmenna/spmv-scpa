#ifndef CUDA_CSR_H
#define CUDA_CSR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "csr.h"

void set_csr_warps_per_block(int wppb);

double csr_spmv_cuda_thread_row(const sparse_csr *A, const double *x, double *y,
                                void *_unused);

double csr_spmv_cuda_warp_row(const sparse_csr *A, const double *x, double *y,
                              void *_unused);

double csr_spmv_cuda_halfwarp_row(const sparse_csr *A, const double *x,
                                  double *y, void *_unused);

double csr_spmv_cuda_block_row(const sparse_csr *A, const double *x, double *y,
                               void *_unused);

double csr_spmv_cuda_halfwarp_row_text(const sparse_csr *A, const double *x,
                                       double *y, void *_unused);

#ifdef __cplusplus
}
#endif

#endif
