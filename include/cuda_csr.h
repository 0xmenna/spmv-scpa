#ifndef CUDA_CSR_H
#define CUDA_CSR_H

#ifdef __cplusplus
extern "C" {
#endif

#include "csr.h"

double csr_spmv_cuda_thread_row(const sparse_csr *A, const double *x,
                                double *y);

double csr_spmv_cuda_warp_row(const sparse_csr *A, const double *x, double *y);

#ifdef __cplusplus
}
#endif

#endif
