#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "cuda_hll.h"
#include "cuda_timer.cuh"
#include "hll.h"

static constexpr int WARP_SIZE = 32;

static int warps_per_block = 4;

void set_warps_per_block(int wppb) { warps_per_block = wppb; }

// -----------------------------------------------------------------------------
// Warp‐wide reduction
// ------------------------------------------------
__inline__ __device__ double warp_reduce_sum(double val, int lane) {
      for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_sync(0xFFFFFFFF, val, lane + offset);
      }
      return val;
}

// -----------------------------------------------------------------------------
// Kernel 0:
// ---------------------------------------------------
__global__ static void spmv_kernel_0(int M, int *IRP, int *JA, double *AS,
                                     double *x, double *y) {}

// -----------------------------------------------------------------------------
// Kernel 1
// ---------------------------------------------------
__global__ static void spmv_kernel_1(int M, const int *IRP, const int *JA,
                                     const double *AS, const double *x,
                                     double *y) {}

// -----------------------------------------------------------------------------
// Kernel 2:
// ---------------------------------------------------
__global__ static void spmv_kernel_2(int M, const int *IRP, const int *JA,
                                     const double *AS, const double *x,
                                     double *y) {}

static void spmv_cuda_up(const sparse_csr *A, const double *x, int **d_IRP,
                         int **d_JA, double **d_AS, double **d_x,
                         double **d_y) {
      int M = A->M, N = A->N, NZ = A->NZ;

      cudaMalloc(d_IRP, (M + 1) * sizeof(int));
      cudaMalloc(d_JA, NZ * sizeof(int));
      cudaMalloc(d_AS, NZ * sizeof(double));
      cudaMalloc(d_x, N * sizeof(double));
      cudaMalloc(d_y, M * sizeof(double));

      cudaMemcpy(*d_IRP, A->IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(*d_JA, A->JA, NZ * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(*d_AS, A->AS, NZ * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(*d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
}

static void spmv_cuda_down(double *y, int M, int *d_IRP, int *d_JA,
                           double *d_AS, double *d_x, double *d_y) {
      cudaMemcpy(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(d_IRP);
      cudaFree(d_JA);
      cudaFree(d_AS);
      cudaFree(d_x);
      cudaFree(d_y);
}

// -----------------------------------------------------------------------------
// Host 1:
// -------------------------------------------------------
double csr_spmv_cuda_1(const sparse_csr *A, const double *x, double *y) {
      // Memory setup
      int *d_IRP, *d_JA;
      double *d_AS, *d_x, *d_y;
      spmv_cuda_up(A, x, &d_IRP, &d_JA, &d_AS, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE * warps_per_block);
      dim3 grid_dim((A->M + block_dim.x - 1) / block_dim.x);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_0<<<grid_dim, block_dim>>>(A->M, d_IRP, d_JA, d_AS, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, A->M, d_IRP, d_JA, d_AS, d_x, d_y);

      return elapsed;
}

// -----------------------------------------------------------------------------
// Host 2:
// ---------------------------------------------------------
double csr_spmv_cuda_2(const sparse_csr *A, const double *x, double *y) {
      // Memory setup
      int *d_IRP, *d_JA;
      double *d_AS, *d_x, *d_y;
      spmv_cuda_up(A, x, &d_IRP, &d_JA, &d_AS, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE, warps_per_block);
      dim3 grid_dim((A->M + warps_per_block - 1) / warps_per_block);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_1<<<grid_dim, block_dim>>>(A->M, d_IRP, d_JA, d_AS, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, A->M, d_IRP, d_JA, d_AS, d_x, d_y);

      return elapsed;
}

// -----------------------------------------------------------------------------
// Host 3: warp‐per‐row: uses load-global to read x to improve read
// performance
// ---------------------------------------------------------
double csr_spmv_cuda_3(const sparse_csr *A, const double *x, double *y) {
      // Memory setup
      int *d_IRP, *d_JA;
      double *d_AS, *d_x, *d_y;
      spmv_cuda_up(A, x, &d_IRP, &d_JA, &d_AS, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE, warps_per_block);
      dim3 grid_dim((A->M + warps_per_block - 1) / warps_per_block);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_2<<<grid_dim, block_dim>>>(A->M, d_IRP, d_JA, d_AS, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, A->M, d_IRP, d_JA, d_AS, d_x, d_y);

      return elapsed;
}
