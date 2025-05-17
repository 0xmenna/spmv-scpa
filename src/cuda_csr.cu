#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "csr.h"
#include "cuda_csr.h"
#include "cuda_timer.cuh"

static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_BLOCK = 8;

// -----------------------------------------------------------------------------
// Warp‐wide reduction
// ------------------------------------------------
__inline__ __device__ double warpReduceSum(double val) {
      unsigned mask = 0xFFFFFFFFu;
      for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
      }
      return val;
}

// -----------------------------------------------------------------------------
// Kernel 0: 1 thread → 1 row
// ---------------------------------------------------
__global__ static void spmv_kernel_0(int M, int *IRP, int *JA, double *AS,
                                     double *x, double *y) {
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      if (row < M) {
            double sum = 0.0;
            for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                  sum += AS[j] * x[JA[j]];
            }
            y[row] = sum;
      }
}

// -----------------------------------------------------------------------------
// Kernel 1: 1 warp → 1 row
// ---------------------------------------------------
__global__ static void spmv_kernel_1(int M, const int *IRP, const int *JA,
                                     const double *AS, const double *x,
                                     double *y) {

      int lane = threadIdx.x;
      int row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;

      if (row < M) {
            double sum = 0.0;
            int start = IRP[row], end = IRP[row + 1];
            for (int j = start + lane; j < end; j += WARP_SIZE) {
                  sum += AS[j] * x[JA[j]];
            }
            sum = warpReduceSum(sum);
            if (lane == 0) {
                  y[row] = sum;
            }
      }
}

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
// Host 1: thread‐per‐row
// -------------------------------------------------------
double csr_spmv_cuda_thread_row(const sparse_csr *A, const double *x,
                                double *y) {
      // Memory setup
      int *d_IRP, *d_JA;
      double *d_AS, *d_x, *d_y;
      spmv_cuda_up(A, x, &d_IRP, &d_JA, &d_AS, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(256);
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
// Host 2: warp‐per‐row
// ---------------------------------------------------------
double csr_spmv_cuda_warp_row(const sparse_csr *A, const double *x, double *y) {
      // Memory setup
      int *d_IRP, *d_JA;
      double *d_AS, *d_x, *d_y;
      spmv_cuda_up(A, x, &d_IRP, &d_JA, &d_AS, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE, WARPS_PER_BLOCK);
      dim3 grid_dim((A->M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_1<<<grid_dim, block_dim>>>(A->M, d_IRP, d_JA, d_AS, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, A->M, d_IRP, d_JA, d_AS, d_x, d_y);

      return elapsed;
}
