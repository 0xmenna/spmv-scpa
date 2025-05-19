#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "cuda_hll.h"
#include "cuda_timer.cuh"
#include "hll.h"

static constexpr int WARP_SIZE = 32;

static int warps_per_block = 4;

void set_hll_warps_per_block(int wppb) { warps_per_block = wppb; }

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
// Kernel 0: Assign each thread to a row within a block (uses row-major storage)
// ---------------------------------------------------
__global__ static void spmv_kernel_0(int M, int hack_size,
                                     const ellpack_block *blocks,
                                     int num_blocks, const double *x,
                                     double *y) {
      // tid corresponds to the global matrix row
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= M)
            return;

      int block_id = tid / hack_size;
      int row_in_block = tid % hack_size;

      const ellpack_block blk = blocks[block_id];
      int off = row_in_block * blk.max_NZ;

      double sum = 0.0;
      // Row-major storage: each of max_NZ "rows" holds N entries
      for (int j = 0; j < blk.max_NZ; ++j) {
            int idx = off + j;
            int col = blk.JA[idx];
            if (col < 0)
                  break;
            double val = blk.AS[idx];
            sum += val * x[col];
      }

      y[tid] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 1: Assign each thread to a row within a block (uses col-major
// storage -> acccesses are coalesced)
// ---------------------------------------------------
__global__ static void spmv_kernel_1(int M, int hack_size,
                                     const ellpack_block *blocks,
                                     int num_blocks, const double *x,
                                     double *y) {
      // tid corresponds to the global matrix row
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid >= M)
            return;

      int block_id = tid / hack_size;
      int row_in_block = tid % hack_size;

      const ellpack_block blk = blocks[block_id];

      double sum = 0.0;
      // Column-major storage: each of max_NZ "columns" holds M entries
      for (int j = 0; j < blk.max_NZ; ++j) {
            int idx = j * blk.M + row_in_block;
            int col = blk.JA[idx];
            if (col < 0)
                  break;
            double val = blk.AS[idx];
            sum += val * x[col];
      }

      y[tid] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 2: Assign each warp to a block (uses col-major storage). Logically
// this implementation is different from the first kernel, but physically at
// runtime is mapped to the same behavior so we expect basically the same
// performance (except for the fact we exploit the cache here).
// ---------------------------------------------------
__global__ static void spmv_kernel_2(int M, int hack_size,
                                     const ellpack_block *blocks,
                                     int num_blocks, const double *x,
                                     double *y) {

      int lane = threadIdx.x;
      int warp_id = blockIdx.x * blockDim.y + threadIdx.y;

      if (warp_id >= num_blocks)
            return;

      const ellpack_block blk = blocks[warp_id];

      if (lane >= blk.M)
            return;

      double sum = 0.0;
      for (int j = 0; j < blk.max_NZ; ++j) {
            int idx = j * blk.M + lane;
            int col = blk.JA[idx];
            if (col < 0)
                  break;
            double val = blk.AS[idx];
            sum += val * __ldg(&x[col]);
      }

      y[warp_id * hack_size + lane] = sum;
}

static void spmv_cuda_up(const sparse_hll *H, const double *x,
                         ellpack_block **d_blocks, double **d_x, double **d_y) {
      int M = H->M, N = H->N, B = H->num_blocks;

      cudaMalloc(d_blocks, B * sizeof(ellpack_block));

      for (int i = 0; i < B; i++) {
            const ellpack_block *src = &H->blocks[i];

            ellpack_block tmp = {
                .M = src->M, .N = src->N, .NZ = src->NZ, .max_NZ = src->max_NZ};

            int total_len = src->M * src->max_NZ;

            cudaMalloc(&tmp.JA, total_len * sizeof(int));
            cudaMalloc(&tmp.AS, total_len * sizeof(double));
            cudaMemcpy(tmp.JA, src->JA, total_len * sizeof(int),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(tmp.AS, src->AS, total_len * sizeof(double),
                       cudaMemcpyHostToDevice);

            cudaMemcpy(&((*d_blocks)[i]), &tmp, sizeof(ellpack_block),
                       cudaMemcpyHostToDevice);
      }

      cudaMalloc(d_x, N * sizeof(double));
      cudaMalloc(d_y, M * sizeof(double));
      cudaMemcpy(*d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemset(*d_y, 0, M * sizeof(double));
}

static void spmv_cuda_down(double *y, int M, ellpack_block *d_blocks,
                           int num_blocks, double *d_x, double *d_y) {
      cudaMemcpy(y, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

      for (int i = 0; i < num_blocks; i++) {
            ellpack_block tmp;
            cudaMemcpy(&tmp, &d_blocks[i], sizeof(ellpack_block),
                       cudaMemcpyDeviceToHost);
            cudaFree(tmp.JA);
            cudaFree(tmp.AS);
      }

      cudaFree(d_blocks);
      cudaFree(d_x);
      cudaFree(d_y);
}

// -----------------------------------------------------------------------------
// Host 1: Each thread is assigned to a row within a block (uses row-major
// storage)
// -------------------------------------------------------
double hll_spmv_cuda_threads_row_major(const sparse_hll *H, const double *x,
                                       double *y) {
      // Memory setup
      ellpack_block *d_blocks;
      double *d_x, *d_y;
      spmv_cuda_up(H, x, &d_blocks, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(warps_per_block * WARP_SIZE);
      dim3 grid_dim((H->M + block_dim.x - 1) / block_dim.x);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_0<<<grid_dim, block_dim>>>(H->M, H->hack_size, d_blocks,
                                             H->num_blocks, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, H->M, d_blocks, H->num_blocks, d_x, d_y);

      return elapsed;
}

// -----------------------------------------------------------------------------
// Host 2: Each thread is assigned to a row within a block (uses col-major
// storage -> acccesses are coalesced)
// -------------------------------------------------------
double hll_spmv_cuda_threads_col_major(const sparse_hll *H, const double *x,
                                       double *y) {
      // Memory setup
      ellpack_block *d_blocks;
      double *d_x, *d_y;
      spmv_cuda_up(H, x, &d_blocks, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE * warps_per_block);
      dim3 grid_dim((H->M + block_dim.x - 1) / block_dim.x);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_1<<<grid_dim, block_dim>>>(H->M, H->hack_size, d_blocks,
                                             H->num_blocks, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, H->M, d_blocks, H->num_blocks, d_x, d_y);

      return elapsed;
}

// -----------------------------------------------------------------------------
// Host 3: Each warp is assigned to an ellpack block
// -------------------------------------------------------
double hll_spmv_cuda_warp_block(const sparse_hll *H, const double *x,
                                double *y) {
      // Memory setup
      ellpack_block *d_blocks;
      double *d_x, *d_y;
      spmv_cuda_up(H, x, &d_blocks, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE, warps_per_block);
      dim3 grid_dim((H->num_blocks + warps_per_block - 1) / warps_per_block);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_2<<<grid_dim, block_dim>>>(H->M, H->hack_size, d_blocks,
                                             H->num_blocks, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, H->M, d_blocks, H->num_blocks, d_x, d_y);

      return elapsed;
}
