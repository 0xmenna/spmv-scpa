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
            if (col != -1)
                  sum += blk.AS[idx] * x[col];
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
            if (col != -1)
                  sum += blk.AS[idx] * x[col];
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
            if (col != -1)
                  sum += blk.AS[idx] * __ldg(&x[col]);
      }

      y[warp_id * hack_size + lane] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 3: Assign half warp to one row within a block (using row-major for
// colaesced accesses)
// ---------------------------------------------------
__global__ static void spmv_kernel_3(int M, int hack_size,
                                     const ellpack_block *blocks,
                                     int num_blocks, const double *x,
                                     double *y) {

      int lane = threadIdx.x;
      int warp_local = threadIdx.y;
      int halfwarp_in_warp = lane >> 4; // 0 or 1

      // Global half-warp ID across the grid. It correspods to the global matrix
      // row
      int halfwarp_id_global =
          ((blockIdx.x * blockDim.y + warp_local) << 1) + halfwarp_in_warp;

      if (halfwarp_id_global >= M)
            return;

      // Determine which row of which block to compute
      int block_id = halfwarp_id_global / hack_size;
      int row_in_block = halfwarp_id_global % hack_size;

      const ellpack_block blk = blocks[block_id];
      int off = row_in_block * blk.max_NZ;

      int half_lane = lane & 15;

      // Cooperative row processing with half warp
      double sum = 0.0;
      for (int j = half_lane; j < blk.max_NZ; j += 16) {
            int idx = off + j;
            int col = blk.JA[idx];
            if (col != -1)
                  sum += blk.AS[idx] * __ldg(&x[col]);
      }

      // Reduction across half-warp
      for (int offset = 8; offset > 0; offset >>= 1)
            sum += __shfl_sync(0xFFFF, sum, lane + offset, 16);

      // Only first thread in half-warp writes the result
      if (half_lane == 0) {
            y[halfwarp_id_global] = sum;
      }
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
                                       double *y, void *_unused) {
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
                                       double *y, void *_unused) {
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
double hll_spmv_cuda_warp_block(const sparse_hll *H, const double *x, double *y,
                                void *_unused) {
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

// -----------------------------------------------------------------------------
// Host 3: Each half-warp is assigned to a row (uses row-major for coalesced)
// -------------------------------------------------------
double hll_spmv_cuda_halfwarp_row(const sparse_hll *H, const double *x,
                                  double *y, void *_unused) {
      // Memory setup
      ellpack_block *d_blocks;
      double *d_x, *d_y;
      spmv_cuda_up(H, x, &d_blocks, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE, warps_per_block);
      dim3 grid_dim((H->M + warps_per_block * 2 - 1) / (warps_per_block * 2));

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_3<<<grid_dim, block_dim>>>(H->M, H->hack_size, d_blocks,
                                             H->num_blocks, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, H->M, d_blocks, H->num_blocks, d_x, d_y);

      return elapsed;
}
