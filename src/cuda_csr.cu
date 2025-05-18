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
__inline__ __device__ double warp_reduce_sum(double val, int lane) {
      for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_sync(0xFFFFFFFF, val, lane + offset);
      }
      return val;
}

// -----------------------------------------------------------------------------
// Kernel 0: 1 thread → 1 row
// ---------------------------------------------------
__global__ static void spmv_kernel_0(int M, int *IRP, int *JA, double *AS,
                                     double *x, double *y) {
      int row = blockIdx.x * blockDim.x + threadIdx.x;

      if (row >= M)
            return;

      double sum = 0.0;
      for (int j = IRP[row]; j < IRP[row + 1]; j++) {
            sum += AS[j] * x[JA[j]];
      }
      y[row] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 1: 1 warp → 1 row
// ---------------------------------------------------
__global__ static void spmv_kernel_1(int M, const int *IRP, const int *JA,
                                     const double *AS, const double *x,
                                     double *y) {

      int lane = threadIdx.x;
      int row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;

      if (row >= M)
            return;

      int start = IRP[row], end = IRP[row + 1];

      double sum = 0.0;
      for (int j = start + lane; j < end; j += WARP_SIZE) {
            sum += AS[j] * x[JA[j]];
      }
      sum = warp_reduce_sum(sum, lane);
      if (lane == 0)
            y[row] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 2: 1 warp → 1 row. Uses __ldg
// ---------------------------------------------------
__global__ static void spmv_kernel_2(int M, const int *IRP, const int *JA,
                                     const double *AS, const double *x,
                                     double *y) {

      int lane = threadIdx.x;
      int warp_id = threadIdx.y;
      int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

      if (row >= M)
            return;

      int start = __ldg(&IRP[row]);
      int end = __ldg(&IRP[row + 1]);

      double sum = 0.0;
      for (int j = start + lane; j < end; j += WARP_SIZE) {
            sum += AS[j] * __ldg(&x[JA[j]]);
      }

      sum = warp_reduce_sum(sum, lane);
      if (lane == 0) {
            y[row] = sum;
      }
}

// -----------------------------------------------------------------------------
// Kernel 3: 1 block → 1 row. More warps per row. Worst performance: suffers
// from high syncronization across warps in the same block.
// ---------------------------------------------------
__global__ void spmv_kernel_3(int M, const int *IRP, const int *JA,
                              const double *AS, const double *x, double *y) {

      int lane = threadIdx.x;
      int warp_id = threadIdx.y;
      int row = blockIdx.x;

      // Allocate one shared slot per warp
      __shared__ double shared_partial[WARPS_PER_BLOCK];

      int start = __ldg(&IRP[row]);
      int end = __ldg(&IRP[row + 1]);

      // Divide work across warps in the block
      double local_sum = 0.0;
      for (int j = start + warp_id * WARP_SIZE + lane; j < end;
           j += WARPS_PER_BLOCK * WARP_SIZE)
            local_sum += AS[j] * __ldg(&x[JA[j]]);

      // Intra-warp reduction
      local_sum = warp_reduce_sum(local_sum, lane);

      // One thread per warp writes to shared memory
      if (lane == 0) {
            shared_partial[warp_id] = local_sum;
      }
      __syncthreads();

      // Warp 0 does the final sum
      if (warp_id == 0) {
            double final_sum =
                (lane < WARPS_PER_BLOCK) ? shared_partial[lane] : 0.0;
            final_sum = warp_reduce_sum(final_sum, lane);
            if (lane == 0) {
                  y[row] = final_sum;
            }
      }
}

// -----------------------------------------------------------------------------
// Kernel 4: 1 warp → 1 row. Exploits texture cache for x.
// ---------------------------------------------------
__global__ static void spmv_kernel_4(int M, const int *IRP, const int *JA,
                                     const double *AS,
                                     cudaTextureObject_t tex_x, double *y) {

      int lane = threadIdx.x;
      int warp_id = threadIdx.y;
      int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;

      if (row >= M)
            return;

      int start = __ldg(&IRP[row]);
      int end = __ldg(&IRP[row + 1]);

      double sum = 0.0;
      for (int j = start + lane; j < end; j += WARP_SIZE) {
            int2 x_val_i2 = tex1Dfetch<int2>(tex_x, JA[j]);
            double x_val = __longlong_as_double(
                (static_cast<long long>(x_val_i2.y) << 32) |
                static_cast<unsigned int>(x_val_i2.x));
            sum += AS[j] * x_val;
      }

      sum = warp_reduce_sum(sum, lane);
      if (lane == 0)
            y[row] = sum;
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

// -----------------------------------------------------------------------------
// Host 3: warp‐per‐row: uses load-global to read x to improve read
// performance
// ---------------------------------------------------------
double csr_spmv_cuda_warp_row_ldg(const sparse_csr *A, const double *x,
                                  double *y) {
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
      spmv_kernel_2<<<grid_dim, block_dim>>>(A->M, d_IRP, d_JA, d_AS, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, A->M, d_IRP, d_JA, d_AS, d_x, d_y);

      return elapsed;
}

// -----------------------------------------------------------------------------
// Host 4: block‐per‐row: multiple warps of the same block access the same row
// ---------------------------------------------------------
double csr_spmv_cuda_block_row(const sparse_csr *A, const double *x,
                               double *y) {
      // Memory setup
      int *d_IRP, *d_JA;
      double *d_AS, *d_x, *d_y;
      spmv_cuda_up(A, x, &d_IRP, &d_JA, &d_AS, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE, WARPS_PER_BLOCK);
      dim3 grid_dim(A->M);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_3<<<grid_dim, block_dim>>>(A->M, d_IRP, d_JA, d_AS, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, A->M, d_IRP, d_JA, d_AS, d_x, d_y);

      return elapsed;
}

// -----------------------------------------------------------------------------
// Host 5: warp‐per‐row: exploits texture cache for x
// ---------------------------------------------------------
double csr_spmv_cuda_warp_row_text(const sparse_csr *A, const double *x,
                                   double *y) {
      // Memory setup
      int *d_IRP, *d_JA;
      double *d_AS, *d_x, *d_y;
      spmv_cuda_up(A, x, &d_IRP, &d_JA, &d_AS, &d_x, &d_y);

      // Texture Object Setup for x
      cudaResourceDesc resDesc = {};
      resDesc.resType = cudaResourceTypeLinear;
      resDesc.res.linear.devPtr = d_x;
      resDesc.res.linear.sizeInBytes = A->N * sizeof(double);
      resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;
      resDesc.res.linear.desc.x = 32;
      resDesc.res.linear.desc.y = 32;

      cudaTextureDesc texDesc = {};
      texDesc.readMode = cudaReadModeElementType;

      cudaTextureObject_t tex_x = 0;
      cudaCreateTextureObject(&tex_x, &resDesc, &texDesc, nullptr);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE, WARPS_PER_BLOCK);
      dim3 grid_dim((A->M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_4<<<grid_dim, block_dim>>>(A->M, d_IRP, d_JA, d_AS, tex_x,
                                             d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      cudaDestroyTextureObject(tex_x);

      // Copy back to host and cleanup
      spmv_cuda_down(y, A->M, d_IRP, d_JA, d_AS, d_x, d_y);

      return elapsed;
}
