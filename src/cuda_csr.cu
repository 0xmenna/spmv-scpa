#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

#include "csr.h"
#include "cuda_csr.h"
#include "cuda_timer.cuh"

static constexpr int WARP_SIZE = 32;

static int warps_per_block = 4;

void set_csr_warps_per_block(int wppb) { warps_per_block = wppb; }

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
      int row = blockIdx.x * blockDim.y + threadIdx.y;

      if (row >= M)
            return;

      int start = IRP[row], end = IRP[row + 1];

      double sum = 0.0;
      for (int j = start + lane; j < end; j += WARP_SIZE) {
            sum += AS[j] * x[JA[j]];
      }
      for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
            sum += __shfl_sync(0xFFFFFFFF, sum, lane + offset);

      if (lane == 0)
            y[row] = sum;
}

// -----------------------------------------------------------------------------
// Kernel 2: half warp → 1 row.
// ---------------------------------------------------
__global__ void spmv_kernel_2(int M, const int *IRP, const int *JA,
                              const double *AS, const double *x, double *y) {
      int lane = threadIdx.x;           // 0..31
      int warp_local = threadIdx.y;     // Warp index in block
      int halfwarp_in_warp = lane >> 4; // 0 or 1
      int halfwarp_local = lane & 15;   // 0..15

      // Global half-warp ID
      int halfwarp_id =
          (blockIdx.x * blockDim.y + warp_local) * 2 + halfwarp_in_warp;
      if (halfwarp_id >= M)
            return;

      int row = halfwarp_id;
      int start = __ldg(&IRP[row]);
      int end = __ldg(&IRP[row + 1]);

      double sum = 0.0;
      for (int j = start + halfwarp_local; j < end; j += 16)
            sum += AS[j] * __ldg(&x[JA[j]]);

      // Half-warp reduction (only among 16 threads)
      for (int offset = 8; offset > 0; offset >>= 1)
            sum += __shfl_sync(0xFFFF, sum, lane + offset, 16);

      // First thread of the half-warp writes result
      if (halfwarp_local == 0)
            y[row] = sum;
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

      int warps_per_block = blockDim.y;

      // Allocate one shared slot per each warp within a block
      extern __shared__ double shared_partial[];

      int start = __ldg(&IRP[row]);
      int end = __ldg(&IRP[row + 1]);

      // Divide work across warps in the block
      double local_sum = 0.0;
      for (int j = start + warp_id * WARP_SIZE + lane; j < end;
           j += warps_per_block * WARP_SIZE)
            local_sum += AS[j] * __ldg(&x[JA[j]]);

      // Intra-warp reduction
      for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
            local_sum += __shfl_sync(0xFFFFFFFF, local_sum, lane + offset);

      // One thread per warp writes to shared memory
      if (lane == 0) {
            shared_partial[warp_id] = local_sum;
      }
      __syncthreads();

      // Warp 0 does the final sum
      if (warp_id == 0) {
            double final_sum =
                (lane < warps_per_block) ? shared_partial[lane] : 0.0;

            for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
                  final_sum +=
                      __shfl_sync(0xFFFFFFFF, final_sum, lane + offset);

            if (lane == 0) {
                  y[row] = final_sum;
            }
      }
}

// -----------------------------------------------------------------------------
// Kernel 4: half warp → 1 row. Exploits texture cache for x.
// ---------------------------------------------------
__global__ static void spmv_kernel_4(int M, const int *IRP, const int *JA,
                                     const double *AS,
                                     cudaTextureObject_t tex_x, double *y) {

      int lane = threadIdx.x;
      int warp_local = threadIdx.y;
      int halfwarp_in_warp = lane >> 4;
      int halfwarp_local = lane & 15;

      // Global half-warp ID
      int halfwarp_id =
          (blockIdx.x * blockDim.y + warp_local) * 2 + halfwarp_in_warp;
      if (halfwarp_id >= M)
            return;

      int row = halfwarp_id;
      int start = __ldg(&IRP[row]);
      int end = __ldg(&IRP[row + 1]);

      double sum = 0.0;
      for (int j = start + halfwarp_local; j < end; j += 16) {
            int2 x_val_i2 = tex1Dfetch<int2>(tex_x, JA[j]);
            double x_val = __longlong_as_double(
                (static_cast<long long>(x_val_i2.y) << 32) |
                static_cast<unsigned int>(x_val_i2.x));
            sum += AS[j] * x_val;
      }

      for (int offset = 8; offset > 0; offset >>= 1)
            sum += __shfl_sync(0xFFFF, sum, lane + offset, 16);

      if (halfwarp_local == 0)
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
// Host 2: warp‐per‐row
// ---------------------------------------------------------
double csr_spmv_cuda_warp_row(const sparse_csr *A, const double *x, double *y) {
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
// Host 3: half-warp‐per‐row
// ---------------------------------------------------------
double csr_spmv_cuda_halfwarp_row(const sparse_csr *A, const double *x,
                                  double *y) {
      // Memory setup
      int *d_IRP, *d_JA;
      double *d_AS, *d_x, *d_y;
      spmv_cuda_up(A, x, &d_IRP, &d_JA, &d_AS, &d_x, &d_y);

      cuda_timer timer;
      timer_init(&timer);

      dim3 block_dim(WARP_SIZE, warps_per_block);
      dim3 grid_dim((A->M + (warps_per_block * 2) - 1) / (warps_per_block * 2));

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

      dim3 block_dim(WARP_SIZE, warps_per_block);
      dim3 grid_dim(A->M);

      int shared_size = warps_per_block * sizeof(double);

      // Launch kernel
      timer_start(&timer, 0);
      spmv_kernel_3<<<grid_dim, block_dim, shared_size>>>(A->M, d_IRP, d_JA,
                                                          d_AS, d_x, d_y);
      double elapsed = timer_stop(&timer, 0);

      timer_destroy(&timer);

      // Copy back to host and cleanup
      spmv_cuda_down(y, A->M, d_IRP, d_JA, d_AS, d_x, d_y);

      return elapsed;
}

// -----------------------------------------------------------------------------
// Host 5: half-warp‐per‐row: exploits texture cache for x
// ---------------------------------------------------------
double csr_spmv_cuda_halfwarp_row_text(const sparse_csr *A, const double *x,
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

      dim3 block_dim(WARP_SIZE, warps_per_block);
      dim3 grid_dim((A->M + (warps_per_block * 2) - 1) / (warps_per_block * 2));

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
