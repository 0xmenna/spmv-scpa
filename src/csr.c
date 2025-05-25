// csr.c

#include <assert.h>
#include <errno.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csr.h"
#include "cuda_csr.h"
#include "err.h"
#include "mmio.h"
#include "utils.h"
#include "vector.h"

void extract_matrix_name(const char *path, char *name_out) {
      const char *base = strrchr(path, '/');
      base = (base) ? base + 1 : path;

      size_t len = strlen(base);
      if (len > 4 && strcmp(base + len - 4, ".mtx") == 0) {
            len -= 4;
      }

      size_t copy_len = (len < (MAX_NAME - 1)) ? len : (MAX_NAME - 1);
      memcpy(name_out, base, copy_len);
      name_out[copy_len] = '\0';
}
sparse_csr *io_load_csr(const char *path) {
      char m_name[MAX_NAME];
      FILE *f = NULL;
      int *row_counts = NULL, *IRP = NULL, *JA = NULL;
      double *AS = NULL;
      sparse_csr *A = NULL;
      MM_typecode tt;
      int M, N, nz0;
      long total_nnz = 0;
      int is_sym, is_pat;
      int ret_errno = 0;

      extract_matrix_name(path, m_name);

      if (!(f = fopen(path, "r")))
            return ERR_PTR(-errno);

      if (mm_read_banner(f, &tt) || !mm_is_matrix(tt) || !mm_is_sparse(tt) ||
          !(mm_is_real(tt) || mm_is_pattern(tt))) {
            ret_errno = -EINVAL;
            goto cleanup;
      }

      if (mm_read_mtx_crd_size(f, &M, &N, &nz0)) {
            ret_errno = -EINVAL;
            goto cleanup;
      }
      is_sym = mm_is_symmetric(tt);
      is_pat = mm_is_pattern(tt);

      // First pass: count entries per row
      long header_pos = ftell(f);
      if (!(row_counts = calloc((size_t)M, sizeof *row_counts))) {
            ret_errno = -ENOMEM;
            goto cleanup;
      }

      for (int e = 0, i, j; e < nz0; ++e) {
            double v;
            if (is_pat) {
                  if (fscanf(f, "%d %d", &i, &j) != 2) {
                        ret_errno = -EIO;
                        goto cleanup;
                  }
                  v = 1.0;
            } else {
                  if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) {
                        ret_errno = -EIO;
                        goto cleanup;
                  }
            }
            --i;
            --j;
            if (i < 0 || i >= M || j < 0 || j >= N) {
                  ret_errno = -ERANGE;
                  goto cleanup;
            }

            row_counts[i]++;
            total_nnz++;
            if (is_sym && i != j) {
                  row_counts[j]++;
                  total_nnz++;
            }
      }

      // Build IRP
      IRP = aligned_malloc((size_t)(M + 1) * sizeof(int));
      if (!IRP) {
            ret_errno = -ENOMEM;
            goto cleanup;
      }
      IRP[0] = 0;
      for (int r = 0; r < M; ++r)
            IRP[r + 1] = IRP[r] + row_counts[r];

      // Build JA and AS
      JA = aligned_malloc((size_t)total_nnz * sizeof(int));
      AS = aligned_malloc((size_t)total_nnz * sizeof(double));
      if (!JA || !AS) {
            ret_errno = -ENOMEM;
            goto cleanup;
      }

      // Second pass: fill CSR
      memset(row_counts, 0, (size_t)M * sizeof *row_counts);
      if (fseek(f, header_pos, SEEK_SET) != 0) {
            ret_errno = -EIO;
            goto cleanup;
      }

      for (int e = 0, i, j; e < nz0; ++e) {
            double v;
            if (is_pat) {
                  if (fscanf(f, "%d %d", &i, &j) != 2) {
                        ret_errno = -EIO;
                        goto cleanup;
                  }
                  v = 1.0;
            } else {
                  if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) {
                        ret_errno = -EIO;
                        goto cleanup;
                  }
            }
            --i;
            --j;
            long idx = (long)IRP[i] + row_counts[i]++;
            JA[idx] = j;
            AS[idx] = v;
            if (is_sym && i != j) {
                  long idx2 = (long)IRP[j] + row_counts[j]++;
                  JA[idx2] = i;
                  AS[idx2] = v;
            }
      }

      A = malloc(sizeof(sparse_csr));
      if (!A) {
            ret_errno = -ENOMEM;
            goto cleanup;
      }
      init_csr(A, m_name, M, N, (int)total_nnz, IRP, JA, AS);

cleanup:
      if (row_counts)
            free(row_counts);
      if (ret_errno && IRP)
            free(IRP);
      if (ret_errno && JA)
            free(JA);
      if (ret_errno && AS)
            free(AS);
      if (f)
            fclose(f);

      if (ret_errno)
            return ERR_PTR(ret_errno);
      else
            return A;
}

void csr_free(sparse_csr *A) {
      if (!A)
            return;
      free(A->IRP);
      free(A->JA);
      free(A->AS);
      free(A);
}

static int compute_benchmark_csr(const sparse_csr *A, const double *x,
                                 bench *out, void *add_arg,
                                 double (*spmv_f)(const sparse_csr *A,
                                                  const double *x, double *y,
                                                  void *add_ardg)) {
      vec y = vec_create(A->M);
      if (!y.data) {
            return -ENOMEM;
      }

      double duration = spmv_f(A, x, y.data, add_arg);

      out->duration_ms = duration;
      out->gflops = compute_gflops(duration, A->NZ);
      out->data = y;

      return 0;
}

static inline double csr_spmv_serial(const sparse_csr *A, const double *x,
                                     double *y, void *_unused) {

      double start = now();
      for (int i = 0; i < A->M; i++) {
            double sum = 0.0;
            for (int j = A->IRP[i]; j < A->IRP[i + 1]; j++) {
                  sum += A->AS[j] * x[A->JA[j]];
            }

            y[i] = sum;
      }
      double end = now();

      return end - start;
}

static int *partition_csr_rows(const sparse_csr *A, int *num_threads) {

      int M = A->M;
      int max_t = *num_threads;

      // Count nonzeros per row
      int *nz_per_row = malloc(M * sizeof(int));
      if (!nz_per_row) {
            return ERR_PTR(-ENOMEM);
      }

      long total_nz = 0;
      for (int r = 0; r < M; r++) {
            nz_per_row[r] = A->IRP[r + 1] - A->IRP[r];
            total_nz += nz_per_row[r];
      }

      int *starts = malloc((max_t + 1) * sizeof(int));
      if (!starts) {
            free(nz_per_row);
            return ERR_PTR(-ENOMEM);
      }

      // Target workload per thread
      double target = (double)total_nz / max_t;

      int curr_tid = 0;
      starts[curr_tid] = 0;
      double running = 0.0;

      for (int r = 0; r < M && curr_tid < max_t - 1; r++) {
            running += nz_per_row[r];

            if (running >= target) {
                  curr_tid++;
                  starts[curr_tid] = r + 1;

                  running = 0.0;
            }
      }

      starts[curr_tid + 1] = M;

      int used_threads = curr_tid + 1;
      if (used_threads < max_t) {
            *num_threads = used_threads;
            int *shrunken = realloc(starts, (used_threads + 1) * sizeof(int));
            if (!shrunken) {
                  free(starts);
                  free(nz_per_row);
                  return ERR_PTR(-ENOMEM);
            }
            starts = shrunken;
      }

      free(nz_per_row);

      return starts;
}

static double csr_spmv_omp_guided(const sparse_csr *restrict A,
                                  const double *restrict x, double *restrict y,
                                  void *add_arg) {

      int num_threads = *(int *)add_arg;

      double start = omp_get_wtime();

#pragma omp parallel for schedule(guided) num_threads(num_threads)
      for (int i = 0; i < A->M; i++) {
            double sum = 0.0;
            for (int j = A->IRP[i]; j < A->IRP[i + 1]; j++) {
                  sum += A->AS[j] * x[A->JA[j]];
            }
            y[i] = sum;
      }

      double end = omp_get_wtime();

      return (end - start) * 1e3;
}

struct omp_nnz_balancing_arg {
      int num_threads;
      int *partitions;
};

static double csr_spmv_omp_nnz_balancing(const sparse_csr *restrict A,
                                         const double *restrict x,
                                         double *restrict y, void *add_arg) {

      // Ensure input arrays are cache-line aligned
      assert(((uintptr_t)A->AS % ALIGNMENT) == 0);
      assert(((uintptr_t)A->JA % ALIGNMENT) == 0);
      assert(((uintptr_t)x % ALIGNMENT) == 0);

      struct omp_nnz_balancing_arg *arg =
          (struct omp_nnz_balancing_arg *)add_arg;

      int num_threads = arg->num_threads;
      int *partitions = arg->partitions;

      assert(num_threads <= omp_get_max_threads());

      double start = omp_get_wtime();
#pragma omp parallel num_threads(num_threads)
      {
            int tid = omp_get_thread_num();
            int r0 = partitions[tid];
            int r1 = partitions[tid + 1];
            for (int i = r0; i < r1; i++) {
                  double sum = 0.0;
                  for (int j = A->IRP[i]; j < A->IRP[i + 1]; j++) {
                        sum += A->AS[j] * x[A->JA[j]];
                  }
                  y[i] = sum;
            }
      }
      double end = omp_get_wtime();

      return (end - start) * 1e3;
}

// Run CSR SpMV serial benchmark
inline int bench_csr_serial(const sparse_csr *A, const double *x, bench *out) {
      return compute_benchmark_csr(A, x, out, NULL, csr_spmv_serial);
}

// Run CSR SpMV OpenMP benchmark (guided scheduling)
int bench_csr_omp_guided(const sparse_csr *A, const double *x, bench_omp *out) {

      snprintf(out->name, sizeof(out->name), "omp_guided");

      return compute_benchmark_csr(A, x, &out->bench, &out->num_threads,
                                   csr_spmv_omp_guided);
}

// Run CSR SpMV OpenMP benchmark (schduled based on nnz)
int bench_csr_omp_nnz_balancing(const sparse_csr *A, const double *x,
                                bench_omp *out) {
      int ret;

      int *partitions = partition_csr_rows(A, &out->num_threads);
      if (IS_ERR(partitions)) {
            return PTR_ERR(partitions);
      }

      struct omp_nnz_balancing_arg arg = {
          .num_threads = out->num_threads,
          .partitions = partitions,
      };

      ret = compute_benchmark_csr(A, x, &out->bench, &arg,
                                  csr_spmv_omp_nnz_balancing);

      // Clean up partitions
      free(partitions);
      partitions = NULL;

      snprintf(out->name, sizeof(out->name), "omp_nnz");

      return ret;
}

inline int bench_csr_cuda_thread_row(const sparse_csr *A, const double *x,
                                     bench_cuda *out) {
      set_csr_warps_per_block(out->warps_per_block);
      return compute_benchmark_csr(A, x, &out->bench, NULL,
                                   csr_spmv_cuda_thread_row);
}

inline int bench_csr_cuda_warp_row(const sparse_csr *A, const double *x,
                                   bench_cuda *out) {
      set_csr_warps_per_block(out->warps_per_block);
      return compute_benchmark_csr(A, x, &out->bench, NULL,
                                   csr_spmv_cuda_warp_row);
}

inline int bench_csr_cuda_halfwarp_row(const sparse_csr *A, const double *x,
                                       bench_cuda *out) {
      set_csr_warps_per_block(out->warps_per_block);
      return compute_benchmark_csr(A, x, &out->bench, NULL,
                                   csr_spmv_cuda_halfwarp_row);
}

inline int bench_csr_cuda_block_row(const sparse_csr *A, const double *x,
                                    bench_cuda *out) {
      set_csr_warps_per_block(out->warps_per_block);
      return compute_benchmark_csr(A, x, &out->bench, NULL,
                                   csr_spmv_cuda_block_row);
}

inline int bench_csr_cuda_halfwarp_row_text(const sparse_csr *A,
                                            const double *x, bench_cuda *out) {
      set_csr_warps_per_block(out->warps_per_block);
      return compute_benchmark_csr(A, x, &out->bench, NULL,
                                   csr_spmv_cuda_halfwarp_row_text);
}
