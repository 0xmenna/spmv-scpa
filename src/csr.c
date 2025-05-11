// csr.c

#include <errno.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/csr.h"
#include "../include/err.h"
#include "../include/mmio.h"
#include "../include/utils.h"
#include "../include/vector.h"

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

static int compute_benchmark_csr(const sparse_csr *A, bench *out,
                                 void (*spmv_f)(const sparse_csr *A,
                                                const double *x, double *y)) {
      vec x = vec_create(A->N);
      if (!x.data)
            return -ENOMEM;

      vec_fill(&x, 1.0);

      vec y = vec_create(A->M);
      if (!y.data) {
            vec_put(&x);
            return -ENOMEM;
      }

      double start = now();
      spmv_f(A, x.data, y.data);
      double end = now();

      vec_put(&x);

      out->duration_ms = end - start;
      out->gflops = compute_gflops(end - start, A->NZ);
      out->data = y;

      return 0;
}

static inline void csr_spmv_serial(const sparse_csr *A, const double *x,
                                   double *y) {
      for (int i = 0; i < A->M; i++) {
            double sum = 0.0;
            for (int j = A->IRP[i]; j < A->IRP[i + 1]; j++) {
                  sum += A->AS[j] * x[A->JA[j]];
            }

            y[i] = sum;
      }
}

static inline void csr_spmv_omp(const sparse_csr *A, const double *x,
                                double *y) {
      int M = A->M;
      int T = omp_get_max_threads();

      // 1) Build prefix-sum of NNZ per row
      int *row_counts = malloc((M + 1) * sizeof(int));
      row_counts[0] = 0;
      for (int i = 0; i < M; i++)
            row_counts[i + 1] = A->IRP[i + 1] - A->IRP[i];
      for (int i = 1; i <= M; i++)
            row_counts[i] += row_counts[i - 1];
      long total_nnz = row_counts[M];

      // 2) Determine per-thread row slices
      int *row_start = malloc((T + 1) * sizeof(int));
      row_start[0] = 0;
      row_start[T] = M;
      for (int t = 1; t < T; t++) {
            long target = (total_nnz * t) / T;
            // binary search in row_counts[0..M]
            int lo = 0, hi = M;
            while (lo < hi) {
                  int mid = (lo + hi) >> 1;
                  if (row_counts[mid] < target)
                        lo = mid + 1;
                  else
                        hi = mid;
            }
            row_start[t] = lo;
      }

// 3) Parallel region: each thread its slice
#pragma omp parallel
      {
            int tid = omp_get_thread_num();
            int r0 = row_start[tid];
            int r1 = row_start[tid + 1];

            const int *IRP = A->IRP;
            const int *JA = A->JA;
            const double *AS = A->AS;
            const double *x_ = x;
            double *y_ = y;

            for (int i = r0; i < r1; i++) {
                  double sum = 0.0;
                  for (int jj = IRP[i]; jj < IRP[i + 1]; jj++)
                        sum += AS[jj] * x_[JA[jj]];
                  y_[i] = sum;
            }
      }

      free(row_counts);
      free(row_start);
}

// Run CSR SpMV serial benchmark
inline int bench_csr_serial(const sparse_csr *A, bench *out) {
      return compute_benchmark_csr(A, out, csr_spmv_serial);
}

// Run CSR SpMV OpenMP benchmark
inline int bench_csr_omp(const sparse_csr *A, bench *out) {
      return compute_benchmark_csr(A, out, csr_spmv_omp);
}
