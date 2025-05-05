// csr.c

#include <errno.h>
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

SparseCSR *io_load_csr(const char *path) {
      char m_name[MAX_NAME];
      FILE *f = NULL;

      extract_matrix_name(path, m_name);
      f = fopen(path, "r");
      if (!f) {
            return ERR_PTR(-errno);
      }

      MM_typecode tt;
      if (mm_read_banner(f, &tt) || !mm_is_matrix(tt) || !mm_is_sparse(tt) ||
          !(mm_is_real(tt) || mm_is_pattern(tt))) {
            fclose(f);
            return ERR_PTR(-EINVAL);
      }

      int M, N, nz0;
      if (mm_read_mtx_crd_size(f, &M, &N, &nz0)) {
            fclose(f);
            return ERR_PTR(-EINVAL);
      }
      const int is_sym = mm_is_symmetric(tt);
      const int is_pat = mm_is_pattern(tt);

      // First pass: count entries per row
      long header_pos = ftell(f);
      int *row_counts = calloc((size_t)M, sizeof(int));
      if (!row_counts) {
            fclose(f);
            return ERR_PTR(-ENOMEM);
      }

      int i, j;
      double v;
      long total_nnz = 0;
      for (int e = 0; e < nz0; ++e) {
            if (is_pat) {
                  if (fscanf(f, "%d %d", &i, &j) != 2) {
                        errno = EIO;
                        goto failure;
                  }
                  v = 1.0;
            } else {
                  if (fscanf(f, "%d %d %lf", &i, &j, &v) != 3) {
                        errno = EIO;
                        goto failure;
                  }
            }
            --i;
            --j;
            if (i < 0 || i >= M || j < 0 || j >= N) {
                  errno = ERANGE;
                  goto failure;
            }
            row_counts[i]++;
            total_nnz++;
            if (is_sym && i != j) {
                  row_counts[j]++;
                  total_nnz++;
            }
      }

      // Allocate aligned row offsets
      int *row_off;
      if (posix_memalign((void **)&row_off, 64,
                         (size_t)(M + 1) * sizeof(int))) {
            errno = ENOMEM;
            goto failure;
      }
      row_off[0] = 0;
      for (int r = 0; r < M; ++r) {
            row_off[r + 1] = row_off[r] + row_counts[r];
      }

      // Allocate aligned interleaved entries
      Entry *entries;
      if (posix_memalign((void **)&entries, 64,
                         (size_t)total_nnz * sizeof(Entry))) {
            free(row_off);
            errno = ENOMEM;
            goto failure;
      }

      // Second pass: fill CSR data
      memset(row_counts, 0, (size_t)M * sizeof(int));
      if (fseek(f, header_pos, SEEK_SET) != 0) {
            errno = EIO;
            free(row_off);
            free(entries);
            goto failure;
      }

      for (int e = 0; e < nz0; ++e) {
            if (is_pat) {
                  fscanf(f, "%d %d", &i, &j);
                  v = 1.0;
            } else {
                  fscanf(f, "%d %d %lf", &i, &j, &v);
            }
            --i;
            --j;
            long idx = (long)row_off[i] + row_counts[i]++;
            entries[idx].col = j;
            entries[idx].val = v;
            if (is_sym && i != j) {
                  long idx2 = (long)row_off[j] + row_counts[j]++;
                  entries[idx2].col = i;
                  entries[idx2].val = v;
            }
      }
      fclose(f);
      free(row_counts);

      SparseCSR *A = malloc(sizeof *A);
      if (!A) {
            free(row_off);
            free(entries);
            return ERR_PTR(ENOMEM);
      }

      init_csr(A, m_name, M, N, (int)total_nnz, row_off, entries);
      return A;

failure:
      free(row_counts);
      fclose(f);
      return ERR_PTR(-errno);
}

void csr_free(SparseCSR *A) {
      if (!A)
            return;
      free(A->row_off);
      free(A->entries);
      free(A);
}

static inline void csr_spmv_serial(const SparseCSR *A, const double *x,
                                   double *y) {
      for (int i = 0; i < A->rows; i++) {
            double sum = 0.0;
            for (int k = A->row_off[i]; k < A->row_off[i + 1]; k++) {
                  Entry entry = A->entries[k];
                  sum += entry.val * x[entry.col];
            }

            y[i] = sum;
      }
}

// Run CSR SpMV serial benchmark
Bench bench_csr_serial(const SparseCSR *A) {
      VecD *x = vec_create(A->cols);
      VecD *y = vec_create(A->rows);
      vec_fill(x, 1.0);

      double start = now();
      csr_spmv_serial(A, x->data, y->data);
      double end = now();

      Bench result = {.duration = end - start,
                      .gflops = compute_gflops(end - start, A->nnz)};

      vec_free(x);
      vec_free(y);

      return result;
}
