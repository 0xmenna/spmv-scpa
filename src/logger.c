#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "err.h"
#include "logger.h"

static FILE *log_serial = NULL;
static FILE *log_omp = NULL;
static FILE *log_cuda = NULL;

enum LOGGER_TYPE {
      SERIAL,
      OMP,
      CUDA,
};

static FILE *open_log(const char *path, const enum LOGGER_TYPE type) {
      struct stat st;
      int file_exists = (stat(path, &st) == 0);

      FILE *f = fopen(path, "a");
      if (!f)
            return NULL;

      if (!file_exists) {

            switch (type) {
            case SERIAL:
                  fprintf(f, "matrix,format,rows,cols,nnz,num_blocks,"
                             "duration_ms,gflops\n");
                  break;
            case OMP:
                  fprintf(f, "matrix,format,bench,rows,cols,nnz,num_blocks,"
                             "num_threads,duration_ms,gflops\n");
                  break;
            case CUDA:
                  fprintf(f, "matrix,format,kernel,warps_per_block,rows,cols,"
                             "nnz,num_blocks,"
                             "duration_ms,gflops\n");
                  break;
            default:
                  LOG_ERR("Invalid logger type");
                  fclose(f);
                  return NULL;
                  break;
            }

            fflush(f);
      }

      return f;
}

int logger_init(const char *base_path) {
      char path_serial[256], path_omp[256], path_cuda[256];

      snprintf(path_serial, sizeof(path_serial), "%s/serial.csv", base_path);
      snprintf(path_omp, sizeof(path_omp), "%s/omp.csv", base_path);
      snprintf(path_cuda, sizeof(path_cuda), "%s/cuda.csv", base_path);

      log_serial = open_log(path_serial, SERIAL);
      log_omp = open_log(path_omp, OMP);
      log_cuda = open_log(path_cuda, CUDA);

      if (!log_serial || !log_omp || !log_cuda) {
            return -1;
      }

      return 0;
}

void logger_close(void) {
      if (log_serial) {
            fclose(log_serial);
            log_serial = NULL;
      }
      if (log_omp) {
            fclose(log_omp);
            log_omp = NULL;
      }
      if (log_cuda) {
            fclose(log_cuda);
            log_cuda = NULL;
      }
}

void log_csr_serial_benchmark(const sparse_csr *A, bench res) {
      if (!log_serial) {
            LOG_ERR("Serial log not initialized");
            return;
      }
      fprintf(log_serial, "%s,CSR,%d,%d,%d,,%f,%f\n", A->name, A->M, A->N,
              A->NZ, res.duration_ms, res.gflops);
      fflush(log_serial);
}

void log_hll_serial_benchmark(const sparse_hll *H, bench res) {
      if (!log_serial) {
            LOG_ERR("Serial log not initialized");
            return;
      }
      fprintf(log_serial, "%s,HLL,%d,%d,%d,%d,%f,%f\n", H->name, H->M, H->N,
              H->NZ, H->num_blocks, res.duration_ms, res.gflops);
      fflush(log_serial);
}

void log_csr_omp_benchmark(const sparse_csr *A, bench_omp res) {
      if (!log_omp) {
            LOG_ERR("OMP log not initialized");
            return;
      }
      fprintf(log_omp, "%s,CSR,%s,%d,%d,%d,,%d,%f,%f\n", A->name, res.name,
              A->M, A->N, A->NZ, res.num_threads, res.bench.duration_ms,
              res.bench.gflops);
      fflush(log_omp);
}

void log_hll_omp_benchmark(const sparse_hll *H, bench_omp res) {
      if (!log_omp) {
            LOG_ERR("OMP log not initialized");
            return;
      }
      fprintf(log_omp, "%s,HLL,%s,%d,%d,%d,%d,%d,%f,%f\n", H->name, res.name,
              H->M, H->N, H->NZ, H->num_blocks, res.num_threads,
              res.bench.duration_ms, res.bench.gflops);
      fflush(log_omp);
}

void log_csr_cuda_benchmark(const sparse_csr *A, bench_cuda res,
                            int kernel_id) {
      if (!log_cuda) {
            LOG_ERR("CUDA log not initialized");
            return;
      }
      fprintf(log_cuda, "%s,CSR,%d,%d,%d,%d,%d,,%f,%f\n", A->name, kernel_id,
              res.warps_per_block, A->M, A->N, A->NZ, res.bench.duration_ms,
              res.bench.gflops);
      fflush(log_cuda);
}

void log_hll_cuda_benchmark(const sparse_hll *H, bench_cuda res,
                            int kernel_id) {
      if (!log_cuda) {
            LOG_ERR("CUDA log not initialized");
            return;
      }
      fprintf(log_cuda, "%s,HLL,%d,%d,%d,%d,%d,%d,%f,%f\n", H->name, kernel_id,
              res.warps_per_block, H->M, H->N, H->NZ, H->num_blocks,
              res.bench.duration_ms, res.bench.gflops);
      fflush(log_cuda);
}
