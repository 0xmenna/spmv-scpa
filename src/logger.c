#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "../include/err.h"
#include "../include/logger.h"

static FILE *log = NULL;

int logger_init(const char *log_path) {
      if (log) {
            return 0; // already initialized
      }

      // Check if the file already exists
      struct stat st;
      int file_exists = (stat(log_path, &st) == 0);

      // Open file in append mode
      log = fopen(log_path, "a");
      if (!log) {
            return -1;
      }

      // If file did not exist, write the header
      if (!file_exists) {
            fprintf(
                log,
                "matrix,format,benchmark,rows,cols,nnz,num_blocks,num_threads,"
                "duration_ms,gflops\n");
            fflush(log);
      }

      return 0;
}

void logger_close(void) {
      if (log) {
            fflush(log);
            fclose(log);
            log = NULL;
      }
}

void log_csr_serial_benchmark(const sparse_csr *A, bench res) {

      if (!log) {
            LOG_ERR("Log file not initialized");
            return;
      }

      fprintf(log, "%s,CSR,serial,%d,%d,%d,,%d,%f,%f\n", A->name, A->M, A->N,
              A->NZ, 1, res.duration_ms, res.gflops);
      fflush(log);
}

void log_hll_serial_benchmark(const sparse_hll *H, bench res) {
      if (!log) {
            LOG_ERR("Log file not initialized");
            return;
      }

      fprintf(log, "%s,HLL,serial,%d,%d,%d,%d,%d,%f,%f\n", H->name, H->M, H->N,
              H->NZ, H->num_blocks, 1, res.duration_ms, res.gflops);
      fflush(log);
}

void log_csr_omp_benchmark(const sparse_csr *A, bench_omp res) {
      if (!log) {
            LOG_ERR("Log file not initialized");
            return;
      }

      fprintf(log, "%s,CSR,%s,%d,%d,%d,,%d,%f,%f\n", A->name, res.name, A->M,
              A->N, A->NZ, res.num_threads, res.bench.duration_ms,
              res.bench.gflops);
      fflush(log);
}

void log_hll_omp_benchmark(const sparse_hll *H, bench_omp res) {
      if (!log) {
            LOG_ERR("Log file not initialized");
            return;
      }

      fprintf(log, "%s,HLL,omp,%d,%d,%d,%d,%d,%f,%f\n", H->name, H->M, H->N,
              H->NZ, H->num_blocks, res.num_threads, res.bench.duration_ms,
              res.bench.gflops);
      fflush(log);
}
