// logger.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/err.h"
#include "../include/logger.h"

static FILE *log = NULL;

int logger_init(const char *log_path) {
      if (log) {
            // already opened
            return 0;
      }
      log = fopen(log_path, "w");
      if (!log) {
            return -1;
      }
      // CSV header
      fprintf(log, "matrix,format,benchmark,rows,cols,nnz,num_blocks,"
                   "duration_ms,gflops\n");
      fflush(log);
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
      fprintf(log, "%s,CSR,serial,%d,%d,%d,,%f,%f\n", A->name, A->M, A->N,
              A->NZ, res.duration_ms, res.gflops);
      fflush(log);
}

void log_hll_serial_benchmark(const sparse_hll *H, bench res) {
      if (!log) {
            LOG_ERR("Log file not initialized");
            return;
      }

      fprintf(log, "%s,HLL,serial,%d,%d,%d,%d,%f,%f\n", H->name, H->M, H->N,
              H->NZ, H->num_blocks, res.duration_ms, res.gflops);
      fflush(log);
}
