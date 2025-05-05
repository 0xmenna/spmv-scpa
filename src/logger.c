//logger.c

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
      fprintf(log, "matrix,format,benchmark,rows,cols,nnz,"
                   "duration,gflops\n");
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

void log_csr_serial_benchmark(const SparseCSR *A, Bench res) {
      if (!log) {
            LOG_ERR("Log file not initialized");
            return;
      }
      // For CSR we leave num_blocks and block_size empty
      fprintf(log,
              "%s,CSR,serial,%d,%d,%d,"
              "%f,%f\n",
              A->name, A->rows, A->cols, A->nnz, res.duration, res.gflops);
      fflush(log);
}

void log_hll_serial_benchmark(const BlockELLPACK *H, Bench res) {
      if (!log) {
            LOG_ERR("Log file not initialized");
            return;
      }
      // HLL has num_blocks and block_size
      fprintf(log, "%s,HLL,serial,%d,%d,%d,%f,%f\n", H->name, H->rows, H->cols,
              H->nnz, res.duration, res.gflops);
      fflush(log);
}
