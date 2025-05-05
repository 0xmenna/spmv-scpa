// utils.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void log_prog_usage(const char *prog) {
      fprintf(
          stderr,
          "Usage: %s <matrix-file.mtx> <log-file>\n"
          "  <matrix-file.mtx>   Path to the Matrix Market file to process\n"
          "  <log-file>          Path where benchmark logs will be written\n",
          prog);
}

double compute_gflops(double duration, int nnz) {
      return (2.0 * nnz) / duration / 1e9;
}
