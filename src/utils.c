// utils.c
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/utils.h"

#define ALIGNMENT 64

void log_prog_usage(const char *prog) {
      fprintf(
          stderr,
          "Usage: %s -m <matrix-file.mtx> -o <out-file> [-d] [-h]"
          "  -m, --matrix <file>   Path to the Matrix Market file to process"
          "  -o, --out <file>      Path where benchmark logs will be written"
          "  -d, --debug           Print the result vector y after each run"
          "  -h, --help            Show this help message and exit",
          prog);
}

void print_result_vector(const vec res, const char *fmt) {
      printf("Result vector y (length %d) - Format %s.\n", res.len, fmt);
      for (int i = 0; i < res.len; i++) {
            printf("  y[%d] = %.4f\n", i, res.data[i]);
      }
      printf("\n");
}

void *aligned_malloc(size_t size) {
      void *ptr = NULL;
      int err = posix_memalign(&ptr, ALIGNMENT, size);
      if (err) {
            return NULL;
      }
      return ptr;
}
