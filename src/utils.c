// utils.c
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"

void log_prog_usage(const char *prog) {
      fprintf(
          stderr,
          "Usage: %s -m <matrix-file.mtx> -o <out-dir> [-d] [-h]"
          "  -m, --matrix <file>   Path to the Matrix Market file to process"
          "  -o, --out <file>      Path where benchmark csv files will be saved"
          "  -d, --debug           Validate results against serial "
          "implementation"
          "  -h, --help            Show this help message and exit",
          prog);
}

void print_result_vector(const vec res) {
      printf("Result vector y (length %d)\n", res.len);
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

int validation_vec_result(const vec expected, const vec res) {
      if (res.len != expected.len) {
            return -1;
      }

      double l2_norm = 0.0;

      for (int i = 0; i < res.len; i++) {
            double diff = expected.data[i] - res.data[i];
            l2_norm += diff * diff;
      }

      l2_norm = sqrt(l2_norm);

      const double epsilon = 1e-1;

      if (l2_norm > epsilon) {
            return -1;
      }

      return 0;
}
