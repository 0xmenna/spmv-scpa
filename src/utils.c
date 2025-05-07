// utils.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/utils.h"

#define ALIGNMENT 64

void log_prog_usage(const char *prog) {
      fprintf(
          stderr,
          "Usage: %s <matrix-file.mtx> <log-file>\n"
          "  <matrix-file.mtx>   Path to the Matrix Market file to process\n"
          "  <log-file>          Path where benchmark logs will be written\n",
          prog);
}

void *aligned_malloc(size_t size) {
      void *ptr = NULL;
      int err = posix_memalign(&ptr, ALIGNMENT, size);
      if (err) {
            return NULL;
      }
      return ptr;
}
