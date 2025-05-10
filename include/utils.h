#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "vector.h"

#define MAX_PATH 256

typedef struct benchmark_result {
      double duration_ms;
      double gflops;
      vec data;
} bench;

#define LOG_WARN(fmt, ...)                                                     \
      do {                                                                     \
            fprintf(stdout, "[WARN ] %s:%d: " fmt "\n", __FILE__, __LINE__,    \
                    ##__VA_ARGS__);                                            \
      } while (0)

#define LOG_INFO(fmt, ...)                                                     \
      do {                                                                     \
            fprintf(stdout, "[INFO ] %s:%d: " fmt "\n", __FILE__, __LINE__,    \
                    ##__VA_ARGS__);                                            \
      } while (0)

void log_prog_usage(const char *prog);

void print_result_vector(const vec res, const char *fmt);

// Get the current time in milliseconds
inline double now(void) { return (double)clock() * 1e3 / CLOCKS_PER_SEC; }

inline double compute_gflops(double duration, int nnz) {
      return (2.0 * nnz) / (duration * 1e6);
}

void *aligned_malloc(size_t size);

#endif