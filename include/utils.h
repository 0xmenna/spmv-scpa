#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MAX_PATH 256

typedef struct benchmark_result {
      double duration; // in seconds
      double gflops;
} Bench;

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

static inline double now() { return (double)clock() / CLOCKS_PER_SEC; }

double compute_gflops(double duration, int nnz);

#endif