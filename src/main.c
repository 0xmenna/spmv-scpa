#include <errno.h>
#include <getopt.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/csr.h"
#include "../include/err.h"
#include "../include/hll.h"
#include "../include/logger.h"
#include "../include/utils.h"

static bool debug = false;
static bool is_logger_open = false;

// Benchmarks
int run_benchmark(sparse_csr *A, sparse_hll *H, enum BENCH_TYPE bench);

int main(int argc, char **argv) {
      sparse_csr *A = NULL;
      sparse_hll *H = NULL;
      bench res;
      int ret;

      const char *matrix_path = NULL;
      const char *log_path = NULL;
      enum BENCH_TYPE bench = CSR_SERIAL;

      // Define long options
      static struct option long_opts[] = {
          {"matrix", required_argument, NULL, 'm'},
          {"out", required_argument, NULL, 'o'},
          {"bench", required_argument, NULL, 'b'},
          {"debug", no_argument, NULL, 'd'},
          {"help", no_argument, NULL, 'h'},
          {NULL, 0, NULL, 0}};

      int opt;
      int idx;
      while ((opt = getopt_long(argc, argv, "m:o:b:dh", long_opts, &idx)) !=
             -1) {
            switch (opt) {
            case 'm':
                  matrix_path = optarg;
                  break;
            case 'o':
                  log_path = optarg;
                  break;
            case 'b':
                  if (strcmp(optarg, "csr-serial") == 0) {
                        bench = CSR_SERIAL;
                  } else if (strcmp(optarg, "hll-serial") == 0) {
                        bench = HLL_SERIAL;
                  } else if (strcmp(optarg, "csr-omp-guided") == 0) {
                        bench = CSR_OMP_GUIDED;
                  } else if (strcmp(optarg, "csr-omp-nnz") == 0) {
                        bench = CSR_OMP_CUSTOM;
                  } else if (strcmp(optarg, "hll-omp") == 0) {
                        bench = HLL_OMP;
                  } else {
                        LOG_ERR("Invalid benchmark type: %s", optarg);
                        log_prog_usage(basename(argv[0]));
                        return EXIT_FAILURE;
                  }
                  break;
            case 'd':
                  debug = true;
                  break;
            case 'h':
                  log_prog_usage(basename(argv[0]));
                  return EXIT_SUCCESS;
            default:
                  log_prog_usage(basename(argv[0]));
                  return EXIT_FAILURE;
            }
      }

      if (!matrix_path || !log_path) {
            log_prog_usage(basename(argv[0]));
            return EXIT_FAILURE;
      }

      // Initialize logger
      if (logger_init(log_path)) {
            LOG_ERR("Failed to open log file: %s", log_path);
            return EXIT_FAILURE;
      }
      is_logger_open = true;

      A = io_load_csr(matrix_path);
      if (!A) {
            LOG_ERR("Failed to load matrix: %s (err %d)", matrix_path,
                    PTR_ERR(A));
            goto cleanup;
      }

      H = csr_to_hll(A, false);
      if (IS_ERR(H)) {
            LOG_ERR("Failed to convert CSR to HLL (err %d)", PTR_ERR(H));
            goto cleanup;
      }

      ret = run_benchmark(A, H, bench);
      if (ret) {
            LOG_ERR("Failed to run benchmark (err %d)", ret);
            goto cleanup;
      }

      ret = EXIT_SUCCESS;

cleanup:
      if (A)
            csr_free(A);
      if (H)
            hll_free(H);

      if (is_logger_open) {
            logger_close();
      }
      return ret;
}

static int run_csr_serial_benchmark(sparse_csr *A, vec *expected) {
      bench res;
      int ret;

      ret = bench_csr_serial(A, &res);
      if (ret) {
            LOG_ERR("[CSR serial] failed with error %d", ret);
            return ret;
      }

      log_csr_serial_benchmark(A, res);

      if (debug) {
            // Capture expected output for later validation
            *expected = res.data;
      } else {
            vec_put(&res.data);
      }

      return 0;
}

static int run_hll_serial_benchmark(sparse_hll *H, const vec expected) {
      bench res;
      int ret = bench_hll_serial(H, &res);
      if (ret) {
            LOG_ERR("[HLL serial] failed with error %d", ret);
            return ret;
      }

      log_hll_serial_benchmark(H, res);

      if (debug) {
            ret = validation_vec_result(expected, res.data);
            if (ret) {
                  vec_put(&res.data);
                  LOG_ERR("[HLL serial] validation failed");
                  return ret;
            }
      }

      vec_put(&res.data);

      return 0;
}

typedef int (*csr_bench_fn)(const sparse_csr *, bench_omp *);

static int run_csr_omp_benchmarks(sparse_csr *A, vec expected,
                                  csr_bench_fn bench_fn) {
      bench_omp benchmarks[] = {
          {.num_threads = 2},  {.num_threads = 4},  {.num_threads = 8},
          {.num_threads = 16}, {.num_threads = 32}, {.num_threads = 40},
      };

      for (int i = 0; i < ARRAY_SIZE(benchmarks); i++) {
            int ret = bench_fn(A, &benchmarks[i]);
            if (ret) {
                  LOG_ERR("[CSR OMP] failed with error %d", ret);
                  return ret;
            }

            log_csr_omp_benchmark(A, benchmarks[i]);

            if (debug) {
                  ret =
                      validation_vec_result(expected, benchmarks[i].bench.data);
                  if (ret) {
                        vec_put(&benchmarks[i].bench.data);
                        LOG_ERR("[CSR OMP] validation failed");
                        return ret;
                  }
            }

            vec_put(&benchmarks[i].bench.data);
      }

      return 0;
}

static inline int run_csr_omp_nnz_balancing_benchmarks(sparse_csr *A,
                                                       vec expected) {
      return run_csr_omp_benchmarks(A, expected, bench_csr_omp_nnz_balancing);
}

static inline int run_csr_omp_guided_benchmarks(sparse_csr *A, vec expected) {
      return run_csr_omp_benchmarks(A, expected, bench_csr_omp_guided);
}

static int run_hll_omp_benchmarks(sparse_hll *H, vec expected) {
      bench_omp benchmarks[] = {
          {.num_threads = 2},  {.num_threads = 4},  {.num_threads = 8},
          {.num_threads = 16}, {.num_threads = 32}, {.num_threads = 40},
      };

      for (int i = 0; i < ARRAY_SIZE(benchmarks); i++) {

            int ret = bench_hll_omp(H, &benchmarks[i]);
            if (ret) {
                  LOG_ERR("[HLL OMP] failed with error %d", ret);
                  return ret;
            }

            log_hll_omp_benchmark(H, benchmarks[i]);

            if (debug) {
                  // Validate against serial result
                  ret =
                      validation_vec_result(expected, benchmarks[i].bench.data);
                  if (ret) {
                        vec_put(&benchmarks[i].bench.data);
                        LOG_ERR("[HLL OMP] validation failed");
                        return ret;
                  }
            }

            vec_put(&benchmarks[i].bench.data);
      }
      return 0;
}

int run_benchmark(sparse_csr *A, sparse_hll *H, enum BENCH_TYPE bench) {
      vec expected = {0};
      int ret;

      if (debug) {
            // Always run the serial benchmark for validation
            ret = run_csr_serial_benchmark(A, &expected);
            if (ret)
                  goto cleanup;
      }

      switch (bench) {
      case CSR_SERIAL:
            if (debug) {
                  // Already run in debug mode
                  break;
            }
            ret = run_csr_serial_benchmark(A, &expected);
            break;
      case HLL_SERIAL:
            ret = run_hll_serial_benchmark(H, expected);
            break;
      case CSR_OMP_CUSTOM:
            ret = run_csr_omp_nnz_balancing_benchmarks(A, expected);
            break;
      case CSR_OMP_GUIDED:
            ret = run_csr_omp_guided_benchmarks(A, expected);
            break;
      case HLL_OMP:
            // sparse_hll *H_col_major = csr_to_hll(A, true);
            ret = run_hll_omp_benchmarks(H, expected);
            // free();
            break;
      default:
            LOG_ERR("Invalid benchmark type");
            return -EINVAL;
      }

cleanup:
      if (expected.data) {
            vec_put(&expected);
      }
      return ret;
}
