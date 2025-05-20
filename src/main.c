#include <errno.h>
#include <getopt.h>
#include <libgen.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csr.h"
#include "err.h"
#include "hll.h"
#include "logger.h"
#include "utils.h"

static bool debug = false;
static bool is_logger_open = false;

static sparse_csr *A = NULL;
static sparse_hll *H_row_major = NULL;
static sparse_hll *H_col_major = NULL;

static vec expected_res = {0};

static void run_benchmarks(void);
static void cleanup(void);

int main(int argc, char **argv) {
      bench res;

      const char *matrix_path = NULL;
      const char *log_path = NULL;

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
            cleanup();

            return EXIT_FAILURE;
      }

      H_row_major = csr_to_hll(A, false);
      H_col_major = csr_to_hll(A, true);

      if (IS_ERR(H_row_major) || IS_ERR(H_col_major)) {
            LOG_ERR("Failed to convert CSR to HLL");
            cleanup();

            return EXIT_FAILURE;
      }

      run_benchmarks();

      cleanup();

      return EXIT_SUCCESS;
}

static void cleanup(void) {
      if (expected_res.data)
            vec_put(&expected_res);
      if (H_col_major)
            hll_free(H_col_major);
      if (H_row_major)
            hll_free(H_row_major);
      if (A)
            csr_free(A);
      if (is_logger_open)
            logger_close();
}

static void run_csr_serial_benchmark(void) {
      bench res;
      int ret;

      ret = bench_csr_serial(A, &res);
      if (ret) {
            LOG_ERR("[CSR serial] failed with error %d", ret);
            cleanup();

            exit(EXIT_FAILURE);
      }

      log_csr_serial_benchmark(A, res);

      if (debug) {
            // Capture expected output for later validation
            expected_res = res.data;
      } else {
            vec_put(&res.data);
      }
}

static void run_hll_serial_benchmark(void) {
      bench res;
      int ret = bench_hll_serial(H_row_major, &res);
      if (ret) {
            LOG_ERR("[HLL serial] failed with error %d", ret);
            cleanup();

            exit(EXIT_FAILURE);
      }

      log_hll_serial_benchmark(H_row_major, res);

      if (debug) {
            ret = validation_vec_result(expected_res, res.data);
            if (ret) {
                  vec_put(&res.data);
                  LOG_ERR("[HLL serial] validation failed");
                  cleanup();

                  exit(EXIT_FAILURE);
            }
      }

      vec_put(&res.data);
}

typedef int (*csr_bench_fn)(const sparse_csr *, bench_omp *);

static void run_csr_omp_benchmarks(csr_bench_fn bench_fn) {
      bench_omp benchmarks[] = {
          {.num_threads = 2},  {.num_threads = 4},  {.num_threads = 8},
          {.num_threads = 16}, {.num_threads = 32}, {.num_threads = 40},
      };

      for (int i = 0; i < ARRAY_SIZE(benchmarks); i++) {
            int ret = bench_fn(A, &benchmarks[i]);
            if (ret) {
                  LOG_ERR("[CSR OMP] failed with error %d", ret);
                  cleanup();

                  exit(EXIT_FAILURE);
            }

            log_csr_omp_benchmark(A, benchmarks[i]);

            if (debug) {
                  ret = validation_vec_result(expected_res,
                                              benchmarks[i].bench.data);
                  if (ret) {
                        vec_put(&benchmarks[i].bench.data);
                        LOG_ERR("[CSR OMP] validation failed");
                        cleanup();

                        exit(EXIT_FAILURE);
                  }
            }

            vec_put(&benchmarks[i].bench.data);
      }
}

static inline void run_csr_omp_nnz_balancing_benchmarks(void) {
      return run_csr_omp_benchmarks(bench_csr_omp_nnz_balancing);
}

static inline void run_csr_omp_guided_benchmarks(void) {
      return run_csr_omp_benchmarks(bench_csr_omp_guided);
}

static void run_hll_omp_benchmarks(void) {
      bench_omp benchmarks[] = {
          {.num_threads = 2},  {.num_threads = 4},  {.num_threads = 8},
          {.num_threads = 16}, {.num_threads = 32}, {.num_threads = 40},
      };

      for (int i = 0; i < ARRAY_SIZE(benchmarks); i++) {

            int ret = bench_hll_omp(H_row_major, &benchmarks[i]);
            if (ret) {
                  LOG_ERR("[HLL OMP] failed with error %d", ret);
                  cleanup();

                  exit(EXIT_FAILURE);
            }

            log_hll_omp_benchmark(H_row_major, benchmarks[i]);

            if (debug) {
                  // Validate against serial result
                  ret = validation_vec_result(expected_res,
                                              benchmarks[i].bench.data);
                  if (ret) {
                        vec_put(&benchmarks[i].bench.data);
                        LOG_ERR("[HLL OMP] validation failed\n");
                        cleanup();

                        exit(EXIT_FAILURE);
                  }
            }

            vec_put(&benchmarks[i].bench.data);
      }
}

typedef int (*csr_cuda_bench_fn)(const sparse_csr *A, bench_cuda *out);

static inline void run_csr_cuda_benchmarks(void) {
      csr_cuda_bench_fn kernels[] = {
          bench_csr_cuda_thread_row,        bench_csr_cuda_warp_row,
          bench_csr_cuda_halfwarp_row,      bench_csr_cuda_block_row,
          bench_csr_cuda_halfwarp_row_text,
      };

      bench_cuda benchmarks[] = {
          {.warps_per_block = 2},
          {.warps_per_block = 4},
          {.warps_per_block = 8},
      };

      for (int kid = 0; kid < ARRAY_SIZE(kernels); ++kid) {
            for (int i = 0; i < ARRAY_SIZE(benchmarks); ++i) {

                  int ret = kernels[kid](A, &benchmarks[i]);
                  if (ret) {
                        LOG_ERR(
                            "Failed CSR CUDA [kernel %d, warps_per_block %d]",
                            kid, benchmarks[i].warps_per_block);
                        goto err;
                  }

                  if (debug) {
                        // Validate against serial result
                        ret = validation_vec_result(expected_res,
                                                    benchmarks[i].bench.data);
                        if (ret) {
                              vec_put(&benchmarks[i].bench.data);
                              LOG_ERR(
                                  "Failed validation of CSR CUDA [kernel %d]",
                                  kid);
                              goto err;
                        }
                  }

                  vec_put(&benchmarks[i].bench.data);
                  log_csr_cuda_benchmark(A, benchmarks[i], kid);
            }
      }
      return;

err:
      cleanup();
      exit(EXIT_FAILURE);
}

typedef int (*hll_cuda_bench_fn)(const sparse_hll *H, bench_cuda *out);

static inline void run_hll_cuda_benchmarks(void) {
      hll_cuda_bench_fn kernels[] = {
          bench_hll_cuda_threads_row_major,
          bench_hll_cuda_threads_col_major,
          bench_hll_cuda_warp_block,
          bench_hll_cuda_halfwarp_row,
      };

      bench_cuda benchmarks[] = {
          {.warps_per_block = 2},
          {.warps_per_block = 4},
          {.warps_per_block = 8},
      };

      for (int kid = 0; kid < ARRAY_SIZE(kernels); ++kid) {
            const sparse_hll *H =
                (kid == 0 || kid == 3) ? H_row_major : H_col_major;

            for (int i = 0; i < ARRAY_SIZE(benchmarks); ++i) {

                  int ret = kernels[kid](H, &benchmarks[i]);
                  if (ret) {
                        LOG_ERR(
                            "Failed HLL CUDA [kernel %d, warps_per_block %d]",
                            kid, benchmarks[i].warps_per_block);
                        goto err;
                  }

                  if (debug) {
                        // Validate against serial result
                        ret = validation_vec_result(expected_res,
                                                    benchmarks[i].bench.data);
                        if (ret) {
                              vec_put(&benchmarks[i].bench.data);
                              LOG_ERR("Failed validation of HLL CUDA [kernel "
                                      "%d, warps_per_block %d]",
                                      kid, benchmarks[i].warps_per_block);
                              goto err;
                        }
                  }

                  vec_put(&benchmarks[i].bench.data);
                  log_hll_cuda_benchmark(H, benchmarks[i], kid);
            }
      }
      return;

err:
      cleanup();
      exit(EXIT_FAILURE);
}

static inline void run_benchmarks(void) {

      run_csr_serial_benchmark();

      run_hll_serial_benchmark();

      run_csr_omp_nnz_balancing_benchmarks();

      run_csr_omp_guided_benchmarks();

      run_hll_omp_benchmarks();

      run_csr_cuda_benchmarks();

      run_hll_cuda_benchmarks();
}
