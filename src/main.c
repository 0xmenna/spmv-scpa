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

int main(int argc, char **argv) {
      sparse_csr *A = NULL;
      sparse_hll *H = NULL;
      bench res;
      bool is_logger_open = false;
      int ret;

      const char *matrix_path = NULL;
      const char *log_path = NULL;

      bool debug = false;

      // Define long options
      static struct option long_opts[] = {
          {"matrix", required_argument, NULL, 'm'},
          {"out", required_argument, NULL, 'o'},
          {"debug", no_argument, NULL, 'd'},
          {"help", no_argument, NULL, 'h'},
          {NULL, 0, NULL, 0}};

      int opt;
      int idx;
      while ((opt = getopt_long(argc, argv, "m:o:f:dh", long_opts, &idx)) !=
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

      // Load the CSR matrix
      A = io_load_csr(matrix_path);
      if (!A) {
            LOG_ERR("Failed to load matrix: %s (err %d)", matrix_path,
                    PTR_ERR(A));
            goto cleanup;
      }

      // CSR serial benchmark
      ret = bench_csr_serial(A, &res);
      if (ret) {
            LOG_ERR("Failed to run CSR benchmark (err %d)", ret);
            goto cleanup;
      }
      log_csr_serial_benchmark(A, res);
      if (debug) {
            print_result_vector(res.data, "CSR");
      }
      vec_put(&res.data);

      // HLL serial benchmark
      H = csr_to_hll(A, false);
      if (IS_ERR(H)) {
            LOG_ERR("Failed to convert CSR to HLL (err %d)", PTR_ERR(H));
            goto cleanup;
      }
      ret = bench_hll_serial(H, &res, false);
      if (ret) {
            LOG_ERR("Failed to run HLL benchmark (err %d)", ret);
            goto cleanup;
      }
      log_hll_serial_benchmark(H, res);
      if (debug) {
            print_result_vector(res.data, "HLL");
      }
      vec_put(&res.data);

      ret = EXIT_SUCCESS;

cleanup:
      if (A) {
            csr_free(A);
      }
      if (H) {
            hll_free(H);
      }
      if (is_logger_open) {
            logger_close();
      }
      return ret;
}
