//main.c

#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../include/csr.h"
#include "../include/err.h"
#include "../include/hll.h"
#include "../include/logger.h"
#include "../include/utils.h"

int main(int argc, char **argv) {
      if (argc != 3) {
            log_prog_usage(argv[0]);
            return EXIT_FAILURE;
      }

      const char *matrix_path = argv[1];
      const char *log_path = argv[2];

      if (logger_init(log_path)) {
            LOG_ERR("Failed to open log file: %s", log_path);
            return EXIT_FAILURE;
      }

      // Load the CSR matrix
      SparseCSR *A = io_load_csr(matrix_path);
      if (!A) {
            LOG_ERR("Failed to load matrix: %s (err %d)", matrix_path,
                    PTR_ERR(A));
            logger_close();
            return EXIT_FAILURE;
      }

      Bench res;

      // CSR serial benchmark
      res = bench_csr_serial(A);
      log_csr_serial_benchmark(A, res);

      // HLL serial benchmark
      BlockELLPACK *H = csr_to_hll(A);
      res = bench_hll_serial(H);
      log_hll_serial_benchmark(H, res);

      // Cleanup
      csr_free(A);
      hll_free(H);
      logger_close();

      return EXIT_SUCCESS;
}
