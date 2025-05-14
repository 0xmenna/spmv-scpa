#ifndef LOGGER_H
#define LOGGER_H

#include "../include/csr.h"
#include "../include/hll.h"
#include "../include/utils.h"

/**
 * Opens (or creates) the CSV log at `log_path` and writes the header row.
 * Returns 0 on success, non-zero on failure.
 */
int logger_init(const char *log_path);

/** Flushes and closes the log file. */
void logger_close(void);

void log_csr_serial_benchmark(const sparse_csr *A, bench res);

void log_hll_serial_benchmark(const sparse_hll *H, bench res);

void log_csr_omp_benchmark(const sparse_csr *A, bench_omp res);

void log_hll_omp_benchmark(const sparse_hll *H, bench_omp res);

#endif // LOGGER_H
