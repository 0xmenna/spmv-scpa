#ifndef LOGGER_H
#define LOGGER_H

#include "../include/csr.h"
#include "../include/hll.h"
#include "../include/utils.h"

int logger_init(const char *base_path);
void logger_close(void);

void log_csr_serial_benchmark(const sparse_csr *A, bench res);
void log_hll_serial_benchmark(const sparse_hll *H, bench res);

void log_csr_omp_benchmark(const sparse_csr *A, bench_omp res);
void log_hll_omp_benchmark(const sparse_hll *H, bench_omp res);

void log_csr_cuda_benchmark(const sparse_csr *A, bench res, int kernel_id);
void log_hll_cuda_benchmark(const sparse_hll *H, bench res, int kernel_id);

#endif // LOGGER_H
