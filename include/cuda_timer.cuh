#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda_runtime.h>

typedef struct {
      cudaEvent_t start;
      cudaEvent_t stop;
} cuda_timer;

// Initializes the timer (creates events)
int timer_init(cuda_timer *t);

// Starts timing
void timer_start(cuda_timer *t, cudaStream_t stream);

// Stops timing and returns elapsed time in milliseconds
double timer_stop(cuda_timer *t, cudaStream_t stream);

// Frees resources
void timer_destroy(cuda_timer *t);

#endif
