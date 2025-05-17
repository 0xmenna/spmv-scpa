#include "cuda_timer.cuh"

int timer_init(cuda_timer *t) {
      if (cudaEventCreate(&t->start) != cudaSuccess)
            return -1;
      if (cudaEventCreate(&t->stop) != cudaSuccess)
            return -1;
      return 0;
}

void timer_start(cuda_timer *t, cudaStream_t stream) {
      cudaEventRecord(t->start, stream);
}

double timer_stop(cuda_timer *t, cudaStream_t stream) {
      cudaEventRecord(t->stop, stream);
      cudaEventSynchronize(t->stop);
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, t->start, t->stop);
      return (double)ms;
}

void timer_destroy(cuda_timer *t) {
      cudaEventDestroy(t->start);
      cudaEventDestroy(t->stop);
}
