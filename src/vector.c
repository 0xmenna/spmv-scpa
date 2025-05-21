// vector.c

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "err.h"
#include "utils.h"
#include "vector.h"

vec vec_create(size_t n) {
      vec v;

      // Be sure the caller checks for memory allocation errors
      v.data = aligned_malloc(n * sizeof *v.data);
      memset(v.data, 0, n * sizeof(double));
      v.len = n;

      return v;
}

void vec_put(vec *v) {
      if (!v)
            return;
      free(v->data);
      v->data = NULL;
}

void vec_fill(vec *v, double value) {
      if (!v || !v->data)
            return;
      for (size_t i = 0; i < v->len; i++)
            v->data[i] = value;
}

void vec_fill_random(vec *v) {
      if (!v || !v->data)
            return;
      for (size_t i = 0; i < v->len; i++)
            v->data[i] = (double)rand() / RAND_MAX;
}
