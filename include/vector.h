#ifndef VECTOR_H
#define VECTOR_H

#include <stddef.h>

/** Simple dynamically‐sized vector of doubles */
typedef struct {
      size_t len;
      double *data;
} vec;

vec vec_create(size_t n);

void vec_put(vec *v);

void vec_fill(vec *v, double value);

void vec_fill_random(vec *v);

#endif /* VECTOR_H */
