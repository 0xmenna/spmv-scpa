#ifndef VECTOR_H
#define VECTOR_H

#include <stddef.h>

/** Simple dynamically‐sized vector of doubles */
typedef struct {
      size_t len; 
      double *data;
} VecD;


VecD *vec_create(size_t n);

void vec_free(VecD *v);

void vec_fill(VecD *v, double value);

#endif /* VECTOR_H */
