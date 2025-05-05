// vector.c

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "../include/err.h"
#include "../include/vector.h"

VecD *vec_create(size_t n) {
      VecD *v = malloc(sizeof *v);
      if (!v)
            goto failure;

      v->data = malloc(n * sizeof *v->data);
      if (!v->data) {
            goto failure;
      }
      v->len = n;

      return v;

failure:
      if (v) {
            free(v);
      }
      return ERR_PTR(-ENOMEM);
}

void vec_free(VecD *v) {
      if (!v)
            return;
      free(v->data);
      free(v);
}

void vec_fill(VecD *v, double value) {
      if (!v || !v->data)
            return;
      for (size_t i = 0; i < v->len; i++)
            v->data[i] = value;
}
