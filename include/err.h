#ifndef ERR_H
#define ERR_H

#include <stdint.h>
#include <stdio.h>

/*
 * Encode/decode small negative errno-style codes into pointers.
 */
#define ERR_PTR(err) ((void *)(intptr_t)(err))
#define PTR_ERR(ptr) ((int)(intptr_t)(ptr))
#define IS_ERR(ptr) ((uintptr_t)(ptr) > (uintptr_t)(-4096))

#define LOG_ERR(fmt, ...)                                                      \
      do {                                                                     \
            fprintf(stderr, "[ERROR] %s:%d: " fmt "\n", __FILE__, __LINE__,    \
                    ##__VA_ARGS__);                                            \
      } while (0)

#endif /* ERR_H */