#ifndef NEURAL_PARSE_H
#define NEURAL_PARSE_H

#include <stddef.h>
#include <stdint.h>

#include "neural/types.h"

int neural_parse_size(const char *text, size_t *value);
int neural_parse_uint64(const char *text, uint64_t *value);
int neural_parse_real(const char *text, neural_real *value);

#endif
