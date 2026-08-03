#ifndef NEURAL_INTERNAL_ATOMIC_FILE_H
#define NEURAL_INTERNAL_ATOMIC_FILE_H

#include <stdio.h>

#include "neural/error.h"

typedef int (*NeuralAtomicFileWriter)(FILE *stream,
                                      void *context,
                                      NeuralError *error);

int neural_atomic_file_write(const char *path,
                             NeuralAtomicFileWriter writer,
                             void *context,
                             NeuralError *error);

#endif
