#ifndef NEURAL_INTERNAL_PATH_H
#define NEURAL_INTERNAL_PATH_H

#include "neural/error.h"

char *neural_path_join(const char *directory,
                       const char *filename,
                       NeuralError *error);

#endif
