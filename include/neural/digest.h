#ifndef NEURAL_DIGEST_H
#define NEURAL_DIGEST_H

#include <stddef.h>

#include "neural/checkpoint.h"
#include "neural/error.h"
#include "neural/project.h"

int neural_sha256_hex(const void *data,
                      size_t size,
                      char output[NEURAL_SHA256_TEXT_CAPACITY],
                      NeuralError *error);

int neural_project_digests_compute(const NeuralProject *project,
                                   NeuralProjectDigests *digests,
                                   NeuralError *error);

#endif
