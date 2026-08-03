#ifndef NEURAL_INIT_H
#define NEURAL_INIT_H

#include "neural/error.h"
#include "neural/project.h"

int neural_project_initialize(const char *directory,
                              const NeuralModelSpec *model,
                              const NeuralTrainingConfig *training,
                              int force,
                              NeuralError *error);

#endif
