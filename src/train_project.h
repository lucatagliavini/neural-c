#ifndef NEURAL_INTERNAL_TRAIN_PROJECT_H
#define NEURAL_INTERNAL_TRAIN_PROJECT_H

#include "neural/error.h"
#include "neural/parallel.h"
#include "neural/training.h"

int neural_project_train_fresh(const char *directory,
                               const NeuralExecutionConfig *execution,
                               NeuralTrainingResult *result,
                               NeuralError *error);

#endif
