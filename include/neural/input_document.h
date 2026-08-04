#ifndef NEURAL_INPUT_DOCUMENT_H
#define NEURAL_INPUT_DOCUMENT_H

#include <stddef.h>
#include <stdio.h>

#include "neural/error.h"
#include "neural/types.h"

typedef struct NeuralInputDocument NeuralInputDocument;

int neural_input_document_open(const char *path,
                               NeuralInputDocument **document,
                               NeuralError *error);
size_t neural_input_document_sample_count(const NeuralInputDocument *document);
size_t neural_input_document_input_count(const NeuralInputDocument *document);
int neural_input_document_read(NeuralInputDocument *document,
                               neural_real *inputs,
                               size_t batch_capacity,
                               size_t *sample_count,
                               int *complete,
                               NeuralError *error);
void neural_input_document_close(NeuralInputDocument *document);

#endif
