#ifndef NEURAL_INTERNAL_SHA256_H
#define NEURAL_INTERNAL_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t block_size;
} NeuralSha256;

void neural_sha256_init(NeuralSha256 *context);
void neural_sha256_update(NeuralSha256 *context,
                          const void *data,
                          size_t size);
void neural_sha256_final(NeuralSha256 *context, unsigned char digest[32]);

#endif
