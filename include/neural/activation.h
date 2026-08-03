#ifndef NEURAL_ACTIVATION_H
#define NEURAL_ACTIVATION_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/types.h"

typedef enum {
    NEURAL_ACTIVATION_LINEAR,
    NEURAL_ACTIVATION_SIGMOID,
    NEURAL_ACTIVATION_TANH,
    NEURAL_ACTIVATION_RELU,
    NEURAL_ACTIVATION_LEAKY_RELU,
    NEURAL_ACTIVATION_ELU,
    NEURAL_ACTIVATION_SOFTMAX
} NeuralActivationKind;

typedef enum {
    NEURAL_ACTIVATION_PARAMETER_ALPHA,
    NEURAL_ACTIVATION_PARAMETER_BETA,
    NEURAL_ACTIVATION_PARAMETER_THRESHOLD
} NeuralActivationParameterKind;

typedef enum {
    NEURAL_ACTIVATION_SCALAR,
    NEURAL_ACTIVATION_VECTOR
} NeuralActivationShape;

typedef enum {
    NEURAL_INITIALIZER_XAVIER_UNIFORM,
    NEURAL_INITIALIZER_HE_UNIFORM
} NeuralInitializerKind;

typedef struct {
    NeuralActivationParameterKind kind;
    neural_real value;
} NeuralActivationParameter;

typedef struct {
    NeuralActivationKind kind;
    size_t parameter_count;
    NeuralActivationParameter *parameters;
} NeuralActivationSpec;

const char *neural_activation_kind_name(NeuralActivationKind kind);
int neural_activation_kind_from_name(const char *name,
                                     NeuralActivationKind *kind);
const char *neural_activation_parameter_name(
    NeuralActivationParameterKind kind);
int neural_activation_parameter_from_name(
    const char *name,
    NeuralActivationParameterKind *kind);
NeuralActivationShape neural_activation_shape(NeuralActivationKind kind);
NeuralInitializerKind neural_activation_initializer(NeuralActivationKind kind);

int neural_activation_spec_set_parameter(
    NeuralActivationSpec *spec,
    NeuralActivationParameterKind kind,
    neural_real value,
    NeuralError *error);
int neural_activation_spec_validate(const NeuralActivationSpec *spec,
                                    NeuralError *error);
int neural_activation_spec_copy(const NeuralActivationSpec *source,
                                NeuralActivationSpec *destination,
                                NeuralError *error);
void neural_activation_spec_free(NeuralActivationSpec *spec);
int neural_activation_spec_parameter_value(
    const NeuralActivationSpec *spec,
    NeuralActivationParameterKind kind,
    neural_real *value);

int neural_activation_apply(const NeuralActivationSpec *spec,
                            const neural_real *inputs,
                            neural_real *outputs,
                            size_t count,
                            NeuralError *error);

#endif
