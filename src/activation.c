#include "neural/activation.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    NeuralActivationParameterKind kind;
    neural_real minimum;
    neural_real maximum;
    int minimum_inclusive;
    int maximum_inclusive;
} ParameterDefinition;

typedef struct {
    NeuralActivationKind kind;
    const char *name;
    NeuralActivationShape shape;
    NeuralInitializerKind initializer;
    const ParameterDefinition *parameters;
    size_t parameter_count;
} ActivationDefinition;

static const ParameterDefinition leaky_relu_parameters[] = {
    {NEURAL_ACTIVATION_PARAMETER_ALPHA, 0.0, 1.0, 0, 0}
};

static const ParameterDefinition elu_parameters[] = {
    {NEURAL_ACTIVATION_PARAMETER_ALPHA, 0.0, 0.0, 0, 0}
};

static const ActivationDefinition activation_definitions[] = {
    {NEURAL_ACTIVATION_LINEAR, "linear", NEURAL_ACTIVATION_SCALAR,
     NEURAL_INITIALIZER_XAVIER_UNIFORM, NULL, 0U},
    {NEURAL_ACTIVATION_SIGMOID, "sigmoid", NEURAL_ACTIVATION_SCALAR,
     NEURAL_INITIALIZER_XAVIER_UNIFORM, NULL, 0U},
    {NEURAL_ACTIVATION_TANH, "tanh", NEURAL_ACTIVATION_SCALAR,
     NEURAL_INITIALIZER_XAVIER_UNIFORM, NULL, 0U},
    {NEURAL_ACTIVATION_RELU, "relu", NEURAL_ACTIVATION_SCALAR,
     NEURAL_INITIALIZER_HE_UNIFORM, NULL, 0U},
    {NEURAL_ACTIVATION_LEAKY_RELU, "leaky_relu", NEURAL_ACTIVATION_SCALAR,
     NEURAL_INITIALIZER_HE_UNIFORM, leaky_relu_parameters, 1U},
    {NEURAL_ACTIVATION_ELU, "elu", NEURAL_ACTIVATION_SCALAR,
     NEURAL_INITIALIZER_HE_UNIFORM, elu_parameters, 1U},
    {NEURAL_ACTIVATION_SOFTMAX, "softmax", NEURAL_ACTIVATION_VECTOR,
     NEURAL_INITIALIZER_XAVIER_UNIFORM, NULL, 0U}
};

static const ActivationDefinition *find_definition(NeuralActivationKind kind)
{
    size_t index;

    for (index = 0U;
         index < sizeof(activation_definitions) /
                     sizeof(activation_definitions[0]);
         index++) {
        if (activation_definitions[index].kind == kind) {
            return &activation_definitions[index];
        }
    }
    return NULL;
}

static const ParameterDefinition *find_parameter_definition(
    const ActivationDefinition *definition,
    NeuralActivationParameterKind kind)
{
    size_t index;

    for (index = 0U; index < definition->parameter_count; index++) {
        if (definition->parameters[index].kind == kind) {
            return &definition->parameters[index];
        }
    }
    return NULL;
}

const char *neural_activation_kind_name(NeuralActivationKind kind)
{
    const ActivationDefinition *definition = find_definition(kind);
    return definition == NULL ? "unknown" : definition->name;
}

int neural_activation_kind_from_name(const char *name,
                                     NeuralActivationKind *kind)
{
    size_t index;

    if (name == NULL || kind == NULL) {
        return 0;
    }
    for (index = 0U;
         index < sizeof(activation_definitions) /
                     sizeof(activation_definitions[0]);
         index++) {
        if (strcmp(name, activation_definitions[index].name) == 0) {
            *kind = activation_definitions[index].kind;
            return 1;
        }
    }
    return 0;
}

const char *neural_activation_parameter_name(
    NeuralActivationParameterKind kind)
{
    switch (kind) {
    case NEURAL_ACTIVATION_PARAMETER_ALPHA:
        return "alpha";
    case NEURAL_ACTIVATION_PARAMETER_BETA:
        return "beta";
    case NEURAL_ACTIVATION_PARAMETER_THRESHOLD:
        return "threshold";
    }
    return "unknown";
}

int neural_activation_parameter_from_name(
    const char *name,
    NeuralActivationParameterKind *kind)
{
    if (name == NULL || kind == NULL) {
        return 0;
    }
    if (strcmp(name, "alpha") == 0) {
        *kind = NEURAL_ACTIVATION_PARAMETER_ALPHA;
    } else if (strcmp(name, "beta") == 0) {
        *kind = NEURAL_ACTIVATION_PARAMETER_BETA;
    } else if (strcmp(name, "threshold") == 0) {
        *kind = NEURAL_ACTIVATION_PARAMETER_THRESHOLD;
    } else {
        return 0;
    }
    return 1;
}

NeuralActivationShape neural_activation_shape(NeuralActivationKind kind)
{
    const ActivationDefinition *definition = find_definition(kind);
    return definition == NULL ? NEURAL_ACTIVATION_SCALAR : definition->shape;
}

NeuralInitializerKind neural_activation_initializer(NeuralActivationKind kind)
{
    const ActivationDefinition *definition = find_definition(kind);
    return definition == NULL ? NEURAL_INITIALIZER_XAVIER_UNIFORM
                              : definition->initializer;
}

int neural_activation_spec_set_parameter(
    NeuralActivationSpec *spec,
    NeuralActivationParameterKind kind,
    neural_real value,
    NeuralError *error)
{
    NeuralActivationParameter *parameters;
    size_t position = 0U;

    if (spec == NULL || !isfinite(value)) {
        neural_error_set(error, "activation parameter must be finite");
        return 0;
    }
    while (position < spec->parameter_count &&
           spec->parameters[position].kind < kind) {
        position++;
    }
    if (position < spec->parameter_count &&
        spec->parameters[position].kind == kind) {
        neural_error_set(error,
                         "activation parameter '%s' was specified more than once",
                         neural_activation_parameter_name(kind));
        return 0;
    }
    if (spec->parameter_count >= SIZE_MAX / sizeof(*parameters)) {
        neural_error_set(error, "too many activation parameters");
        return 0;
    }
    parameters = realloc(spec->parameters,
                         (spec->parameter_count + 1U) * sizeof(*parameters));
    if (parameters == NULL) {
        neural_error_set(error, "unable to allocate activation parameters");
        return 0;
    }
    spec->parameters = parameters;
    if (position < spec->parameter_count) {
        memmove(&spec->parameters[position + 1U],
                &spec->parameters[position],
                (spec->parameter_count - position) *
                    sizeof(*spec->parameters));
    }
    spec->parameters[position].kind = kind;
    spec->parameters[position].value = value;
    spec->parameter_count++;
    return 1;
}

int neural_activation_spec_parameter_value(
    const NeuralActivationSpec *spec,
    NeuralActivationParameterKind kind,
    neural_real *value)
{
    size_t index;

    if (spec == NULL || value == NULL) {
        return 0;
    }
    for (index = 0U; index < spec->parameter_count; index++) {
        if (spec->parameters[index].kind == kind) {
            *value = spec->parameters[index].value;
            return 1;
        }
    }
    return 0;
}

int neural_activation_spec_validate(const NeuralActivationSpec *spec,
                                    NeuralError *error)
{
    const ActivationDefinition *definition;
    size_t index;

    if (spec == NULL ||
        (spec->parameter_count != 0U && spec->parameters == NULL)) {
        neural_error_set(error, "invalid activation specification");
        return 0;
    }
    definition = find_definition(spec->kind);
    if (definition == NULL) {
        neural_error_set(error, "unknown activation kind");
        return 0;
    }
    if (spec->parameter_count != definition->parameter_count) {
        neural_error_set(error,
                         "activation '%s' requires %zu parameter(s)",
                         definition->name,
                         definition->parameter_count);
        return 0;
    }
    for (index = 0U; index < spec->parameter_count; index++) {
        const NeuralActivationParameter *parameter = &spec->parameters[index];
        const ParameterDefinition *parameter_definition =
            find_parameter_definition(definition, parameter->kind);
        int below_minimum;
        int above_maximum;

        if (parameter_definition == NULL || !isfinite(parameter->value)) {
            neural_error_set(error,
                             "parameter '%s' is invalid for activation '%s'",
                             neural_activation_parameter_name(parameter->kind),
                             definition->name);
            return 0;
        }
        below_minimum = parameter_definition->minimum_inclusive
                            ? parameter->value < parameter_definition->minimum
                            : parameter->value <= parameter_definition->minimum;
        above_maximum = 0;
        if (parameter_definition->maximum != 0.0) {
            above_maximum = parameter_definition->maximum_inclusive
                                ? parameter->value > parameter_definition->maximum
                                : parameter->value >= parameter_definition->maximum;
        }
        if (below_minimum || above_maximum) {
            neural_error_set(error,
                             "parameter '%s' is outside the valid range for '%s'",
                             neural_activation_parameter_name(parameter->kind),
                             definition->name);
            return 0;
        }
    }
    return 1;
}

int neural_activation_spec_copy(const NeuralActivationSpec *source,
                                NeuralActivationSpec *destination,
                                NeuralError *error)
{
    size_t bytes;

    if (source == NULL || destination == NULL) {
        neural_error_set(error, "invalid activation copy arguments");
        return 0;
    }
    memset(destination, 0, sizeof(*destination));
    destination->kind = source->kind;
    if (!neural_activation_spec_validate(source, error)) {
        return 0;
    }
    if (source->parameter_count == 0U) {
        return 1;
    }
    bytes = source->parameter_count * sizeof(*source->parameters);
    destination->parameters = malloc(bytes);
    if (destination->parameters == NULL) {
        neural_error_set(error, "unable to copy activation parameters");
        return 0;
    }
    memcpy(destination->parameters, source->parameters, bytes);
    destination->parameter_count = source->parameter_count;
    return 1;
}

void neural_activation_spec_free(NeuralActivationSpec *spec)
{
    if (spec != NULL) {
        free(spec->parameters);
        memset(spec, 0, sizeof(*spec));
    }
}

static neural_real apply_scalar(const NeuralActivationSpec *spec,
                                neural_real input)
{
    neural_real exponential;
    neural_real alpha = 0.0;

    switch (spec->kind) {
    case NEURAL_ACTIVATION_LINEAR:
        return input;
    case NEURAL_ACTIVATION_SIGMOID:
        if (input >= 0.0) {
            return 1.0 / (1.0 + exp(-input));
        }
        exponential = exp(input);
        return exponential / (1.0 + exponential);
    case NEURAL_ACTIVATION_TANH:
        return tanh(input);
    case NEURAL_ACTIVATION_RELU:
        return input > 0.0 ? input : 0.0;
    case NEURAL_ACTIVATION_LEAKY_RELU:
        (void)neural_activation_spec_parameter_value(
            spec,
            NEURAL_ACTIVATION_PARAMETER_ALPHA,
            &alpha);
        return input >= 0.0 ? input : alpha * input;
    case NEURAL_ACTIVATION_ELU:
        (void)neural_activation_spec_parameter_value(
            spec,
            NEURAL_ACTIVATION_PARAMETER_ALPHA,
            &alpha);
        return input >= 0.0 ? input : alpha * expm1(input);
    case NEURAL_ACTIVATION_SOFTMAX:
        break;
    }
    return NAN;
}

int neural_activation_apply(const NeuralActivationSpec *spec,
                            const neural_real *inputs,
                            neural_real *outputs,
                            size_t count,
                            NeuralError *error)
{
    size_t index;

    if (inputs == NULL || outputs == NULL || count == 0U ||
        !neural_activation_spec_validate(spec, error)) {
        if (inputs == NULL || outputs == NULL || count == 0U) {
            neural_error_set(error, "invalid activation input or output buffer");
        }
        return 0;
    }
    for (index = 0U; index < count; index++) {
        if (!isfinite(inputs[index])) {
            neural_error_set(error, "activation input must be finite");
            return 0;
        }
    }
    if (spec->kind == NEURAL_ACTIVATION_SOFTMAX) {
        neural_real maximum = inputs[0];
        neural_real sum = 0.0;

        for (index = 1U; index < count; index++) {
            if (inputs[index] > maximum) {
                maximum = inputs[index];
            }
        }
        for (index = 0U; index < count; index++) {
            outputs[index] = exp(inputs[index] - maximum);
            sum += outputs[index];
        }
        if (!isfinite(sum) || sum <= 0.0) {
            neural_error_set(error, "softmax normalization is not finite");
            return 0;
        }
        for (index = 0U; index < count; index++) {
            outputs[index] /= sum;
        }
        return 1;
    }
    for (index = 0U; index < count; index++) {
        outputs[index] = apply_scalar(spec, inputs[index]);
        if (!isfinite(outputs[index])) {
            neural_error_set(error, "activation output is not finite");
            return 0;
        }
    }
    return 1;
}

int neural_activation_backward(const NeuralActivationSpec *spec,
                               const neural_real *pre_activations,
                               const neural_real *activations,
                               const neural_real *output_gradients,
                               neural_real *input_gradients,
                               size_t count,
                               NeuralError *error)
{
    neural_real alpha = 0.0;
    size_t index;

    if (pre_activations == NULL || activations == NULL ||
        output_gradients == NULL || input_gradients == NULL || count == 0U ||
        !neural_activation_spec_validate(spec, error)) {
        if (pre_activations == NULL || activations == NULL ||
            output_gradients == NULL || input_gradients == NULL ||
            count == 0U) {
            neural_error_set(error, "invalid activation backward buffers");
        }
        return 0;
    }
    for (index = 0U; index < count; index++) {
        if (!isfinite(pre_activations[index]) ||
            !isfinite(activations[index]) ||
            !isfinite(output_gradients[index])) {
            neural_error_set(error,
                             "activation backward values must be finite");
            return 0;
        }
    }
    if (spec->kind == NEURAL_ACTIVATION_SOFTMAX) {
        neural_real dot_product = 0.0;

        for (index = 0U; index < count; index++) {
            dot_product += output_gradients[index] * activations[index];
        }
        if (!isfinite(dot_product)) {
            neural_error_set(error,
                             "softmax backward reduction is not finite");
            return 0;
        }
        for (index = 0U; index < count; index++) {
            input_gradients[index] =
                activations[index] *
                (output_gradients[index] - dot_product);
            if (!isfinite(input_gradients[index])) {
                neural_error_set(error,
                                 "softmax input gradient is not finite");
                return 0;
            }
        }
        return 1;
    }
    if (spec->kind == NEURAL_ACTIVATION_LEAKY_RELU ||
        spec->kind == NEURAL_ACTIVATION_ELU) {
        (void)neural_activation_spec_parameter_value(
            spec,
            NEURAL_ACTIVATION_PARAMETER_ALPHA,
            &alpha);
    }
    for (index = 0U; index < count; index++) {
        neural_real derivative = 0.0;

        switch (spec->kind) {
        case NEURAL_ACTIVATION_LINEAR:
            derivative = 1.0;
            break;
        case NEURAL_ACTIVATION_SIGMOID:
            derivative = activations[index] * (1.0 - activations[index]);
            break;
        case NEURAL_ACTIVATION_TANH:
            derivative = 1.0 - activations[index] * activations[index];
            break;
        case NEURAL_ACTIVATION_RELU:
            derivative = pre_activations[index] > 0.0 ? 1.0 : 0.0;
            break;
        case NEURAL_ACTIVATION_LEAKY_RELU:
            derivative = pre_activations[index] >= 0.0 ? 1.0 : alpha;
            break;
        case NEURAL_ACTIVATION_ELU:
            derivative = pre_activations[index] >= 0.0
                             ? 1.0
                             : activations[index] + alpha;
            break;
        case NEURAL_ACTIVATION_SOFTMAX:
            derivative = 0.0;
            break;
        }
        input_gradients[index] = output_gradients[index] * derivative;
        if (!isfinite(input_gradients[index])) {
            neural_error_set(error, "activation input gradient is not finite");
            return 0;
        }
    }
    return 1;
}
