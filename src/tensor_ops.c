#include "neural/tensor_ops.h"

#include <math.h>

static int valid_buffer(const neural_real *values, size_t count)
{
    return count == 0U || values != NULL;
}

int neural_tensor_zero(neural_real *values,
                       size_t count,
                       NeuralError *error)
{
    size_t index;

    if (!valid_buffer(values, count)) {
        neural_error_set(error, "invalid tensor zero buffer");
        return 0;
    }
    for (index = 0U; index < count; index++) {
        values[index] = 0.0;
    }
    return 1;
}

int neural_tensor_add(neural_real *destination,
                      const neural_real *source,
                      size_t count,
                      NeuralError *error)
{
    size_t index;

    if (!valid_buffer(destination, count) ||
        !valid_buffer(source, count)) {
        neural_error_set(error, "invalid tensor addition buffers");
        return 0;
    }
    for (index = 0U; index < count; index++) {
        if (!isfinite(destination[index]) || !isfinite(source[index]) ||
            !isfinite(destination[index] + source[index])) {
            neural_error_set(error, "tensor addition must remain finite");
            return 0;
        }
    }
    for (index = 0U; index < count; index++) {
        destination[index] += source[index];
    }
    return 1;
}

int neural_tensor_scale(neural_real *values,
                        size_t count,
                        neural_real factor,
                        NeuralError *error)
{
    size_t index;

    if (!valid_buffer(values, count) || !isfinite(factor)) {
        neural_error_set(error, "invalid tensor scaling arguments");
        return 0;
    }
    for (index = 0U; index < count; index++) {
        if (!isfinite(values[index]) ||
            !isfinite(values[index] * factor)) {
            neural_error_set(error, "tensor scaling must remain finite");
            return 0;
        }
    }
    for (index = 0U; index < count; index++) {
        values[index] *= factor;
    }
    return 1;
}
