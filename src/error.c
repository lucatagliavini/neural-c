#include "neural/error.h"

#include <stdarg.h>
#include <stdio.h>

void neural_error_clear(NeuralError *error)
{
    if (error != NULL) {
        error->message[0] = '\0';
    }
}

void neural_error_set(NeuralError *error, const char *format, ...)
{
    va_list arguments;

    if (error == NULL) {
        return;
    }

    va_start(arguments, format);
    (void)vsnprintf(error->message,
                    sizeof(error->message),
                    format,
                    arguments);
    va_end(arguments);
}
