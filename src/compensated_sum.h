#ifndef NEURAL_INTERNAL_COMPENSATED_SUM_H
#define NEURAL_INTERNAL_COMPENSATED_SUM_H

#include "neural/types.h"

int neural_compensated_add(neural_real sum,
                           neural_real compensation,
                           neural_real addend,
                           neural_real *new_sum,
                           neural_real *new_compensation);

#endif
