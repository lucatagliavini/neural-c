#include "compensated_sum.h"

#include <math.h>
#include <stddef.h>

int neural_compensated_add(neural_real sum,
                           neural_real compensation,
                           neural_real addend,
                           neural_real *new_sum,
                           neural_real *new_compensation)
{
    neural_real total;
    neural_real correction;
    neural_real corrected;

    if (new_sum == NULL || new_compensation == NULL || !isfinite(sum) ||
        !isfinite(compensation) || !isfinite(addend)) {
        return 0;
    }
    total = sum + addend;
    if (!isfinite(total)) {
        return 0;
    }
    if (fabs(sum) >= fabs(addend)) {
        correction = (sum - total) + addend;
    } else {
        correction = (addend - total) + sum;
    }
    corrected = compensation + correction;
    if (!isfinite(correction) || !isfinite(corrected) ||
        !isfinite(total + corrected)) {
        return 0;
    }
    *new_sum = total;
    *new_compensation = corrected;
    return 1;
}
