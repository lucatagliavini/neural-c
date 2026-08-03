#define _GNU_SOURCE

#include "neural/parse.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <threads.h>

static once_flag numeric_locale_once = ONCE_FLAG_INIT;
static locale_t c_numeric_locale = (locale_t)0;

static void initialize_numeric_locale(void)
{
    c_numeric_locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
}

int neural_parse_size(const char *text, size_t *value)
{
    char *end;
    uintmax_t parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-' || text[0] == '+') {
        return 0;
    }
    errno = 0;
    parsed = strtoumax(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || parsed > SIZE_MAX) {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

int neural_parse_uint64(const char *text, uint64_t *value)
{
    char *end;
    uintmax_t parsed;

    if (text == NULL || value == NULL || text[0] == '\0' ||
        text[0] == '-' || text[0] == '+') {
        return 0;
    }
    errno = 0;
    parsed = strtoumax(text, &end, 10);
    if (errno == ERANGE || *end != '\0' || parsed > UINT64_MAX) {
        return 0;
    }
    *value = (uint64_t)parsed;
    return 1;
}

static int has_decimal_syntax(const char *text)
{
    const unsigned char *cursor = (const unsigned char *)text;
    int digits = 0;

    if (*cursor == '+' || *cursor == '-') {
        cursor++;
    }
    while (isdigit(*cursor) != 0) {
        digits = 1;
        cursor++;
    }
    if (*cursor == '.') {
        cursor++;
        while (isdigit(*cursor) != 0) {
            digits = 1;
            cursor++;
        }
    }
    if (!digits) {
        return 0;
    }
    if (*cursor == 'e' || *cursor == 'E') {
        int exponent_digits = 0;
        cursor++;
        if (*cursor == '+' || *cursor == '-') {
            cursor++;
        }
        while (isdigit(*cursor) != 0) {
            exponent_digits = 1;
            cursor++;
        }
        if (!exponent_digits) {
            return 0;
        }
    }
    return *cursor == '\0';
}

int neural_parse_real(const char *text, neural_real *value)
{
    char *end;
    double parsed;

    if (text == NULL || value == NULL || !has_decimal_syntax(text)) {
        return 0;
    }
    call_once(&numeric_locale_once, initialize_numeric_locale);
    if (c_numeric_locale == (locale_t)0) {
        return 0;
    }
    errno = 0;
    parsed = strtod_l(text, &end, c_numeric_locale);
    if (errno == ERANGE || *end != '\0' || !isfinite(parsed)) {
        return 0;
    }
    *value = parsed;
    return 1;
}
