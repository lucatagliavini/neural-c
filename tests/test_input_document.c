#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neural/input_document.h"

static int failures;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static int write_text(const char *path, const char *text)
{
    FILE *stream = fopen(path, "w");
    int success = stream != NULL && fputs(text, stream) != EOF;
    if (stream != NULL && fclose(stream) != 0) success = 0;
    return success;
}

static void test_bounded_input_document(void)
{
    const char *path = "build/tests/prediction-input.txt";
    NeuralInputDocument *document = NULL;
    NeuralError error;
    neural_real values[4];
    size_t count = 0U;
    int complete = 0;

    check(write_text(path,
                     "neural-c inputs 1\n"
                     "samples 3\ninputs 2\n"
                     "sample 0 1 2\n"
                     "sample 1 ? 4\n"
                     "sample 2 5 6\nend\n"),
          "input fixture must be writable");
    check(neural_input_document_open(path, &document, &error) &&
              neural_input_document_sample_count(document) == 3U &&
              neural_input_document_input_count(document) == 2U,
          "versioned input header must load");
    check(neural_input_document_read(document, values, 2U, &count,
                                     &complete, &error) &&
              count == 2U && !complete && values[0] == 1.0 &&
              values[1] == 2.0 && isnan(values[2]) && values[3] == 4.0,
          "reader must preserve order and explicit missing values per batch");
    check(neural_input_document_read(document, values, 2U, &count,
                                     &complete, &error) &&
              count == 1U && complete && values[0] == 5.0 && values[1] == 6.0,
          "reader must validate the end marker before completion");
    neural_input_document_close(document);
    (void)remove(path);
}

static void test_malformed_input_document(void)
{
    const char *path = "build/tests/prediction-input-bad.txt";
    NeuralInputDocument *document = NULL;
    NeuralError error;
    neural_real values[2];
    size_t count;
    int complete;

    check(write_text(path,
                     "neural-c inputs 1\nsamples 1\ninputs 2\n"
                     "sample 1 1 2\nend\n"),
          "malformed input fixture must be writable");
    check(neural_input_document_open(path, &document, &error) &&
              !neural_input_document_read(document, values, 1U, &count,
                                          &complete, &error) &&
              strstr(error.message, "expected sample 0") != NULL,
          "reader must reject non-sequential sample indices actionably");
    neural_input_document_close(document);
    (void)remove(path);
}

int main(void)
{
    test_bounded_input_document();
    test_malformed_input_document();
    if (failures != 0) {
        fprintf(stderr, "%d input document test(s) failed\n", failures);
        return EXIT_FAILURE;
    }
    puts("input document tests passed");
    return EXIT_SUCCESS;
}
