#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <locale.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "neural/cli_options.h"
#include "neural/checkpoint.h"
#include "neural/defaults.h"
#include "neural/init.h"
#include "neural/parse.h"
#include "neural/project.h"
#include "neural/training.h"
#include "neural/types.h"
#include "neural/version.h"

static int failures = 0;

static void check(int condition, const char *description)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", description);
        failures++;
    }
}

static void check_error_contains(const NeuralError *error,
                                 const char *expected,
                                 const char *description)
{
    check(strstr(error->message, expected) != NULL, description);
}

static void test_core_contract(void)
{
    check(sizeof(neural_real) == sizeof(double),
          "neural_real must use double precision");
    check(strcmp(neural_name(), "neural-c") == 0,
          "library name must be stable");
    check(strlen(neural_version()) > 0U,
          "library version must not be empty");
    check(NEURAL_FORMAT_VERSION == 1,
          "initial text format version must be 1");
    check(NEURAL_DEFAULT_TEXT_INITIAL_CAPACITY >= 2U,
          "initial line capacity must leave room for content and terminator");
    check(NEURAL_DEFAULT_TEXT_MAX_LINE_LENGTH >=
              NEURAL_DEFAULT_TEXT_INITIAL_CAPACITY,
          "maximum line length must cover the initial capacity");
    check(NEURAL_DEFAULT_CAPACITY_GROWTH_FACTOR >= 2U,
          "dynamic capacities must grow by at least a factor of two");
    check(NEURAL_DEFAULT_ERROR_MESSAGE_CAPACITY > 1U,
          "error messages must have a usable capacity");
    check(strcmp(NEURAL_DEFAULT_MODEL_FILENAME, "model.txt") == 0,
          "default model filename must match the project convention");
    check(strcmp(NEURAL_DEFAULT_CHECKPOINT_FILENAME, "checkpoint.txt") == 0,
          "default checkpoint filename must match the persistence convention");
    check(strlen(NEURAL_DEFAULT_ATOMIC_TEMP_SUFFIX) >= 6U &&
              strcmp(NEURAL_DEFAULT_ATOMIC_TEMP_SUFFIX +
                         strlen(NEURAL_DEFAULT_ATOMIC_TEMP_SUFFIX) - 6U,
                     "XXXXXX") == 0,
          "atomic temporary suffix must provide a mkstemp template");
    check(NEURAL_SHA256_TEXT_CAPACITY == 65U,
          "SHA-256 text storage must include its terminator");
    check(NEURAL_DEFAULT_THREAD_COUNT >= 1U,
          "default execution thread count must be positive");
    check(isfinite(NEURAL_DEFAULT_GRADIENT_CHECK_EPSILON) &&
              NEURAL_DEFAULT_GRADIENT_CHECK_EPSILON > 0.0,
          "default gradient-check epsilon must be positive and finite");
    check(isfinite(NEURAL_DEFAULT_GRADIENT_CHECK_ABSOLUTE_TOLERANCE) &&
              NEURAL_DEFAULT_GRADIENT_CHECK_ABSOLUTE_TOLERANCE >= 0.0 &&
              isfinite(NEURAL_DEFAULT_GRADIENT_CHECK_RELATIVE_TOLERANCE) &&
              NEURAL_DEFAULT_GRADIENT_CHECK_RELATIVE_TOLERANCE >= 0.0,
          "default gradient-check tolerances must be finite and non-negative");
}

static void test_real_parser_locale_independence(void)
{
    static const char *const candidate_locales[] = {
        "it_IT.UTF-8",
        "it_IT.utf8",
        "de_DE.UTF-8",
        "de_DE.utf8",
        "fr_FR.UTF-8",
        "fr_FR.utf8"
    };
    const char *current_locale = setlocale(LC_NUMERIC, NULL);
    char saved_locale[128];
    neural_real value;
    size_t index;

    check(current_locale != NULL &&
              strlen(current_locale) < sizeof(saved_locale),
          "numeric locale must be available for parser tests");
    if (current_locale == NULL ||
        strlen(current_locale) >= sizeof(saved_locale)) {
        return;
    }
    (void)strcpy(saved_locale, current_locale);

    check(neural_parse_real("0.125", &value) && value == 0.125,
          "real parser must accept the format decimal point");
    check(!neural_parse_real("0,125", &value),
          "real parser must reject locale-specific decimal separators");

    for (index = 0U;
         index < sizeof(candidate_locales) / sizeof(candidate_locales[0]);
         index++) {
        if (setlocale(LC_NUMERIC, candidate_locales[index]) != NULL) {
            check(neural_parse_real("0.125", &value) && value == 0.125,
                  "real parser must ignore the process numeric locale");
            check(!neural_parse_real("0,125", &value),
                  "real parser grammar must always require a decimal point");
            break;
        }
    }
    check(setlocale(LC_NUMERIC, saved_locale) != NULL,
          "numeric locale must be restored after parser tests");
}

static void test_option_parser(void)
{
    enum {
        OPT_VERBOSE,
        OPT_OUTPUT,
        OPT_LAYER,
        OPT_COUNT
    };
    const NeuralOptionDefinition definitions[OPT_COUNT] = {
        {"verbose", 'v', NEURAL_OPTION_FLAG, 0},
        {"output", 'o', NEURAL_OPTION_VALUE, 0},
        {"layer", 'l', NEURAL_OPTION_VALUE, 1}
    };
    char *valid[] = {
        "train", "project", "--output=weights.txt", "-v",
        "--layer=2:sigmoid", "--layer", "1:tanh", "--", "-1"
    };
    char *duplicate[] = {"-v", "--verbose"};
    char *unknown[] = {"--missing"};
    char *missing_value[] = {"--output"};
    char *flag_value[] = {"--verbose=yes"};
    char *grouped[] = {"-vo"};
    NeuralParsedOptions parsed;
    NeuralError error;

    check(neural_options_parse(9,
                               valid,
                               definitions,
                               OPT_COUNT,
                               &parsed,
                               &error),
          "option parser must accept flags, values, and separator");
    check(neural_option_is_present(&parsed, OPT_VERBOSE),
          "short flag must be recorded");
    check(strcmp(neural_option_value(&parsed, OPT_OUTPUT), "weights.txt") == 0,
          "inline long-option value must be recorded");
    check(parsed.positional_count == 3U,
          "option parser must preserve positional arguments");
    check(strcmp(parsed.positionals[2], "-1") == 0,
          "separator must preserve negative positional values");
    check(neural_option_count(&parsed, OPT_LAYER) == 2U,
          "repeatable options must retain every occurrence");
    check(strcmp(neural_option_value_at(&parsed, OPT_LAYER, 1U),
                 "1:tanh") == 0,
          "repeatable option values must retain their order");
    neural_options_free(&parsed);

    check(!neural_options_parse(2,
                                duplicate,
                                definitions,
                                OPT_COUNT,
                                &parsed,
                                &error),
          "duplicate options must be rejected");
    check_error_contains(&error, "more than once", "duplicate error must be clear");
    check(!neural_options_parse(1,
                                unknown,
                                definitions,
                                OPT_COUNT,
                                &parsed,
                                &error),
          "unknown options must be rejected");
    check_error_contains(&error, "unknown option", "unknown-option error must be clear");
    check(!neural_options_parse(1,
                                missing_value,
                                definitions,
                                OPT_COUNT,
                                &parsed,
                                &error),
          "missing option values must be rejected");
    check_error_contains(&error, "requires a value", "missing-value error must be clear");
    check(!neural_options_parse(1,
                                flag_value,
                                definitions,
                                OPT_COUNT,
                                &parsed,
                                &error),
          "flag values must be rejected");
    check(!neural_options_parse(1,
                                grouped,
                                definitions,
                                OPT_COUNT,
                                &parsed,
                                &error),
          "grouped short options must be rejected explicitly");
}

static void test_valid_loaders(void)
{
    NeuralModelSpec model;
    NeuralTrainingConfig config;
    NeuralDataset dataset;
    NeuralProject project;
    NeuralError error;

    check(neural_model_spec_load("tests/fixtures/model_valid.txt",
                                 &model,
                                 &error),
          "valid dynamic model must load");
    check(model.input_count == 3U, "model input count must be parsed");
    check(model.layer_count == 3U, "all dynamic layers must be parsed");
    check(model.layers[0].neuron_count == 5U,
          "hidden layer size must be dynamic");
    check(model.layers[0].activation.kind == NEURAL_ACTIVATION_LEAKY_RELU,
          "parameterized activation kind must be parsed");
    check(model.layers[0].activation.parameter_count == 1U &&
              model.layers[0].activation.parameters[0].value == 0.01,
          "activation parameters must be parsed");
    check(model.layers[1].activation.kind == NEURAL_ACTIVATION_TANH,
          "layer activation must be parsed");
    check(model.layers[2].neuron_count == 2U,
          "last layer must define output count");
    neural_model_spec_free(&model);

    check(neural_training_config_load("tests/fixtures/config_valid.txt",
                                      &config,
                                      &error),
          "valid training configuration must load");
    check(config.epochs == 250U, "epochs must be parsed");
    check(config.learning_rate == 0.25, "learning rate must be parsed");
    check(config.seed == UINT64_C(18446744073709551615),
          "full unsigned 64-bit seed range must be supported");
    check(config.checkpoint_interval == 25U,
          "checkpoint interval must be parsed");
    check(neural_training_config_load(
              "tests/fixtures/config_zero_checkpoint.txt",
              &config,
              &error) &&
              config.checkpoint_interval == 0U,
          "zero checkpoint interval must disable periodic saves");

    check(neural_dataset_load("tests/fixtures/dataset_valid.txt",
                              3U,
                              2U,
                              &dataset,
                              &error),
          "valid dataset must load");
    check(dataset.sample_count == 2U, "dataset sample count must be derived");
    check(dataset.inputs[4] == -2.5, "signed decimal inputs must be parsed");
    check(dataset.outputs[3] == 1.0, "scientific notation must be parsed");
    neural_dataset_free(&dataset);

    check(neural_project_load("projects/xor", &project, &error),
          "complete XOR project must load");
    check(project.model.input_count == 2U, "XOR must have two inputs");
    check(project.dataset.sample_count == 4U, "XOR must have four samples");
    check(project.model.layers[project.model.layer_count - 1U].neuron_count == 1U,
          "XOR output count must come from the final layer");
    neural_project_free(&project);
}

static void test_invalid_models(void)
{
    static const struct {
        const char *path;
        const char *message;
        const char *location;
    } cases[] = {
        {"tests/fixtures/model_no_input.txt", "before layers", NULL},
        {"tests/fixtures/model_no_layers.txt", "at least one layer", NULL},
        {"tests/fixtures/model_zero.txt", "positive integer", NULL},
        {"tests/fixtures/model_bad_activation.txt", "unknown activation", NULL},
        {"tests/fixtures/model_duplicate_input.txt", "more than once", NULL},
        {"tests/fixtures/model_missing_parameter.txt",
         "requires 1 parameter",
         ":3:"},
        {"tests/fixtures/model_unknown_parameter.txt",
         "invalid for activation",
         ":3:"},
        {"tests/fixtures/model_duplicate_parameter.txt",
         "more than once",
         ":3:"},
        {"tests/fixtures/model_invalid_parameter.txt",
         "outside the valid range",
         ":3:"},
        {"tests/fixtures/model_parameter_on_relu.txt",
         "requires 0 parameter",
         ":3:"},
        {"tests/fixtures/model_overflow.txt", "exceed addressable memory", NULL},
        {"tests/fixtures/model_unknown.txt", "unknown model directive", NULL},
        {"tests/fixtures/model_bad_header.txt",
         "expected 'neural-c model 1'",
         NULL}
    };
    size_t index;

    for (index = 0U; index < sizeof(cases) / sizeof(cases[0]); index++) {
        NeuralModelSpec model;
        NeuralError error;
        check(!neural_model_spec_load(cases[index].path, &model, &error),
              cases[index].path);
        check_error_contains(&error, cases[index].message, cases[index].path);
        if (cases[index].location != NULL) {
            check(strncmp(error.message,
                          cases[index].path,
                          strlen(cases[index].path)) == 0 &&
                      strstr(error.message, cases[index].location) != NULL,
                  "activation errors must identify their source line");
        }
    }
}

static void test_invalid_configs(void)
{
    static const struct {
        const char *path;
        const char *message;
    } cases[] = {
        {"tests/fixtures/config_duplicate.txt", "more than once"},
        {"tests/fixtures/config_missing.txt", "required properties"},
        {"tests/fixtures/config_nan.txt", "finite and positive"},
        {"tests/fixtures/config_zero_epochs.txt", "positive integer"},
        {"tests/fixtures/config_unknown.txt", "unknown configuration property"}
    };
    size_t index;

    for (index = 0U; index < sizeof(cases) / sizeof(cases[0]); index++) {
        NeuralTrainingConfig config;
        NeuralError error;
        check(!neural_training_config_load(cases[index].path, &config, &error),
              cases[index].path);
        check_error_contains(&error, cases[index].message, cases[index].path);
    }
}

static void test_invalid_datasets(void)
{
    static const struct {
        const char *path;
        const char *message;
    } cases[] = {
        {"tests/fixtures/dataset_empty.txt", "at least one sample"},
        {"tests/fixtures/dataset_bad_width.txt", "expected 2 inputs"},
        {"tests/fixtures/dataset_no_arrow.txt", "expected 2 inputs"},
        {"tests/fixtures/dataset_nan.txt", "invalid finite input"},
        {"tests/fixtures/dataset_extra_output.txt", "and 1 outputs"},
        {"tests/fixtures/dataset_bad_header.txt", "expected 'neural-c dataset 1'"}
    };
    size_t index;

    for (index = 0U; index < sizeof(cases) / sizeof(cases[0]); index++) {
        NeuralDataset dataset;
        NeuralError error;
        check(!neural_dataset_load(cases[index].path,
                                   2U,
                                   1U,
                                   &dataset,
                                   &error),
              cases[index].path);
        check_error_contains(&error, cases[index].message, cases[index].path);
    }
}

static void test_missing_project(void)
{
    NeuralProject project;
    NeuralError error;

    check(!neural_project_load("tests/fixtures/does-not-exist",
                               &project,
                               &error),
          "missing project directory must be rejected");
    check_error_contains(&error, "unable to open file", "missing-file error must be clear");
}

static void test_optional_error_output(void)
{
    NeuralModelSpec model;

    check(!neural_model_spec_load("tests/fixtures/model_bad_activation.txt",
                                  &model,
                                  NULL),
          "loaders must permit callers to omit error details");
}

static void remove_init_fixture(const char *directory)
{
    static const char *const names[] = {
        "model.txt",
        "project.conf",
        "train.txt",
        "weights.txt",
        "checkpoint.txt",
        ".neural-c.lock",
        "unrelated.txt",
        ".neural-c-model.new",
        ".neural-c-project.new",
        ".neural-c-dataset.new",
        ".neural-c-weights.new",
        ".neural-c-checkpoint.new",
        ".neural-c-model.old",
        ".neural-c-project.old",
        ".neural-c-dataset.old",
        ".neural-c-weights.old",
        ".neural-c-checkpoint.old"
    };
    size_t index;

    for (index = 0U; index < sizeof(names) / sizeof(names[0]); index++) {
        char path[256];
        (void)snprintf(path, sizeof(path), "%s/%s", directory, names[index]);
        (void)remove(path);
    }
    (void)rmdir(directory);
}

static int write_marker_file(const char *path, const char *content)
{
    FILE *stream = fopen(path, "w");
    int write_status;
    int close_status;

    if (stream == NULL) {
        return 0;
    }
    write_status = fputs(content, stream);
    close_status = fclose(stream);
    return write_status >= 0 && close_status == 0;
}

static void test_project_initialization(void)
{
    static const char *const directory = "build/tests/init-project";
    NeuralLayerSpec initial_layers[] = {
        {4U, {NEURAL_ACTIVATION_RELU, 0U, NULL}},
        {2U, {NEURAL_ACTIVATION_SIGMOID, 0U, NULL}}
    };
    NeuralLayerSpec replacement_layers[] = {
        {1U, {NEURAL_ACTIVATION_TANH, 0U, NULL}}
    };
    NeuralModelSpec initial_model = {3U, 2U, initial_layers};
    NeuralModelSpec replacement_model = {2U, 1U, replacement_layers};
    NeuralModelSpec invalid_model = {0U, 1U, replacement_layers};
    NeuralTrainingConfig training = {
        500U, 0.125, UINT64_C(7), NEURAL_LOSS_MSE, 50U
    };
    NeuralModelSpec loaded_model;
    NeuralTrainingConfig loaded_training;
    NeuralError error;
    struct stat status;
    char path[256];
    FILE *stream;

    remove_init_fixture(directory);
    check(!neural_project_initialize(directory,
                                     &invalid_model,
                                     &training,
                                     0,
                                     &error),
          "invalid initialization must fail before filesystem changes");
    check(lstat(directory, &status) != 0 && errno == ENOENT,
          "invalid initialization must not create its directory");

    check(neural_project_initialize(directory,
                                    &initial_model,
                                    &training,
                                    0,
                                    &error),
          "initialization must create a new project");
    (void)snprintf(path, sizeof(path), "%s/model.txt", directory);
    check(neural_model_spec_load(path, &loaded_model, &error),
          "generated model must load");
    check(loaded_model.input_count == 3U && loaded_model.layer_count == 2U,
          "generated model must preserve dynamic architecture");
    neural_model_spec_free(&loaded_model);
    (void)snprintf(path, sizeof(path), "%s/project.conf", directory);
    check(neural_training_config_load(path, &loaded_training, &error),
          "generated training configuration must load");
    check(loaded_training.epochs == 500U &&
              loaded_training.learning_rate == 0.125 &&
              loaded_training.checkpoint_interval == 50U,
          "generated training values must round-trip");

    check(!neural_project_initialize(directory,
                                     &replacement_model,
                                     &training,
                                     0,
                                     &error),
          "existing project must require force");
    check_error_contains(&error, "--force", "existing-project error must explain force");

    (void)snprintf(path, sizeof(path), "%s/.neural-c-model.new", directory);
    check(write_marker_file(path, "foreign temporary file\n"),
          "test must create a preexisting temporary file");
    check(!neural_project_initialize(directory,
                                     &replacement_model,
                                     &training,
                                     1,
                                     &error),
          "initialization must reject stale transaction files");
    stream = fopen(path, "r");
    check(stream != NULL,
          "failed initialization must preserve foreign temporary files");
    if (stream != NULL) {
        (void)fclose(stream);
    }
    check(remove(path) == 0, "test must remove its temporary marker");

    (void)snprintf(path, sizeof(path), "%s/unrelated.txt", directory);
    check(write_marker_file(path, "preserve\n"),
          "test must create an unrelated file");
    (void)snprintf(path, sizeof(path), "%s/weights.txt", directory);
    check(write_marker_file(path, "stale weights\n"),
          "test must create stale weights");
    (void)snprintf(path, sizeof(path), "%s/checkpoint.txt", directory);
    check(write_marker_file(path, "stale checkpoint\n"),
          "test must create a stale checkpoint");
    check(neural_project_initialize(directory,
                                    &replacement_model,
                                    &training,
                                    1,
                                    &error),
          "force must replace a managed project");
    (void)snprintf(path, sizeof(path), "%s/model.txt", directory);
    check(neural_model_spec_load(path, &loaded_model, &error),
          "forced model must remain valid");
    check(loaded_model.input_count == 2U && loaded_model.layer_count == 1U,
          "force must install the replacement architecture");
    neural_model_spec_free(&loaded_model);
    (void)snprintf(path, sizeof(path), "%s/weights.txt", directory);
    check(lstat(path, &status) != 0 && errno == ENOENT,
          "force must remove incompatible weights");
    (void)snprintf(path, sizeof(path), "%s/checkpoint.txt", directory);
    check(lstat(path, &status) != 0 && errno == ENOENT,
          "force must remove an incompatible checkpoint");
    (void)snprintf(path, sizeof(path), "%s/unrelated.txt", directory);
    stream = fopen(path, "r");
    check(stream != NULL, "force must preserve unrelated files");
    if (stream != NULL) {
        (void)fclose(stream);
    }
    remove_init_fixture(directory);
}

static void test_training_requests(void)
{
    NeuralTrainingRequest request;
    NeuralError error;

    request.mode = NEURAL_TRAIN_FRESH;
    request.additional_epochs = 0U;
    check(neural_training_request_validate(&request, &error),
          "fresh training request must be valid");
    request.mode = NEURAL_TRAIN_RESUME;
    check(neural_training_request_validate(&request, &error),
          "resume training request must be valid");
    request.mode = NEURAL_TRAIN_ADDITIONAL;
    request.additional_epochs = 200U;
    check(neural_training_request_validate(&request, &error),
          "additional training request must accept positive epochs");
    request.additional_epochs = 0U;
    check(!neural_training_request_validate(&request, &error),
          "additional training must reject zero epochs");
    request.mode = NEURAL_TRAIN_RESUME;
    request.additional_epochs = 1U;
    check(!neural_training_request_validate(&request, &error),
          "resume must reject additional epochs");
}

int main(void)
{
    test_core_contract();
    test_real_parser_locale_independence();
    test_option_parser();
    test_valid_loaders();
    test_invalid_models();
    test_invalid_configs();
    test_invalid_datasets();
    test_missing_project();
    test_optional_error_output();
    test_project_initialization();
    test_training_requests();

    if (failures != 0) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }

    puts("All core tests passed");
    return 0;
}
