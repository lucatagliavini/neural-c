#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neural/cli_options.h"
#include "neural/defaults.h"
#include "neural/digest.h"
#include "neural/init.h"
#include "neural/model.h"
#include "neural/parallel.h"
#include "neural/parse.h"
#include "neural/project.h"
#include "neural/training.h"
#include "neural/version.h"
#include "predict_project.h"
#include "project_lock.h"
#include "train_project.h"

enum exit_code {
    EXIT_OK = 0,
    EXIT_RUNTIME = 1,
    EXIT_USAGE = 2
};

typedef struct {
    struct sigaction previous_interrupt;
    struct sigaction previous_terminate;
    int interrupt_installed;
    int terminate_installed;
} TrainingSignalGuard;

static volatile sig_atomic_t training_stop_signal = 0;

static void handle_training_signal(int signal_number)
{
    if (training_stop_signal == 0) {
        training_stop_signal = signal_number;
    }
}

static int training_signals_install(TrainingSignalGuard *guard,
                                    NeuralError *error)
{
    struct sigaction action;

    memset(guard, 0, sizeof(*guard));
    memset(&action, 0, sizeof(action));
    action.sa_handler = handle_training_signal;
    action.sa_flags = SA_RESTART;
    if (sigemptyset(&action.sa_mask) != 0) {
        neural_error_set(error,
                         "unable to prepare training signal handlers: %s",
                         strerror(errno));
        return 0;
    }
    training_stop_signal = 0;
    if (sigaction(SIGINT, &action, &guard->previous_interrupt) != 0) {
        neural_error_set(error,
                         "unable to install SIGINT handler: %s",
                         strerror(errno));
        return 0;
    }
    guard->interrupt_installed = 1;
    if (sigaction(SIGTERM, &action, &guard->previous_terminate) != 0) {
        neural_error_set(error,
                         "unable to install SIGTERM handler: %s",
                         strerror(errno));
        (void)sigaction(SIGINT, &guard->previous_interrupt, NULL);
        guard->interrupt_installed = 0;
        return 0;
    }
    guard->terminate_installed = 1;
    return 1;
}

static void training_signals_restore(TrainingSignalGuard *guard)
{
    if (guard->terminate_installed) {
        (void)sigaction(SIGTERM, &guard->previous_terminate, NULL);
    }
    if (guard->interrupt_installed) {
        (void)sigaction(SIGINT, &guard->previous_interrupt, NULL);
    }
}

enum option_index {
    OPTION_HELP,
    OPTION_VERSION,
    OPTION_FORCE,
    OPTION_INPUTS,
    OPTION_LAYER,
    OPTION_EPOCHS,
    OPTION_LEARNING_RATE,
    OPTION_SEED,
    OPTION_LOSS,
    OPTION_CHECKPOINT_INTERVAL,
    OPTION_RESUME,
    OPTION_ADDITIONAL_EPOCHS,
    OPTION_THREADS,
    OPTION_COUNT
};

static const NeuralOptionDefinition option_definitions[OPTION_COUNT] = {
    {"help", 'h', NEURAL_OPTION_FLAG, 0},
    {"version", 'V', NEURAL_OPTION_FLAG, 0},
    {"force", 'f', NEURAL_OPTION_FLAG, 0},
    {"inputs", 'i', NEURAL_OPTION_VALUE, 0},
    {"layer", 'l', NEURAL_OPTION_VALUE, 1},
    {"epochs", '\0', NEURAL_OPTION_VALUE, 0},
    {"learning-rate", '\0', NEURAL_OPTION_VALUE, 0},
    {"seed", '\0', NEURAL_OPTION_VALUE, 0},
    {"loss", '\0', NEURAL_OPTION_VALUE, 0},
    {"checkpoint-interval", '\0', NEURAL_OPTION_VALUE, 0},
    {"resume", '\0', NEURAL_OPTION_FLAG, 0},
    {"additional-epochs", '\0', NEURAL_OPTION_VALUE, 0},
    {"threads", 'j', NEURAL_OPTION_VALUE, 0}
};

static void print_usage(FILE *stream)
{
    fprintf(stream,
            "Usage:\n"
            "  %s init <project-directory> --inputs N --layer N:ACT [...]\n"
            "  %s train <project-directory>\n"
            "  %s predict <project-directory> <INPUT ...>\n"
            "  %s inspect <project-directory>\n"
            "  %s --version\n"
            "\nInit options:\n"
            "  -i, --inputs N             Required input width\n"
            "  -l, --layer N:ACT          Required and repeatable layer\n"
            "  -f, --force                Replace an existing project\n"
            "      --epochs N             Default: %zu\n"
            "      --learning-rate VALUE  Default: %.*g\n"
            "      --seed N               Default: %" PRIu64 "\n"
            "      --loss NAME            Default: %s\n"
            "      --checkpoint-interval N Periodic interval; 0 disables\n"
            "                              Default: %zu\n"
            "\nTraining continuation:\n"
            "      --resume                Resume checkpoint.txt\n"
            "      --additional-epochs N  Refine weights.txt\n"
            "\nExecution options:\n"
            "  -j, --threads N            Worker threads; default: %zu\n",
            neural_name(),
            neural_name(),
            neural_name(),
            neural_name(),
            neural_name(),
            (size_t)NEURAL_DEFAULT_INIT_EPOCHS,
            DBL_DECIMAL_DIG,
            (double)NEURAL_DEFAULT_INIT_LEARNING_RATE,
            (uint64_t)NEURAL_DEFAULT_INIT_SEED,
            NEURAL_DEFAULT_INIT_LOSS,
            (size_t)NEURAL_DEFAULT_INIT_CHECKPOINT_INTERVAL,
            (size_t)NEURAL_DEFAULT_THREAD_COUNT);
}

static int parse_layer_option(const char *text,
                              NeuralLayerSpec *layer,
                              NeuralError *error)
{
    char *copy;
    char *activation_name;
    char *parameter_cursor;
    char *separator;
    int success = 0;

    if (text == NULL || layer == NULL) {
        neural_error_set(error, "invalid layer option");
        return 0;
    }
    memset(layer, 0, sizeof(*layer));
    copy = malloc(strlen(text) + 1U);
    if (copy == NULL) {
        neural_error_set(error, "unable to parse layer option");
        return 0;
    }
    (void)strcpy(copy, text);
    separator = strchr(copy, ':');
    if (separator == NULL || separator == copy || separator[1] == '\0') {
        neural_error_set(error,
                         "invalid layer '%s'; expected NEURONS:ACTIVATION",
                         text);
        goto cleanup;
    }
    *separator = '\0';
    activation_name = separator + 1;
    parameter_cursor = strchr(activation_name, ':');
    if (parameter_cursor != NULL) {
        *parameter_cursor = '\0';
        parameter_cursor++;
    }
    if (!neural_parse_size(copy, &layer->neuron_count) ||
        layer->neuron_count == 0U) {
        neural_error_set(error,
                         "layer neuron count must be a positive integer: '%s'",
                         text);
        goto cleanup;
    }
    if (!neural_activation_kind_from_name(activation_name,
                                          &layer->activation.kind)) {
        neural_error_set(error,
                         "unknown layer activation in '%s'",
                         text);
        goto cleanup;
    }
    while (parameter_cursor != NULL) {
        char *next_parameter = strchr(parameter_cursor, ':');
        char *assignment;
        NeuralActivationParameterKind parameter_kind;
        neural_real parameter_value;

        if (next_parameter != NULL) {
            *next_parameter = '\0';
        }
        assignment = strchr(parameter_cursor, '=');
        if (assignment == NULL || assignment == parameter_cursor ||
            assignment[1] == '\0' || strchr(assignment + 1, '=') != NULL) {
            neural_error_set(error,
                             "invalid activation parameter in layer '%s'",
                             text);
            goto cleanup;
        }
        *assignment = '\0';
        if (!neural_activation_parameter_from_name(parameter_cursor,
                                                   &parameter_kind)) {
            neural_error_set(error,
                             "unknown activation parameter '%s'",
                             parameter_cursor);
            goto cleanup;
        }
        if (!neural_parse_real(assignment + 1, &parameter_value)) {
            neural_error_set(error,
                             "invalid activation parameter value '%s'",
                             assignment + 1);
            goto cleanup;
        }
        if (!neural_activation_spec_set_parameter(&layer->activation,
                                                  parameter_kind,
                                                  parameter_value,
                                                  error)) {
            goto cleanup;
        }
        parameter_cursor = next_parameter == NULL ? NULL : next_parameter + 1;
        if (parameter_cursor != NULL && parameter_cursor[0] == '\0') {
            neural_error_set(error,
                             "empty activation parameter in layer '%s'",
                             text);
            goto cleanup;
        }
    }
    if (!neural_activation_spec_validate(&layer->activation, error)) {
        goto cleanup;
    }
    success = 1;

cleanup:
    free(copy);
    if (!success) {
        neural_activation_spec_free(&layer->activation);
    }
    return success;
}

static int command_init(const char *directory,
                        const NeuralParsedOptions *options)
{
    NeuralModelSpec model = {0};
    NeuralTrainingConfig training = {
        NEURAL_DEFAULT_INIT_EPOCHS,
        NEURAL_DEFAULT_INIT_LEARNING_RATE,
        NEURAL_DEFAULT_INIT_SEED,
        NEURAL_LOSS_MSE,
        NEURAL_DEFAULT_INIT_CHECKPOINT_INTERVAL
    };
    NeuralError error;
    const char *loss_name;
    size_t layer_index;
    int result = EXIT_USAGE;

    if (!neural_option_is_present(options, OPTION_INPUTS)) {
        fprintf(stderr, "%s: init requires --inputs\n", neural_name());
        return EXIT_USAGE;
    }
    model.layer_count = neural_option_count(options, OPTION_LAYER);
    if (model.layer_count == 0U) {
        fprintf(stderr,
                "%s: init requires at least one --layer\n",
                neural_name());
        return EXIT_USAGE;
    }
    if (model.layer_count > SIZE_MAX / sizeof(*model.layers)) {
        fprintf(stderr, "%s: too many layers\n", neural_name());
        return EXIT_USAGE;
    }
    model.layers = calloc(model.layer_count, sizeof(*model.layers));
    if (model.layers == NULL) {
        fprintf(stderr, "%s: unable to allocate layer specifications\n",
                neural_name());
        return EXIT_USAGE;
    }

    if (!neural_parse_size(neural_option_value(options, OPTION_INPUTS),
                           &model.input_count) ||
        model.input_count == 0U) {
        neural_error_set(&error, "inputs must be a positive integer");
        goto invalid;
    }
    for (layer_index = 0U; layer_index < model.layer_count; layer_index++) {
        if (!parse_layer_option(neural_option_value_at(options,
                                                       OPTION_LAYER,
                                                       layer_index),
                                &model.layers[layer_index],
                                &error)) {
            goto invalid;
        }
    }
    if (neural_option_is_present(options, OPTION_EPOCHS) &&
        (!neural_parse_size(neural_option_value(options, OPTION_EPOCHS),
                            &training.epochs) ||
         training.epochs == 0U)) {
        neural_error_set(&error, "epochs must be a positive integer");
        goto invalid;
    }
    if (neural_option_is_present(options, OPTION_LEARNING_RATE) &&
        (!neural_parse_real(neural_option_value(options,
                                                OPTION_LEARNING_RATE),
                            &training.learning_rate) ||
         training.learning_rate <= 0.0)) {
        neural_error_set(&error, "learning-rate must be finite and positive");
        goto invalid;
    }
    if (neural_option_is_present(options, OPTION_SEED) &&
        !neural_parse_uint64(neural_option_value(options, OPTION_SEED),
                             &training.seed)) {
        neural_error_set(&error, "seed must be an unsigned 64-bit integer");
        goto invalid;
    }
    loss_name = neural_option_is_present(options, OPTION_LOSS)
                    ? neural_option_value(options, OPTION_LOSS)
                    : NEURAL_DEFAULT_INIT_LOSS;
    if (!neural_loss_from_name(loss_name, &training.loss)) {
        neural_error_set(&error,
                         "unsupported loss '%s'",
                         loss_name);
        goto invalid;
    }
    if (neural_option_is_present(options, OPTION_CHECKPOINT_INTERVAL) &&
        !neural_parse_size(
            neural_option_value(options, OPTION_CHECKPOINT_INTERVAL),
            &training.checkpoint_interval)) {
        neural_error_set(&error,
                         "checkpoint-interval must be a non-negative integer");
        goto invalid;
    }

    if (!neural_project_initialize(
            directory,
            &model,
            &training,
            neural_option_is_present(options, OPTION_FORCE),
            &error)) {
        goto invalid;
    }
    printf("Initialized %s project: %s\n", neural_name(), directory);
    result = EXIT_OK;
    goto cleanup;

invalid:
    fprintf(stderr, "%s: %s\n", neural_name(), error.message);

cleanup:
    neural_model_spec_free(&model);
    return result;
}

static int has_options_in_range(const NeuralParsedOptions *options,
                                size_t first,
                                size_t end)
{
    size_t index;

    for (index = first; index < end; index++) {
        if (neural_option_is_present(options, index)) {
            return 1;
        }
    }
    return 0;
}

static int command_train(const char *directory,
                         const NeuralParsedOptions *options)
{
    NeuralTrainingRequest request = {NEURAL_TRAIN_FRESH, 0U};
    NeuralExecutionConfig execution = {NEURAL_DEFAULT_THREAD_COUNT};
    NeuralTrainingResult result;
    NeuralError error;
    TrainingSignalGuard signal_guard;
    int interrupted_signal = 0;
    int trained;

    if (neural_option_is_present(options, OPTION_RESUME) &&
        neural_option_is_present(options, OPTION_ADDITIONAL_EPOCHS)) {
        fprintf(stderr,
                "%s: --resume and --additional-epochs are mutually exclusive\n",
                neural_name());
        return EXIT_USAGE;
    }
    if (neural_option_is_present(options, OPTION_RESUME)) {
        request.mode = NEURAL_TRAIN_RESUME;
    } else if (neural_option_is_present(options, OPTION_ADDITIONAL_EPOCHS)) {
        request.mode = NEURAL_TRAIN_ADDITIONAL;
        if (!neural_parse_size(
                neural_option_value(options, OPTION_ADDITIONAL_EPOCHS),
                &request.additional_epochs)) {
            fprintf(stderr,
                    "%s: additional epochs must be a positive integer\n",
                    neural_name());
            return EXIT_USAGE;
        }
    }
    if (!neural_training_request_validate(&request, &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        return EXIT_USAGE;
    }
    if (neural_option_is_present(options, OPTION_THREADS) &&
        !neural_parse_size(neural_option_value(options, OPTION_THREADS),
                           &execution.thread_count)) {
        fprintf(stderr, "%s: threads must be a positive integer\n", neural_name());
        return EXIT_USAGE;
    }
    if (!neural_execution_config_validate(&execution, &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        return EXIT_USAGE;
    }
    if (request.mode == NEURAL_TRAIN_FRESH ||
        request.mode == NEURAL_TRAIN_RESUME ||
        request.mode == NEURAL_TRAIN_ADDITIONAL) {
        if (!training_signals_install(&signal_guard, &error)) {
            fprintf(stderr, "%s: %s\n", neural_name(), error.message);
            return EXIT_RUNTIME;
        }
        if (request.mode == NEURAL_TRAIN_FRESH) {
            trained = neural_project_train_fresh_controlled(
                directory,
                &execution,
                &training_stop_signal,
                &interrupted_signal,
                &result,
                &error);
        } else if (request.mode == NEURAL_TRAIN_RESUME) {
            trained = neural_project_train_resume_controlled(
                directory,
                &execution,
                &training_stop_signal,
                &interrupted_signal,
                &result,
                &error);
        } else {
            trained = neural_project_train_additional_controlled(
                directory,
                &execution,
                request.additional_epochs,
                &training_stop_signal,
                &interrupted_signal,
                &result,
                &error);
        }
        training_signals_restore(&signal_guard);
        if (!trained) {
            fprintf(stderr, "%s: %s\n", neural_name(), error.message);
            if (interrupted_signal > 0 && interrupted_signal < 128) {
                return 128 + interrupted_signal;
            }
            return EXIT_RUNTIME;
        }
        printf("Training complete: %zu epochs, loss %.*g, workers %zu\n",
               result.completed_epochs,
               DBL_DECIMAL_DIG,
               result.final_loss,
               result.worker_count);
        return EXIT_OK;
    }
    fprintf(stderr, "%s: unknown training mode\n", neural_name());
    return EXIT_RUNTIME;
}

static int command_predict(const char *project,
                           const NeuralParsedOptions *options)
{
    NeuralExecutionConfig execution = {NEURAL_DEFAULT_THREAD_COUNT};
    NeuralPredictionSnapshot snapshot =
        NEURAL_PREDICTION_SNAPSHOT_INITIALIZER;
    neural_real *inputs = NULL;
    neural_real *outputs = NULL;
    NeuralError error;
    size_t input_value_count;
    size_t sample_count;
    size_t output_value_count;
    size_t worker_count;
    size_t value_index;
    int result = EXIT_RUNTIME;

    if (neural_option_is_present(options, OPTION_THREADS) &&
        !neural_parse_size(neural_option_value(options, OPTION_THREADS),
                           &execution.thread_count)) {
        fprintf(stderr, "%s: threads must be a positive integer\n", neural_name());
        return EXIT_USAGE;
    }
    if (!neural_execution_config_validate(&execution, &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        return EXIT_USAGE;
    }
    if (!neural_project_prediction_load(project, &snapshot, &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        goto cleanup;
    }
    input_value_count = options->positional_count - 2U;
    if (input_value_count == 0U ||
        input_value_count % snapshot.input_count != 0U) {
        fprintf(stderr,
                "%s: predict requires one or more complete samples of %zu inputs\n",
                neural_name(),
                snapshot.input_count);
        result = EXIT_USAGE;
        goto cleanup;
    }
    sample_count = input_value_count / snapshot.input_count;
    if (input_value_count > SIZE_MAX / sizeof(*inputs) ||
        sample_count > SIZE_MAX / snapshot.output_count ||
        sample_count * snapshot.output_count > SIZE_MAX / sizeof(*outputs)) {
        fprintf(stderr, "%s: prediction input is too large\n", neural_name());
        result = EXIT_USAGE;
        goto cleanup;
    }
    output_value_count = sample_count * snapshot.output_count;
    inputs = malloc(input_value_count * sizeof(*inputs));
    outputs = malloc(output_value_count * sizeof(*outputs));
    if (inputs == NULL || outputs == NULL) {
        fprintf(stderr, "%s: unable to allocate prediction buffers\n",
                neural_name());
        goto cleanup;
    }
    for (value_index = 0U; value_index < input_value_count; value_index++) {
        const char *text = options->positionals[value_index + 2U];

        if (!neural_parse_real(text, &inputs[value_index])) {
            fprintf(stderr,
                    "%s: invalid prediction input %zu: '%s'\n",
                    neural_name(),
                    value_index,
                    text);
            result = EXIT_USAGE;
            goto cleanup;
        }
    }
    if (!neural_prediction_run(&snapshot,
                               inputs,
                               sample_count,
                               &execution,
                               outputs,
                               &worker_count,
                               &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        goto cleanup;
    }
    printf("%s predictions %d\n", NEURAL_FORMAT_MAGIC, NEURAL_FORMAT_VERSION);
    printf("completed_epochs %zu\n", snapshot.completed_epochs);
    printf("samples %zu\n", sample_count);
    printf("inputs %zu\n", snapshot.input_count);
    printf("outputs %zu\n", snapshot.output_count);
    for (value_index = 0U; value_index < sample_count; value_index++) {
        size_t output_index;

        printf("sample %zu", value_index);
        for (output_index = 0U;
             output_index < snapshot.output_count;
             output_index++) {
            printf(" %.*g",
                   DBL_DECIMAL_DIG,
                   outputs[value_index * snapshot.output_count + output_index]);
        }
        putchar('\n');
    }
    puts("end");
    (void)worker_count;
    result = EXIT_OK;

cleanup:
    free(outputs);
    free(inputs);
    neural_prediction_snapshot_free(&snapshot);
    return result;
}

static int command_inspect(const char *directory)
{
    NeuralProject project;
    NeuralProjectDigests digests;
    NeuralModel *runtime_model = NULL;
    NeuralProjectLock project_lock = NEURAL_PROJECT_LOCK_INITIALIZER;
    NeuralError error;
    size_t layer_index;
    size_t output_count;

    if (!neural_project_lock_acquire(directory,
                                     NEURAL_PROJECT_LOCK_SHARED,
                                     &project_lock,
                                     &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        return EXIT_RUNTIME;
    }
    if (!neural_project_load(directory, &project, &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        neural_project_lock_release(&project_lock);
        return EXIT_USAGE;
    }
    if (!neural_model_create(&project.model,
                             project.training.seed,
                             &runtime_model,
                             &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        neural_project_free(&project);
        neural_project_lock_release(&project_lock);
        return EXIT_USAGE;
    }
    if (!neural_project_digests_compute(&project, &digests, &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        neural_model_free(runtime_model);
        neural_project_free(&project);
        neural_project_lock_release(&project_lock);
        return EXIT_USAGE;
    }

    output_count = project.model.layers[project.model.layer_count - 1U]
                       .neuron_count;
    printf("Project: %s\n", directory);
    printf("Format version: %d\n", NEURAL_FORMAT_VERSION);
    printf("Inputs: %zu\n", project.model.input_count);
    printf("Layers: %zu\n", project.model.layer_count);
    for (layer_index = 0U;
         layer_index < project.model.layer_count;
         layer_index++) {
        const NeuralLayerSpec *layer = &project.model.layers[layer_index];
        size_t parameter_index;

        printf("  Layer %zu: dense, %zu neurons, %s, weights %zux%zu",
               layer_index,
               layer->neuron_count,
               neural_activation_kind_name(layer->activation.kind),
               neural_model_layer_neuron_count(runtime_model, layer_index),
               neural_model_layer_input_count(runtime_model, layer_index));
        for (parameter_index = 0U;
             parameter_index < layer->activation.parameter_count;
             parameter_index++) {
            const NeuralActivationParameter *parameter =
                &layer->activation.parameters[parameter_index];
            printf(" %s=%.*g",
                   neural_activation_parameter_name(parameter->kind),
                   DBL_DECIMAL_DIG,
                   parameter->value);
        }
        putchar('\n');
    }
    printf("Outputs: %zu\n", output_count);
    printf("Parameters: %zu\n",
           neural_model_parameter_count(runtime_model));
    printf("Training samples: %zu\n", project.dataset.sample_count);
    printf("Loss: %s\n", neural_loss_name(project.training.loss));
    printf("Epochs: %zu\n", project.training.epochs);
    printf("Learning rate: %.*g\n",
           DBL_DECIMAL_DIG,
           project.training.learning_rate);
    printf("Seed: %" PRIu64 "\n", project.training.seed);
    printf("Checkpoint interval: %zu\n",
           project.training.checkpoint_interval);
    printf("Model digest: sha256:%s\n", digests.model);
    printf("Dataset digest: sha256:%s\n", digests.dataset);
    printf("Training digest: sha256:%s\n", digests.training);
    puts("Validation: OK");

    neural_model_free(runtime_model);
    neural_project_free(&project);
    neural_project_lock_release(&project_lock);
    return EXIT_OK;
}

int main(int argc, char **argv)
{
    NeuralParsedOptions options;
    NeuralError error;
    const char *command;

    if (!neural_options_parse(argc - 1,
                              argv + 1,
                              option_definitions,
                              OPTION_COUNT,
                              &options,
                              &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        return EXIT_USAGE;
    }

    if (neural_option_is_present(&options, OPTION_VERSION)) {
        printf("%s %s\n", neural_name(), neural_version());
        neural_options_free(&options);
        return EXIT_OK;
    }

    if (neural_option_is_present(&options, OPTION_HELP)) {
        print_usage(stdout);
        neural_options_free(&options);
        return EXIT_OK;
    }

    if (options.positional_count < 2U) {
        print_usage(stderr);
        neural_options_free(&options);
        return EXIT_USAGE;
    }

    command = options.positionals[0];
    if (strcmp(command, "init") == 0) {
        int result;
        if (options.positional_count != 2U) {
            fprintf(stderr,
                    "%s: init requires one project directory\n",
                    neural_name());
            neural_options_free(&options);
            return EXIT_USAGE;
        }
        if (has_options_in_range(&options,
                                 OPTION_RESUME,
                                 OPTION_COUNT)) {
            fprintf(stderr,
                    "%s: training and execution options are invalid with init\n",
                    neural_name());
            neural_options_free(&options);
            return EXIT_USAGE;
        }
        result = command_init(options.positionals[1], &options);
        neural_options_free(&options);
        return result;
    }
    if (has_options_in_range(&options,
                             OPTION_FORCE,
                             OPTION_RESUME)) {
        fprintf(stderr,
                "%s: initialization options are valid only with init\n",
                neural_name());
        neural_options_free(&options);
        return EXIT_USAGE;
    }
    if (strcmp(command, "train") == 0) {
        int result;
        if (options.positional_count != 2U) {
            fprintf(stderr,
                    "%s: train requires one project directory\n",
                    neural_name());
            neural_options_free(&options);
            return EXIT_USAGE;
        }
        result = command_train(options.positionals[1], &options);
        neural_options_free(&options);
        return result;
    }
    if (neural_option_is_present(&options, OPTION_RESUME) ||
        neural_option_is_present(&options, OPTION_ADDITIONAL_EPOCHS)) {
        fprintf(stderr,
                "%s: training continuation options are valid only with train\n",
                neural_name());
        neural_options_free(&options);
        return EXIT_USAGE;
    }
    if (strcmp(command, "inspect") == 0) {
        int result;
        if (neural_option_is_present(&options, OPTION_THREADS)) {
            fprintf(stderr,
                    "%s: --threads is valid only with train or predict\n",
                    neural_name());
            neural_options_free(&options);
            return EXIT_USAGE;
        }
        if (options.positional_count != 2U) {
            fprintf(stderr,
                    "%s: inspect requires one project directory\n",
                    neural_name());
            neural_options_free(&options);
            return EXIT_USAGE;
        }
        result = command_inspect(options.positionals[1]);
        neural_options_free(&options);
        return result;
    }
    if (strcmp(command, "predict") == 0) {
        int result = command_predict(options.positionals[1], &options);
        neural_options_free(&options);
        return result;
    }

    fprintf(stderr, "%s: unknown command '%s'\n", neural_name(), command);
    print_usage(stderr);
    neural_options_free(&options);
    return EXIT_USAGE;
}
