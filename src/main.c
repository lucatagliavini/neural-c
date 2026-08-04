#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "neural/cli_options.h"
#include "neural/data_import.h"
#include "neural/defaults.h"
#include "neural/digest.h"
#include "neural/evaluation.h"
#include "neural/init.h"
#include "neural/input_document.h"
#include "neural/model.h"
#include "neural/parallel.h"
#include "neural/parse.h"
#include "neural/persistence.h"
#include "neural/project.h"
#include "neural/training.h"
#include "neural/version.h"
#include "predict_project.h"
#include "path.h"
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
    OPTION_EARLY_STOPPING_PATIENCE,
    OPTION_EARLY_STOPPING_MIN_DELTA,
    OPTION_RESUME,
    OPTION_ADDITIONAL_EPOCHS,
    OPTION_THREADS,
    OPTION_REPORT_INTERVAL,
    OPTION_DATASET,
    OPTION_HISTORY,
    OPTION_STATE,
    OPTION_INPUT_FILE,
    OPTION_BATCH_SIZE,
    OPTION_SCHEMA,
    OPTION_VALIDATION_RATIO,
    OPTION_TEST_RATIO,
    OPTION_SPLIT_SEED,
    OPTION_NORMALIZATION,
    OPTION_MISSING,
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
    {"early-stopping-patience", '\0', NEURAL_OPTION_VALUE, 0},
    {"early-stopping-min-delta", '\0', NEURAL_OPTION_VALUE, 0},
    {"resume", '\0', NEURAL_OPTION_FLAG, 0},
    {"additional-epochs", '\0', NEURAL_OPTION_VALUE, 0},
    {"threads", 'j', NEURAL_OPTION_VALUE, 0},
    {"report-interval", '\0', NEURAL_OPTION_VALUE, 0},
    {"dataset", '\0', NEURAL_OPTION_VALUE, 0},
    {"history", '\0', NEURAL_OPTION_FLAG, 0},
    {"state", '\0', NEURAL_OPTION_FLAG, 0},
    {"input", '\0', NEURAL_OPTION_VALUE, 0},
    {"batch-size", '\0', NEURAL_OPTION_VALUE, 0},
    {"schema", '\0', NEURAL_OPTION_VALUE, 0},
    {"validation-ratio", '\0', NEURAL_OPTION_VALUE, 0},
    {"test-ratio", '\0', NEURAL_OPTION_VALUE, 0},
    {"split-seed", '\0', NEURAL_OPTION_VALUE, 0},
    {"normalization", '\0', NEURAL_OPTION_VALUE, 0},
    {"missing", '\0', NEURAL_OPTION_VALUE, 0}
};

static void print_usage(FILE *stream)
{
    fprintf(stream,
            "Usage:\n"
            "  %s init <project-directory> --inputs N --layer N:ACT [...]\n"
            "  %s train <project-directory>\n"
            "  %s predict <project-directory> <INPUT ...>\n"
            "  %s predict <project-directory> --input FILE|- [--batch-size N]\n"
            "  %s evaluate <project-directory> [--dataset NAME]\n"
            "  %s import-csv <project-directory> <CSV> --schema FILE\n"
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
            "      --early-stopping-patience N Validation patience; 0 disables\n"
            "      --early-stopping-min-delta VALUE Minimum validation improvement\n"
            "\nTraining continuation:\n"
            "      --resume                Resume checkpoint.txt\n"
            "      --additional-epochs N  Refine weights.txt\n"
            "\nExecution options:\n"
            "  -j, --threads N            Worker threads; default: %zu\n"
            "      --report-interval N    Training progress; 0 disables\n"
            "                              Default: %zu\n"
            "      --history              Write progress to history.txt; requires\n"
            "                              a positive --report-interval\n"
            "      --dataset NAME         Evaluation dataset: train, validation,\n"
            "                              or test (default: test)\n"
            "      --state                Inspect finalized/checkpoint state\n"
            "      --input FILE|-         Versioned prediction input document\n"
            "      --batch-size N         Prediction stream batch; default: %zu\n"
            "\nCSV import options:\n"
            "      --schema FILE          Required versioned CSV schema\n"
            "      --validation-ratio R   Validation fraction; default: 0\n"
            "      --test-ratio R         Test fraction; default: 0\n"
            "      --split-seed N         Deterministic split seed; default: 0\n"
            "      --normalization NAME   none, standardize, or minmax\n"
            "      --missing POLICY       reject or mean\n",
            neural_name(),
            neural_name(),
            neural_name(),
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
            (size_t)NEURAL_DEFAULT_THREAD_COUNT,
            (size_t)NEURAL_DEFAULT_REPORT_INTERVAL,
            (size_t)NEURAL_DEFAULT_PREDICTION_BATCH_SIZE);
}

typedef struct {
    size_t interval;
    neural_real best_loss;
    neural_real last_reported_loss;
    int has_best;
    int has_reported;
    int history_enabled;
    int history_append;
    int history_failed;
    const char *directory;
    char *history_path;
    FILE *history;
} TrainingProgress;

static void training_progress_close(TrainingProgress *progress)
{
    if (progress != NULL) {
        if (progress->history != NULL) {
            (void)fclose(progress->history);
            progress->history = NULL;
        }
        free(progress->history_path);
        progress->history_path = NULL;
    }
}

static void write_training_history(TrainingProgress *progress,
                                   const NeuralEpochReport *report)
{
    long position;

    if (!progress->history_enabled || progress->history_failed) {
        return;
    }
    if (progress->history == NULL) {
        progress->history_path = neural_path_join(
            progress->directory, NEURAL_DEFAULT_HISTORY_FILENAME, NULL);
        if (progress->history_path != NULL) {
            progress->history = fopen(progress->history_path,
                                      progress->history_append ? "a+" : "w");
        }
        if (progress->history == NULL) {
            fprintf(stderr,
                    "%s: warning: unable to open training history\n",
                    neural_name());
            progress->history_failed = 1;
            return;
        }
        if (fseek(progress->history, 0L, SEEK_END) != 0 ||
            (position = ftell(progress->history)) < 0L) {
            fprintf(stderr,
                    "%s: warning: unable to inspect training history\n",
                    neural_name());
            progress->history_failed = 1;
            return;
        }
        if (position == 0L &&
            fprintf(progress->history,
                    "%s history %d\n",
                    NEURAL_FORMAT_MAGIC,
                    NEURAL_FORMAT_VERSION) < 0) {
            progress->history_failed = 1;
            return;
        }
    }
    if (fprintf(progress->history,
                "epoch %zu target %zu loss %.*g best %.*g",
                report->completed_epochs,
                report->target_epochs,
                DBL_DECIMAL_DIG,
                report->loss,
                DBL_DECIMAL_DIG,
                progress->best_loss) < 0 ||
        (report->has_validation_loss &&
         fprintf(progress->history,
                 " validation_loss %.*g best_validation_loss %.*g",
                 DBL_DECIMAL_DIG,
                 report->validation_loss,
                 DBL_DECIMAL_DIG,
                 report->best_validation_loss) < 0) ||
        fputc('\n', progress->history) == EOF ||
        fflush(progress->history) != 0) {
        fprintf(stderr,
                "%s: warning: unable to write training history\n",
                neural_name());
        progress->history_failed = 1;
    }
}

static int report_training_progress(const NeuralEpochReport *report,
                                    void *context,
                                    NeuralError *error)
{
    TrainingProgress *progress = context;
    neural_real convergence_loss;
    int should_report;

    if (report == NULL || progress == NULL || progress->interval == 0U ||
        report->completed_epochs == 0U || report->target_epochs == 0U ||
        report->completed_epochs > report->target_epochs ||
        !isfinite(report->loss)) {
        neural_error_set(error, "training progress report is invalid");
        return 0;
    }
    convergence_loss = report->has_validation_loss
                           ? report->validation_loss
                           : report->loss;
    if (report->has_validation_loss) {
        progress->best_loss = report->best_validation_loss;
        progress->has_best = 1;
    } else if (!progress->has_best || convergence_loss < progress->best_loss) {
        progress->best_loss = convergence_loss;
        progress->has_best = 1;
    }
    should_report = report->completed_epochs == report->target_epochs ||
                    report->stopped_early ||
                    report->completed_epochs % progress->interval == 0U;
    if (!should_report) {
        return 1;
    }
    if (fprintf(stderr,
                "Training progress: epoch %zu/%zu, loss %.*g, best %.*g",
                report->completed_epochs,
                report->target_epochs,
                DBL_DECIMAL_DIG,
                report->loss,
                DBL_DECIMAL_DIG,
                progress->best_loss) < 0) {
        neural_error_set(error, "unable to write training progress");
        return 0;
    }
    if (report->has_validation_loss &&
        fprintf(stderr,
                ", validation %.*g, best validation %.*g",
                DBL_DECIMAL_DIG,
                report->validation_loss,
                DBL_DECIMAL_DIG,
                report->best_validation_loss) < 0) {
        neural_error_set(error, "unable to write training progress");
        return 0;
    }
    if (progress->has_reported) {
        neural_real improvement =
            progress->last_reported_loss - convergence_loss;
        neural_real denominator = fabs(progress->last_reported_loss);
        neural_real relative = denominator == 0.0
                                   ? 0.0
                                   : improvement / denominator;

        if (fprintf(stderr,
                    ", improvement %.*g, relative %.*g",
                    DBL_DECIMAL_DIG,
                    improvement,
                    DBL_DECIMAL_DIG,
                    relative) < 0) {
            neural_error_set(error, "unable to write training progress");
            return 0;
        }
    } else if (fputs(", improvement n/a, relative n/a", stderr) == EOF) {
        neural_error_set(error, "unable to write training progress");
        return 0;
    }
    if (fputc('\n', stderr) == EOF) {
        neural_error_set(error, "unable to write training progress");
        return 0;
    }
    write_training_history(progress, report);
    progress->last_reported_loss = convergence_loss;
    progress->has_reported = 1;
    return 1;
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
        NEURAL_DEFAULT_INIT_CHECKPOINT_INTERVAL,
        NEURAL_DEFAULT_INIT_EARLY_STOPPING_PATIENCE,
        NEURAL_DEFAULT_INIT_EARLY_STOPPING_MIN_DELTA
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
    if (neural_option_is_present(options, OPTION_EARLY_STOPPING_PATIENCE) &&
        !neural_parse_size(
            neural_option_value(options, OPTION_EARLY_STOPPING_PATIENCE),
            &training.early_stopping_patience)) {
        neural_error_set(
            &error,
            "early-stopping-patience must be a non-negative integer");
        goto invalid;
    }
    if (neural_option_is_present(options, OPTION_EARLY_STOPPING_MIN_DELTA) &&
        (!neural_parse_real(
             neural_option_value(options, OPTION_EARLY_STOPPING_MIN_DELTA),
             &training.early_stopping_min_delta) ||
         training.early_stopping_min_delta < 0.0)) {
        neural_error_set(
            &error,
            "early-stopping-min-delta must be finite and non-negative");
        goto invalid;
    }
    if (!neural_training_config_validate(&training, &error)) {
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
    TrainingProgress progress = {
        NEURAL_DEFAULT_REPORT_INTERVAL, 0.0, 0.0, 0, 0,
        0, 0, 0, directory, NULL, NULL
    };
    NeuralTrainingResult result;
    NeuralError error;
    TrainingSignalGuard signal_guard;
    int interrupted_signal = 0;
    int trained;

    if (neural_option_is_present(options, OPTION_DATASET)) {
        fprintf(stderr, "%s: --dataset is valid only with evaluate\n",
                neural_name());
        return EXIT_USAGE;
    }
    if (neural_option_is_present(options, OPTION_STATE)) {
        fprintf(stderr, "%s: --state is valid only with inspect\n",
                neural_name());
        return EXIT_USAGE;
    }

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
    if (neural_option_is_present(options, OPTION_REPORT_INTERVAL) &&
        !neural_parse_size(
            neural_option_value(options, OPTION_REPORT_INTERVAL),
            &progress.interval)) {
        fprintf(stderr,
                "%s: report-interval must be a non-negative integer\n",
                neural_name());
        return EXIT_USAGE;
    }
    progress.history_enabled =
        neural_option_is_present(options, OPTION_HISTORY);
    progress.history_append = request.mode != NEURAL_TRAIN_FRESH;
    if (progress.history_enabled && progress.interval == 0U) {
        fprintf(stderr,
                "%s: --history requires a positive --report-interval\n",
                neural_name());
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
                progress.interval == 0U ? NULL : report_training_progress,
                &progress,
                &interrupted_signal,
                &result,
                &error);
        } else if (request.mode == NEURAL_TRAIN_RESUME) {
            trained = neural_project_train_resume_controlled(
                directory,
                &execution,
                &training_stop_signal,
                progress.interval == 0U ? NULL : report_training_progress,
                &progress,
                &interrupted_signal,
                &result,
                &error);
        } else {
            trained = neural_project_train_additional_controlled(
                directory,
                &execution,
                request.additional_epochs,
                &training_stop_signal,
                progress.interval == 0U ? NULL : report_training_progress,
                &progress,
                &interrupted_signal,
                &result,
                &error);
        }
        training_signals_restore(&signal_guard);
        training_progress_close(&progress);
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

static int write_prediction_header(FILE *stream,
                                   const NeuralPredictionSnapshot *snapshot,
                                   size_t sample_count)
{
    if (fprintf(stream,
                "%s predictions %zu\n"
                "completed_epochs %zu\n",
                NEURAL_FORMAT_MAGIC,
                snapshot->format_version == 2U ? (size_t)2U : (size_t)1U,
                snapshot->completed_epochs) < 0) {
        return 0;
    }
    if (snapshot->format_version == 2U &&
        fprintf(stream,
                "selected_epoch %zu\n"
                "target_epochs %zu\n"
                "completion %s\n",
                snapshot->selected_epoch,
                snapshot->target_epochs,
                snapshot->completion_reason ==
                        NEURAL_COMPLETION_EARLY_STOPPING
                    ? "early_stopping" : "target") < 0) {
        return 0;
    }
    return fprintf(stream,
                   "samples %zu\ninputs %zu\noutputs %zu\n",
                   sample_count,
                   snapshot->input_count,
                   snapshot->output_count) >= 0;
}

static int write_prediction_samples(FILE *stream,
                                    const NeuralPredictionSnapshot *snapshot,
                                    const neural_real *outputs,
                                    size_t first_sample,
                                    size_t sample_count)
{
    size_t sample_index;

    for (sample_index = 0U; sample_index < sample_count; sample_index++) {
        size_t output_index;

        if (fprintf(stream, "sample %zu", first_sample + sample_index) < 0) {
            return 0;
        }
        for (output_index = 0U;
             output_index < snapshot->output_count;
             output_index++) {
            if (fprintf(stream,
                        " %.*g",
                        DBL_DECIMAL_DIG,
                        outputs[sample_index * snapshot->output_count +
                                output_index]) < 0) {
                return 0;
            }
        }
        if (fputc('\n', stream) == EOF) {
            return 0;
        }
    }
    return 1;
}

static int copy_stream(FILE *source, FILE *destination, NeuralError *error)
{
    unsigned char buffer[8192];

    if (fflush(source) != 0 || fseek(source, 0L, SEEK_SET) != 0) {
        neural_error_set(error, "unable to finalize prediction output");
        return 0;
    }
    for (;;) {
        size_t count = fread(buffer, 1U, sizeof(buffer), source);

        if (count != 0U && fwrite(buffer, 1U, count, destination) != count) {
            neural_error_set(error, "unable to write prediction output");
            return 0;
        }
        if (count < sizeof(buffer)) {
            if (ferror(source) != 0) {
                neural_error_set(error, "unable to read staged prediction output");
                return 0;
            }
            break;
        }
    }
    return 1;
}

static int command_predict_document(
    const char *project,
    const NeuralParsedOptions *options,
    const NeuralExecutionConfig *execution,
    NeuralPredictionSnapshot *snapshot,
    NeuralError *error)
{
    NeuralInputDocument *document = NULL;
    neural_real *inputs = NULL;
    neural_real *outputs = NULL;
    FILE *staged_output = NULL;
    size_t batch_size = NEURAL_DEFAULT_PREDICTION_BATCH_SIZE;
    size_t total_samples;
    size_t first_sample = 0U;
    size_t worker_count;
    int complete = 0;
    int result = EXIT_RUNTIME;

    (void)project;
    if (options->positional_count != 2U) {
        fprintf(stderr,
                "%s: --input cannot be combined with positional samples\n",
                neural_name());
        return EXIT_USAGE;
    }
    if (neural_option_is_present(options, OPTION_BATCH_SIZE) &&
        (!neural_parse_size(neural_option_value(options, OPTION_BATCH_SIZE),
                            &batch_size) ||
         batch_size == 0U)) {
        fprintf(stderr, "%s: batch-size must be a positive integer\n",
                neural_name());
        return EXIT_USAGE;
    }
    if (!neural_input_document_open(
            neural_option_value(options, OPTION_INPUT_FILE),
            &document,
            error)) {
        goto cleanup;
    }
    total_samples = neural_input_document_sample_count(document);
    if (neural_input_document_input_count(document) != snapshot->input_count) {
        neural_error_set(error,
                         "input document width %zu does not match model width %zu",
                         neural_input_document_input_count(document),
                         snapshot->input_count);
        goto cleanup;
    }
    if (batch_size > total_samples) {
        batch_size = total_samples;
    }
    if (batch_size > SIZE_MAX / snapshot->input_count ||
        batch_size * snapshot->input_count > SIZE_MAX / sizeof(*inputs) ||
        batch_size > SIZE_MAX / snapshot->output_count ||
        batch_size * snapshot->output_count > SIZE_MAX / sizeof(*outputs)) {
        neural_error_set(error, "prediction batch dimensions are too large");
        goto cleanup;
    }
    inputs = malloc(batch_size * snapshot->input_count * sizeof(*inputs));
    outputs = malloc(batch_size * snapshot->output_count * sizeof(*outputs));
    staged_output = tmpfile();
    if (inputs == NULL || outputs == NULL || staged_output == NULL) {
        neural_error_set(error, "unable to allocate streamed prediction state");
        goto cleanup;
    }
    if (!write_prediction_header(staged_output, snapshot, total_samples)) {
        neural_error_set(error, "unable to stage prediction output");
        goto cleanup;
    }
    while (!complete) {
        size_t loaded = 0U;

        if (!neural_input_document_read(document,
                                        inputs,
                                        batch_size,
                                        &loaded,
                                        &complete,
                                        error) ||
            loaded == 0U ||
            !neural_prediction_prepare_inputs(snapshot,
                                               inputs,
                                               loaded,
                                               error) ||
            !neural_prediction_run(snapshot,
                                   inputs,
                                   loaded,
                                   execution,
                                   outputs,
                                   &worker_count,
                                   error) ||
            !write_prediction_samples(staged_output,
                                      snapshot,
                                      outputs,
                                      first_sample,
                                      loaded)) {
            if (error->message[0] == '\0') {
                neural_error_set(error, "unable to stage prediction samples");
            }
            goto cleanup;
        }
        first_sample += loaded;
    }
    if (first_sample != total_samples || fputs("end\n", staged_output) == EOF ||
        !copy_stream(staged_output, stdout, error)) {
        if (error->message[0] == '\0') {
            neural_error_set(error, "prediction document was incomplete");
        }
        goto cleanup;
    }
    result = EXIT_OK;

cleanup:
    if (result != EXIT_OK && error->message[0] != '\0') {
        fprintf(stderr, "%s: %s\n", neural_name(), error->message);
    }
    if (staged_output != NULL) {
        (void)fclose(staged_output);
    }
    free(outputs);
    free(inputs);
    neural_input_document_close(document);
    return result;
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
    if (neural_option_is_present(options, OPTION_INPUT_FILE)) {
        result = command_predict_document(project,
                                          options,
                                          &execution,
                                          &snapshot,
                                          &error);
        goto cleanup;
    }
    if (neural_option_is_present(options, OPTION_BATCH_SIZE)) {
        fprintf(stderr, "%s: --batch-size requires --input\n", neural_name());
        result = EXIT_USAGE;
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

        if (strcmp(text, "?") == 0) {
            inputs[value_index] = NAN;
        } else if (!neural_parse_real(text, &inputs[value_index])) {
            fprintf(stderr,
                    "%s: invalid prediction input %zu: '%s'\n",
                    neural_name(),
                    value_index,
                    text);
            result = EXIT_USAGE;
            goto cleanup;
        }
    }
    if (!neural_prediction_prepare_inputs(&snapshot,
                                          inputs,
                                          sample_count,
                                          &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        goto cleanup;
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
    if (!write_prediction_header(stdout, &snapshot, sample_count) ||
        !write_prediction_samples(stdout,
                                  &snapshot,
                                  outputs,
                                  0U,
                                  sample_count) ||
        fputs("end\n", stdout) == EOF) {
        fprintf(stderr, "%s: unable to write prediction output\n", neural_name());
        goto cleanup;
    }
    (void)worker_count;
    result = EXIT_OK;

cleanup:
    free(outputs);
    free(inputs);
    neural_prediction_snapshot_free(&snapshot);
    return result;
}

static int command_import_csv(const char *project,
                              const char *csv_path,
                              const NeuralParsedOptions *options)
{
    NeuralDataImportConfig config = {
        NULL,
        0.0,
        0.0,
        UINT64_C(0),
        NEURAL_NORMALIZATION_NONE,
        NEURAL_MISSING_REJECT
    };
    NeuralDataImportResult imported;
    NeuralError error;

    if (!neural_option_is_present(options, OPTION_SCHEMA)) {
        fprintf(stderr, "%s: import-csv requires --schema FILE\n",
                neural_name());
        return EXIT_USAGE;
    }
    config.schema_path = neural_option_value(options, OPTION_SCHEMA);
    if (neural_option_is_present(options, OPTION_VALIDATION_RATIO) &&
        !neural_parse_real(neural_option_value(options,
                                               OPTION_VALIDATION_RATIO),
                           &config.validation_ratio)) {
        fprintf(stderr, "%s: validation-ratio must be finite\n", neural_name());
        return EXIT_USAGE;
    }
    if (neural_option_is_present(options, OPTION_TEST_RATIO) &&
        !neural_parse_real(neural_option_value(options, OPTION_TEST_RATIO),
                           &config.test_ratio)) {
        fprintf(stderr, "%s: test-ratio must be finite\n", neural_name());
        return EXIT_USAGE;
    }
    if (neural_option_is_present(options, OPTION_SPLIT_SEED) &&
        !neural_parse_uint64(neural_option_value(options, OPTION_SPLIT_SEED),
                             &config.split_seed)) {
        fprintf(stderr, "%s: split-seed must be an unsigned integer\n",
                neural_name());
        return EXIT_USAGE;
    }
    if (neural_option_is_present(options, OPTION_NORMALIZATION) &&
        !neural_normalization_from_name(
            neural_option_value(options, OPTION_NORMALIZATION),
            &config.normalization)) {
        fprintf(stderr,
                "%s: normalization must be none, standardize, or minmax\n",
                neural_name());
        return EXIT_USAGE;
    }
    if (neural_option_is_present(options, OPTION_MISSING) &&
        !neural_missing_policy_from_name(
            neural_option_value(options, OPTION_MISSING),
            &config.missing_policy)) {
        fprintf(stderr, "%s: missing must be reject or mean\n", neural_name());
        return EXIT_USAGE;
    }
    if (config.validation_ratio < 0.0 || config.test_ratio < 0.0 ||
        config.validation_ratio + config.test_ratio >= 1.0) {
        fprintf(stderr,
                "%s: split ratios must be non-negative and sum to less than 1\n",
                neural_name());
        return EXIT_USAGE;
    }
    if (!neural_data_import_csv(project,
                                csv_path,
                                &config,
                                &imported,
                                &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        return EXIT_RUNTIME;
    }
    printf("CSV import complete: %zu samples, train %zu, validation %zu, "
           "test %zu, stratified %s\n",
           imported.total_samples,
           imported.training_samples,
           imported.validation_samples,
           imported.test_samples,
           imported.stratified ? "yes" : "no");
    return EXIT_OK;
}

static int command_evaluate(const char *project,
                            const NeuralParsedOptions *options)
{
    NeuralExecutionConfig execution = {NEURAL_DEFAULT_THREAD_COUNT};
    NeuralEvaluationSnapshot snapshot = NEURAL_EVALUATION_SNAPSHOT_INITIALIZER;
    NeuralEvaluationResult evaluation = {0};
    neural_real *outputs = NULL;
    NeuralError error;
    const char *dataset_name = neural_option_is_present(options, OPTION_DATASET)
                                   ? neural_option_value(options, OPTION_DATASET)
                                   : "test";
    const char *dataset_filename;
    size_t output_value_count;
    size_t worker_count;
    size_t class_index;
    int result = EXIT_RUNTIME;

    if (strcmp(dataset_name, "train") == 0) {
        dataset_filename = NEURAL_DEFAULT_DATASET_FILENAME;
    } else if (strcmp(dataset_name, "validation") == 0) {
        dataset_filename = NEURAL_DEFAULT_VALIDATION_FILENAME;
    } else if (strcmp(dataset_name, "test") == 0) {
        dataset_filename = NEURAL_DEFAULT_TEST_FILENAME;
    } else {
        fprintf(stderr,
                "%s: dataset must be train, validation, or test\n",
                neural_name());
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
    if (!neural_project_evaluation_load(project,
                                        dataset_filename,
                                        &snapshot,
                                        &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        goto cleanup;
    }
    if (snapshot.dataset.sample_count > SIZE_MAX /
                                            snapshot.dataset.output_count ||
        snapshot.dataset.sample_count * snapshot.dataset.output_count >
            SIZE_MAX / sizeof(*outputs)) {
        fprintf(stderr, "%s: evaluation dataset is too large\n", neural_name());
        goto cleanup;
    }
    output_value_count = snapshot.dataset.sample_count *
                         snapshot.dataset.output_count;
    outputs = malloc(output_value_count * sizeof(*outputs));
    if (outputs == NULL) {
        fprintf(stderr, "%s: unable to allocate evaluation outputs\n",
                neural_name());
        goto cleanup;
    }
    if (!neural_prediction_run(&snapshot.prediction,
                               snapshot.dataset.inputs,
                               snapshot.dataset.sample_count,
                               &execution,
                               outputs,
                               &worker_count,
                               &error) ||
        !neural_evaluation_compute(snapshot.loss,
                                   snapshot.output_activation,
                                   outputs,
                                   snapshot.dataset.outputs,
                                   snapshot.dataset.sample_count,
                                   snapshot.dataset.output_count,
                                   &evaluation,
                                   &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        goto cleanup;
    }
    printf("%s evaluation %zu\n",
           NEURAL_FORMAT_MAGIC,
           snapshot.prediction.format_version == 2U
               ? (size_t)2U : (size_t)1U);
    printf("completed_epochs %zu\n", snapshot.prediction.completed_epochs);
    if (snapshot.prediction.format_version == 2U) {
        printf("selected_epoch %zu\n", snapshot.prediction.selected_epoch);
        printf("target_epochs %zu\n", snapshot.prediction.target_epochs);
        printf("completion %s\n",
               snapshot.prediction.completion_reason ==
                       NEURAL_COMPLETION_EARLY_STOPPING
                   ? "early_stopping" : "target");
    }
    printf("dataset %s\n", dataset_name);
    printf("samples %zu\n", snapshot.dataset.sample_count);
    printf("inputs %zu\n", snapshot.dataset.input_count);
    printf("outputs %zu\n", snapshot.dataset.output_count);
    printf("loss %.*g\n", DBL_DECIMAL_DIG, evaluation.loss);
    printf("classification %s\n",
           evaluation.is_classification ? "yes" : "no");
    if (evaluation.is_classification) {
        printf("correct %zu\n", evaluation.correct_count);
        printf("accuracy %.*g\n", DBL_DECIMAL_DIG, evaluation.accuracy);
        printf("classes %zu\n", evaluation.class_count);
        for (class_index = 0U;
             class_index < evaluation.class_count;
             class_index++) {
            size_t predicted_class;

            printf("confusion %zu", class_index);
            for (predicted_class = 0U;
                 predicted_class < evaluation.class_count;
                 predicted_class++) {
                printf(" %zu",
                       evaluation.confusion[
                           class_index * evaluation.class_count +
                           predicted_class]);
            }
            putchar('\n');
        }
        for (class_index = 0U;
             class_index < evaluation.class_count;
             class_index++) {
            printf("class %zu precision %.*g recall %.*g f1 %.*g\n",
                   class_index,
                   DBL_DECIMAL_DIG,
                   evaluation.precision[class_index],
                   DBL_DECIMAL_DIG,
                   evaluation.recall[class_index],
                   DBL_DECIMAL_DIG,
                   evaluation.f1[class_index]);
        }
    }
    puts("end");
    (void)worker_count;
    result = EXIT_OK;

cleanup:
    neural_evaluation_result_free(&evaluation);
    free(outputs);
    neural_evaluation_snapshot_free(&snapshot);
    return result;
}

static int inspect_path_exists(const char *path,
                               int *exists,
                               NeuralError *error)
{
    struct stat status;

    if (lstat(path, &status) == 0) {
        *exists = 1;
        return 1;
    }
    if (errno == ENOENT) {
        *exists = 0;
        return 1;
    }
    neural_error_set(error,
                     "%s: unable to inspect state: %s",
                     path,
                     strerror(errno));
    return 0;
}

static int print_project_state(const char *directory,
                               const NeuralProject *project,
                               const NeuralProjectDigests *digests,
                               NeuralError *error)
{
    NeuralWeightsMetadata weights_metadata;
    NeuralCheckpointMetadata checkpoint_metadata;
    NeuralModel *weights_model = NULL;
    NeuralModel *checkpoint_model = NULL;
    NeuralModel *best_model = NULL;
    char *weights_path = NULL;
    char *checkpoint_path = NULL;
    int weights_exists = 0;
    int checkpoint_exists = 0;
    int success = 0;

    weights_path = neural_path_join(directory,
                                    NEURAL_DEFAULT_WEIGHTS_FILENAME,
                                    error);
    checkpoint_path = neural_path_join(directory,
                                       NEURAL_DEFAULT_CHECKPOINT_FILENAME,
                                       error);
    if (weights_path == NULL || checkpoint_path == NULL ||
        !inspect_path_exists(weights_path, &weights_exists, error) ||
        !inspect_path_exists(checkpoint_path, &checkpoint_exists, error)) {
        goto cleanup;
    }
    if (weights_exists &&
        (!neural_model_create(&project->model,
                              project->training.seed,
                              &weights_model,
                              error) ||
         !neural_weights_load(weights_path,
                              weights_model,
                              digests,
                              &weights_metadata,
                              error))) {
        goto cleanup;
    }
    if (checkpoint_exists) {
        if (!neural_model_create(&project->model,
                                 project->training.seed,
                                 &checkpoint_model,
                                 error)) {
            goto cleanup;
        }
        if (project->training.early_stopping_patience != 0U) {
            if (!neural_model_create(&project->model,
                                     project->training.seed,
                                     &best_model,
                                     error) ||
                !neural_early_checkpoint_load(checkpoint_path,
                                              checkpoint_model,
                                              best_model,
                                              digests,
                                              &checkpoint_metadata,
                                              error)) {
                goto cleanup;
            }
        } else if (!neural_checkpoint_load(checkpoint_path,
                                           checkpoint_model,
                                           digests,
                                           &checkpoint_metadata,
                                           error)) {
            goto cleanup;
        }
    }
    printf("Weights state: %s\n", weights_exists ? "present" : "absent");
    if (weights_exists) {
        printf("Weights completed epochs: %zu\n",
               weights_metadata.completed_epochs);
        printf("Weights selected epoch: %zu\n",
               weights_metadata.selected_epoch);
        printf("Weights target epochs: %zu\n",
               weights_metadata.target_epochs);
        printf("Weights completion: %s\n",
               weights_metadata.completion_reason ==
                       NEURAL_COMPLETION_EARLY_STOPPING
                   ? "early_stopping" : "target");
    }
    printf("Checkpoint state: %s\n",
           checkpoint_exists ? "present" : "absent");
    if (checkpoint_exists) {
        printf("Checkpoint completed epochs: %zu\n",
               checkpoint_metadata.completed_epochs);
        printf("Checkpoint target epochs: %zu\n",
               checkpoint_metadata.target_epochs);
        if (checkpoint_metadata.format_version == 2U) {
            printf("Checkpoint best epoch: %zu\n",
                   checkpoint_metadata.best_epoch);
            printf("Checkpoint best validation loss: %.*g\n",
                   DBL_DECIMAL_DIG,
                   checkpoint_metadata.best_loss);
            printf("Checkpoint stale epochs: %zu\n",
                   checkpoint_metadata.stale_epochs);
        }
    }
    success = 1;

cleanup:
    free(checkpoint_path);
    free(weights_path);
    neural_model_free(checkpoint_model);
    neural_model_free(best_model);
    neural_model_free(weights_model);
    return success;
}

static int command_inspect(const char *directory, int include_state)
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
    printf("Early stopping patience: %zu\n",
           project.training.early_stopping_patience);
    printf("Early stopping min delta: %.*g\n",
           DBL_DECIMAL_DIG,
           project.training.early_stopping_min_delta);
    if (project.has_validation) {
        printf("Validation samples: %zu\n",
               project.validation.sample_count);
    }
    if (project.has_preprocessing) {
        printf("Preprocessing: present\n");
        printf("Normalization: %s\n",
               neural_normalization_name(project.preprocessing.normalization));
        printf("Missing policy: %s\n",
               neural_missing_policy_name(project.preprocessing.missing_policy));
        printf("Split seed: %" PRIu64 "\n",
               project.preprocessing.split_seed);
        printf("Validation ratio: %.*g\n",
               DBL_DECIMAL_DIG,
               project.preprocessing.validation_ratio);
        printf("Test ratio: %.*g\n",
               DBL_DECIMAL_DIG,
               project.preprocessing.test_ratio);
        printf("Stratified split: %s\n",
               project.preprocessing.stratified ? "yes" : "no");
    } else {
        printf("Preprocessing: absent (identity)\n");
    }
    printf("Model digest: sha256:%s\n", digests.model);
    printf("Dataset digest: sha256:%s\n", digests.dataset);
    printf("Training digest: sha256:%s\n", digests.training);
    if (include_state &&
        !print_project_state(directory, &project, &digests, &error)) {
        fprintf(stderr, "%s: %s\n", neural_name(), error.message);
        neural_model_free(runtime_model);
        neural_project_free(&project);
        neural_project_lock_release(&project_lock);
        return EXIT_USAGE;
    }
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
    if ((neural_option_is_present(&options, OPTION_INPUT_FILE) ||
         neural_option_is_present(&options, OPTION_BATCH_SIZE)) &&
        strcmp(command, "predict") != 0) {
        fprintf(stderr,
                "%s: --input and --batch-size are valid only with predict\n",
                neural_name());
        neural_options_free(&options);
        return EXIT_USAGE;
    }
    if (strcmp(command, "import-csv") == 0) {
        int result;

        if (options.positional_count != 3U) {
            fprintf(stderr,
                    "%s: import-csv requires a project directory and CSV file\n",
                    neural_name());
            neural_options_free(&options);
            return EXIT_USAGE;
        }
        if (has_options_in_range(&options, OPTION_RESUME, OPTION_SCHEMA)) {
            fprintf(stderr,
                    "%s: training and execution options are invalid with import-csv\n",
                    neural_name());
            neural_options_free(&options);
            return EXIT_USAGE;
        }
        result = command_import_csv(options.positionals[1],
                                    options.positionals[2],
                                    &options);
        neural_options_free(&options);
        return result;
    }
    if (has_options_in_range(&options, OPTION_SCHEMA, OPTION_COUNT)) {
        fprintf(stderr, "%s: CSV import options are valid only with import-csv\n",
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
    if (neural_option_is_present(&options, OPTION_REPORT_INTERVAL)) {
        fprintf(stderr,
                "%s: --report-interval is valid only with train\n",
                neural_name());
        neural_options_free(&options);
        return EXIT_USAGE;
    }
    if (neural_option_is_present(&options, OPTION_HISTORY)) {
        fprintf(stderr,
                "%s: --history is valid only with train\n",
                neural_name());
        neural_options_free(&options);
        return EXIT_USAGE;
    }
    if (neural_option_is_present(&options, OPTION_STATE) &&
        strcmp(command, "inspect") != 0) {
        fprintf(stderr,
                "%s: --state is valid only with inspect\n",
                neural_name());
        neural_options_free(&options);
        return EXIT_USAGE;
    }
    if (strcmp(command, "evaluate") == 0) {
        int result;

        if (options.positional_count != 2U) {
            fprintf(stderr,
                    "%s: evaluate requires one project directory\n",
                    neural_name());
            neural_options_free(&options);
            return EXIT_USAGE;
        }
        result = command_evaluate(options.positionals[1], &options);
        neural_options_free(&options);
        return result;
    }
    if (neural_option_is_present(&options, OPTION_DATASET)) {
        fprintf(stderr,
                "%s: --dataset is valid only with evaluate\n",
                neural_name());
        neural_options_free(&options);
        return EXIT_USAGE;
    }
    if (strcmp(command, "inspect") == 0) {
        int result;
        if (neural_option_is_present(&options, OPTION_THREADS)) {
            fprintf(stderr,
                    "%s: --threads is valid only with train, predict, or evaluate\n",
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
        result = command_inspect(
            options.positionals[1],
            neural_option_is_present(&options, OPTION_STATE));
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
