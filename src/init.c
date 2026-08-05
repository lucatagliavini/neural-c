#define _POSIX_C_SOURCE 200809L

#include "neural/init.h"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "neural/defaults.h"
#include "neural/loss.h"
#include "neural/version.h"
#include "path.h"
#include "project_lock.h"

enum managed_file_index {
    MANAGED_MODEL,
    MANAGED_PROJECT,
    MANAGED_DATASET,
    MANAGED_WEIGHTS,
    MANAGED_CHECKPOINT,
    MANAGED_PREPROCESSING,
    MANAGED_FILE_COUNT
};

typedef struct {
    char *final_path;
    char *new_path;
    char *old_path;
    int has_original;
    int new_created;
    int committed;
} ManagedFile;

static int write_model(FILE *stream, const NeuralModelSpec *model)
{
    size_t layer_index;

    if (fprintf(stream,
                "%s model %d\n\ninput %zu\n",
                NEURAL_FORMAT_MAGIC,
                NEURAL_FORMAT_VERSION,
                model->input_count) < 0) {
        return 0;
    }
    for (layer_index = 0U; layer_index < model->layer_count; layer_index++) {
        const NeuralLayerSpec *layer = &model->layers[layer_index];
        size_t parameter_index;

        if (fprintf(stream,
                    "dense %zu %s",
                    layer->neuron_count,
                    neural_activation_kind_name(layer->activation.kind)) < 0) {
            return 0;
        }
        for (parameter_index = 0U;
             parameter_index < layer->activation.parameter_count;
             parameter_index++) {
            const NeuralActivationParameter *parameter =
                &layer->activation.parameters[parameter_index];
            if (fprintf(stream,
                        " %s=%.*g",
                        neural_activation_parameter_name(parameter->kind),
                        DBL_DECIMAL_DIG,
                        parameter->value) < 0) {
                return 0;
            }
        }
        if (fputc('\n', stream) == EOF) {
            return 0;
        }
    }
    return 1;
}

static int write_project(FILE *stream, const NeuralTrainingConfig *training)
{
    return fprintf(stream,
                   "%s project %d\n\n"
                   "epochs %zu\n"
                   "learning_rate %.*g\n"
                   "seed %" PRIu64 "\n"
                   "loss %s\n"
                   "checkpoint_interval %zu\n"
                   "early_stopping_patience %zu\n"
                   "early_stopping_min_delta %.*g\n"
                   "batch_size %zu\n"
                   "shuffle %d\n"
                   "gradient_clip_norm %.*g\n",
                   NEURAL_FORMAT_MAGIC,
                   NEURAL_FORMAT_VERSION,
                   training->epochs,
                   DBL_DECIMAL_DIG,
                   training->learning_rate,
                   training->seed,
                   neural_loss_name(training->loss),
                   training->checkpoint_interval,
                   training->early_stopping_patience,
                   DBL_DECIMAL_DIG,
                   training->early_stopping_min_delta,
                   training->batch_size,
                   training->shuffle,
                   DBL_DECIMAL_DIG,
                   training->gradient_clip_norm) >= 0;
}

static int write_dataset(FILE *stream, const NeuralModelSpec *model)
{
    size_t output_count = model->layers[model->layer_count - 1U].neuron_count;

    return fprintf(stream,
                   "%s dataset %d\n\n"
                   "# Add one sample per line using this shape:\n"
                   "# <%zu inputs> -> <%zu outputs>\n",
                   NEURAL_FORMAT_MAGIC,
                   NEURAL_FORMAT_VERSION,
                   model->input_count,
                   output_count) >= 0;
}

static int write_new_file(ManagedFile *file,
                          enum managed_file_index index,
                          const NeuralModelSpec *model,
                          const NeuralTrainingConfig *training,
                          NeuralError *error)
{
    FILE *stream = fopen(file->new_path, "wx");
    int written;
    int close_status;

    if (stream == NULL) {
        neural_error_set(error,
                         "%s: unable to create temporary file: %s",
                         file->new_path,
                         strerror(errno));
        return 0;
    }
    if (index == MANAGED_MODEL) {
        written = write_model(stream, model);
    } else if (index == MANAGED_PROJECT) {
        written = write_project(stream, training);
    } else {
        written = write_dataset(stream, model);
    }
    close_status = fclose(stream);
    if (!written || close_status != 0) {
        neural_error_set(error,
                         "%s: unable to write temporary file",
                         file->new_path);
        (void)remove(file->new_path);
        return 0;
    }
    file->new_created = 1;
    return 1;
}

static void remove_new_files(ManagedFile *files)
{
    size_t index;

    for (index = 0U; index < MANAGED_FILE_COUNT; index++) {
        if (files[index].new_created && files[index].new_path != NULL) {
            (void)remove(files[index].new_path);
        }
    }
}

static void restore_originals(ManagedFile *files)
{
    size_t index;

    for (index = 0U; index < MANAGED_FILE_COUNT; index++) {
        if (files[index].committed) {
            (void)remove(files[index].final_path);
        }
    }
    for (index = 0U; index < MANAGED_FILE_COUNT; index++) {
        if (files[index].has_original) {
            (void)rename(files[index].old_path, files[index].final_path);
        }
    }
}

static void free_managed_files(ManagedFile *files)
{
    size_t index;

    for (index = 0U; index < MANAGED_FILE_COUNT; index++) {
        free(files[index].final_path);
        free(files[index].new_path);
        free(files[index].old_path);
    }
}

static int require_absent(const char *path,
                          const char *directory,
                          NeuralError *error)
{
    struct stat status;

    if (lstat(path, &status) == 0) {
        neural_error_set(error,
                         "%s: stale initialization file already exists",
                         directory);
        return 0;
    }
    if (errno != ENOENT) {
        neural_error_set(error,
                         "%s: unable to inspect initialization files: %s",
                         directory,
                         strerror(errno));
        return 0;
    }
    return 1;
}

static int prepare_paths(const char *directory,
                         ManagedFile *files,
                         NeuralError *error)
{
    static const char *const final_names[MANAGED_FILE_COUNT] = {
        NEURAL_DEFAULT_MODEL_FILENAME,
        NEURAL_DEFAULT_PROJECT_FILENAME,
        NEURAL_DEFAULT_DATASET_FILENAME,
        NEURAL_DEFAULT_WEIGHTS_FILENAME,
        NEURAL_DEFAULT_CHECKPOINT_FILENAME,
        NEURAL_DEFAULT_PREPROCESSING_FILENAME
    };
    static const char *const new_names[MANAGED_FILE_COUNT] = {
        ".neural-c-model.new",
        ".neural-c-project.new",
        ".neural-c-dataset.new",
        ".neural-c-weights.new",
        ".neural-c-checkpoint.new",
        ".neural-c-preprocessing.new"
    };
    static const char *const old_names[MANAGED_FILE_COUNT] = {
        ".neural-c-model.old",
        ".neural-c-project.old",
        ".neural-c-dataset.old",
        ".neural-c-weights.old",
        ".neural-c-checkpoint.old",
        ".neural-c-preprocessing.old"
    };
    size_t index;

    for (index = 0U; index < MANAGED_FILE_COUNT; index++) {
        struct stat status;

        files[index].final_path = neural_path_join(directory,
                                                   final_names[index],
                                                   error);
        files[index].new_path = neural_path_join(directory,
                                                 new_names[index],
                                                 error);
        files[index].old_path = neural_path_join(directory,
                                                 old_names[index],
                                                 error);
        if (files[index].final_path == NULL ||
            files[index].new_path == NULL || files[index].old_path == NULL) {
            return 0;
        }
        if (!require_absent(files[index].new_path, directory, error) ||
            !require_absent(files[index].old_path, directory, error)) {
            return 0;
        }
        if (lstat(files[index].final_path, &status) == 0) {
            if (S_ISDIR(status.st_mode)) {
                neural_error_set(error,
                                 "%s: managed path is a directory",
                                 files[index].final_path);
                return 0;
            }
            files[index].has_original = 1;
        } else if (errno != ENOENT) {
            neural_error_set(error,
                             "%s: unable to inspect file: %s",
                             files[index].final_path,
                             strerror(errno));
            return 0;
        }
    }
    return 1;
}

int neural_project_initialize(const char *directory,
                              const NeuralModelSpec *model,
                              const NeuralTrainingConfig *training,
                              int force,
                              NeuralError *error)
{
    ManagedFile files[MANAGED_FILE_COUNT] = {{0}};
    NeuralProjectLock project_lock = NEURAL_PROJECT_LOCK_INITIALIZER;
    struct stat directory_status;
    int directory_created = 0;
    int directory_exists;
    size_t index;
    int success = 0;
    locale_t c_numeric_locale = (locale_t)0;
    locale_t previous_locale = (locale_t)0;
    int locale_active = 0;

    neural_error_clear(error);
    if (directory == NULL || directory[0] == '\0') {
        neural_error_set(error, "project directory must not be empty");
        return 0;
    }
    if (!neural_model_spec_validate(model, error) ||
        !neural_training_config_validate(training, error) ||
        !neural_loss_validate_output(
            training->loss,
            model->layers[model->layer_count - 1U].activation.kind,
            model->layers[model->layer_count - 1U].neuron_count,
            error)) {
        return 0;
    }

    directory_exists = lstat(directory, &directory_status) == 0;
    if (!directory_exists && errno != ENOENT) {
        neural_error_set(error,
                         "%s: unable to inspect project directory: %s",
                         directory,
                         strerror(errno));
        return 0;
    }
    if (directory_exists) {
        if (!S_ISDIR(directory_status.st_mode)) {
            neural_error_set(error,
                             "%s: project path is not a directory",
                             directory);
            return 0;
        }
        if (!force) {
            neural_error_set(error,
                             "%s: directory already exists; use --force to overwrite it",
                             directory);
            return 0;
        }
    } else {
        if (mkdir(directory, 0777) != 0) {
            neural_error_set(error,
                             "%s: unable to create project directory: %s",
                             directory,
                             strerror(errno));
            return 0;
        }
        directory_created = 1;
    }

    if (!neural_project_lock_acquire(directory,
                                     NEURAL_PROJECT_LOCK_EXCLUSIVE,
                                     &project_lock,
                                     error) ||
        !prepare_paths(directory, files, error)) {
        goto cleanup;
    }
    c_numeric_locale = newlocale(LC_NUMERIC_MASK, "C", (locale_t)0);
    if (c_numeric_locale == (locale_t)0) {
        neural_error_set(error,
                         "%s: unable to create numeric locale: %s",
                         directory,
                         strerror(errno));
        goto cleanup;
    }
    previous_locale = uselocale(c_numeric_locale);
    if (previous_locale == (locale_t)0) {
        neural_error_set(error,
                         "%s: unable to select numeric locale: %s",
                         directory,
                         strerror(errno));
        goto cleanup;
    }
    locale_active = 1;
    for (index = 0U; index < MANAGED_WEIGHTS; index++) {
        if (!write_new_file(&files[index],
                            (enum managed_file_index)index,
                            model,
                            training,
                            error)) {
            goto cleanup;
        }
    }
    if (uselocale(previous_locale) == (locale_t)0) {
        locale_active = 0;
        /* The C locale may still be active, so it must remain allocated. */
        c_numeric_locale = (locale_t)0;
        neural_error_set(error,
                         "%s: unable to restore numeric locale: %s",
                         directory,
                         strerror(errno));
        goto cleanup;
    }
    locale_active = 0;
    freelocale(c_numeric_locale);
    c_numeric_locale = (locale_t)0;

    for (index = 0U; index < MANAGED_FILE_COUNT; index++) {
        if (files[index].has_original &&
            rename(files[index].final_path, files[index].old_path) != 0) {
            neural_error_set(error,
                             "%s: unable to stage existing file: %s",
                             files[index].final_path,
                             strerror(errno));
            restore_originals(files);
            goto cleanup;
        }
    }
    for (index = 0U; index < MANAGED_WEIGHTS; index++) {
        if (rename(files[index].new_path, files[index].final_path) != 0) {
            neural_error_set(error,
                             "%s: unable to install generated file: %s",
                             files[index].final_path,
                             strerror(errno));
            restore_originals(files);
            goto cleanup;
        }
        files[index].new_created = 0;
        files[index].committed = 1;
    }

    for (index = 0U; index < MANAGED_FILE_COUNT; index++) {
        if (files[index].has_original) {
            (void)remove(files[index].old_path);
        }
    }
    success = 1;

cleanup:
    if (locale_active) {
        if (uselocale(previous_locale) == (locale_t)0) {
            /* Do not free a locale that may still be active. */
            c_numeric_locale = (locale_t)0;
        }
    }
    if (c_numeric_locale != (locale_t)0) {
        freelocale(c_numeric_locale);
    }
    remove_new_files(files);
    free_managed_files(files);
    if (!success && directory_created) {
        neural_project_lock_discard(&project_lock);
        (void)rmdir(directory);
    } else {
        neural_project_lock_release(&project_lock);
    }
    return success;
}
