#define _POSIX_C_SOURCE 200809L

#include "train_project.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "neural/defaults.h"
#include "neural/digest.h"
#include "neural/model.h"
#include "neural/persistence.h"
#include "neural/project.h"
#include "path.h"

static int persistence_path_is_absent(const char *path, NeuralError *error)
{
    struct stat status;

    if (lstat(path, &status) == 0) {
        neural_error_set(error,
                         "%s already exists; fresh training requires no saved state",
                         path);
        return 0;
    }
    if (errno != ENOENT) {
        neural_error_set(error,
                         "%s: unable to inspect saved state: %s",
                         path,
                         strerror(errno));
        return 0;
    }
    return 1;
}

int neural_project_train_fresh(const char *directory,
                               const NeuralExecutionConfig *execution,
                               NeuralTrainingResult *result,
                               NeuralError *error)
{
    NeuralProject project;
    NeuralModel *model = NULL;
    NeuralProjectDigests digests;
    NeuralWeightsMetadata metadata;
    NeuralTrainingResult completed = {0U, 0U, 0.0};
    char *weights_path = NULL;
    char *checkpoint_path = NULL;
    int project_loaded = 0;
    int success = 0;

    neural_error_clear(error);
    if (directory == NULL || directory[0] == '\0' || execution == NULL ||
        result == NULL) {
        neural_error_set(error, "fresh project training arguments are required");
        return 0;
    }
    *result = completed;
    weights_path = neural_path_join(directory,
                                    NEURAL_DEFAULT_WEIGHTS_FILENAME,
                                    error);
    checkpoint_path = neural_path_join(directory,
                                       NEURAL_DEFAULT_CHECKPOINT_FILENAME,
                                       error);
    if (weights_path == NULL || checkpoint_path == NULL ||
        !persistence_path_is_absent(weights_path, error) ||
        !persistence_path_is_absent(checkpoint_path, error) ||
        !neural_project_load(directory, &project, error)) {
        goto cleanup;
    }
    project_loaded = 1;
    if (!neural_project_digests_compute(&project, &digests, error) ||
        !neural_model_create(&project.model,
                             project.training.seed,
                             &model,
                             error) ||
        !neural_model_train_full_batch(model,
                                       &project.dataset,
                                       &project.training,
                                       execution,
                                       NULL,
                                       NULL,
                                       &completed,
                                       error)) {
        goto cleanup;
    }
    if (!persistence_path_is_absent(weights_path, error) ||
        !persistence_path_is_absent(checkpoint_path, error)) {
        goto cleanup;
    }
    metadata.completed_epochs = completed.completed_epochs;
    metadata.digests = digests;
    if (!neural_weights_save_atomic(weights_path,
                                    model,
                                    &metadata,
                                    error)) {
        goto cleanup;
    }
    *result = completed;
    success = 1;

cleanup:
    neural_model_free(model);
    if (project_loaded) {
        neural_project_free(&project);
    }
    free(checkpoint_path);
    free(weights_path);
    return success;
}
