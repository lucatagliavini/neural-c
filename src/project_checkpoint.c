#include "project_checkpoint.h"

#include <string.h>

#include "neural/persistence.h"

int neural_project_checkpoint_observer_initialize(
    NeuralProjectCheckpointObserver *observer,
    const char *path,
    const NeuralModel *model,
    const NeuralProjectDigests *digests,
    NeuralOptimizerKind optimizer,
    size_t interval,
    size_t target_epochs,
    NeuralError *error)
{
    if (observer == NULL || path == NULL || path[0] == '\0' ||
        model == NULL || digests == NULL || target_epochs == 0U ||
        strcmp(neural_optimizer_name(optimizer), "unknown") == 0) {
        neural_error_set(error,
                         "periodic checkpoint configuration is invalid");
        return 0;
    }
    memset(observer, 0, sizeof(*observer));
    observer->path = path;
    observer->model = model;
    observer->interval = interval;
    observer->metadata.target_epochs = target_epochs;
    observer->metadata.optimizer = optimizer;
    observer->metadata.digests = *digests;
    return 1;
}

void neural_project_checkpoint_observer_set_stop_request(
    NeuralProjectCheckpointObserver *observer,
    const volatile sig_atomic_t *stop_request)
{
    if (observer != NULL) {
        observer->stop_request = stop_request;
    }
}

int neural_project_checkpoint_observe(const NeuralEpochReport *report,
                                      void *context,
                                      NeuralError *error)
{
    NeuralProjectCheckpointObserver *observer = context;
    sig_atomic_t stop_signal;

    if (report == NULL || observer == NULL || observer->path == NULL ||
        observer->model == NULL ||
        observer->metadata.target_epochs == 0U ||
        report->completed_epochs == 0U ||
        report->completed_epochs > observer->metadata.target_epochs) {
        neural_error_set(error, "periodic checkpoint report is invalid");
        return 0;
    }
    stop_signal = observer->stop_request == NULL
                      ? 0
                      : *observer->stop_request;
    if (observer->interval == 0U && stop_signal == 0) {
        return 1;
    }
    if (stop_signal == 0 &&
        report->completed_epochs % observer->interval != 0U) {
        return 1;
    }
    observer->metadata.completed_epochs = report->completed_epochs;
    observer->metadata.rng_state =
        neural_model_random_state(observer->model);
    if ((observer->metadata.optimizer !=
             NEURAL_OPTIMIZER_GRADIENT_DESCENT &&
         (report->optimizer == NULL ||
          neural_optimizer_kind(report->optimizer) !=
              observer->metadata.optimizer)) ||
        !neural_checkpoint_save_atomic_with_optimizer(
             observer->path,
             observer->model,
             report->optimizer,
             &observer->metadata,
             error)) {
        if (error != NULL && error->message[0] == '\0') {
            neural_error_set(error, "checkpoint optimizer state is missing");
        }
        return 0;
    }
    if (stop_signal != 0) {
        observer->interrupted_signal = (int)stop_signal;
        neural_error_set(error,
                         "training interrupted by signal %d after epoch %zu; "
                         "checkpoint saved",
                         observer->interrupted_signal,
                         report->completed_epochs);
        return 0;
    }
    return 1;
}
