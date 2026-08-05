#include "neural/optimizer.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "neural/gradient.h"
#include "neural/model.h"

struct NeuralOptimizer {
    const NeuralModel *model;
    NeuralOptimizerOptions options;
    size_t version;
    size_t timestep;
    neural_real beta1_power;
    neural_real beta2_power;
    NeuralGradient *state1;
    NeuralGradient *state2;
    NeuralGradient *candidate1;
    NeuralGradient *candidate2;
    NeuralGradient *update;
    neural_real current_learning_rate;
    size_t schedule_completed_epochs;
    size_t schedule_next_transition;
    size_t schedule_stale_epochs;
    int schedule_has_best;
    neural_real schedule_best_metric;
    size_t convergence_stale_epochs;
    int convergence_has_best;
    neural_real convergence_best_loss;
    NeuralConvergenceReason convergence_reason;
};

const char *neural_optimizer_name(NeuralOptimizerKind kind)
{
    switch (kind) {
    case NEURAL_OPTIMIZER_GRADIENT_DESCENT:
        return "gradient_descent";
    case NEURAL_OPTIMIZER_MOMENTUM:
        return "momentum";
    case NEURAL_OPTIMIZER_ADAM:
        return "adam";
    }
    return "unknown";
}

int neural_optimizer_from_name(const char *name, NeuralOptimizerKind *kind)
{
    if (name == NULL || kind == NULL) {
        return 0;
    }
    if (strcmp(name, "gradient_descent") == 0) {
        *kind = NEURAL_OPTIMIZER_GRADIENT_DESCENT;
    } else if (strcmp(name, "momentum") == 0) {
        *kind = NEURAL_OPTIMIZER_MOMENTUM;
    } else if (strcmp(name, "adam") == 0) {
        *kind = NEURAL_OPTIMIZER_ADAM;
    } else {
        return 0;
    }
    return 1;
}

const char *neural_learning_rate_schedule_name(
    NeuralLearningRateScheduleKind kind)
{
    switch (kind) {
    case NEURAL_LR_SCHEDULE_CONSTANT:
        return "constant";
    case NEURAL_LR_SCHEDULE_STEP:
        return "step";
    case NEURAL_LR_SCHEDULE_EXPONENTIAL:
        return "exponential";
    case NEURAL_LR_SCHEDULE_PLATEAU:
        return "plateau";
    }
    return "unknown";
}

int neural_learning_rate_schedule_from_name(
    const char *name,
    NeuralLearningRateScheduleKind *kind)
{
    if (name == NULL || kind == NULL) {
        return 0;
    }
    if (strcmp(name, "constant") == 0) {
        *kind = NEURAL_LR_SCHEDULE_CONSTANT;
    } else if (strcmp(name, "step") == 0) {
        *kind = NEURAL_LR_SCHEDULE_STEP;
    } else if (strcmp(name, "exponential") == 0) {
        *kind = NEURAL_LR_SCHEDULE_EXPONENTIAL;
    } else if (strcmp(name, "plateau") == 0) {
        *kind = NEURAL_LR_SCHEDULE_PLATEAU;
    } else {
        return 0;
    }
    return 1;
}

static int options_valid(const NeuralOptimizerOptions *options)
{
    if (options == NULL) {
        return 0;
    }
    if (!isfinite(options->learning_rate) || options->learning_rate <= 0.0 ||
        strcmp(neural_learning_rate_schedule_name(options->schedule),
               "unknown") == 0 ||
        (options->schedule != NEURAL_LR_SCHEDULE_CONSTANT &&
         (!isfinite(options->schedule_decay) ||
          options->schedule_decay <= 0.0 ||
          options->schedule_decay >= 1.0)) ||
        (options->schedule == NEURAL_LR_SCHEDULE_STEP &&
         options->schedule_step_epochs == 0U) ||
        (options->schedule == NEURAL_LR_SCHEDULE_PLATEAU &&
         (options->schedule_plateau_patience == 0U ||
          !isfinite(options->schedule_plateau_min_delta) ||
          options->schedule_plateau_min_delta < 0.0)) ||
        !isfinite(options->divergence_threshold) ||
        options->divergence_threshold < 0.0 ||
        !isfinite(options->target_loss) || options->target_loss < -1.0 ||
        !isfinite(options->no_improvement_min_delta) ||
        options->no_improvement_min_delta < 0.0) {
        return 0;
    }
    switch (options->kind) {
    case NEURAL_OPTIMIZER_GRADIENT_DESCENT:
        return 1;
    case NEURAL_OPTIMIZER_MOMENTUM:
        return isfinite(options->momentum) && options->momentum >= 0.0 &&
               options->momentum < 1.0;
    case NEURAL_OPTIMIZER_ADAM:
        return isfinite(options->adam_beta1) &&
               options->adam_beta1 >= 0.0 && options->adam_beta1 < 1.0 &&
               isfinite(options->adam_beta2) &&
               options->adam_beta2 >= 0.0 && options->adam_beta2 < 1.0 &&
               isfinite(options->adam_epsilon) &&
               options->adam_epsilon > 0.0;
    }
    return 0;
}

static int create_gradient(const NeuralModel *model,
                           NeuralGradient **gradient,
                           NeuralError *error)
{
    return neural_gradient_create(model, gradient, error) &&
           neural_gradient_zero(*gradient, error);
}

int neural_optimizer_create(const NeuralModel *model,
                            const NeuralOptimizerOptions *options,
                            NeuralOptimizer **optimizer,
                            NeuralError *error)
{
    NeuralOptimizer *created;

    if (model == NULL || optimizer == NULL || !options_valid(options)) {
        neural_error_set(error, "optimizer configuration is invalid");
        return 0;
    }
    *optimizer = NULL;
    created = calloc(1U, sizeof(*created));
    if (created == NULL) {
        neural_error_set(error, "unable to allocate optimizer");
        return 0;
    }
    created->model = model;
    created->options = *options;
    created->version = NEURAL_OPTIMIZER_ABSTRACTION_VERSION;
    created->beta1_power = 1.0;
    created->beta2_power = 1.0;
    created->current_learning_rate = options->learning_rate;
    created->schedule_best_metric = INFINITY;
    created->convergence_best_loss = INFINITY;
    created->schedule_next_transition =
        options->schedule == NEURAL_LR_SCHEDULE_STEP
            ? options->schedule_step_epochs
            : 1U;
    if (options->kind == NEURAL_OPTIMIZER_MOMENTUM &&
        (!create_gradient(model, &created->state1, error) ||
         !create_gradient(model, &created->candidate1, error))) {
        neural_optimizer_free(created);
        return 0;
    }
    if (options->kind == NEURAL_OPTIMIZER_ADAM &&
        (!create_gradient(model, &created->state1, error) ||
         !create_gradient(model, &created->state2, error) ||
         !create_gradient(model, &created->candidate1, error) ||
         !create_gradient(model, &created->candidate2, error) ||
         !create_gradient(model, &created->update, error))) {
        neural_optimizer_free(created);
        return 0;
    }
    *optimizer = created;
    return 1;
}

void neural_optimizer_free(NeuralOptimizer *optimizer)
{
    if (optimizer != NULL) {
        neural_gradient_free(optimizer->update);
        neural_gradient_free(optimizer->candidate2);
        neural_gradient_free(optimizer->candidate1);
        neural_gradient_free(optimizer->state2);
        neural_gradient_free(optimizer->state1);
        free(optimizer);
    }
}

NeuralOptimizerKind neural_optimizer_kind(const NeuralOptimizer *optimizer)
{
    return optimizer == NULL
               ? (NeuralOptimizerKind)-1
               : optimizer->options.kind;
}

size_t neural_optimizer_version(const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0U : optimizer->version;
}

size_t neural_optimizer_timestep(const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0U : optimizer->timestep;
}

neural_real neural_optimizer_beta1_power(const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0.0 : optimizer->beta1_power;
}

neural_real neural_optimizer_beta2_power(const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0.0 : optimizer->beta2_power;
}

const NeuralGradient *neural_optimizer_state1(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? NULL : optimizer->state1;
}

const NeuralGradient *neural_optimizer_state2(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? NULL : optimizer->state2;
}

int neural_optimizer_requires_checkpoint_state(
    const NeuralOptimizer *optimizer)
{
    return optimizer != NULL &&
           (optimizer->options.kind != NEURAL_OPTIMIZER_GRADIENT_DESCENT ||
            optimizer->options.schedule != NEURAL_LR_SCHEDULE_CONSTANT ||
            optimizer->options.divergence_threshold > 0.0 ||
            optimizer->options.target_loss >= 0.0 ||
            optimizer->options.no_improvement_epochs != 0U);
}

neural_real neural_optimizer_current_learning_rate(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0.0 : optimizer->current_learning_rate;
}

NeuralLearningRateScheduleKind neural_optimizer_schedule_kind(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL
               ? (NeuralLearningRateScheduleKind)-1
               : optimizer->options.schedule;
}

size_t neural_optimizer_schedule_completed_epochs(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0U : optimizer->schedule_completed_epochs;
}

size_t neural_optimizer_schedule_next_transition(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0U : optimizer->schedule_next_transition;
}

size_t neural_optimizer_schedule_stale_epochs(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0U : optimizer->schedule_stale_epochs;
}

int neural_optimizer_schedule_has_best(const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0 : optimizer->schedule_has_best;
}

neural_real neural_optimizer_schedule_best_metric(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0.0 : optimizer->schedule_best_metric;
}

int neural_optimizer_observe_epoch(NeuralOptimizer *optimizer,
                                   neural_real metric,
                                   NeuralError *error)
{
    size_t completed;
    size_t next_transition;
    size_t stale;
    int has_best;
    int convergence_has_best;
    neural_real best;
    neural_real convergence_best;
    neural_real rate;
    size_t convergence_stale;
    NeuralConvergenceReason convergence_reason;

    if (optimizer == NULL || !isfinite(metric) ||
        optimizer->schedule_completed_epochs == SIZE_MAX) {
        neural_error_set(error, "learning-rate schedule state is invalid");
        return 0;
    }
    if (optimizer->options.divergence_threshold > 0.0 &&
        metric > optimizer->options.divergence_threshold) {
        neural_error_set(error,
                         "training diverged: loss %.*g exceeds threshold %.*g",
                         DBL_DECIMAL_DIG,
                         metric,
                         DBL_DECIMAL_DIG,
                         optimizer->options.divergence_threshold);
        return 0;
    }
    completed = optimizer->schedule_completed_epochs + 1U;
    next_transition = optimizer->schedule_next_transition;
    stale = optimizer->schedule_stale_epochs;
    has_best = optimizer->schedule_has_best;
    best = optimizer->schedule_best_metric;
    rate = optimizer->current_learning_rate;
    convergence_has_best = optimizer->convergence_has_best;
    convergence_best = optimizer->convergence_best_loss;
    convergence_stale = optimizer->convergence_stale_epochs;
    convergence_reason = optimizer->convergence_reason;
    if (optimizer->options.schedule == NEURAL_LR_SCHEDULE_STEP &&
        completed == next_transition) {
        rate *= optimizer->options.schedule_decay;
        next_transition = next_transition >
                                  SIZE_MAX -
                                      optimizer->options.schedule_step_epochs
                              ? SIZE_MAX
                              : next_transition +
                                    optimizer->options.schedule_step_epochs;
    } else if (optimizer->options.schedule ==
               NEURAL_LR_SCHEDULE_EXPONENTIAL) {
        rate *= optimizer->options.schedule_decay;
        next_transition = completed == SIZE_MAX ? SIZE_MAX : completed + 1U;
    } else if (optimizer->options.schedule == NEURAL_LR_SCHEDULE_PLATEAU) {
        if (!has_best || best - metric >
                             optimizer->options.schedule_plateau_min_delta) {
            best = metric;
            has_best = 1;
            stale = 0U;
        } else {
            if (stale == SIZE_MAX) {
                neural_error_set(error,
                                 "learning-rate plateau counter overflow");
                return 0;
            }
            stale++;
            if (stale >= optimizer->options.schedule_plateau_patience) {
                rate *= optimizer->options.schedule_decay;
                stale = 0U;
                next_transition = completed == SIZE_MAX
                                      ? SIZE_MAX
                                      : completed + 1U;
            }
        }
    }
    if (!isfinite(rate) || rate <= 0.0) {
        neural_error_set(error, "learning rate became non-finite or zero");
        return 0;
    }
    if (!convergence_has_best ||
        convergence_best - metric >
            optimizer->options.no_improvement_min_delta) {
        convergence_best = metric;
        convergence_has_best = 1;
        convergence_stale = 0U;
    } else if (optimizer->options.no_improvement_epochs != 0U) {
        if (convergence_stale == SIZE_MAX) {
            neural_error_set(error, "no-improvement counter overflow");
            return 0;
        }
        convergence_stale++;
    }
    if (optimizer->options.target_loss >= 0.0 &&
        metric <= optimizer->options.target_loss) {
        convergence_reason = NEURAL_CONVERGENCE_LOSS_TARGET;
    } else if (optimizer->options.no_improvement_epochs != 0U &&
               convergence_stale >=
                   optimizer->options.no_improvement_epochs) {
        convergence_reason = NEURAL_CONVERGENCE_NO_IMPROVEMENT;
    }
    optimizer->schedule_completed_epochs = completed;
    optimizer->schedule_next_transition = next_transition;
    optimizer->schedule_stale_epochs = stale;
    optimizer->schedule_has_best = has_best;
    optimizer->schedule_best_metric = best;
    optimizer->current_learning_rate = rate;
    optimizer->convergence_has_best = convergence_has_best;
    optimizer->convergence_best_loss = convergence_best;
    optimizer->convergence_stale_epochs = convergence_stale;
    optimizer->convergence_reason = convergence_reason;
    return 1;
}

NeuralConvergenceReason neural_optimizer_convergence_reason(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? NEURAL_CONVERGENCE_NONE
                             : optimizer->convergence_reason;
}

int neural_optimizer_convergence_has_best(const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0 : optimizer->convergence_has_best;
}

neural_real neural_optimizer_convergence_best_loss(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0.0 : optimizer->convergence_best_loss;
}

size_t neural_optimizer_convergence_stale_epochs(
    const NeuralOptimizer *optimizer)
{
    return optimizer == NULL ? 0U : optimizer->convergence_stale_epochs;
}

int neural_optimizer_restore_schedule(
    NeuralOptimizer *optimizer,
    size_t completed_epochs,
    neural_real current_learning_rate,
    size_t next_transition,
    int has_best,
    neural_real best_metric,
    size_t stale_epochs,
    NeuralError *error)
{
    if (optimizer == NULL || !isfinite(current_learning_rate) ||
        current_learning_rate <= 0.0 || (has_best != 0 && has_best != 1) ||
        (has_best && !isfinite(best_metric)) ||
        (!has_best && stale_epochs != 0U)) {
        neural_error_set(error, "learning-rate schedule restore is invalid");
        return 0;
    }
    optimizer->schedule_completed_epochs = completed_epochs;
    optimizer->current_learning_rate = current_learning_rate;
    optimizer->schedule_next_transition = next_transition;
    optimizer->schedule_has_best = has_best;
    optimizer->schedule_best_metric = has_best ? best_metric : INFINITY;
    optimizer->schedule_stale_epochs = stale_epochs;
    return 1;
}

int neural_optimizer_restore_convergence(
    NeuralOptimizer *optimizer,
    int has_best,
    neural_real best_loss,
    size_t stale_epochs,
    NeuralConvergenceReason reason,
    NeuralError *error)
{
    if (optimizer == NULL || (has_best != 0 && has_best != 1) ||
        (has_best && !isfinite(best_loss)) || (!has_best && stale_epochs != 0U) ||
        (reason != NEURAL_CONVERGENCE_NONE &&
         reason != NEURAL_CONVERGENCE_LOSS_TARGET &&
         reason != NEURAL_CONVERGENCE_NO_IMPROVEMENT)) {
        neural_error_set(error, "convergence restore state is invalid");
        return 0;
    }
    optimizer->convergence_has_best = has_best;
    optimizer->convergence_best_loss = has_best ? best_loss : INFINITY;
    optimizer->convergence_stale_epochs = stale_epochs;
    optimizer->convergence_reason = reason;
    return 1;
}

int neural_optimizer_restore(NeuralOptimizer *optimizer,
                             size_t timestep,
                             neural_real beta1_power,
                             neural_real beta2_power,
                             const NeuralGradient *state1,
                             const NeuralGradient *state2,
                             NeuralError *error)
{
    if (optimizer == NULL ||
        optimizer->options.kind == NEURAL_OPTIMIZER_GRADIENT_DESCENT ||
        state1 == NULL ||
        !neural_gradient_is_compatible(state1, optimizer->model) ||
        (optimizer->options.kind == NEURAL_OPTIMIZER_ADAM &&
         (state2 == NULL ||
          !neural_gradient_is_compatible(state2, optimizer->model))) ||
        (optimizer->options.kind == NEURAL_OPTIMIZER_MOMENTUM &&
         state2 != NULL) ||
        (optimizer->options.kind == NEURAL_OPTIMIZER_ADAM &&
         (!isfinite(beta1_power) || !isfinite(beta2_power) ||
          beta1_power < 0.0 || beta1_power > 1.0 ||
          beta2_power < 0.0 || beta2_power > 1.0))) {
        neural_error_set(error, "optimizer restore state is invalid");
        return 0;
    }
    if (!neural_gradient_copy(optimizer->candidate1, state1, error) ||
        (optimizer->options.kind == NEURAL_OPTIMIZER_ADAM &&
         !neural_gradient_copy(optimizer->candidate2, state2, error))) {
        return 0;
    }
    {
        NeuralGradient *swap = optimizer->state1;
        optimizer->state1 = optimizer->candidate1;
        optimizer->candidate1 = swap;
        if (optimizer->options.kind == NEURAL_OPTIMIZER_ADAM) {
            swap = optimizer->state2;
            optimizer->state2 = optimizer->candidate2;
            optimizer->candidate2 = swap;
            optimizer->beta1_power = beta1_power;
            optimizer->beta2_power = beta2_power;
        }
    }
    optimizer->timestep = timestep;
    return 1;
}

static int momentum_values(neural_real *candidate,
                           const neural_real *state,
                           const neural_real *gradient,
                           size_t count,
                           neural_real coefficient)
{
    size_t index;

    for (index = 0U; index < count; index++) {
        candidate[index] = coefficient * state[index] + gradient[index];
        if (!isfinite(candidate[index])) {
            return 0;
        }
    }
    return 1;
}

static int adam_values(neural_real *candidate_m,
                       neural_real *candidate_v,
                       neural_real *update,
                       const neural_real *state_m,
                       const neural_real *state_v,
                       const neural_real *gradient,
                       size_t count,
                       const NeuralOptimizerOptions *options,
                       neural_real correction1,
                       neural_real correction2)
{
    size_t index;

    for (index = 0U; index < count; index++) {
        neural_real square = gradient[index] * gradient[index];
        neural_real denominator;

        candidate_m[index] = options->adam_beta1 * state_m[index] +
                             (1.0 - options->adam_beta1) * gradient[index];
        candidate_v[index] = options->adam_beta2 * state_v[index] +
                             (1.0 - options->adam_beta2) * square;
        denominator = sqrt(candidate_v[index] / correction2) +
                      options->adam_epsilon;
        update[index] = (candidate_m[index] / correction1) / denominator;
        if (!isfinite(square) || !isfinite(candidate_m[index]) ||
            !isfinite(candidate_v[index]) || denominator <= 0.0 ||
            !isfinite(denominator) || !isfinite(update[index])) {
            return 0;
        }
    }
    return 1;
}

static int build_momentum_candidate(NeuralOptimizer *optimizer,
                                    const NeuralGradient *gradient)
{
    size_t layer_index;

    for (layer_index = 0U;
         layer_index < neural_model_layer_count(optimizer->model);
         layer_index++) {
        size_t weight_count;
        size_t bias_count;
        neural_real *candidate_weights = neural_gradient_layer_weights(
            optimizer->candidate1, layer_index, &weight_count);
        neural_real *candidate_biases = neural_gradient_layer_biases(
            optimizer->candidate1, layer_index, &bias_count);
        const neural_real *state_weights = neural_gradient_layer_weights_const(
            optimizer->state1, layer_index, NULL);
        const neural_real *state_biases = neural_gradient_layer_biases_const(
            optimizer->state1, layer_index, NULL);
        const neural_real *gradient_weights =
            neural_gradient_layer_weights_const(gradient, layer_index, NULL);
        const neural_real *gradient_biases =
            neural_gradient_layer_biases_const(gradient, layer_index, NULL);

        if (!momentum_values(candidate_weights,
                             state_weights,
                             gradient_weights,
                             weight_count,
                             optimizer->options.momentum) ||
            !momentum_values(candidate_biases,
                             state_biases,
                             gradient_biases,
                             bias_count,
                             optimizer->options.momentum)) {
            return 0;
        }
    }
    return 1;
}

static int build_adam_candidates(NeuralOptimizer *optimizer,
                                 const NeuralGradient *gradient,
                                 neural_real correction1,
                                 neural_real correction2)
{
    size_t layer_index;

    for (layer_index = 0U;
         layer_index < neural_model_layer_count(optimizer->model);
         layer_index++) {
        size_t weight_count;
        size_t bias_count;
        neural_real *candidate_m_weights = neural_gradient_layer_weights(
            optimizer->candidate1, layer_index, &weight_count);
        neural_real *candidate_m_biases = neural_gradient_layer_biases(
            optimizer->candidate1, layer_index, &bias_count);
        neural_real *candidate_v_weights = neural_gradient_layer_weights(
            optimizer->candidate2, layer_index, NULL);
        neural_real *candidate_v_biases = neural_gradient_layer_biases(
            optimizer->candidate2, layer_index, NULL);
        neural_real *update_weights = neural_gradient_layer_weights(
            optimizer->update, layer_index, NULL);
        neural_real *update_biases = neural_gradient_layer_biases(
            optimizer->update, layer_index, NULL);
        const neural_real *state_m_weights =
            neural_gradient_layer_weights_const(
                optimizer->state1, layer_index, NULL);
        const neural_real *state_m_biases = neural_gradient_layer_biases_const(
            optimizer->state1, layer_index, NULL);
        const neural_real *state_v_weights =
            neural_gradient_layer_weights_const(
                optimizer->state2, layer_index, NULL);
        const neural_real *state_v_biases = neural_gradient_layer_biases_const(
            optimizer->state2, layer_index, NULL);
        const neural_real *gradient_weights =
            neural_gradient_layer_weights_const(gradient, layer_index, NULL);
        const neural_real *gradient_biases =
            neural_gradient_layer_biases_const(gradient, layer_index, NULL);

        if (!adam_values(candidate_m_weights,
                         candidate_v_weights,
                         update_weights,
                         state_m_weights,
                         state_v_weights,
                         gradient_weights,
                         weight_count,
                         &optimizer->options,
                         correction1,
                         correction2) ||
            !adam_values(candidate_m_biases,
                         candidate_v_biases,
                         update_biases,
                         state_m_biases,
                         state_v_biases,
                         gradient_biases,
                         bias_count,
                         &optimizer->options,
                         correction1,
                         correction2)) {
            return 0;
        }
    }
    return 1;
}

int neural_optimizer_step(NeuralOptimizer *optimizer,
                          NeuralModel *model,
                          const NeuralGradient *gradient,
                          neural_real learning_rate,
                          NeuralError *error)
{
    NeuralGradient *swap;

    if (optimizer == NULL || model == NULL || model != optimizer->model ||
        gradient == NULL ||
        !neural_gradient_is_compatible(gradient, model)) {
        neural_error_set(error, "optimizer step is incompatible");
        return 0;
    }
    if (optimizer->options.kind == NEURAL_OPTIMIZER_GRADIENT_DESCENT) {
        return neural_model_apply_gradient(model,
                                           gradient,
                                           learning_rate,
                                           error);
    }
    if (optimizer->timestep == SIZE_MAX) {
        neural_error_set(error, "optimizer timestep overflow");
        return 0;
    }
    if (optimizer->options.kind == NEURAL_OPTIMIZER_MOMENTUM) {
        if (!build_momentum_candidate(optimizer, gradient)) {
            neural_error_set(error, "momentum state became non-finite");
            return 0;
        }
        if (!neural_model_apply_gradient(model,
                                         optimizer->candidate1,
                                         learning_rate,
                                         error)) {
            return 0;
        }
        swap = optimizer->state1;
        optimizer->state1 = optimizer->candidate1;
        optimizer->candidate1 = swap;
        optimizer->timestep++;
        return 1;
    }
    if (optimizer->options.kind == NEURAL_OPTIMIZER_ADAM) {
        neural_real beta1_power = optimizer->beta1_power *
                                  optimizer->options.adam_beta1;
        neural_real beta2_power = optimizer->beta2_power *
                                  optimizer->options.adam_beta2;
        neural_real correction1 = 1.0 - beta1_power;
        neural_real correction2 = 1.0 - beta2_power;

        if (!isfinite(beta1_power) || !isfinite(beta2_power) ||
            correction1 <= 0.0 || correction2 <= 0.0 ||
            !build_adam_candidates(optimizer,
                                   gradient,
                                   correction1,
                                   correction2)) {
            neural_error_set(error, "Adam state became non-finite");
            return 0;
        }
        if (!neural_model_apply_gradient(model,
                                         optimizer->update,
                                         learning_rate,
                                         error)) {
            return 0;
        }
        swap = optimizer->state1;
        optimizer->state1 = optimizer->candidate1;
        optimizer->candidate1 = swap;
        swap = optimizer->state2;
        optimizer->state2 = optimizer->candidate2;
        optimizer->candidate2 = swap;
        optimizer->beta1_power = beta1_power;
        optimizer->beta2_power = beta2_power;
        optimizer->timestep++;
        return 1;
    }
    neural_error_set(error, "optimizer step is incompatible");
    return 0;
}
