#ifndef NEURAL_OPTIMIZER_H
#define NEURAL_OPTIMIZER_H

#include <stddef.h>

#include "neural/error.h"
#include "neural/types.h"

#define NEURAL_OPTIMIZER_ABSTRACTION_VERSION 1U

typedef enum {
    NEURAL_OPTIMIZER_GRADIENT_DESCENT,
    NEURAL_OPTIMIZER_MOMENTUM,
    NEURAL_OPTIMIZER_ADAM
} NeuralOptimizerKind;

typedef enum {
    NEURAL_LR_SCHEDULE_CONSTANT,
    NEURAL_LR_SCHEDULE_STEP,
    NEURAL_LR_SCHEDULE_EXPONENTIAL,
    NEURAL_LR_SCHEDULE_PLATEAU
} NeuralLearningRateScheduleKind;

typedef enum {
    NEURAL_CONVERGENCE_NONE,
    NEURAL_CONVERGENCE_LOSS_TARGET,
    NEURAL_CONVERGENCE_NO_IMPROVEMENT
} NeuralConvergenceReason;

typedef struct {
    NeuralOptimizerKind kind;
    neural_real momentum;
    neural_real adam_beta1;
    neural_real adam_beta2;
    neural_real adam_epsilon;
    neural_real learning_rate;
    NeuralLearningRateScheduleKind schedule;
    neural_real schedule_decay;
    size_t schedule_step_epochs;
    size_t schedule_plateau_patience;
    neural_real schedule_plateau_min_delta;
    neural_real divergence_threshold;
    neural_real target_loss;
    size_t no_improvement_epochs;
    neural_real no_improvement_min_delta;
} NeuralOptimizerOptions;

typedef struct NeuralModel NeuralModel;
typedef struct NeuralGradient NeuralGradient;
typedef struct NeuralOptimizer NeuralOptimizer;

const char *neural_optimizer_name(NeuralOptimizerKind kind);
int neural_optimizer_from_name(const char *name, NeuralOptimizerKind *kind);
const char *neural_learning_rate_schedule_name(
    NeuralLearningRateScheduleKind kind);
int neural_learning_rate_schedule_from_name(
    const char *name,
    NeuralLearningRateScheduleKind *kind);

int neural_optimizer_create(const NeuralModel *model,
                            const NeuralOptimizerOptions *options,
                            NeuralOptimizer **optimizer,
                            NeuralError *error);
void neural_optimizer_free(NeuralOptimizer *optimizer);
NeuralOptimizerKind neural_optimizer_kind(const NeuralOptimizer *optimizer);
size_t neural_optimizer_version(const NeuralOptimizer *optimizer);
size_t neural_optimizer_timestep(const NeuralOptimizer *optimizer);
neural_real neural_optimizer_beta1_power(const NeuralOptimizer *optimizer);
neural_real neural_optimizer_beta2_power(const NeuralOptimizer *optimizer);
const NeuralGradient *neural_optimizer_state1(
    const NeuralOptimizer *optimizer);
const NeuralGradient *neural_optimizer_state2(
    const NeuralOptimizer *optimizer);
int neural_optimizer_requires_checkpoint_state(
    const NeuralOptimizer *optimizer);
neural_real neural_optimizer_current_learning_rate(
    const NeuralOptimizer *optimizer);
NeuralLearningRateScheduleKind neural_optimizer_schedule_kind(
    const NeuralOptimizer *optimizer);
size_t neural_optimizer_schedule_completed_epochs(
    const NeuralOptimizer *optimizer);
size_t neural_optimizer_schedule_next_transition(
    const NeuralOptimizer *optimizer);
size_t neural_optimizer_schedule_stale_epochs(
    const NeuralOptimizer *optimizer);
int neural_optimizer_schedule_has_best(const NeuralOptimizer *optimizer);
neural_real neural_optimizer_schedule_best_metric(
    const NeuralOptimizer *optimizer);
int neural_optimizer_observe_epoch(NeuralOptimizer *optimizer,
                                   neural_real metric,
                                   NeuralError *error);
NeuralConvergenceReason neural_optimizer_convergence_reason(
    const NeuralOptimizer *optimizer);
int neural_optimizer_convergence_has_best(const NeuralOptimizer *optimizer);
neural_real neural_optimizer_convergence_best_loss(
    const NeuralOptimizer *optimizer);
size_t neural_optimizer_convergence_stale_epochs(
    const NeuralOptimizer *optimizer);
int neural_optimizer_restore_schedule(
    NeuralOptimizer *optimizer,
    size_t completed_epochs,
    neural_real current_learning_rate,
    size_t next_transition,
    int has_best,
    neural_real best_metric,
    size_t stale_epochs,
    NeuralError *error);
int neural_optimizer_restore_convergence(
    NeuralOptimizer *optimizer,
    int has_best,
    neural_real best_loss,
    size_t stale_epochs,
    NeuralConvergenceReason reason,
    NeuralError *error);
int neural_optimizer_restore(NeuralOptimizer *optimizer,
                             size_t timestep,
                             neural_real beta1_power,
                             neural_real beta2_power,
                             const NeuralGradient *state1,
                             const NeuralGradient *state2,
                             NeuralError *error);
int neural_optimizer_step(NeuralOptimizer *optimizer,
                          NeuralModel *model,
                          const NeuralGradient *gradient,
                          neural_real learning_rate,
                          NeuralError *error);

#endif
