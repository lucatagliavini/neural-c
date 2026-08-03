#include "neural/digest.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "neural/activation.h"
#include "neural/version.h"
#include "sha256.h"

#if DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024
#error "canonical persistence requires IEEE 754 binary64 double"
#endif

_Static_assert(sizeof(neural_real) == sizeof(uint64_t),
               "canonical persistence requires a 64-bit neural_real");

static void digest_to_hex(const unsigned char digest[32],
                          char output[NEURAL_SHA256_TEXT_CAPACITY])
{
    static const char hexadecimal[] = "0123456789abcdef";
    size_t index;

    for (index = 0U; index < 32U; index++) {
        output[index * 2U] = hexadecimal[digest[index] >> 4U];
        output[index * 2U + 1U] = hexadecimal[digest[index] & 0x0fU];
    }
    output[NEURAL_SHA256_HEX_LENGTH] = '\0';
}

int neural_sha256_hex(const void *data,
                      size_t size,
                      char output[NEURAL_SHA256_TEXT_CAPACITY],
                      NeuralError *error)
{
    NeuralSha256 context;
    unsigned char digest[32];

    if ((data == NULL && size != 0U) || output == NULL) {
        neural_error_set(error, "invalid SHA-256 arguments");
        return 0;
    }
    neural_sha256_init(&context);
    if (size != 0U) {
        neural_sha256_update(&context, data, size);
    }
    neural_sha256_final(&context, digest);
    digest_to_hex(digest, output);
    return 1;
}

static void update_u64(NeuralSha256 *context, uint64_t value)
{
    unsigned char encoded[8];
    size_t index;

    for (index = 0U; index < sizeof(encoded); index++) {
        unsigned int shift = (unsigned int)((7U - index) * 8U);
        encoded[index] = (unsigned char)(value >> shift);
    }
    neural_sha256_update(context, encoded, sizeof(encoded));
}

static void update_size(NeuralSha256 *context, size_t value)
{
    update_u64(context, (uint64_t)value);
}

static void update_text(NeuralSha256 *context, const char *text)
{
    size_t length = strlen(text);

    update_size(context, length);
    neural_sha256_update(context, text, length);
}

static void update_real(NeuralSha256 *context, neural_real value)
{
    uint64_t bits;

    memcpy(&bits, &value, sizeof(bits));
    update_u64(context, bits);
}

static int validate_dataset(const NeuralProject *project, NeuralError *error)
{
    const NeuralDataset *dataset = &project->dataset;
    size_t expected_output_count;
    size_t input_value_count;
    size_t output_value_count;
    size_t index;

    expected_output_count =
        project->model.layers[project->model.layer_count - 1U].neuron_count;
    if (dataset->sample_count == 0U ||
        dataset->input_count != project->model.input_count ||
        dataset->output_count != expected_output_count ||
        dataset->sample_count > SIZE_MAX / dataset->input_count ||
        dataset->sample_count > SIZE_MAX / dataset->output_count) {
        neural_error_set(error, "dataset dimensions are invalid for digesting");
        return 0;
    }
    input_value_count = dataset->sample_count * dataset->input_count;
    output_value_count = dataset->sample_count * dataset->output_count;
    if (dataset->inputs == NULL || dataset->outputs == NULL) {
        neural_error_set(error, "dataset values are required for digesting");
        return 0;
    }
    for (index = 0U; index < input_value_count; index++) {
        if (!isfinite(dataset->inputs[index])) {
            neural_error_set(error, "dataset inputs must be finite for digesting");
            return 0;
        }
    }
    for (index = 0U; index < output_value_count; index++) {
        if (!isfinite(dataset->outputs[index])) {
            neural_error_set(error, "dataset outputs must be finite for digesting");
            return 0;
        }
    }
    return 1;
}

static int digest_model(const NeuralModelSpec *model,
                        char output[NEURAL_SHA256_TEXT_CAPACITY],
                        NeuralError *error)
{
    NeuralSha256 context;
    unsigned char digest[32];
    size_t layer_index;

    if (!neural_model_spec_validate(model, error)) {
        return 0;
    }
    neural_sha256_init(&context);
    update_text(&context, NEURAL_FORMAT_MAGIC ":canonical:model");
    update_u64(&context, (uint64_t)NEURAL_FORMAT_VERSION);
    update_size(&context, model->input_count);
    update_size(&context, model->layer_count);
    for (layer_index = 0U; layer_index < model->layer_count; layer_index++) {
        const NeuralLayerSpec *layer = &model->layers[layer_index];
        size_t parameter_count = 0U;
        int parameter_kind;

        update_size(&context, layer->neuron_count);
        update_text(&context,
                    neural_activation_kind_name(layer->activation.kind));
        update_size(&context, layer->activation.parameter_count);
        for (parameter_kind = (int)NEURAL_ACTIVATION_PARAMETER_ALPHA;
             parameter_kind <= (int)NEURAL_ACTIVATION_PARAMETER_THRESHOLD;
             parameter_kind++) {
            neural_real value;
            NeuralActivationParameterKind kind =
                (NeuralActivationParameterKind)parameter_kind;

            if (neural_activation_spec_parameter_value(&layer->activation,
                                                       kind,
                                                       &value)) {
                update_text(&context,
                            neural_activation_parameter_name(kind));
                update_real(&context, value);
                parameter_count++;
            }
        }
        if (parameter_count != layer->activation.parameter_count) {
            neural_error_set(error,
                             "activation parameters are not uniquely canonical");
            return 0;
        }
    }
    neural_sha256_final(&context, digest);
    digest_to_hex(digest, output);
    return 1;
}

static void digest_dataset(const NeuralDataset *dataset,
                           char output[NEURAL_SHA256_TEXT_CAPACITY])
{
    NeuralSha256 context;
    unsigned char digest[32];
    size_t sample_index;

    neural_sha256_init(&context);
    update_text(&context, NEURAL_FORMAT_MAGIC ":canonical:dataset");
    update_u64(&context, (uint64_t)NEURAL_FORMAT_VERSION);
    update_size(&context, dataset->sample_count);
    update_size(&context, dataset->input_count);
    update_size(&context, dataset->output_count);
    for (sample_index = 0U;
         sample_index < dataset->sample_count;
         sample_index++) {
        size_t value_index;

        for (value_index = 0U;
             value_index < dataset->input_count;
             value_index++) {
            update_real(&context,
                        dataset->inputs[sample_index * dataset->input_count +
                                        value_index]);
        }
        for (value_index = 0U;
             value_index < dataset->output_count;
             value_index++) {
            update_real(&context,
                        dataset->outputs[sample_index * dataset->output_count +
                                         value_index]);
        }
    }
    neural_sha256_final(&context, digest);
    digest_to_hex(digest, output);
}

static void digest_training(const NeuralTrainingConfig *training,
                            char output[NEURAL_SHA256_TEXT_CAPACITY])
{
    NeuralSha256 context;
    unsigned char digest[32];

    neural_sha256_init(&context);
    update_text(&context, NEURAL_FORMAT_MAGIC ":canonical:training");
    update_u64(&context, (uint64_t)NEURAL_FORMAT_VERSION);
    update_size(&context, training->epochs);
    update_real(&context, training->learning_rate);
    update_u64(&context, training->seed);
    update_text(&context, neural_loss_name(training->loss));
    update_size(&context, training->checkpoint_interval);
    neural_sha256_final(&context, digest);
    digest_to_hex(digest, output);
}

int neural_project_digests_compute(const NeuralProject *project,
                                   NeuralProjectDigests *digests,
                                   NeuralError *error)
{
    if (project == NULL || digests == NULL) {
        neural_error_set(error, "project and digest output are required");
        return 0;
    }
    if (!digest_model(&project->model, digests->model, error) ||
        !neural_training_config_validate(&project->training, error) ||
        !validate_dataset(project, error)) {
        return 0;
    }
    digest_dataset(&project->dataset, digests->dataset);
    digest_training(&project->training, digests->training);
    return 1;
}
