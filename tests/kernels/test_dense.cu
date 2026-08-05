#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include"../../network/dense.cu"

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
    g_tests_run++; \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << msg << "\n"; \
        g_tests_failed++; \
    } else { \
        std::cout << "[ OK ] " << msg << "\n"; \
    } \
} while(0)

float sigmoid_cpu(float x) { return 1.0f/(1.0f+expf(-x)); }

void dense_forward_cpu(float* inputs, float* weights, float* biases, float* outputs, int in_size, int out_size) {
    for (int j = 0; j < out_size; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < in_size; ++k) sum += inputs[k] * weights[k*out_size+j];
        outputs[j] = sigmoid_cpu(sum + biases[j]);
    }
}

void dense_forward_cpu_batch(float* input_batch, float* weights, float* biases, float* output_batch,
                              int batch_size, int in_size, int out_size) {
    for (int b = 0; b < batch_size; ++b)
        dense_forward_cpu(input_batch + (size_t)b*in_size, weights, biases, output_batch + (size_t)b*out_size, in_size, out_size);
}

void dense_backward_cpu_batch(float* input_batch, float* weights, float* biases, float* output_batch, float* next_errors_batch,
                               float* dInputs_batch, float* weight_grad_avg, float* bias_grad_avg,
                               int batch_size, int in_size, int out_size) {
    std::vector<float> local_grad((size_t)batch_size * out_size);
    for (size_t i = 0; i < local_grad.size(); ++i)
        local_grad[i] = next_errors_batch[i] * output_batch[i] * (1.0f - output_batch[i]);

    std::vector<float> wg_sum(in_size*out_size, 0.0f);
    std::vector<float> bg_sum(out_size, 0.0f);

    for (int i = 0; i < in_size; ++i)
        for (int j = 0; j < out_size; ++j) {
            float s = 0.0f;
            for (int b = 0; b < batch_size; ++b) s += input_batch[b*in_size+i] * local_grad[b*out_size+j];
            wg_sum[i*out_size+j] = s;
        }

    for (int j = 0; j < out_size; ++j) {
        float s = 0.0f;
        for (int b = 0; b < batch_size; ++b) s += local_grad[b*out_size+j];
        bg_sum[j] = s;
    }

    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < in_size; ++i) {
            float s = 0.0f;
            for (int j = 0; j < out_size; ++j) s += local_grad[b*out_size+j] * weights[i*out_size+j];
            dInputs_batch[b*in_size+i] = s;
        }

    for (int idx = 0; idx < in_size*out_size; ++idx) weight_grad_avg[idx] = wg_sum[idx] / (float)batch_size;
    for (int j = 0; j < out_size; ++j) bias_grad_avg[j] = bg_sum[j] / (float)batch_size;
}

void test_dense_forward_batch_size_1_equivalence() {
    const int in_size = 5, out_size = 4;
    Dense dense(in_size, out_size);

    for (int i = 0; i < in_size*out_size; ++i) dense.weights[i] = 0.05f * ((i % 13) - 6);
    for (int j = 0; j < out_size; ++j) dense.biases[j] = 0.02f * j;

    float input[in_size];
    for (int i = 0; i < in_size; ++i) input[i] = 0.1f * ((i*3) % 7) - 0.3f;

    dense.forward(input);

    float expected[out_size];
    dense_forward_cpu(input, dense.weights, dense.biases, expected, in_size, out_size);

    bool match = true;
    for (int j = 0; j < out_size; ++j) if (std::fabs(dense.outputs[j] - expected[j]) > 1e-4f) match = false;
    CHECK(match, "batch_size=1: forward output identical to legacy single-image implementation");
}

void test_dense_forward_batch() {
    const int in_size = 5, out_size = 4, batch_size = 6;
    Dense dense(in_size, out_size);
    dense.set_batch_size(batch_size);

    for (int i = 0; i < in_size*out_size; ++i) dense.weights[i] = 0.03f * ((i % 11) - 5);
    for (int j = 0; j < out_size; ++j) dense.biases[j] = 0.04f * j - 0.1f;

    std::vector<float> input_batch(batch_size * in_size);
    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < in_size; ++i)
            input_batch[b*in_size+i] = 0.07f * ((i*(b+1) + b*5) % 17) - 0.5f;

    dense.forward(input_batch.data());

    std::vector<float> expected(batch_size * out_size);
    dense_forward_cpu_batch(input_batch.data(), dense.weights, dense.biases, expected.data(), batch_size, in_size, out_size);

    bool match = true;
    for (size_t i = 0; i < expected.size(); ++i)
        if (std::fabs(dense.outputs[i] - expected[i]) > 1e-4f) match = false;
    CHECK(match, "batch forward: GPU output matches CPU reference for all 6 images");
}

void test_dense_backward_batch() {
    const int in_size = 5, out_size = 4, batch_size = 6;
    Dense dense(in_size, out_size);
    dense.set_batch_size(batch_size);

    for (int i = 0; i < in_size*out_size; ++i) dense.weights[i] = 0.03f * ((i % 11) - 5);
    for (int j = 0; j < out_size; ++j) dense.biases[j] = 0.04f * j - 0.1f;

    std::vector<float> weights_before(dense.weights, dense.weights + in_size*out_size);
    std::vector<float> biases_before(dense.biases, dense.biases + out_size);

    std::vector<float> input_batch(batch_size * in_size);
    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < in_size; ++i)
            input_batch[b*in_size+i] = 0.07f * ((i*(b+1) + b*5) % 17) - 0.5f;

    dense.forward(input_batch.data());
    std::vector<float> outputs_snapshot(dense.outputs, dense.outputs + batch_size*out_size);

    std::vector<float> next_errors_batch(batch_size * out_size);
    for (size_t i = 0; i < next_errors_batch.size(); ++i) next_errors_batch[i] = 0.3f + 0.05f * (i % 7);

    std::vector<float> dInputs_cpu(batch_size * in_size);
    std::vector<float> wg_avg_cpu(in_size*out_size);
    std::vector<float> bg_avg_cpu(out_size);

    dense_backward_cpu_batch(input_batch.data(), weights_before.data(), biases_before.data(),
                              outputs_snapshot.data(), next_errors_batch.data(),
                              dInputs_cpu.data(), wg_avg_cpu.data(), bg_avg_cpu.data(),
                              batch_size, in_size, out_size);

    float lr = dense.learning_rate;
    std::vector<float> weights_expected(in_size*out_size), biases_expected(out_size);
    for (int i = 0; i < in_size*out_size; ++i) weights_expected[i] = weights_before[i] + lr * wg_avg_cpu[i];
    for (int j = 0; j < out_size; ++j) biases_expected[j] = biases_before[j] + lr * bg_avg_cpu[j];

    float* prev_errors = dense.backward(input_batch.data(), next_errors_batch.data());

    bool weights_match = true;
    for (int i = 0; i < in_size*out_size; ++i) if (std::fabs(dense.weights[i] - weights_expected[i]) > 1e-4f) weights_match = false;
    CHECK(weights_match, "batch backward: updated weights match one averaged SGD step over the batch");

    bool biases_match = true;
    for (int j = 0; j < out_size; ++j) if (std::fabs(dense.biases[j] - biases_expected[j]) > 1e-4f) biases_match = false;
    CHECK(biases_match, "batch backward: updated biases match one averaged SGD step over the batch");

    bool dinputs_match = true;
    for (int i = 0; i < batch_size*in_size; ++i) if (std::fabs(prev_errors[i] - dInputs_cpu[i]) > 1e-4f) dinputs_match = false;
    CHECK(dinputs_match, "batch backward: dInputs are per-image (not reduced), match CPU reference for all 6 images");

    free(prev_errors);
}

int main() {
    std::cout << "=== Dense batch tests ===\n\n";
    test_dense_forward_batch_size_1_equivalence();
    test_dense_forward_batch();
    test_dense_backward_batch();
    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}
