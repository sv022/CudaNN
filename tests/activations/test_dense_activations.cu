#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include "../../network/dense.cu"

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

// ---------------- CPU reference implementations ----------------

float sigmoid_cpu(float x) { return 1.0f/(1.0f+expf(-x)); }
float relu_cpu(float x) { return x > 0.0f ? x : 0.0f; }

void dense_forward_cpu(float* inputs, float* weights, float* biases, float* outputs,
                        int in_size, int out_size, ActivationType act) {
    std::vector<float> z(out_size);
    for (int j = 0; j < out_size; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < in_size; ++k) sum += inputs[k] * weights[k*out_size+j];
        z[j] = sum + biases[j];
    }
    switch (act) {
        case ActivationType::Sigmoid:
            for (int j = 0; j < out_size; ++j) outputs[j] = sigmoid_cpu(z[j]);
            break;
        case ActivationType::ReLU:
            for (int j = 0; j < out_size; ++j) outputs[j] = relu_cpu(z[j]);
            break;
        case ActivationType::Linear:
            for (int j = 0; j < out_size; ++j) outputs[j] = z[j];
            break;
        case ActivationType::Softmax: {
            float m = z[0];
            for (int j = 1; j < out_size; ++j) if (z[j] > m) m = z[j];
            float sum = 0.0f;
            std::vector<float> e(out_size);
            for (int j = 0; j < out_size; ++j) { e[j] = expf(z[j]-m); sum += e[j]; }
            for (int j = 0; j < out_size; ++j) outputs[j] = e[j]/sum;
            break;
        }
    }
}

void dense_backward_cpu(float* inputs, float* weights, float* biases, float* outputs,
                         float* next_errors, float* dInputs, float* weight_grad,
                         float* bias_grad, int in_size, int out_size,
                         ActivationType act, bool raw_gradient) {
    std::vector<float> local_grad(out_size);
    if (raw_gradient) {
        for (int j = 0; j < out_size; ++j) local_grad[j] = next_errors[j];
    } else {
        switch (act) {
            case ActivationType::Sigmoid:
                for (int j = 0; j < out_size; ++j)
                    local_grad[j] = next_errors[j] * outputs[j] * (1.0f - outputs[j]);
                break;
            case ActivationType::ReLU:
                for (int j = 0; j < out_size; ++j)
                    local_grad[j] = next_errors[j] * (outputs[j] > 0.0f ? 1.0f : 0.0f);
                break;
            case ActivationType::Linear:
                for (int j = 0; j < out_size; ++j) local_grad[j] = next_errors[j];
                break;
            case ActivationType::Softmax:
                break;
        }
    }

    for (int i = 0; i < in_size; ++i)
        for (int j = 0; j < out_size; ++j)
            weight_grad[i*out_size+j] = inputs[i] * local_grad[j];

    for (int j = 0; j < out_size; ++j) bias_grad[j] = local_grad[j];

    for (int i = 0; i < in_size; ++i) {
        float s = 0.0f;
        for (int j = 0; j < out_size; ++j) s += local_grad[j] * weights[i*out_size+j];
        dInputs[i] = s;
    }
}

// ---------------- Shared fixture ----------------

const int IN_SIZE = 5, OUT_SIZE = 4;

void make_weights(float* W, float* b_sigmoid_style) {
    for (int i = 0; i < IN_SIZE*OUT_SIZE; ++i) W[i] = 0.05f * ((i % 13) - 6);
    for (int j = 0; j < OUT_SIZE; ++j) b_sigmoid_style[j] = 0.02f * j;
}

void make_input(float* x) {
    for (int i = 0; i < IN_SIZE; ++i) x[i] = 0.1f * ((i*3) % 7) - 0.3f;
}

// ---------------- Test 1: Sigmoid forward+backward (raw_gradient=false) ----------------

void test_sigmoid_forward_backward() {
    Dense dense(IN_SIZE, OUT_SIZE, ActivationType::Sigmoid);

    float W[IN_SIZE*OUT_SIZE], b[OUT_SIZE], x[IN_SIZE];
    make_weights(W, b);
    make_input(x);
    memcpy(dense.weights, W, sizeof(W));
    memcpy(dense.biases, b, sizeof(b));
    dense.sync_weights_to_device();

    dense.forward(x);

    float expected_out[OUT_SIZE];
    dense_forward_cpu(x, W, b, expected_out, IN_SIZE, OUT_SIZE, ActivationType::Sigmoid);

    bool fwd_match = true;
    for (int j = 0; j < OUT_SIZE; ++j)
        if (std::fabs(dense.outputs[j] - expected_out[j]) > 1e-4f) fwd_match = false;
    CHECK(fwd_match, "Sigmoid: forward output matches CPU reference");

    float next_errors[OUT_SIZE] = {0.3f, -0.2f, 0.1f, 0.4f};
    float expected_dInputs[IN_SIZE], expected_wg[IN_SIZE*OUT_SIZE], expected_bg[OUT_SIZE];
    dense_backward_cpu(x, W, b, expected_out, next_errors, expected_dInputs,
                        expected_wg, expected_bg, IN_SIZE, OUT_SIZE,
                        ActivationType::Sigmoid, false);

    float lr = dense.learning_rate;
    float expected_W[IN_SIZE*OUT_SIZE], expected_b[OUT_SIZE];
    for (int idx = 0; idx < IN_SIZE*OUT_SIZE; ++idx) expected_W[idx] = W[idx] + lr*expected_wg[idx];
    for (int j = 0; j < OUT_SIZE; ++j) expected_b[j] = b[j] + lr*expected_bg[j];

    float* prev_errors = dense.backward(x, next_errors, false);

    bool w_match = true;
    for (int idx = 0; idx < IN_SIZE*OUT_SIZE; ++idx)
        if (std::fabs(dense.weights[idx]-expected_W[idx]) > 1e-4f) w_match = false;
    CHECK(w_match, "Sigmoid: updated weights match one SGD step");

    bool b_match = true;
    for (int j = 0; j < OUT_SIZE; ++j)
        if (std::fabs(dense.biases[j]-expected_b[j]) > 1e-4f) b_match = false;
    CHECK(b_match, "Sigmoid: updated biases match one SGD step");

    bool din_match = true;
    for (int i = 0; i < IN_SIZE; ++i)
        if (std::fabs(prev_errors[i]-expected_dInputs[i]) > 1e-4f) din_match = false;
    CHECK(din_match, "Sigmoid: dInputs match CPU reference");

    free(prev_errors);
}

// ---------------- Test 2: ReLU forward+backward (mixed-sign z, raw_gradient=false) ----------------

void test_relu_forward_backward() {
    Dense dense(IN_SIZE, OUT_SIZE, ActivationType::ReLU);

    float W[IN_SIZE*OUT_SIZE], b_dummy[OUT_SIZE], x[IN_SIZE];
    make_weights(W, b_dummy);
    make_input(x);
    float b[OUT_SIZE] = {0.5f, -0.3f, 0.1f, -0.6f};
    memcpy(dense.weights, W, sizeof(W));
    memcpy(dense.biases, b, sizeof(b));
    dense.sync_weights_to_device();

    dense.forward(x);

    float expected_out[OUT_SIZE];
    dense_forward_cpu(x, W, b, expected_out, IN_SIZE, OUT_SIZE, ActivationType::ReLU);

    bool fwd_match = true;
    for (int j = 0; j < OUT_SIZE; ++j)
        if (std::fabs(dense.outputs[j] - expected_out[j]) > 1e-4f) fwd_match = false;
    CHECK(fwd_match, "ReLU: forward output matches CPU reference (includes zeroed negatives)");

    bool has_zero = false, has_nonzero = false;
    for (int j = 0; j < OUT_SIZE; ++j) {
        if (dense.outputs[j] == 0.0f) has_zero = true;
        if (dense.outputs[j] > 0.0f) has_nonzero = true;
    }
    CHECK(has_zero && has_nonzero, "ReLU: fixture actually exercises both zero and positive branches");

    float next_errors[OUT_SIZE] = {0.5f, 0.3f, -0.4f, 0.2f};
    float expected_dInputs[IN_SIZE], expected_wg[IN_SIZE*OUT_SIZE], expected_bg[OUT_SIZE];
    dense_backward_cpu(x, W, b, expected_out, next_errors, expected_dInputs,
                        expected_wg, expected_bg, IN_SIZE, OUT_SIZE,
                        ActivationType::ReLU, false);

    float lr = dense.learning_rate;
    float expected_W[IN_SIZE*OUT_SIZE], expected_b[OUT_SIZE];
    for (int idx = 0; idx < IN_SIZE*OUT_SIZE; ++idx) expected_W[idx] = W[idx] + lr*expected_wg[idx];
    for (int j = 0; j < OUT_SIZE; ++j) expected_b[j] = b[j] + lr*expected_bg[j];

    float* prev_errors = dense.backward(x, next_errors, false);

    bool w_match = true;
    for (int idx = 0; idx < IN_SIZE*OUT_SIZE; ++idx)
        if (std::fabs(dense.weights[idx]-expected_W[idx]) > 1e-4f) w_match = false;
    CHECK(w_match, "ReLU: updated weights match one SGD step (gradient blocked where output==0)");

    bool din_match = true;
    for (int i = 0; i < IN_SIZE; ++i)
        if (std::fabs(prev_errors[i]-expected_dInputs[i]) > 1e-4f) din_match = false;
    CHECK(din_match, "ReLU: dInputs match CPU reference");

    free(prev_errors);
}

// ---------------- Test 3: Linear forward+backward (derivative=1, raw_gradient=false) ----------------

void test_linear_forward_backward() {
    Dense dense(IN_SIZE, OUT_SIZE, ActivationType::Linear);

    float W[IN_SIZE*OUT_SIZE], b[OUT_SIZE], x[IN_SIZE];
    make_weights(W, b);
    make_input(x);
    memcpy(dense.weights, W, sizeof(W));
    memcpy(dense.biases, b, sizeof(b));
    dense.sync_weights_to_device();

    dense.forward(x);

    float expected_out[OUT_SIZE];
    dense_forward_cpu(x, W, b, expected_out, IN_SIZE, OUT_SIZE, ActivationType::Linear);

    bool fwd_match = true;
    for (int j = 0; j < OUT_SIZE; ++j)
        if (std::fabs(dense.outputs[j] - expected_out[j]) > 1e-4f) fwd_match = false;
    CHECK(fwd_match, "Linear: forward output equals pre-activation z (no nonlinearity applied)");

    float next_errors[OUT_SIZE] = {0.2f, -0.1f, 0.3f, 0.15f};
    float* prev_errors = dense.backward(x, next_errors, false);

    float expected_dInputs[IN_SIZE];
    for (int i = 0; i < IN_SIZE; ++i) {
        float s = 0.0f;
        for (int j = 0; j < OUT_SIZE; ++j) s += next_errors[j] * W[i*OUT_SIZE+j];
        expected_dInputs[i] = s;
    }

    bool din_match = true;
    for (int i = 0; i < IN_SIZE; ++i)
        if (std::fabs(prev_errors[i]-expected_dInputs[i]) > 1e-4f) din_match = false;
    CHECK(din_match, "Linear: dInputs use next_errors directly (derivative=1)");

    free(prev_errors);
}

// ---------------- Test 4: Softmax forward -- sums to 1 for both normal and extreme logits ----------------

void test_softmax_forward_sums_to_one() {
    Dense dense(IN_SIZE, OUT_SIZE, ActivationType::Softmax);

    float W[IN_SIZE*OUT_SIZE], b[OUT_SIZE], x[IN_SIZE];
    make_weights(W, b);
    make_input(x);
    memcpy(dense.weights, W, sizeof(W));
    memcpy(dense.biases, b, sizeof(b));
    dense.sync_weights_to_device();

    dense.forward(x);

    float expected_out[OUT_SIZE];
    dense_forward_cpu(x, W, b, expected_out, IN_SIZE, OUT_SIZE, ActivationType::Softmax);

    bool fwd_match = true;
    for (int j = 0; j < OUT_SIZE; ++j)
        if (std::fabs(dense.outputs[j] - expected_out[j]) > 1e-4f) fwd_match = false;
    CHECK(fwd_match, "Softmax: forward output matches CPU reference");

    float sum = 0.0f;
    for (int j = 0; j < OUT_SIZE; ++j) sum += dense.outputs[j];
    CHECK(std::fabs(sum - 1.0f) < 1e-4f, "Softmax: output sums to 1.0 (normal logits)");

    Dense dense_extreme(IN_SIZE, OUT_SIZE, ActivationType::Softmax);
    float W_extreme[IN_SIZE*OUT_SIZE];
    for (int i = 0; i < IN_SIZE*OUT_SIZE; ++i) W_extreme[i] = W[i] * 500.0f;
    memcpy(dense_extreme.weights, W_extreme, sizeof(W_extreme));
    memcpy(dense_extreme.biases, b, sizeof(b));

    dense_extreme.forward(x);

    bool no_nan = true, no_inf = true;
    float sum_extreme = 0.0f;
    for (int j = 0; j < OUT_SIZE; ++j) {
        if (std::isnan(dense_extreme.outputs[j])) no_nan = false;
        if (std::isinf(dense_extreme.outputs[j])) no_inf = false;
        sum_extreme += dense_extreme.outputs[j];
    }
    CHECK(no_nan && no_inf, "Softmax: no NaN/Inf with extreme pre-activation values (z~70)");
    CHECK(std::fabs(sum_extreme - 1.0f) < 1e-3f, "Softmax: output sums to 1.0 even with extreme logits");
}

// ---------------- Test 5: Softmax backward requires raw_gradient=true ----------------

void run_softmax_backward_without_raw_gradient_should_abort() {
    Dense dense(IN_SIZE, OUT_SIZE, ActivationType::Softmax);
    float W[IN_SIZE*OUT_SIZE], b[OUT_SIZE], x[IN_SIZE];
    make_weights(W, b);
    make_input(x);
    memcpy(dense.weights, W, sizeof(W));
    memcpy(dense.biases, b, sizeof(b));
    dense.sync_weights_to_device();
    dense.forward(x);

    float next_errors[OUT_SIZE] = {0.1f, 0.1f, 0.1f, 0.1f};
    float *r = dense.backward(x, next_errors, false);
    free(r);
}

// ---------------- Test 6: Softmax backward with raw_gradient=true works correctly ----------------

void test_softmax_backward_with_raw_gradient() {
    Dense dense(IN_SIZE, OUT_SIZE, ActivationType::Softmax);

    float W[IN_SIZE*OUT_SIZE], b[OUT_SIZE], x[IN_SIZE];
    make_weights(W, b);
    make_input(x);
    memcpy(dense.weights, W, sizeof(W));
    memcpy(dense.biases, b, sizeof(b));
    dense.sync_weights_to_device();

    dense.forward(x);

    float target[OUT_SIZE] = {0.0f, 1.0f, 0.0f, 0.0f};
    float next_errors[OUT_SIZE];
    for (int j = 0; j < OUT_SIZE; ++j) next_errors[j] = target[j] - dense.outputs[j];

    float* prev_errors = dense.backward(x, next_errors, true);

    float expected_dInputs[IN_SIZE];
    for (int i = 0; i < IN_SIZE; ++i) {
        float s = 0.0f;
        for (int j = 0; j < OUT_SIZE; ++j) s += next_errors[j] * W[i*OUT_SIZE+j];
        expected_dInputs[i] = s;
    }

    bool din_match = true;
    for (int i = 0; i < IN_SIZE; ++i)
        if (std::fabs(prev_errors[i]-expected_dInputs[i]) > 1e-4f) din_match = false;
    CHECK(din_match, "Softmax+raw_gradient=true: dInputs use next_errors directly, no crash");

    free(prev_errors);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--abort-check") {
        run_softmax_backward_without_raw_gradient_should_abort();
        return 0;
    }

    std::cout << "=== Dense activation tests ===\n\n";
    test_sigmoid_forward_backward();
    test_relu_forward_backward();
    test_linear_forward_backward();
    test_softmax_forward_sums_to_one();
    test_softmax_backward_with_raw_gradient();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    std::cout << "\nNOTE: guard test (Softmax backward without raw_gradient must abort)\n"
                 "is NOT run here since it crashes the process by design.\n"
                 "Run with --abort-check in a separate process and verify non-zero exit code.\n";

    return g_tests_failed == 0 ? 0 : 1;
}
