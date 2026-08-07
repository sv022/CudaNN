#include <cstring>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
#include <fstream>

#include "../../network/conv.cu"
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

// ============================================================
// Test 1: Conv save_weights -> load_weights -> forward
// ============================================================

void test_conv_save_load_roundtrip() {
    const std::string path = "data/test_conv_weights.bin";
    std::remove(path.c_str());

    const int size = 9;
    const int C = 1;
    const int K = 4;
    const int kernel_size = 3;
    const int OH = 7, OW = 7;
    const int input_size = C * size * size;
    const int output_size = K * OH * OW;
    const int kernel_total = K * C * kernel_size * kernel_size;

    Conv conv_orig(size, size, C, kernel_size, K, 1, 0);
    for (int i = 0; i < kernel_total; i++) conv_orig.kernels[i] = 0.01f * (i % 17 - 8);
    for (int i = 0; i < K; i++) conv_orig.biases[i] = 0.05f * i - 0.1f;
    conv_orig.sync_weights_to_device();

    float input[input_size];
    for (int i = 0; i < input_size; i++) input[i] = 0.03f * ((i * 7) % 23) - 0.3f;

    conv_orig.forward(input);
    std::vector<float> output_before(conv_orig.outputs, conv_orig.outputs + output_size);

    conv_orig.save_weights(path);

    Conv conv_loaded(size, size, C, kernel_size, K, 1, 0);
    int returned = conv_loaded.load_weights(path, 0);

    CHECK(returned == kernel_total + K, "Conv::load_weights returns correct element count (kernel_total + num_kernels)");

    bool host_weights_match = true;
    for (int i = 0; i < kernel_total; i++)
        if (std::fabs(conv_loaded.kernels[i] - conv_orig.kernels[i]) > 1e-6f) host_weights_match = false;
    for (int i = 0; i < K; i++)
        if (std::fabs(conv_loaded.biases[i] - conv_orig.biases[i]) > 1e-6f) host_weights_match = false;
    CHECK(host_weights_match, "Conv::load_weights: host kernels/biases match saved values exactly");

    conv_loaded.forward(input);

    bool forward_matches = true;
    for (int i = 0; i < output_size; i++) {
        if (std::fabs(conv_loaded.outputs[i] - output_before[i]) > 1e-4f) {
            forward_matches = false;
            std::cerr << "  mismatch at " << i << ": loaded=" << conv_loaded.outputs[i] << " original=" << output_before[i] << "\n";
        }
    }
    CHECK(forward_matches, "Conv: forward() after load_weights() matches forward() before save_weights(), device weights are synchronized after loading");

    std::remove(path.c_str());
}

// ============================================================
// Test 2: Dense save_weights -> load_weights -> forward
// ============================================================

void test_dense_save_load_roundtrip() {
    const std::string path = "test_dense_weights.bin";
    std::remove(path.c_str());

    const int in_size = 5, out_size = 4;

    Dense dense_orig(in_size, out_size);
    for (int i = 0; i < in_size*out_size; i++) dense_orig.weights[i] = 0.04f * (i % 11 - 5);
    for (int j = 0; j < out_size; j++) dense_orig.biases[j] = 0.03f * j - 0.05f;
    dense_orig.sync_weights_to_device();

    float input[in_size];
    for (int i = 0; i < in_size; i++) input[i] = 0.1f * ((i*5) % 9) - 0.4f;

    dense_orig.forward(input);
    std::vector<float> output_before(dense_orig.outputs, dense_orig.outputs + out_size);

    dense_orig.save_weights(path);

    Dense dense_loaded(in_size, out_size);
    int returned = dense_loaded.load_weights(path, 0);

    CHECK(returned == in_size*out_size + out_size, "Dense::load_weights returns correct element count (size*output_size + output_size)");

    bool host_weights_match = true;
    for (int i = 0; i < in_size*out_size; i++)
        if (std::fabs(dense_loaded.weights[i] - dense_orig.weights[i]) > 1e-6f) host_weights_match = false;
    for (int j = 0; j < out_size; j++)
        if (std::fabs(dense_loaded.biases[j] - dense_orig.biases[j]) > 1e-6f) host_weights_match = false;
    CHECK(host_weights_match, "Dense::load_weights: host weights/biases match saved values exactly");

    dense_loaded.forward(input);

    bool forward_matches = true;
    for (int j = 0; j < out_size; j++) {
        if (std::fabs(dense_loaded.outputs[j] - output_before[j]) > 1e-4f) {
            forward_matches = false;
            std::cerr << "  mismatch at " << j << ": loaded=" << dense_loaded.outputs[j] << " original=" << output_before[j] << "\n";
        }
    }
    CHECK(forward_matches, "Dense: forward() after load_weights() matches forward() before save_weights(), device weights are synchronized after loading");

    std::remove(path.c_str());
}

// ============================================================
// Test 3: Conv + Dense multilayer (as in NeuralNetwork::save_weights/load_weights)
// ============================================================

void test_multi_layer_save_load_offsets() {
    const std::string path = "test_multi_layer_weights.bin";
    std::remove(path.c_str());

    const int size = 9, C = 1, K = 4, kernel_size = 3;
    const int OH = 7, OW = 7;
    const int input_size = C * size * size;
    const int conv_output_size = K * OH * OW;
    const int kernel_total = K * C * kernel_size * kernel_size;

    const int dense_in = conv_output_size;
    const int dense_out = 3;

    Conv conv_orig(size, size, C, kernel_size, K, 1, 0);
    for (int i = 0; i < kernel_total; i++) conv_orig.kernels[i] = 0.02f * (i % 13 - 6);
    for (int i = 0; i < K; i++) conv_orig.biases[i] = 0.04f * i;
    conv_orig.sync_weights_to_device();

    Dense dense_orig(dense_in, dense_out);
    for (int i = 0; i < dense_in*dense_out; i++) dense_orig.weights[i] = 0.01f * (i % 19 - 9);
    for (int j = 0; j < dense_out; j++) dense_orig.biases[j] = 0.02f * j - 0.03f;
    dense_orig.sync_weights_to_device();

    float input[input_size];
    for (int i = 0; i < input_size; i++) input[i] = 0.02f * ((i*3) % 29) - 0.25f;

    conv_orig.forward(input);
    dense_orig.forward(conv_orig.outputs);
    std::vector<float> dense_output_before(dense_orig.outputs, dense_orig.outputs + dense_out);

    conv_orig.save_weights(path);
    dense_orig.save_weights(path);

    Conv conv_loaded(size, size, C, kernel_size, K, 1, 0);
    int conv_elements = conv_loaded.load_weights(path, 0);

    int bytes_to_skip = (int)(sizeof(float) * conv_elements);

    Dense dense_loaded(dense_in, dense_out);
    int dense_elements = dense_loaded.load_weights(path, bytes_to_skip);

    CHECK(conv_elements == kernel_total + K, "multi-layer: Conv element count correct within combined file");
    CHECK(dense_elements == dense_in*dense_out + dense_out, "multi-layer: Dense element count correct within combined file");

    conv_loaded.forward(input);
    dense_loaded.forward(conv_loaded.outputs);

    bool full_pipeline_matches = true;
    for (int j = 0; j < dense_out; j++) {
        if (std::fabs(dense_loaded.outputs[j] - dense_output_before[j]) > 1e-4f) {
            full_pipeline_matches = false;
            std::cerr << "  mismatch at " << j << ": loaded=" << dense_loaded.outputs[j] << " original=" << dense_output_before[j] << "\n";
        }
    }
    CHECK(full_pipeline_matches, "multi-layer: Conv+Dense loaded from combined file reproduce identical end-to-end output");

    std::remove(path.c_str());
}

int main() {
    std::cout << "=== Weights save/load I/O tests ===\n\n";

    test_conv_save_load_roundtrip();
    std::cout << "\n";
    test_dense_save_load_roundtrip();
    std::cout << "\n";
    test_multi_layer_save_load_offsets();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}
