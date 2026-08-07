#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

#include "../../network/conv.cu"

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

float relu(float x) {
    return x > 0 ? x : 0;
}

void conv_forward_cpu(
    float* input, float* kernel, float* bias, float* output,
    int channels, int in_h, int in_w, int k, int stride, int padding,
    int out_h, int out_w
) {
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            float sum = bias[0];
            for (int c = 0; c < channels; ++c) {
                for (int kh = 0; kh < k; ++kh) {
                    for (int kw = 0; kw < k; ++kw) {
                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            int in_index = c * in_h * in_w + ih * in_w + iw;
                            int k_index = c * k * k + kh * k + kw;
                            sum += input[in_index] * kernel[k_index];
                        }
                    }
                }
            }
            output[oh * out_w + ow] = relu(sum);
        }
    }
}

void conv_backward_cpu(
    float* inputs, float* kernels, float* biases, float* outputs, float* next_errors,
    float* dInputs, float* dKernels, float* dBiases,
    int C, int H, int W, int K, int kernel_size, int stride, int padding, int OH, int OW
) {
    int input_size = C * H * W;
    int kernel_total = K * C * kernel_size * kernel_size;

    for (int i = 0; i < input_size; i++) dInputs[i] = 0.0f;
    for (int i = 0; i < kernel_total; i++) dKernels[i] = 0.0f;
    for (int i = 0; i < K; i++) dBiases[i] = 0.0f;

    for (int k = 0; k < K; k++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int out_index = k*OH*OW + oh*OW + ow;
                float relu_der = outputs[out_index] > 0 ? 1.0f : 0.0f;
                float grad = next_errors[out_index] * relu_der;
                dBiases[k] += grad;

                for (int c = 0; c < C; c++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh*stride + kh - padding;
                            int iw = ow*stride + kw - padding;
                            if (ih < 0 || iw < 0 || ih >= H || iw >= W) continue;

                            int input_idx = c*H*W + ih*W + iw;
                            int kernel_idx = k*(C*kernel_size*kernel_size) + c*(kernel_size*kernel_size) + kh*kernel_size + kw;

                            dKernels[kernel_idx] += inputs[input_idx] * grad;
                            dInputs[input_idx] += kernels[kernel_idx] * grad;
                        }
                    }
                }
            }
        }
    }
}

// Layout: input_batch [batch][C*H*W], output_batch [batch][K*OH*OW]
void conv_forward_cpu_batch(
    float* input_batch, float* kernels, float* biases, float* output_batch,
    int batch_size, int channels, int in_h, int in_w, int k_size, int num_kernels,
    int stride, int padding, int out_h, int out_w
) {
    int input_size = channels * in_h * in_w;
    int output_size = num_kernels * out_h * out_w;
    int kernel_size_per_filter = channels * k_size * k_size;

    for (int b = 0; b < batch_size; ++b) {
        float* inp = input_batch + (size_t)b * input_size;
        float* out = output_batch + (size_t)b * output_size;
        for (int kk = 0; kk < num_kernels; ++kk) {
            conv_forward_cpu(
                inp, kernels + kk * kernel_size_per_filter, &biases[kk],
                out + kk * out_h * out_w,
                channels, in_h, in_w, k_size, stride, padding, out_h, out_w
            );
        }
    }
}

void conv_backward_cpu_batch(
    float* input_batch, float* kernels, float* biases, float* output_batch, float* next_errors_batch,
    float* dInputs_batch,      // [batch][C*H*W]
    float* dKernels_avg,        // [K*C*k*k]
    float* dBiases_avg,         // [K]
    int batch_size, int C, int H, int W, int K, int kernel_size, int stride, int padding, int OH, int OW
) {
    int input_size = C * H * W;
    int kernel_total = K * C * kernel_size * kernel_size;
    int output_size = K * OH * OW;

    std::vector<float> dKernels_sum(kernel_total, 0.0f);
    std::vector<float> dBiases_sum(K, 0.0f);

    for (int b = 0; b < batch_size; ++b) {
        float* inp = input_batch + (size_t)b * input_size;
        float* out = output_batch + (size_t)b * output_size;
        float* next_err = next_errors_batch + (size_t)b * output_size;
        float* dIn = dInputs_batch + (size_t)b * input_size;

        std::vector<float> dK_local(kernel_total);
        std::vector<float> dB_local(K);

        conv_backward_cpu(
            inp, kernels, biases, out, next_err,
            dIn, dK_local.data(), dB_local.data(),
            C, H, W, K, kernel_size, stride, padding, OH, OW
        );

        for (int i = 0; i < kernel_total; ++i) dKernels_sum[i] += dK_local[i];
        for (int i = 0; i < K; ++i) dBiases_sum[i] += dB_local[i];
    }

    for (int i = 0; i < kernel_total; ++i) dKernels_avg[i] = dKernels_sum[i] / (float)batch_size;
    for (int i = 0; i < K; ++i) dBiases_avg[i] = dBiases_sum[i] / (float)batch_size;
}

// ============================================================
// Test 1: batch_size=1
// ============================================================

void test_conv_forward_batch_size_1_equivalence() {
    const int size = 9;
    const int C = 1;
    const int K = 4;
    const int kernel_size = 3;
    const int OH = 7, OW = 7;
    const int input_size = C * size * size;
    const int output_size = K * OH * OW;
    const int kernel_total = K * C * kernel_size * kernel_size;

    Conv conv(size, size, C, kernel_size, K, 1, 0);

    float kernels[kernel_total] = {
        -1,-1,-1, -1,8,-1, -1,-1,-1,
         0,1,0,   1,-4,1,   0,1,0,
         1,0,-1,  0,0,0,   -1,0,1,
         0.2f,0.2f,0.2f, 0.2f,0.2f,0.2f, 0.2f,0.2f,0.2f
    };
    float biases[K] = {0, 1, -1, 0.5f};
    memcpy(conv.kernels, kernels, sizeof(kernels));
    memcpy(conv.biases, biases, sizeof(biases));
    conv.sync_weights_to_device();

    float input[input_size] = {
        0.68f,0.50f,0.92f,0.15f,0.16f,0.98f,0.88f,0.54f,0.52f,
        0.20f,0.81f,0.99f,0.65f,0.46f,0.74f,0.55f,0.10f,0.78f,
        0.15f,0.58f,0.49f,0.02f,0.56f,0.24f,0.27f,0.35f,0.29f,
        0.01f,0.11f,0.97f,0.51f,0.95f,0.75f,0.27f,0.74f,0.44f,
        0.93f,0.43f,0.62f,0.90f,0.67f,0.33f,0.39f,0.40f,0.25f,
        0.67f,0.02f,0.15f,0.06f,0.79f,0.20f,0.08f,0.67f,0.94f,
        0.08f,0.97f,0.65f,0.66f,0.43f,0.01f,0.19f,0.74f,0.59f,
        0.70f,0.10f,0.69f,0.49f,0.39f,0.91f,0.95f,0.57f,0.91f,
        0.09f,0.20f,0.97f,0.49f,0.85f,0.38f,0.55f,0.60f,0.95f
    };

    conv.forward(input);

    float expected[output_size];
    for (int k = 0; k < K; k++) {
        conv_forward_cpu(input, &kernels[k * 9], &biases[k], &expected[k * OH * OW], C, size, size, kernel_size, 1, 0, OH, OW);
    }

    bool all_match = true;
    for (int i = 0; i < output_size; ++i) {
        if (std::fabs(conv.outputs[i] - expected[i]) > 1e-3f) all_match = false;
    }
    CHECK(all_match, "batch_size=1: forward output identical to legacy single-image implementation");
}

// ============================================================
// Test 2: batch forward (batch_size=3)
// ============================================================

void test_conv_forward_batch() {
    const int size = 9;
    const int C = 1;
    const int K = 4;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 0;
    const int OH = 7, OW = 7;
    const int batch_size = 3;

    const int input_size = C * size * size;
    const int output_size = K * OH * OW;
    const int kernel_total = K * C * kernel_size * kernel_size;

    Conv conv(size, size, C, kernel_size, K, stride, padding);
    conv.set_batch_size(batch_size);

    float kernels[kernel_total] = {
        -1,-1,-1, -1,8,-1, -1,-1,-1,
         0,1,0,   1,-4,1,   0,1,0,
         1,0,-1,  0,0,0,   -1,0,1,
         0.2f,0.2f,0.2f, 0.2f,0.2f,0.2f, 0.2f,0.2f,0.2f
    };
    float biases[K] = {0, 1, -1, 0.5f};
    memcpy(conv.kernels, kernels, sizeof(kernels));
    memcpy(conv.biases, biases, sizeof(biases));
    conv.sync_weights_to_device();

    std::vector<float> input_batch(batch_size * input_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < input_size; ++i) {
            input_batch[b * input_size + i] = 0.05f * ((i * (b + 1) + b * 3) % 20);
        }
    }

    conv.forward(input_batch.data());

    std::vector<float> expected(batch_size * output_size);
    conv_forward_cpu_batch(
        input_batch.data(), kernels, biases, expected.data(),
        batch_size, C, size, size, kernel_size, K, stride, padding, OH, OW
    );

    bool all_match = true;
    for (int i = 0; i < batch_size * output_size; ++i) {
        if (std::fabs(conv.outputs[i] - expected[i]) > 1e-3f) {
            all_match = false;
            std::cerr << "  mismatch at flat idx " << i << ": GPU=" << conv.outputs[i] << " CPU=" << expected[i] << "\n";
        }
    }
    CHECK(all_match, "batch forward: GPU output matches CPU reference for all 3 images");
}

// ============================================================
// Test 3: Incomplete batch batch_size
// ============================================================

void test_conv_forward_uneven_batch() {
    const int size = 6;
    const int C = 2;
    const int K = 2;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 1;
    const int OH = 6, OW = 6; // padding=1, stride=1, k=3 -> OH=OW=H=W
    const int batch_size = 5;

    const int input_size = C * size * size;
    const int output_size = K * OH * OW;
    const int kernel_total = K * C * kernel_size * kernel_size;

    Conv conv(size, size, C, kernel_size, K, stride, padding);
    conv.set_batch_size(batch_size);

    for (int i = 0; i < kernel_total; i++) conv.kernels[i] = 0.02f * (i % 11 - 5);
    for (int i = 0; i < K; i++) conv.biases[i] = 0.05f * i;
    conv.sync_weights_to_device();

    std::vector<float> kernels_snapshot(conv.kernels, conv.kernels + kernel_total);
    std::vector<float> biases_snapshot(conv.biases, conv.biases + K);

    std::vector<float> input_batch(batch_size * input_size);
    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < input_size; ++i)
            input_batch[b * input_size + i] = 0.03f * ((i * 3 + b * 7) % 17);

    conv.forward(input_batch.data());

    std::vector<float> expected(batch_size * output_size);
    conv_forward_cpu_batch(
        input_batch.data(), kernels_snapshot.data(), biases_snapshot.data(), expected.data(),
        batch_size, C, size, size, kernel_size, K, stride, padding, OH, OW
    );

    bool all_match = true;
    for (int i = 0; i < batch_size * output_size; ++i) {
        if (std::fabs(conv.outputs[i] - expected[i]) > 1e-3f) all_match = false;
    }
    CHECK(all_match, "uneven batch_size=5 with C=2,K=2,padding=1: GPU forward matches CPU for all images");
}

// ============================================================
// Test 4: batch backward
// ============================================================

void test_conv_backward_batch() {
    const int size = 9;
    const int C = 1;
    const int K = 4;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 0;
    const int OH = 7, OW = 7;
    const int batch_size = 3;

    const int input_size = C * size * size;
    const int output_size = K * OH * OW;
    const int kernel_total = K * C * kernel_size * kernel_size;

    Conv conv(size, size, C, kernel_size, K, stride, padding);
    conv.set_batch_size(batch_size);

    for (int i = 0; i < kernel_total; i++) conv.kernels[i] = 0.01f * (i + 1);
    for (int i = 0; i < K; i++) conv.biases[i] = 0.1f * i;
    conv.sync_weights_to_device();

    std::vector<float> kernels_before(conv.kernels, conv.kernels + kernel_total);
    std::vector<float> biases_before(conv.biases, conv.biases + K);

    std::vector<float> input_batch(batch_size * input_size);
    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < input_size; ++i)
            input_batch[b * input_size + i] = 0.1f * ((i + b * 5) % 7);

    conv.forward(input_batch.data());

    std::vector<float> outputs_snapshot(conv.outputs, conv.outputs + batch_size * output_size);

    std::vector<float> next_errors_batch(batch_size * output_size, 1.0f);

    std::vector<float> dInputs_cpu(batch_size * input_size);
    std::vector<float> dKernels_avg_cpu(kernel_total);
    std::vector<float> dBiases_avg_cpu(K);

    conv_backward_cpu_batch(
        input_batch.data(), kernels_before.data(), biases_before.data(),
        outputs_snapshot.data(), next_errors_batch.data(),
        dInputs_cpu.data(), dKernels_avg_cpu.data(), dBiases_avg_cpu.data(),
        batch_size, C, size, size, K, kernel_size, stride, padding, OH, OW
    );

    float lr = conv.learning_rate;
    std::vector<float> kernels_expected(kernel_total), biases_expected(K);
    for (int i = 0; i < kernel_total; ++i) kernels_expected[i] = kernels_before[i] + lr * dKernels_avg_cpu[i];
    for (int i = 0; i < K; ++i) biases_expected[i] = biases_before[i] + lr * dBiases_avg_cpu[i];

    float* prev_errors = conv.backward(input_batch.data(), next_errors_batch.data());

    bool kernels_match = true;
    for (int i = 0; i < kernel_total; ++i) {
        if (std::fabs(conv.kernels[i] - kernels_expected[i]) > 1e-3f) {
            kernels_match = false;
            std::cerr << "  kernel[" << i << "]: GPU=" << conv.kernels[i] << " expected=" << kernels_expected[i] << "\n";
        }
    }
    CHECK(kernels_match, "batch backward: updated kernels match one averaged SGD step over the batch");

    bool biases_match = true;
    for (int i = 0; i < K; ++i) {
        if (std::fabs(conv.biases[i] - biases_expected[i]) > 1e-3f) {
            biases_match = false;
            std::cerr << "  bias[" << i << "]: GPU=" << conv.biases[i] << " expected=" << biases_expected[i] << "\n";
        }
    }
    CHECK(biases_match, "batch backward: updated biases match one averaged SGD step over the batch");

    bool dinputs_match = true;
    for (int i = 0; i < batch_size * input_size; ++i) {
        if (std::fabs(prev_errors[i] - dInputs_cpu[i]) > 1e-3f) {
            dinputs_match = false;
            std::cerr << "  dInput[" << i << "]: GPU=" << prev_errors[i] << " expected=" << dInputs_cpu[i] << "\n";
        }
    }
    CHECK(dinputs_match, "batch backward: dInputs are per-image (not reduced), match CPU reference for all 3 images");

    free(prev_errors);
}

int main() {
    std::cout << "=== Conv batch tests ===\n\n";

    test_conv_forward_batch_size_1_equivalence();
    test_conv_forward_batch();
    test_conv_forward_uneven_batch();
    test_conv_backward_batch();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}
