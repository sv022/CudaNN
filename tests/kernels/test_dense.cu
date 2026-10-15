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

static float rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    float w = 0; for (size_t i = 0; i < a.size(); ++i) w = std::max(w, fabsf(a[i]-b[i]) / (1.0f + fabsf(b[i])));
    return w;
}

static void check_dense_kernels(int B, int IN, int OUT) {
    std::vector<float> X(B*IN), W(IN*OUT), bias(OUT), E(B*OUT);
    for (size_t i = 0; i < X.size(); ++i) X[i] = 0.01f * (float)((i*7) % 29) - 0.1f;
    for (size_t i = 0; i < W.size(); ++i) W[i] = 0.02f * (float)((i*5) % 23) - 0.2f;
    for (int j = 0; j < OUT; ++j) bias[j] = 0.1f * j - 0.3f;
    for (size_t i = 0; i < E.size(); ++i) E[i] = 0.03f * (float)((i*3) % 19) - 0.25f;

    std::vector<float> y(B*OUT), dW(IN*OUT, 0), dB(OUT, 0), dX(B*IN, 0);
    for (int b = 0; b < B; ++b) for (int j = 0; j < OUT; ++j) {
        float s = bias[j];
        for (int k = 0; k < IN; ++k) s += X[b*IN+k] * W[k*OUT+j];
        y[b*OUT+j] = s;
    }
    for (int i = 0; i < IN; ++i) for (int j = 0; j < OUT; ++j)
        for (int b = 0; b < B; ++b) dW[i*OUT+j] += X[b*IN+i] * E[b*OUT+j];
    for (int j = 0; j < OUT; ++j) for (int b = 0; b < B; ++b) dB[j] += E[b*OUT+j];
    for (int b = 0; b < B; ++b) for (int i = 0; i < IN; ++i)
        for (int j = 0; j < OUT; ++j) dX[b*IN+i] += E[b*OUT+j] * W[i*OUT+j];

    float *dx_, *dw_, *db_, *de_, *dy_, *dgw, *dgb, *dgx;
    cudaMalloc(&dx_, X.size()*4); cudaMalloc(&dw_, W.size()*4); cudaMalloc(&db_, bias.size()*4);
    cudaMalloc(&de_, E.size()*4); cudaMalloc(&dy_, y.size()*4);
    cudaMalloc(&dgw, dW.size()*4); cudaMalloc(&dgb, dB.size()*4); cudaMalloc(&dgx, dX.size()*4);
    cudaMemcpy(dx_, X.data(), X.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw_, W.data(), W.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(db_, bias.data(), bias.size()*4, cudaMemcpyHostToDevice);
    cudaMemcpy(de_, E.data(), E.size()*4, cudaMemcpyHostToDevice);

    std::string tag = "B=" + std::to_string(B) + " IN=" + std::to_string(IN) + " OUT=" + std::to_string(OUT);
    std::vector<float> got(y.size());

    dim3 THREADS(32, 32), grid((OUT+31)/32, (B+31)/32);
    Kernel::dot_bias<<<grid, THREADS>>>(dx_, dw_, db_, dy_, B, IN, OUT);
    cudaMemcpy(got.data(), dy_, got.size()*4, cudaMemcpyDeviceToHost);
    CHECK(rel_err(got, y) < 1e-4f, "Kernel::dot_bias (" + tag + ")");

    dense_dweights_kernel<<<dim3((IN+127)/128, OUT), 128>>>(dx_, de_, dgw, B, IN, OUT);
    dense_dbias_kernel<<<OUT, 256>>>(de_, dgb, B, OUT);
    dense_dinput_kernel<<<dim3((IN+127)/128, B), 128>>>(de_, dw_, dgx, B, IN, OUT);
    std::vector<float> gw(dW.size()), gb(dB.size()), gx(dX.size());
    cudaMemcpy(gw.data(), dgw, gw.size()*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gb.data(), dgb, gb.size()*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(gx.data(), dgx, gx.size()*4, cudaMemcpyDeviceToHost);
    CHECK(rel_err(gw, dW) < 1e-4f, "dense_dweights_kernel (" + tag + ")");
    CHECK(rel_err(gb, dB) < 1e-4f, "dense_dbias_kernel (" + tag + ")");
    CHECK(rel_err(gx, dX) < 1e-4f, "dense_dinput_kernel (" + tag + ")");

    cudaFree(dx_); cudaFree(dw_); cudaFree(db_); cudaFree(de_); cudaFree(dy_);
    cudaFree(dgw); cudaFree(dgb); cudaFree(dgx);
}

int main() {
    std::cout << "=== Dense batch tests ===\n\n";
    check_dense_kernels(7, 37, 10);
    check_dense_kernels(5, 20, 100);
    check_dense_kernels(1, 13, 3);
    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}
