#include <cstring>
#include <cmath>
#include <cassert>
#include <iostream>
#include <vector>

#include"../../network/maxpooling.cu"

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

void maxpool_forward_cpu(
    float* inputs,
    float* outputs,
    int* indices,
    int C, int H, int W,
    int pool, int stride,
    int OH, int OW
){
    for(int c=0; c<C; ++c){
        for(int oh=0; oh<OH; ++oh){
            for(int ow=0; ow<OW; ++ow){

                int h0 = oh * stride;
                int w0 = ow * stride;

                float max_val = -1e30f;
                int max_idx = -1;

                for(int kh=0; kh<pool; ++kh){
                    for(int kw=0; kw<pool; ++kw){

                        int ih = h0 + kh;
                        int iw = w0 + kw;

                        int idx = c*H*W + ih*W + iw;

                        if(inputs[idx] > max_val){
                            max_val = inputs[idx];
                            max_idx = idx;
                        }
                    }
                }

                int out_idx = c*OH*OW + oh*OW + ow;
                outputs[out_idx] = max_val;
                indices[out_idx] = max_idx;
            }
        }
    }
}

void maxpool_forward_cpu_batch(
    float* input_batch,
    float* outputs_batch,
    int* indices_batch,
    int batch_size,
    int C, int H, int W,
    int pool, int stride,
    int OH, int OW
){
    int input_size = C * H * W;
    int output_size = C * OH * OW;
    for (int b = 0; b < batch_size; ++b) {
        maxpool_forward_cpu(
            input_batch + (size_t)b * input_size,
            outputs_batch + (size_t)b * output_size,
            indices_batch + (size_t)b * output_size,
            C, H, W, pool, stride, OH, OW
        );
    }
}

void test_maxpool_forward(){

    const int C = 1;
    const int H = 7;
    const int W = 7;

    const int pool = 2;
    const int stride = 2;

    const int OH = 3;
    const int OW = 3;

    const int input_size = C * H * W;
    const int output_size = C * OH * OW;
    const int batch_size = 3;

    MaxPooling pool_layer(
        W, H,
        C,
        pool,
        stride
    );
    pool_layer.set_batch_size(batch_size); // NEW

    float input_orig[H * W] = {
        1.97, 3.8, 1.45, 0.18, 1.82, 0.3, 0, 0.91, 0, 0, 0.16, 0, 0, 0, 0, 4.1, 0, 3.62, 2.32, 0, 3.26, 0, 1.81, 2.48, 0.87, 0, 0, 0, 0, 0, 0, 3.06, 0, 0, 1.78, 4.7, 2.06, 1.63, 0, 0, 0, 1.02, 0, 0.99, 0, 0, 3.53, 3.65, 0
    };

    float input_batch[batch_size * H * W];
    for (int i = 0; i < input_size; ++i) {
        input_batch[0 * input_size + i] = input_orig[i];
        input_batch[1 * input_size + i] = input_orig[i] * 0.5f + 0.1f;
        input_batch[2 * input_size + i] = input_orig[input_size - 1 - i];
    }

    float out_cpu[batch_size * OH * OW];
    int idx_cpu[batch_size * OH * OW];

    maxpool_forward_cpu_batch(
        input_batch,
        out_cpu,
        idx_cpu,
        batch_size,
        C, H, W,
        pool, stride,
        OH, OW
    );

    pool_layer.forward(input_batch);

    std::cout << "\n===== INPUT (batch of 3) =====\n";
    Matrix::log_static(input_batch, batch_size * H, W, 'I');

    std::cout << "\n===== CPU OUTPUT =====\n";
    Matrix::log_static(out_cpu, batch_size * OH, OW, 'C');

    std::cout << "\n===== CUDA OUTPUT =====\n";
    Matrix::log_static(pool_layer.outputs, batch_size * OH, OW, 'G');

    for(int i=0;i<batch_size * output_size;i++){
        assert(fabs(out_cpu[i] - pool_layer.outputs[i]) < 1e-6);
        assert(idx_cpu[i] == pool_layer.max_indices[i]);
    }

    std::cout << "\nMaxPool batch forward test PASSED (batch_size=" << batch_size << ")\n";
}

void maxpool_backward_cpu(
    int* indices,
    float* next_errors,
    float* dInput,
    int input_size,
    int total_outputs
){
    for(int i=0;i<input_size;i++)
        dInput[i]=0.0f;

    for(int o=0;o<total_outputs;o++){
        int idx = indices[o];
        dInput[idx] += next_errors[o];
    }
}

void maxpool_backward_cpu_batch(
    int* indices_batch,
    float* next_errors_batch,
    float* dInput_batch,
    int batch_size,
    int input_size_per_image,
    int output_size_per_image
){
    for (int b = 0; b < batch_size; ++b) {
        maxpool_backward_cpu(
            indices_batch + (size_t)b * output_size_per_image,
            next_errors_batch + (size_t)b * output_size_per_image,
            dInput_batch + (size_t)b * input_size_per_image,
            input_size_per_image,
            output_size_per_image
        );
    }
}

void test_maxpool_backward(){

    const int C = 1;
    const int H = 7;
    const int W = 7;

    const int pool = 2;
    const int stride = 2;

    const int input_size = C * H * W;
    const int batch_size = 3;

    MaxPooling layer(W, H, C, pool, stride);
    layer.set_batch_size(batch_size); 

    float input_orig[H * W] = {
        1.97, 3.8, 1.45, 0.18, 1.82, 0.3, 0, 0.91, 0, 0, 0.16, 0, 0, 0, 0, 4.1, 0, 3.62, 2.32, 0, 3.26, 0, 1.81, 2.48, 0.87, 0, 0, 0, 0, 0, 0, 3.06, 0, 0, 1.78, 4.7, 2.06, 1.63, 0, 0, 0, 1.02, 0, 0.99, 0, 0, 3.53, 3.65, 0
    };

    float input_batch[batch_size * H * W];
    for (int i = 0; i < input_size; ++i) {
        input_batch[0 * input_size + i] = input_orig[i];
        input_batch[1 * input_size + i] = input_orig[i] * 0.5f + 0.1f;
        input_batch[2 * input_size + i] = input_orig[input_size - 1 - i];
    }

    layer.forward(input_batch);

    float next_errors_single[9] = {
        0.5, 0.2, 0.1,
        0.4, 0.3, 0.2,
        0.1, 0.6, 0.7
    };

    float scale[batch_size] = {1.0f, 2.0f, 0.5f};
    float next_errors_batch[batch_size * 9];
    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < 9; ++i)
            next_errors_batch[b * 9 + i] = next_errors_single[i] * scale[b];

    float* dInput_gpu = layer.backward(input_batch, next_errors_batch);

    float dInput_cpu[batch_size * H * W];

    maxpool_backward_cpu_batch(
        layer.max_indices,
        next_errors_batch,
        dInput_cpu,
        batch_size,
        H * W,
        9
    );

    std::cout << "\n===== CPU dInput (batch of 3) =====\n";
    Matrix::log_static(dInput_cpu, batch_size * H, W, 'C');

    std::cout << "\n===== GPU dInput (batch of 3) =====\n";
    Matrix::log_static(dInput_gpu, batch_size * H, W, 'G');

    for(int i=0;i<batch_size * input_size;i++)
        assert(fabs(dInput_cpu[i] - dInput_gpu[i]) < 1e-6);

    for (int i = 0; i < input_size; ++i) {
        float v0 = dInput_gpu[0 * input_size + i];
        float v1 = dInput_gpu[1 * input_size + i];
        assert(fabs(v1 - 2.0f * v0) < 1e-5);
    }

    std::cout << "\nMaxPool batch backward PASSED (batch_size=" << batch_size << ")\n";

    free(dInput_gpu);
}

int main() {
    std::cout << "=== MaxPool batch tests ===\n\n";
    test_maxpool_forward();
    test_maxpool_backward();
    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}