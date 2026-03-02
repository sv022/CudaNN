#include"../network/maxpooling.cu"
#include <cassert>


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

void test_maxpool_backward(){

    const int C = 1;
    const int H = 7;
    const int W = 7;

    const int pool = 2;
    const int stride = 2;

    MaxPooling layer(W, H, C, pool, stride);

    float input[H * W] = {
        1.97, 3.8, 1.45, 0.18, 1.82, 0.3, 0, 0.91, 0, 0, 0.16, 0, 0, 0, 0, 4.1, 0, 3.62, 2.32, 0, 3.26, 0, 1.81, 2.48, 0.87, 0, 0, 0, 0, 0, 0, 3.06, 0, 0, 1.78, 4.7, 2.06, 1.63, 0, 0, 0, 1.02, 0, 0.99, 0, 0, 3.53, 3.65, 0
    };

    layer.forward(input);

    float next_errors[9] = {
        0.5, 0.2, 0.1,
        0.4, 0.3, 0.2,
        0.1, 0.6, 0.7
    };

    float* dInput_gpu = layer.backward(input, next_errors);

    float dInput_cpu[H * W];

    maxpool_backward_cpu(
        layer.max_indices,
        next_errors,
        dInput_cpu,
        H * W,
        9
    );

    std::cout << "\n===== CPU dInput =====\n";
    Matrix::log_static(dInput_cpu, H, W, 'C');

    std::cout << "\n===== GPU dInput =====\n";
    Matrix::log_static(dInput_gpu, H, W, 'G');

    for(int i=0;i<16;i++)
        assert(fabs(dInput_cpu[i] - dInput_gpu[i]) < 1e-6);

    std::cout << "\nMaxPool backward PASSED\n";

    free(dInput_gpu);
}

int main() {
    test_maxpool_backward();
    return 0;
}