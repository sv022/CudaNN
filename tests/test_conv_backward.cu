#include"../network/conv.cu"
#include <cassert>


void conv_backward_cpu(
    float* inputs,
    float* kernels,
    float* biases,
    float* outputs,
    float* next_errors,
    float* dInputs,
    float* dKernels,
    float* dBiases,
    int C, int H, int W,
    int K,
    int kernel_size,
    int stride,
    int padding,
    int OH, int OW
){
    int input_size = C * H * W;
    int kernel_total = K * C * kernel_size * kernel_size;

    for(int i=0;i<input_size;i++) dInputs[i]=0.0f;
    for(int i=0;i<kernel_total;i++) dKernels[i]=0.0f;
    for(int i=0;i<K;i++) dBiases[i]=0.0f;

    for(int k=0;k<K;k++){
        for(int oh=0;oh<OH;oh++){
            for(int ow=0;ow<OW;ow++){

                int out_index = k*OH*OW + oh*OW + ow;

                float relu_der = outputs[out_index] > 0 ? 1.0f : 0.0f;
                float grad = next_errors[out_index] * relu_der;

                dBiases[k] += grad;

                for(int c=0;c<C;c++){
                    for(int kh=0;kh<kernel_size;kh++){
                        for(int kw=0;kw<kernel_size;kw++){

                            int ih = oh*stride + kh - padding;
                            int iw = ow*stride + kw - padding;

                            if(ih < 0 || iw < 0 || ih >= H || iw >= W)
                                continue;

                            int input_idx = c*H*W + ih*W + iw;
                            int kernel_idx =
                                k*(C*kernel_size*kernel_size) +
                                c*(kernel_size*kernel_size) +
                                kh*kernel_size + kw;

                            dKernels[kernel_idx] += inputs[input_idx] * grad;
                            dInputs[input_idx] += kernels[kernel_idx] * grad;
                        }
                    }
                }
            }
        }
    }
}

void test_conv_backward() {

    const int size = 9;
    const int C = 1;
    const int K = 4;
    const int kernel_size = 3;
    const int stride = 1;
    const int padding = 0;

    Conv conv(
        size, size,
        C,
        kernel_size,
        K,
        stride,
        padding
    );

    const int OH = 7;
    const int OW = 7;

    const int input_size = C * size * size;
    const int kernel_total = K * C * kernel_size * kernel_size;
    const int output_size = K * OH * OW;

    float input[input_size];
    for(int i=0;i<input_size;i++)
        input[i] = 0.1f * (i % 7);

    for(int i=0;i<kernel_total;i++)
        conv.kernels[i] = 0.01f * (i+1);

    for(int i=0;i<K;i++)
        conv.biases[i] = 0.1f * i;

    conv.forward(input);

    float next_errors[output_size];
    for(int i=0;i<output_size;i++)
        next_errors[i] = 1.0f;

    float dInputs_cpu[input_size];
    float dKernels_cpu[kernel_total];
    float dBias_cpu[K];

    conv_backward_cpu(
        input,
        conv.kernels,
        conv.biases,
        conv.outputs,
        next_errors,
        dInputs_cpu,
        dKernels_cpu,
        dBias_cpu,
        C, size, size,
        K,
        kernel_size,
        stride,
        padding,
        OH, OW
    );

    // ===== CPU backward =====

    std::cout << "\n===== CPU dInputs =====\n"; 
    Matrix::log_static(dInputs_cpu, size, size, 'D'); 
    
    std::cout << "\n===== CPU dKernels =====\n"; 
    Matrix::log_static(dKernels_cpu, 1, kernel_total, 'K'); 
    
    std::cout << "\n===== CPU dBiases =====\n"; 
    Matrix::log_static(dBias_cpu, 1, K, 'B');
    
    // ===== CUDA backward ===== 
    float* prev_errors = conv.backward(input, next_errors);

    std::cout << "\n===== CUDA dInputs =====\n"; 
    Matrix::log_static(prev_errors, size, size, 'D'); 
    
    std::cout << "\n===== UPDATED KERNEL =====\n"; 
    Matrix::log_static(conv.kernels, 1, kernel_total, 'K'); 

    std::cout << "\n===== UPDATED BIASES =====\n"; 
    Matrix::log_static(conv.biases, 1, K, 'B');

    std::cout << "\n===== CUDA OUTPUTS =====\n";
    Matrix::log_static(conv.outputs, OH, OW, 'O');


    free(prev_errors);
}

int main() {
    test_conv_backward();
    return 0;
}