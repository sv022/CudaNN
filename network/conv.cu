#include"layer.cu"
#include"convolution/convolution.cuh"


class Conv : public Layer
{
    private:
    int input_width;
    int input_height;
    int channels;
    int kernel_size;
    int num_kernels;
    int stride;
    int padding;

    int output_width;
    int output_height;

    float *kernels;
    float *biases;
    float learning_rate = 0.1;
    
    public:
    Conv(int input_height, int input_width, int channels, int kernel_size, int n_kernels = 1, int stride = 1, int padding = 0);

    void set_learning_rate(float lr) { learning_rate = lr; };
    
    float* backward(float *inputs, float *targets);
    void forward(float *inputs) override;

    void save_weights(std::string path) override {};
    void load_weights(std::string path, int start) override {};
};


Conv::Conv(int input_h, int input_w, int c, int k, int n_kernels, int stride, int padding) {
    input_width = input_w;
    input_height = input_h;
    channels = c;

    size = input_w * input_h * channels;

    kernel_size = k;
    this->stride = stride;
    this->padding = padding;
    num_kernels = n_kernels;

    output_width = ((input_width + 2 * padding - kernel_size) / stride) + 1;
    output_height = ((input_height + 2 * padding - kernel_size) / stride) + 1;

    output_size = num_kernels * output_height * output_width;
    outputs = (float*)malloc(sizeof(float) * output_size);

    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    kernels = (float*)malloc(sizeof(float) * kernel_total);
    
    biases = (float*)malloc(sizeof(float) * num_kernels);
    
    float init_range = 1.0f / sqrt(channels * kernel_size * kernel_size);
    Matrix::initRandomf_static(kernels, 1, kernel_total, -init_range, init_range);
    Matrix::initRandomf_static(biases, 1, num_kernels, -init_range, init_range);

    // Matrix::log_static(kernels, 1, kernel_total, 'K');
    // Matrix::log_static(biases, 1, num_kernels, 'B');
}

void Conv::forward(float* inputs)
{
    int input_size  = channels * input_height * input_width;
    int kernel_size_total = num_kernels * channels * kernel_size * kernel_size;
    int output_size_total = num_kernels * output_height * output_width;

    float *d_inputs, *d_kernels, *d_biases, *d_outputs;

    cudaMalloc(&d_inputs,  input_size * sizeof(float));
    cudaMalloc(&d_kernels, kernel_size_total * sizeof(float));
    cudaMalloc(&d_biases,  num_kernels * sizeof(float));
    cudaMalloc(&d_outputs, output_size_total * sizeof(float));

    cudaMemcpy(d_inputs,  inputs,  input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, kernels, kernel_size_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases,  biases, num_kernels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_outputs, 0, output_size_total * sizeof(float));

    dim3 BLOCK(16, 16);
    dim3 GRID(
        (output_width + BLOCK.x - 1) / BLOCK.y,
        (output_height + BLOCK.x - 1) / BLOCK.y,
        num_kernels
    );

    conv_forward_kernel<<<GRID, BLOCK>>>(
        d_inputs, d_kernels, d_biases, d_outputs,
        channels, input_height, input_width, 
        kernel_size, stride, padding, 
        num_kernels, output_height, output_width
    );

    cudaDeviceSynchronize();

    cudaMemcpy(outputs, d_outputs, output_size_total * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputs);
    cudaFree(d_kernels);
    cudaFree(d_biases);
    cudaFree(d_outputs);
}

float* Conv::backward(float* inputs, float* next_errors) {
    int input_size = channels * input_height * input_width;
    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    int output_size_total = num_kernels * output_height * output_width;

    float* local_grad = (float*) malloc(sizeof(float) * output_size_total);
    
    for (int i = 0; i < output_size_total; ++i) {
        float relu_derivative = (outputs[i] > 0.0f) ? 1.0f : 0.0f;
        local_grad[i] = next_errors[i] * relu_derivative;
    }

    float *d_inputs = nullptr;
    float *d_kernels = nullptr;
    float *d_biases = nullptr;
    float *d_local_grad = nullptr;

    float *d_dKernels = nullptr;
    float *d_dBiases = nullptr;
    float *d_dInputs = nullptr;

    cudaMalloc(&d_inputs, input_size * sizeof(float));
    cudaMalloc(&d_kernels, kernel_total * sizeof(float));
    cudaMalloc(&d_biases, num_kernels * sizeof(float));
    cudaMalloc(&d_local_grad, output_size_total * sizeof(float));

    cudaMalloc(&d_dKernels, kernel_total * sizeof(float));
    cudaMalloc(&d_dBiases, num_kernels * sizeof(float));
    cudaMalloc(&d_dInputs, input_size * sizeof(float));

    cudaMemcpy(d_inputs, inputs, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernels, kernels, kernel_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, num_kernels * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_local_grad, local_grad, output_size_total * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_dKernels, 0, kernel_total * sizeof(float));
    cudaMemset(d_dBiases, 0, num_kernels * sizeof(float));
    cudaMemset(d_dInputs, 0, input_size * sizeof(float));

    int threads1d = 256;
    dim3 grid1(num_kernels);
    dbias_kernel<<<grid1, threads1d>>>(d_local_grad, d_dBiases, output_height, output_width);
    

    int blocks = (kernel_total + threads1d - 1) / threads1d;
    dkernel_kernel<<<blocks, threads1d>>>(
        d_inputs, d_local_grad, d_dKernels,
        channels, input_height, input_width, kernel_size, 
        stride, padding, num_kernels, 
        output_height, output_width
    );


    dim3 THREADS(16, 16);
    dim3 grid(
        (output_width + THREADS.x - 1) / THREADS.x, 
        (output_height + THREADS.y - 1) / THREADS.y, 
        num_kernels 
    );
    dinput_kernel<<<grid, THREADS>>>(
        d_kernels, d_local_grad, d_dInputs,
        channels, input_height, input_width, kernel_size, 
        stride, padding, num_kernels, 
        output_height, output_width
    );

    cudaDeviceSynchronize();

    int blocks_w = (kernel_total + threads1d - 1) / threads1d;
    update_kernels_kernel<<<blocks_w, threads1d>>>(d_kernels, d_dKernels, learning_rate, kernel_total);

    int blocks_b = (num_kernels + threads1d - 1) / threads1d;
    update_biases_kernel<<<blocks_b, threads1d>>>(d_biases, d_dBiases, learning_rate, num_kernels);

    cudaDeviceSynchronize();

    free(kernels);
    kernels = (float*) malloc(sizeof(float) * kernel_total);
    cudaMemcpy(kernels, d_kernels, kernel_total * sizeof(float), cudaMemcpyDeviceToHost);

    free(biases);
    biases = (float*) malloc(sizeof(float) * num_kernels);
    cudaMemcpy(biases, d_biases, num_kernels * sizeof(float), cudaMemcpyDeviceToHost);

    float* prev_errors = (float*) malloc(sizeof(float) * input_size);
    cudaMemcpy(prev_errors, d_dInputs, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputs);
    cudaFree(d_kernels);
    cudaFree(d_biases);
    cudaFree(d_local_grad);
    cudaFree(d_dKernels);
    cudaFree(d_dBiases);
    cudaFree(d_dInputs);

    free(local_grad);

    return prev_errors;
}
