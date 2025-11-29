#include"layer.cu"
#include"convolution/convolution_shared_memory.cuh"


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

    dim3 BLOCK(BLOCK_W, BLOCK_H);
    dim3 GRID(
        (output_width + BLOCK_W - 1) / BLOCK_W,
        (output_height + BLOCK_H - 1) / BLOCK_H,
        num_kernels
    );

    int tileH = BLOCK_H + kernel_size - 1;
    int tileW = BLOCK_W + kernel_size - 1;
    size_t SHMEM = sizeof(float) * (channels * tileH * tileW);

    conv_kernel_shared<<<GRID, BLOCK, SHMEM>>>(
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

