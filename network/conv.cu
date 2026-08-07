#include"layer.cuh"


class Conv : public Layer
{
    private:
    int input_width, input_height, channels;
    int kernel_size, num_kernels, stride, padding;
    int output_width, output_height;

    float *kernels, *biases;

    float *d_kernels, *d_biases, *d_dKernels, *d_dBiases;
    float *d_inputs, *d_outputs, *d_local_grad, *d_dInputs;
    bool buffers_allocated;

    void allocate_batch_buffers(int bs);
    void free_batch_buffers();

    public:
    Conv(int input_height, int input_width, int channels, int kernel_size,
         int n_kernels = 1, int stride = 1, int padding = 0, ActivationType activation = ActivationType::ReLU);
    ~Conv();

    void set_batch_size(int bs) override;
    void sync_weights_to_device();

    float* backward(float *inputs, float *targets, bool);
    void forward(float *inputs) override;

    void save_weights(std::string path) override;
    int load_weights(std::string path, int start) override;

    friend void test_conv_forward_batch_size_1_equivalence();
    friend void test_conv_forward_batch();
    friend void test_conv_forward_uneven_batch();
    friend void test_conv_backward_batch();
    
    friend void test_conv_save_load_roundtrip();
    friend void test_multi_layer_save_load_offsets();
};


Conv::Conv(int input_h, int input_w, int c, int k, int n_kernels, int stride, int padding, ActivationType act) {
    input_width = input_w; 
    input_height = input_h; 
    channels = c;

    activation = act;

    size = input_w * input_h * channels;
    kernel_size = k; 
    this->stride = stride; 
    this->padding = padding; 
    num_kernels = n_kernels;

    output_width = ((input_width + 2 * padding - kernel_size) / stride) + 1;
    output_height = ((input_height + 2 * padding - kernel_size) / stride) + 1;
    output_size = num_kernels * output_height * output_width;

    batch_size = 1;
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * output_size);

    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    kernels = (float*)malloc(sizeof(float) * kernel_total);
    biases = (float*)malloc(sizeof(float) * num_kernels);

    float init_range = 1.0f / sqrt(channels * kernel_size * kernel_size);
    Matrix::initRandomf_static(kernels, 1, kernel_total, -init_range, init_range);
    Matrix::initRandomf_static(biases, 1, num_kernels, -init_range, init_range);

    cudaMalloc(&d_kernels, kernel_total * sizeof(float));
    cudaMalloc(&d_biases, num_kernels * sizeof(float));
    cudaMalloc(&d_dKernels, kernel_total * sizeof(float));
    cudaMalloc(&d_dBiases, num_kernels * sizeof(float));

    sync_weights_to_device();

    d_inputs = d_outputs = d_local_grad = d_dInputs = nullptr;
    buffers_allocated = false;
    allocate_batch_buffers(batch_size);
}

Conv::~Conv() {
    free(outputs); 
    free(kernels); 
    free(biases);
    cudaFree(d_kernels); 
    cudaFree(d_biases); 
    cudaFree(d_dKernels); 
    cudaFree(d_dBiases);
    free_batch_buffers();
}

void Conv::set_batch_size(int bs) {
    if (bs == batch_size) return;
    batch_size = bs;
    free(outputs);
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * output_size);
    allocate_batch_buffers(batch_size);
}

void Conv::sync_weights_to_device() {
    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    cudaMemcpy(d_kernels, kernels, kernel_total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, num_kernels * sizeof(float), cudaMemcpyHostToDevice);
}

void Conv::free_batch_buffers() {
    if (!buffers_allocated) return;
    cudaFree(d_inputs); 
    cudaFree(d_outputs); 
    cudaFree(d_local_grad); 
    cudaFree(d_dInputs);
    buffers_allocated = false;
}

void Conv::allocate_batch_buffers(int bs) {
    free_batch_buffers();
    int input_size_per_image = channels * input_height * input_width;
    int output_size_per_image = num_kernels * output_height * output_width;
    size_t total_input = (size_t)bs * input_size_per_image;
    size_t total_output = (size_t)bs * output_size_per_image;

    cudaMalloc(&d_inputs, total_input * sizeof(float));
    cudaMalloc(&d_outputs, total_output * sizeof(float));
    cudaMalloc(&d_local_grad, total_output * sizeof(float));
    cudaMalloc(&d_dInputs, total_input * sizeof(float));
    buffers_allocated = true;
}

void Conv::forward(float* inputs) {
    int input_size_per_image = channels * input_height * input_width;
    int output_size_per_image = num_kernels * output_height * output_width;
    size_t total_input_size = (size_t)batch_size * input_size_per_image;
    size_t total_output_size = (size_t)batch_size * output_size_per_image;

    cudaMemcpy(d_inputs, inputs, total_input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_outputs, 0, total_output_size * sizeof(float));

    dim3 BLOCK(16, 16);
    dim3 GRID(
        (output_width + BLOCK.x - 1) / BLOCK.x,
        (output_height + BLOCK.y - 1) / BLOCK.y,
        num_kernels * batch_size
    );

    conv_forward_kernel<<<GRID, BLOCK>>>(
        d_inputs, d_kernels, d_biases, d_outputs,
        channels, input_height, input_width,
        kernel_size, stride, padding,
        num_kernels, output_height, output_width,
        batch_size, activation
    );

    cudaDeviceSynchronize();
    cudaMemcpy(outputs, d_outputs, total_output_size * sizeof(float), cudaMemcpyDeviceToHost);
}

float* Conv::backward(float* inputs, float* next_errors, bool raw_gradient) {
    int input_size_per_image = channels * input_height * input_width;
    int kernel_total = num_kernels * channels * kernel_size * kernel_size;
    int output_size_per_image = num_kernels * output_height * output_width;
    size_t total_input_size = (size_t)batch_size * input_size_per_image;
    size_t total_output_size = (size_t)batch_size * output_size_per_image;

    float* local_grad = (float*) malloc(sizeof(float) * total_output_size);
    if (raw_gradient) for (size_t i = 0; i < total_output_size; ++i) local_grad[i] = next_errors[i];
    else for (size_t i = 0; i < total_output_size; ++i) local_grad[i] = activation_derivative(next_errors[i], outputs[i], activation);
    

    cudaMemcpy(d_inputs, inputs, total_input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_local_grad, local_grad, total_output_size * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_dKernels, 0, kernel_total * sizeof(float));
    cudaMemset(d_dBiases, 0, num_kernels * sizeof(float));
    cudaMemset(d_dInputs, 0, total_input_size * sizeof(float));

    int threads1d = 256;
    dim3 grid1(num_kernels);
    dbias_kernel<<<grid1, threads1d>>>(d_local_grad, d_dBiases, output_height, output_width, num_kernels, batch_size);

    int blocks = (kernel_total + threads1d - 1) / threads1d;
    dkernel_kernel<<<blocks, threads1d>>>(
        d_inputs, d_local_grad, d_dKernels,
        channels, input_height, input_width, kernel_size,
        stride, padding, num_kernels, output_height, output_width, batch_size
    );

    dim3 THREADS(16, 16);
    dim3 grid(
        (output_width + THREADS.x - 1) / THREADS.x,
        (output_height + THREADS.y - 1) / THREADS.y,
        num_kernels * batch_size
    );
    dinput_kernel<<<grid, THREADS>>>(
        d_kernels, d_local_grad, d_dInputs,
        channels, input_height, input_width, kernel_size,
        stride, padding, num_kernels, output_height, output_width, batch_size
    );

    cudaDeviceSynchronize();

    int blocks_w = (kernel_total + threads1d - 1) / threads1d;
    update_kernels_kernel<<<blocks_w, threads1d>>>(d_kernels, d_dKernels, learning_rate, kernel_total, batch_size);

    int blocks_b = (num_kernels + threads1d - 1) / threads1d;
    update_biases_kernel<<<blocks_b, threads1d>>>(d_biases, d_dBiases, learning_rate, num_kernels, batch_size);

    cudaDeviceSynchronize();

    cudaMemcpy(kernels, d_kernels, kernel_total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(biases, d_biases, num_kernels * sizeof(float), cudaMemcpyDeviceToHost);

    float* prev_errors = (float*) malloc(sizeof(float) * total_input_size);
    cudaMemcpy(prev_errors, d_dInputs, total_input_size * sizeof(float), cudaMemcpyDeviceToHost);

    free(local_grad);
    return prev_errors;
}


void Conv::save_weights(std::string path) {
    std::ofstream file(path, std::ios::binary | std::ios::app);

    int kernel_total = num_kernels * channels * kernel_size * kernel_size;

    file.write(reinterpret_cast<const char*>(kernels), sizeof(float) * kernel_total);
    file.write(reinterpret_cast<const char*>(biases), sizeof(float) * num_kernels);

    file.close();
}

int Conv::load_weights(std::string path, int start) { // return loaded weights size in bytes
    std::ifstream file(path, std::ios::binary);

    file.seekg(start, std::ios::beg);

    int kernel_total = num_kernels * channels * kernel_size * kernel_size;

    free(kernels);
    kernels = (float*)malloc(sizeof(float) * kernel_total);

    file.read(reinterpret_cast<char*>(kernels), sizeof(float) * kernel_total);

    free(biases);
    biases = (float*)malloc(sizeof(float) * num_kernels);

    file.read(reinterpret_cast<char*>(biases), sizeof(float) * num_kernels);

    file.close();

    sync_weights_to_device();

    return kernel_total + num_kernels;
}