#include"layer.cu"
#include"convolution/pooling.cuh"

class MaxPooling : public Layer
{
    private:
    int channels;
    int input_width;
    int input_height;

    int kernel_size;
    int stride;

    int output_width;
    int output_height;

    int* max_indices;

    public:
    MaxPooling(int in_w, int in_h, int channels, int pool=2, int s=2);
    void set_batch_size(int bs) override;

    void forward(float *inputs) override;
    float* backward(float *inputs, float *targets) override;

    ~MaxPooling() {
        free(outputs);
        free(max_indices);
    }

    friend void test_maxpool_forward();
    friend void test_maxpool_backward();
};


MaxPooling::MaxPooling(int in_w, int in_h, int channels, int pool, int s) {
    this->input_width  = in_w;
    this->input_height = in_h;
    this->channels = channels;

    this->kernel_size = pool;
    this->stride = s;

    output_width  = (in_w - pool) / s + 1;
    output_height = (in_h - pool) / s + 1;

    size = channels * in_w * in_h;
    output_size = channels * output_width * output_height;

    batch_size = 1;
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * output_size);
    max_indices = (int*)malloc(sizeof(int) * (size_t)batch_size * output_size);
}

void MaxPooling::set_batch_size(int bs) {
    if (bs == batch_size) return;
    batch_size = bs;
    free(outputs);
    free(max_indices);
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * output_size);
    max_indices = (int*)malloc(sizeof(int) * (size_t)batch_size * output_size);
}

void MaxPooling::forward(float* inputs) {
    int input_size_per_image  = channels * input_height * input_width;
    int output_size_per_image = channels * output_height * output_width;

    size_t total_input_size  = (size_t)batch_size * input_size_per_image;
    size_t total_output_size = (size_t)batch_size * output_size_per_image;

    float *d_inputs, *d_outputs;
    int *d_indices;

    cudaMalloc(&d_inputs, total_input_size * sizeof(float));
    cudaMalloc(&d_outputs, total_output_size * sizeof(float));
    cudaMalloc(&d_indices, total_output_size * sizeof(int));

    cudaMemcpy(d_inputs, inputs, total_input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_outputs, 0, total_output_size * sizeof(float));

    dim3 BLOCK(16, 16);
    dim3 GRID(
        (output_width  + BLOCK.x - 1) / BLOCK.x,
        (output_height + BLOCK.y - 1) / BLOCK.y,
        channels * batch_size
    );

    maxpool_forward_kernel<<<GRID, BLOCK>>>(
        d_inputs,
        d_outputs,
        d_indices,
        channels,
        input_height,
        input_width,
        kernel_size,
        stride,
        output_height,
        output_width,
        batch_size
    );

    cudaDeviceSynchronize();

    cudaMemcpy(outputs, d_outputs, total_output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_indices, d_indices, total_output_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_indices);
}

float* MaxPooling::backward(float* inputs, float* next_errors) {
    int input_size_per_image  = channels * input_height * input_width;
    int output_size_per_image = channels * output_height * output_width;

    size_t total_input_size  = (size_t)batch_size * input_size_per_image;
    size_t total_output_size = (size_t)batch_size * output_size_per_image;

    float *d_next = nullptr;
    float *d_dInput = nullptr;
    int *d_indices = nullptr;

    cudaMalloc(&d_next, total_output_size * sizeof(float));
    cudaMalloc(&d_indices, total_output_size * sizeof(int));
    cudaMalloc(&d_dInput, total_input_size * sizeof(float));

    cudaMemcpy(d_next, next_errors, total_output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, max_indices, total_output_size * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_dInput, 0, total_input_size * sizeof(float));

    int threads = 256;
    int blocks = (total_output_size + threads - 1) / threads;

    maxpool_backward_kernel<<<blocks, threads>>>(
        d_indices,
        d_next,
        d_dInput,
        output_size_per_image,
        input_size_per_image,
        total_output_size
    );

    cudaDeviceSynchronize();

    float* prev_errors = (float*) malloc(sizeof(float) * total_input_size);
    cudaMemcpy(prev_errors, d_dInput, total_input_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_next);
    cudaFree(d_indices);
    cudaFree(d_dInput);

    return prev_errors;
}
