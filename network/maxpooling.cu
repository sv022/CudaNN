#include"layer.cuh"


class MaxPooling : public Layer
{
    private:
    int channels, input_width, input_height;
    int kernel_size, stride;
    int output_width, output_height;

    int* max_indices;

    float *d_inputs, *d_outputs;
    int *d_indices;
    float *d_next_errors, *d_dInput;
    bool buffers_allocated;

    void allocate_batch_buffers(int bs);
    void free_batch_buffers();

    public:
    MaxPooling(int in_w, int in_h, int channels, int pool=2, int s=2);
    void set_batch_size(int bs) override;

    void forward(float *inputs) override;
    float* backward(float *inputs, float *targets) override;

    ~MaxPooling();

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

    d_inputs = d_outputs = d_next_errors = d_dInput = nullptr;
    d_indices = nullptr;
    buffers_allocated = false;
    allocate_batch_buffers(batch_size);
}

MaxPooling::~MaxPooling() {
    free(outputs);
    free(max_indices);
    free_batch_buffers();
}

void MaxPooling::set_batch_size(int bs) {
    if (bs == batch_size && buffers_allocated) return;
    if (bs != batch_size) {
        free(outputs); free(max_indices);
        outputs = (float*)malloc(sizeof(float) * (size_t)bs * output_size);
        max_indices = (int*)malloc(sizeof(int) * (size_t)bs * output_size);
    }
    batch_size = bs;
    allocate_batch_buffers(bs);
}

void MaxPooling::free_batch_buffers() {
    if (!buffers_allocated) return;
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_indices);
    cudaFree(d_next_errors);
    cudaFree(d_dInput);
    buffers_allocated = false;
}

void MaxPooling::allocate_batch_buffers(int bs) {
    free_batch_buffers();
    int input_size_per_image = channels * input_height * input_width;
    int output_size_per_image = channels * output_height * output_width;
    size_t total_input = (size_t)bs * input_size_per_image;
    size_t total_output = (size_t)bs * output_size_per_image;

    cudaMalloc(&d_inputs, total_input * sizeof(float));
    cudaMalloc(&d_outputs, total_output * sizeof(float));
    cudaMalloc(&d_indices, total_output * sizeof(int));
    cudaMalloc(&d_next_errors, total_output * sizeof(float));
    cudaMalloc(&d_dInput, total_input * sizeof(float));
    buffers_allocated = true;
}

void MaxPooling::forward(float* inputs) {
    int input_size_per_image  = channels * input_height * input_width;
    int output_size_per_image = channels * output_height * output_width;
    size_t total_input_size  = (size_t)batch_size * input_size_per_image;
    size_t total_output_size = (size_t)batch_size * output_size_per_image;

    cudaMemcpy(d_inputs, inputs, total_input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_outputs, 0, total_output_size * sizeof(float));

    dim3 BLOCK(16, 16);
    dim3 GRID(
        (output_width  + BLOCK.x - 1) / BLOCK.x,
        (output_height + BLOCK.y - 1) / BLOCK.y,
        channels * batch_size
    );

    maxpool_forward_kernel<<<GRID, BLOCK>>>(
        d_inputs, d_outputs, d_indices,
        channels, input_height, input_width,
        kernel_size, stride, output_height, output_width,
        batch_size
    );

    cudaDeviceSynchronize();
    cudaMemcpy(outputs, d_outputs, total_output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_indices, d_indices, total_output_size * sizeof(int), cudaMemcpyDeviceToHost);
}

float* MaxPooling::backward(float* inputs, float* next_errors) {
    int input_size_per_image  = channels * input_height * input_width;
    int output_size_per_image = channels * output_height * output_width;
    size_t total_input_size  = (size_t)batch_size * input_size_per_image;
    size_t total_output_size = (size_t)batch_size * output_size_per_image;

    cudaMemcpy(d_next_errors, next_errors, total_output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_dInput, 0, total_input_size * sizeof(float));

    int threads = 256;
    int blocks = (total_output_size + threads - 1) / threads;

    maxpool_backward_kernel<<<blocks, threads>>>(
        d_indices, d_next_errors, d_dInput,
        output_size_per_image, input_size_per_image, total_output_size
    );

    cudaDeviceSynchronize();

    float* prev_errors = (float*) malloc(sizeof(float) * total_input_size);
    cudaMemcpy(prev_errors, d_dInput, total_input_size * sizeof(float), cudaMemcpyDeviceToHost);

    return prev_errors;
}
