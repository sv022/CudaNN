#include"layer.cuh"


class MaxPooling : public Layer
{
    private:
    int channels, input_width, input_height;
    int kernel_size, stride;
    int output_width, output_height;

    float *d_inputs;
    int *d_indices;
    float *d_next_errors, *d_dInput;
    bool buffers_allocated;

    void allocate_batch_buffers(int bs);
    void free_batch_buffers();

    public:
    MaxPooling(int in_w, int in_h, int channels, int pool=2, int s=2);
    void set_batch_size(int bs) override;

    void forward(float *inputs) override;
    float* backward(float *inputs, float *targets, bool raw_grad = false) override;

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

    d_inputs = d_outputs = d_next_errors = d_dInput = nullptr;
    d_indices = nullptr;
    buffers_allocated = false;
    allocate_batch_buffers(batch_size);
}

MaxPooling::~MaxPooling() {
    free(outputs);
    free_batch_buffers();
}

void MaxPooling::set_batch_size(int bs) {
    if (bs == batch_size && buffers_allocated) return;
    batch_size = bs;
    allocate_batch_buffers(bs);
}

void MaxPooling::free_batch_buffers() {
    if (!buffers_allocated) return;
    cudaFree(d_indices); d_indices = nullptr;
    free_io();
    buffers_allocated = false;
}

void MaxPooling::allocate_batch_buffers(int bs) {
    free_batch_buffers();
    cudaMalloc(&d_indices, (size_t)bs * output_size * sizeof(int));
    alloc_io(bs);
    buffers_allocated = true;
}

void MaxPooling::forward(float *d_in) {
    cudaMemset(d_outputs, 0, (size_t)batch_size * output_size * sizeof(float));
    dim3 BLOCK(16, 16);
    dim3 GRID((output_width + BLOCK.x - 1) / BLOCK.x, (output_height + BLOCK.y - 1) / BLOCK.y,
              channels * batch_size);
    maxpool_forward_kernel<<<GRID, BLOCK>>>(d_in, d_outputs, d_indices,
        channels, input_height, input_width, kernel_size, stride, output_height, output_width, batch_size);
}

float* MaxPooling::backward(float *, float *d_err, bool) {
    if (!compute_input_grad) return nullptr;
    size_t total_in  = (size_t)batch_size * size;
    size_t total_out = (size_t)batch_size * output_size;
    cudaMemset(d_prev_errors, 0, total_in * sizeof(float));
    int threads = 256, blocks = (int)((total_out + threads - 1) / threads);
    maxpool_backward_kernel<<<blocks, threads>>>(d_indices, d_err, d_prev_errors, output_size, size, total_out);
    return d_prev_errors;
}
