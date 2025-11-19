#include"layer.cu"


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
    
    public:
    Conv(int input_height, int input_width, int channels, int kernel_size, int n_kernels = 1, int stride = 1, int padding = 0);
    
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

void Conv::forward(float *inputs) {
    memset(outputs, 0, sizeof(float) * output_size);

    for (int n = 0; n < num_kernels; n++)
    {
        for (int i = 0; i < output_height; i++)
        {
            for (int j = 0; j < output_width; j++)
            {
                float sum = biases[n];

                int in_y_origin = i * stride - padding;
                int in_x_origin = j * stride - padding;

                for (int c = 0; c < channels; c++)
                {
                    for (int u = 0; u < kernel_size; u++)
                    {
                        for (int v = 0; v < kernel_size; v++)
                        {
                            int in_y = in_y_origin + u;
                            int in_x = in_x_origin + v;

                            if (in_y < 0 || in_y >= input_height || in_x < 0 || in_x >= input_width)
                                continue;

                            float val = inputs[c * input_height * input_width + in_y * input_width + in_x];
                            float w = kernels[
                                n * (channels*kernel_size*kernel_size) + c * (kernel_size*kernel_size) + u * kernel_size + v
                            ];
                            sum += val * w;
                        }
                    }
                }

                outputs[n * (output_height*output_width) + i * output_width + j] = sum;
            }
        }
    }
    // Matrix::log_static(outputs, 1, output_size, 'C');
}
