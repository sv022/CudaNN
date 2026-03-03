#include"../network/conv.cu"

float relu(float x) {
    return x > 0 ? x : 0;
}

void conv_forward_cpu(
    float* input,
    float* kernel,
    float* bias,
    float* output,
    int channels,
    int in_h,
    int in_w,
    int k,
    int stride,
    int padding,
    int out_h,
    int out_w
) {
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {

            float sum = bias[0];

            for (int c = 0; c < channels; ++c) {
                for (int kh = 0; kh < k; ++kh) {
                    for (int kw = 0; kw < k; ++kw) {

                        int ih = oh * stride + kh - padding;
                        int iw = ow * stride + kw - padding;

                        if (ih >= 0 && ih < in_h &&
                            iw >= 0 && iw < in_w)
                        {
                            int in_index =
                                c * in_h * in_w +
                                ih * in_w + iw;

                            int k_index =
                                c * k * k +
                                kh * k + kw;

                            sum += input[in_index] * kernel[k_index];
                        }
                    }
                }
            }

            output[oh * out_w + ow] = relu(sum);
        }
    }
}

void test_conv_forward() {
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
    const int output_size = K * OH * OW;
    const int kernel_total = K * C * kernel_size * kernel_size;

    float input[size * size] = {
        0.68,0.50,0.92,0.15,0.16,0.98,0.88,0.54,0.52,
        0.20,0.81,0.99,0.65,0.46,0.74,0.55,0.10,0.78,
        0.15,0.58,0.49,0.02,0.56,0.24,0.27,0.35,0.29,
        0.01,0.11,0.97,0.51,0.95,0.75,0.27,0.74,0.44,
        0.93,0.43,0.62,0.90,0.67,0.33,0.39,0.40,0.25,
        0.67,0.02,0.15,0.06,0.79,0.20,0.08,0.67,0.94,
        0.08,0.97,0.65,0.66,0.43,0.01,0.19,0.74,0.59,
        0.70,0.10,0.69,0.49,0.39,0.91,0.95,0.57,0.91,
        0.09,0.20,0.97,0.49,0.85,0.38,0.55,0.60,0.95
    };

    float kernels[kernel_total] = {
        -1,-1,-1,
        -1, 8,-1,
        -1,-1,-1,
        0,1,0,
        1,-4,1,
        0,1,0,
        1,0,-1,
        0,0,0,
        -1,0,1,
        0.2,0.2,0.2,
        0.2,0.2,0.2,
        0.2,0.2,0.2
    };

    float biases[K] = {0, 1, -1, 0.5};

    memcpy(conv.kernels, kernels, sizeof(kernels));
    memcpy(conv.biases, biases, sizeof(biases));

    conv.forward(input);

    float expected[output_size];

    for(int k=0;k<K;k++){
        conv_forward_cpu(
            input,
            &kernels[k * 9],
            &biases[k],
            &expected[k * OH * OW],
            C,
            size, size,
            kernel_size,
            stride,
            padding,
            OH, OW
        );
    }

    Matrix::log_static(expected, 1, output_size, 'E');

    Matrix::log_static(conv.outputs, 1, output_size, 'G');
}

int main() {
    test_conv_forward();
    return 0;
}