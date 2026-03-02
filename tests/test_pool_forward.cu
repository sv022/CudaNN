#include"../network/maxpooling.cu"
#include <cassert>


void maxpool_forward_cpu(
    float* inputs,
    float* outputs,
    int* indices,
    int C, int H, int W,
    int pool, int stride,
    int OH, int OW
){
    for(int c=0; c<C; ++c){
        for(int oh=0; oh<OH; ++oh){
            for(int ow=0; ow<OW; ++ow){

                int h0 = oh * stride;
                int w0 = ow * stride;

                float max_val = -1e30f;
                int max_idx = -1;

                for(int kh=0; kh<pool; ++kh){
                    for(int kw=0; kw<pool; ++kw){

                        int ih = h0 + kh;
                        int iw = w0 + kw;

                        int idx = c*H*W + ih*W + iw;

                        if(inputs[idx] > max_val){
                            max_val = inputs[idx];
                            max_idx = idx;
                        }
                    }
                }

                int out_idx = c*OH*OW + oh*OW + ow;
                outputs[out_idx] = max_val;
                indices[out_idx] = max_idx;
            }
        }
    }
}

void test_maxpool_forward(){

    const int C = 1;
    const int H = 7;
    const int W = 7;

    const int pool = 2;
    const int stride = 2;

    MaxPooling pool_layer(
        W, H,
        C,
        pool,
        stride
    );

    float input[H * W] = {
        1.97, 3.8, 1.45, 0.18, 1.82, 0.3, 0, 0.91, 0, 0, 0.16, 0, 0, 0, 0, 4.1, 0, 3.62, 2.32, 0, 3.26, 0, 1.81, 2.48, 0.87, 0, 0, 0, 0, 0, 0, 3.06, 0, 0, 1.78, 4.7, 2.06, 1.63, 0, 0, 0, 1.02, 0, 0.99, 0, 0, 3.53, 3.65, 0
    };

    const int OH = 3;
    const int OW = 3;

    float out_cpu[OH * OW];
    int idx_cpu[OH * OW];

    maxpool_forward_cpu(
        input,
        out_cpu,
        idx_cpu,
        C, H, W,
        pool, stride,
        OH, OW
    );

    pool_layer.forward(input);

    std::cout << "\n===== INPUT =====\n";
    Matrix::log_static(input, H, W, 'I');

    std::cout << "\n===== CPU OUTPUT =====\n";
    Matrix::log_static(out_cpu, OH, OW, 'C');

    std::cout << "\n===== CUDA OUTPUT =====\n";
    Matrix::log_static(pool_layer.outputs, OH, OW, 'G');

    for(int i=0;i<4;i++){
        assert(fabs(out_cpu[i] - pool_layer.outputs[i]) < 1e-6);
        assert(idx_cpu[i] == pool_layer.max_indices[i]);
    }

    std::cout << "\nMaxPool forward test PASSED\n";
}

int main() {
    test_maxpool_forward();
    return 0;
}
