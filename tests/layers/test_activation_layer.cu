#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>
#include"../../network/activation.cu"

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
g_tests_run++; \
if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; g_tests_failed++; } \
else { std::cout << "[ OK ] " << msg << "\n"; } \
} while(0)

// ---------------- CPU reference (batch, по аналогии с dense_forward_cpu_batch) ----------------

float sigmoid_cpu(float x) { return 1.0f/(1.0f+expf(-x)); }
float relu_cpu(float x) { return x > 0.0f ? x : 0.0f; }

void activation_forward_cpu_batch(float* z_batch, float* out_batch, int batch_size, int size, ActivationType act) {
    for (int b = 0; b < batch_size; ++b) {
        float* z = z_batch + (size_t)b*size;
        float* out = out_batch + (size_t)b*size;
        switch (act) {
            case ActivationType::Sigmoid:
                for (int i=0;i<size;++i) out[i] = sigmoid_cpu(z[i]);
                break;
            case ActivationType::ReLU:
                for (int i=0;i<size;++i) out[i] = relu_cpu(z[i]);
                break;
            case ActivationType::Linear:
                for (int i=0;i<size;++i) out[i] = z[i];
                break;
            case ActivationType::Softmax: {
                float m = z[0];
                for (int i=1;i<size;++i) if (z[i]>m) m=z[i];
                float sum=0.0f;
                std::vector<float> e(size);
                for (int i=0;i<size;++i) { e[i]=expf(z[i]-m); sum+=e[i]; }
                for (int i=0;i<size;++i) out[i]=e[i]/sum;
                break;
            }
        }
    }
}

void activation_backward_cpu_batch(float* out_batch, float* next_errors_batch, float* local_grad_batch,
                                    int batch_size, int size, ActivationType act, bool raw_gradient) {
    for (int b = 0; b < batch_size; ++b) {
        float* out = out_batch + (size_t)b*size;
        float* ne = next_errors_batch + (size_t)b*size;
        float* lg = local_grad_batch + (size_t)b*size;
        if (raw_gradient) {
            for (int i=0;i<size;++i) lg[i] = ne[i];
        } else {
            switch (act) {
                case ActivationType::Sigmoid:
                    for (int i=0;i<size;++i) lg[i] = ne[i]*out[i]*(1.0f-out[i]);
                    break;
                case ActivationType::ReLU:
                    for (int i=0;i<size;++i) lg[i] = ne[i]*(out[i]>0.0f?1.0f:0.0f);
                    break;
                case ActivationType::Linear:
                    for (int i=0;i<size;++i) lg[i] = ne[i];
                    break;
                case ActivationType::Softmax:
                    break;
            }
        }
    }
}

const int SIZE = 4, BATCH = 6;

void make_z_batch(float* z) {
    for (int b=0;b<BATCH;++b)
        for (int i=0;i<SIZE;++i)
            z[b*SIZE+i] = 0.15f * ((i*3 + b*5) % 11) - 0.7f;
}

// ============================================================
// Test 1: Sigmoid forward+backward (batch, raw_gradient=false)
// ============================================================
void test_activation_sigmoid_batch() {
    Activation act(SIZE, ActivationType::Sigmoid);
    act.set_batch_size(BATCH);

    float z[BATCH*SIZE];
    make_z_batch(z);
    act.forward(z);

    float expected_out[BATCH*SIZE];
    activation_forward_cpu_batch(z, expected_out, BATCH, SIZE, ActivationType::Sigmoid);

    bool fwd_match = true;
    for (int i=0;i<BATCH*SIZE;++i)
        if (std::fabs(act.outputs[i]-expected_out[i]) > 1e-4f) fwd_match = false;
    CHECK(fwd_match, "Activation(Sigmoid): batch forward matches CPU reference");

    float next_errors[BATCH*SIZE];
    for (int i=0;i<BATCH*SIZE;++i) next_errors[i] = 0.3f - 0.05f*(i%7);

    float* local_grad = act.backward(nullptr, next_errors, false);

    float expected_lg[BATCH*SIZE];
    activation_backward_cpu_batch(expected_out, next_errors, expected_lg, BATCH, SIZE, ActivationType::Sigmoid, false);

    bool bwd_match = true;
    for (int i=0;i<BATCH*SIZE;++i)
        if (std::fabs(local_grad[i]-expected_lg[i]) > 1e-4f) bwd_match = false;
    CHECK(bwd_match, "Activation(Sigmoid): batch backward (raw_gradient=false) matches CPU reference");

    free(local_grad);
}

// ============================================================
// Test 2: ReLU forward+backward
// ============================================================
void test_activation_relu_batch() {
    Activation act(SIZE, ActivationType::ReLU);
    act.set_batch_size(BATCH);

    float z[BATCH*SIZE];
    make_z_batch(z);
    act.forward(z);

    float expected_out[BATCH*SIZE];
    activation_forward_cpu_batch(z, expected_out, BATCH, SIZE, ActivationType::ReLU);

    bool fwd_match = true;
    bool has_zero=false, has_pos=false;
    for (int i=0;i<BATCH*SIZE;++i) {
        if (std::fabs(act.outputs[i]-expected_out[i]) > 1e-4f) fwd_match = false;
        if (act.outputs[i]==0.0f) has_zero=true;
        if (act.outputs[i]>0.0f) has_pos=true;
    }
    CHECK(fwd_match, "Activation(ReLU): batch forward matches CPU reference");
    CHECK(has_zero && has_pos, "Activation(ReLU): fixture exercises both zero and positive branches");

    float next_errors[BATCH*SIZE];
    for (int i=0;i<BATCH*SIZE;++i) next_errors[i] = 0.4f + 0.1f*(i%5);

    float* local_grad = act.backward(nullptr, next_errors, false);
    float expected_lg[BATCH*SIZE];
    activation_backward_cpu_batch(expected_out, next_errors, expected_lg, BATCH, SIZE, ActivationType::ReLU, false);

    bool bwd_match = true;
    for (int i=0;i<BATCH*SIZE;++i)
        if (std::fabs(local_grad[i]-expected_lg[i]) > 1e-4f) bwd_match = false;
    CHECK(bwd_match, "Activation(ReLU): batch backward blocks gradient where output==0");

    free(local_grad);
}

// ============================================================
// Test 3: Linear forward+backward (identity, derivative=1)
// ============================================================
void test_activation_linear_batch() {
    Activation act(SIZE, ActivationType::Linear);
    act.set_batch_size(BATCH);

    float z[BATCH*SIZE];
    make_z_batch(z);
    act.forward(z);

    bool identity_match = true;
    for (int i=0;i<BATCH*SIZE;++i)
        if (std::fabs(act.outputs[i]-z[i]) > 1e-6f) identity_match = false;
    CHECK(identity_match, "Activation(Linear): forward is identity (output == input)");

    float next_errors[BATCH*SIZE];
    for (int i=0;i<BATCH*SIZE;++i) next_errors[i] = 0.2f - 0.03f*i;

    float* local_grad = act.backward(nullptr, next_errors, false);
    bool bwd_match = true;
    for (int i=0;i<BATCH*SIZE;++i)
        if (std::fabs(local_grad[i]-next_errors[i]) > 1e-6f) bwd_match = false;
    CHECK(bwd_match, "Activation(Linear): backward passes next_errors through unchanged (derivative=1)");

    free(local_grad);
}

// ============================================================
// Test 4: Softmax forward -- sums to 1, numerically stable on extreme z
// ============================================================
void test_activation_softmax_batch() {
    Activation act(SIZE, ActivationType::Softmax);
    act.set_batch_size(BATCH);

    float z[BATCH*SIZE];
    make_z_batch(z);
    act.forward(z);

    float expected_out[BATCH*SIZE];
    activation_forward_cpu_batch(z, expected_out, BATCH, SIZE, ActivationType::Softmax);

    bool fwd_match = true;
    for (int i=0;i<BATCH*SIZE;++i)
        if (std::fabs(act.outputs[i]-expected_out[i]) > 1e-4f) fwd_match = false;
    CHECK(fwd_match, "Activation(Softmax): batch forward matches CPU reference");

    bool all_rows_sum_to_one = true;
    for (int b=0;b<BATCH;++b) {
        float sum=0.0f;
        for (int i=0;i<SIZE;++i) sum += act.outputs[b*SIZE+i];
        if (std::fabs(sum-1.0f) > 1e-4f) all_rows_sum_to_one = false;
    }
    CHECK(all_rows_sum_to_one, "Activation(Softmax): every row in the batch sums to 1.0");

    Activation act_extreme(SIZE, ActivationType::Softmax);
    act_extreme.set_batch_size(1);
    float z_extreme[SIZE] = {30.0f, 65.02f, 67.54f, 70.06f};
    act_extreme.forward(z_extreme);

    bool no_nan=true, no_inf=true;
    float sum_extreme=0.0f;
    for (int i=0;i<SIZE;++i) {
        if (std::isnan(act_extreme.outputs[i])) no_nan=false;
        if (std::isinf(act_extreme.outputs[i])) no_inf=false;
        sum_extreme += act_extreme.outputs[i];
    }
    CHECK(no_nan && no_inf, "Activation(Softmax): no NaN/Inf with extreme pre-activation values (z~70)");
    CHECK(std::fabs(sum_extreme-1.0f) < 1e-3f, "Activation(Softmax): sums to 1.0 even with extreme logits");
}

// ============================================================
// Test 5: Softmax backward with raw_gradient=true
// ============================================================
void test_activation_softmax_raw_gradient() {
    Activation act(SIZE, ActivationType::Softmax);
    act.set_batch_size(BATCH);

    float z[BATCH*SIZE];
    make_z_batch(z);
    act.forward(z);

    float next_errors[BATCH*SIZE];
    for (int b=0;b<BATCH;++b) {
        int target_class = b % SIZE;
        for (int i=0;i<SIZE;++i) {
            float target = (i==target_class) ? 1.0f : 0.0f;
            next_errors[b*SIZE+i] = target - act.outputs[b*SIZE+i];
        }
    }

    float* local_grad = act.backward(nullptr, next_errors, true);

    bool copy_match = true;
    for (int i=0;i<BATCH*SIZE;++i)
        if (std::fabs(local_grad[i]-next_errors[i]) > 1e-6f) copy_match = false;
    CHECK(copy_match, "Activation(Softmax)+raw_gradient=true: local_grad is an exact copy of next_errors");

    free(local_grad);
}

// ============================================================
// Test 6 - guard
// ============================================================
void run_softmax_backward_without_raw_gradient_should_abort() {
    Activation act(SIZE, ActivationType::Softmax);
    float z[SIZE] = {0.1f, 0.2f, -0.1f, 0.05f};
    act.forward(z);
    float next_errors[SIZE] = {0.1f,0.1f,0.1f,0.1f};
    float* r = act.backward(nullptr, next_errors, false); 
    free(r); 
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--abort-check") {
        run_softmax_backward_without_raw_gradient_should_abort();
        return 0;
    }

    std::cout << "=== Activation layer tests ===\n\n";
    test_activation_sigmoid_batch();
    test_activation_relu_batch();
    test_activation_linear_batch();
    test_activation_softmax_batch();
    test_activation_softmax_raw_gradient();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    std::cout << "\nNOTE: guard test (Softmax backward without raw_gradient must abort)\n"
                 "is NOT run here since it crashes the process by design.\n"
                 "Run with --abort-check in a separate process and verify non-zero exit code.\n";

    return g_tests_failed == 0 ? 0 : 1;
}
