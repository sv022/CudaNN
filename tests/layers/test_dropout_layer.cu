#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include"../../network/dropout.cu"

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
g_tests_run++; \
if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; g_tests_failed++; } \
else { std::cout << "[ OK ] " << msg << "\n"; } \
} while(0)

const float DROP_FRACTION_TOLERANCE = 0.02f;
const float MEAN_OUTPUT_TOLERANCE = 0.05f;

// ============================================================
// Test 1: Train mode -- drop_prob
// ============================================================
void test_dropout_fraction_matches_probability() {
    const int SIZE = 100000;
    const float P = 0.5f;

    Dropout d(SIZE, P);
    d.is_training = true;

    std::vector<float> x(SIZE, 1.0f);
    d.forward(x.data());

    int zeros = 0;
    for (int i = 0; i < SIZE; ++i) if (d.outputs[i] == 0.0f) zeros++;
    float observed_fraction = (float)zeros / SIZE;

    CHECK(std::fabs(observed_fraction - P) < DROP_FRACTION_TOLERANCE,
        "Dropout(train): observed drop fraction (" + std::to_string(observed_fraction) +
        ") matches drop_prob=" + std::to_string(P));
}

// ============================================================
// Test 2: Train mode -- inverted dropout
// ============================================================
void test_dropout_mean_preserved_by_inverted_scaling() {
    const int SIZE = 100000;
    const float P = 0.5f;
    const float X_VAL = 2.0f;

    Dropout d(SIZE, P);
    d.is_training = true;

    std::vector<float> x(SIZE, X_VAL);
    d.forward(x.data());

    float sum = 0.0f;
    for (int i = 0; i < SIZE; ++i) sum += d.outputs[i];
    float mean = sum / SIZE;

    CHECK(std::fabs(mean - X_VAL) < MEAN_OUTPUT_TOLERANCE,
        "Dropout(train): inverted scaling preserves expected output mean "
        "(observed=" + std::to_string(mean) + ", target=" + std::to_string(X_VAL) + ")");
}

// ============================================================
// Test 3: Train mode -- forward and backward mask
// ============================================================
void test_dropout_forward_backward_mask_consistency() {
    const int SIZE = 1000;
    const float P = 0.4f;

    Dropout d(SIZE, P);
    d.is_training = true;

    std::vector<float> x(SIZE, 1.0f);
    d.forward(x.data());

    std::vector<float> next_errors(SIZE, 0.5f);
    float* local_grad = d.backward(nullptr, next_errors.data(), false);

    bool consistent = true;
    for (int i = 0; i < SIZE; ++i) {
        bool output_zeroed = (d.outputs[i] == 0.0f);
        bool grad_zeroed = (local_grad[i] == 0.0f);
        if (output_zeroed != grad_zeroed) consistent = false;
    }

    CHECK(consistent,
        "Dropout(train): backward zeroes gradient exactly where forward zeroed output");

    free(local_grad);
}

// ============================================================
// Test 4: Train mode -- local_grad == next_errors * scale
// ============================================================
void test_dropout_backward_exact_value_where_kept() {
    const int SIZE = 1000;
    const float P = 0.4f;
    const float scale = 1.0f / (1.0f - P);

    Dropout d(SIZE, P);
    d.is_training = true;

    std::vector<float> x(SIZE, 1.0f);
    d.forward(x.data());

    std::vector<float> next_errors(SIZE);
    for (int i = 0; i < SIZE; ++i) next_errors[i] = 0.3f + 0.01f * i;

    float* local_grad = d.backward(nullptr, next_errors.data(), false);

    bool exact_match = true;
    for (int i = 0; i < SIZE; ++i) {
        if (d.outputs[i] != 0.0f) {
            float expected = next_errors[i] * scale;
            if (std::fabs(local_grad[i] - expected) > 1e-4f) exact_match = false;
        }
    }

    CHECK(exact_match,
        "Dropout(train): local_grad exactly equals next_errors*scale for every kept element");

    free(local_grad);
}

// ============================================================
// Test 5: Eval mode -- forward -- identity
// ============================================================
void test_dropout_eval_forward_is_identity() {
    const int SIZE = 500;
    Dropout d(SIZE, 0.5f);
    d.is_training = false;

    std::vector<float> x(SIZE);
    for (int i = 0; i < SIZE; ++i) x[i] = 0.1f * i - 25.0f;

    d.forward(x.data());

    bool identity = true;
    for (int i = 0; i < SIZE; ++i) if (d.outputs[i] != x[i]) identity = false;

    CHECK(identity, "Dropout(eval): forward is exact identity");
}

// ============================================================
// Test 6: Eval mode -- backward -- identity
// ============================================================
void test_dropout_eval_backward_is_identity() {
    const int SIZE = 500;
    Dropout d(SIZE, 0.5f);
    d.is_training = false;

    std::vector<float> x(SIZE, 1.0f);
    d.forward(x.data());

    std::vector<float> next_errors(SIZE);
    for (int i = 0; i < SIZE; ++i) next_errors[i] = 0.2f * i - 10.0f;

    float* local_grad = d.backward(nullptr, next_errors.data(), false);

    bool identity = true;
    for (int i = 0; i < SIZE; ++i) if (local_grad[i] != next_errors[i]) identity = false;

    CHECK(identity, "Dropout(eval): backward is exact identity");

    free(local_grad);
}

// ============================================================
// Test 7: Batch 
// ============================================================
void test_dropout_batch_rows_get_independent_masks() {
    const int SIZE = 2000, BATCH = 4;
    Dropout d(SIZE, 0.5f);
    d.set_batch_size(BATCH);
    d.is_training = true;

    std::vector<float> x(BATCH * SIZE, 1.0f);
    d.forward(x.data());

    bool rows_differ = false;
    for (int i = 0; i < SIZE; ++i) {
        bool row0_zeroed = (d.outputs[0*SIZE+i] == 0.0f);
        bool row1_zeroed = (d.outputs[1*SIZE+i] == 0.0f);
        if (row0_zeroed != row1_zeroed) { rows_differ = true; break; }
    }

    CHECK(rows_differ,
        "Dropout(train, batch>1): different rows in the batch receive independent masks");
}

// ============================================================
// Test 8a: drop_prob=0.0
// ============================================================
void test_dropout_zero_probability_is_identity_even_in_train() {
    const int SIZE = 1000;
    Dropout d(SIZE, 0.0f);
    d.is_training = true;

    std::vector<float> x(SIZE);
    for (int i = 0; i < SIZE; ++i) x[i] = 0.05f * i;

    d.forward(x.data());

    bool identity = true;
    for (int i = 0; i < SIZE; ++i) if (std::fabs(d.outputs[i] - x[i]) > 1e-4f) identity = false;

    CHECK(identity,
        "Dropout(train, drop_prob=0.0): forward is identity (scale=1/(1-0)=1, no element dropped)");
}

// ============================================================
// Test 8b: drop_prob -> 1.0
// ============================================================
void test_dropout_high_probability_stays_finite() {
    const int SIZE = 10000;
    Dropout d(SIZE, 0.99f);
    d.is_training = true;

    std::vector<float> x(SIZE, 1.0f);
    d.forward(x.data());

    bool all_finite = true;
    for (int i = 0; i < SIZE; ++i) {
        if (std::isnan(d.outputs[i]) || std::isinf(d.outputs[i])) all_finite = false;
    }

    CHECK(all_finite,
        "Dropout(train, drop_prob=0.99): scale=1/(1-0.99)=100 stays finite, no NaN/Inf in output");
}

// ============================================================
// Test 9: Batch reallocate buffers
// ============================================================
void test_dropout_set_batch_size_reallocates_buffers() {
    const int SIZE = 50;
    Dropout d(SIZE, 0.3f);

    d.set_batch_size(1);
    std::vector<float> x1(1 * SIZE, 1.0f);
    d.forward(x1.data());

    d.set_batch_size(8);
    std::vector<float> x8(8 * SIZE, 1.0f);
    d.forward(x8.data());

    int zeros = 0;
    for (int i = 0; i < 8 * SIZE; ++i) if (d.outputs[i] == 0.0f) zeros++;

    CHECK(zeros > 0 && zeros < 8*SIZE,
        "Dropout: set_batch_size() correctly reallocates outputs/mask buffers "
        "for the full batch_size*size range");
}

int main() {
    srand(123);

    std::cout << "=== Dropout layer tests ===\n\n";
    test_dropout_fraction_matches_probability();
    test_dropout_mean_preserved_by_inverted_scaling();
    test_dropout_forward_backward_mask_consistency();
    test_dropout_backward_exact_value_where_kept();
    test_dropout_eval_forward_is_identity();
    test_dropout_eval_backward_is_identity();
    test_dropout_batch_rows_get_independent_masks();
    test_dropout_zero_probability_is_identity_even_in_train();
    test_dropout_high_probability_stays_finite();
    test_dropout_set_batch_size_reallocates_buffers();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}
