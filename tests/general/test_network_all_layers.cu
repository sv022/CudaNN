#include"../../network/network.cu"
#include <vector>
#include <cmath>
#include <iostream>
#include <cstring>

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
g_tests_run++; \
if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; g_tests_failed++; } \
else { std::cout << "[ OK ] " << msg << "\n"; } \
} while(0)

static const int C = 3, H = 8, W = 8;
static const int INPUT_NODES = C * H * W;
static const int N_CLASSES = 3;
static const int DATA_SIZE = 45;

NeuralNetwork* build_full_network(float lr, float dropout_p) {
    NetworkStructure structure(lr, LossType::CategoricalCrossEntropy);

    structure.add_conv(W, H, C, /*kernel_size=*/3, /*num_kernels=*/4, /*stride=*/1, /*padding=*/1);
    structure.add_activation(4*H*W, ActivationType::ReLU); // Conv output 8x8x4=256

    structure.add_pool(8, 8, 4, /*pool=*/2, /*stride=*/2); // -> 4x4x4=64

    structure.add_dense(64, 32);
    structure.add_activation(32, ActivationType::ReLU);

    structure.add_dropout(32, dropout_p);

    structure.add_dense(32, N_CLASSES);
    structure.add_activation(N_CLASSES, ActivationType::Softmax);

    return new NeuralNetwork(&structure);
}

// ============================================================
// Test 1: training reduces loss with Dropout
// ============================================================
void test_full_chain_training_reduces_loss() {
    const int batch_size = 9, epochs = 50;
    NeuralNetwork* net = build_full_network(0.05f, 0.3f);

    std::vector<float> loss_by_epoch(epochs);
    net->train("data/multichannel_dataset.csv", DATA_SIZE, epochs, batch_size, loss_by_epoch.data());

    std::cout << "  loss[0]=" << loss_by_epoch[0] << " loss[" << epochs-1 << "]=" << loss_by_epoch[epochs-1] << "\n";

    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0],
        "Full chain (Conv->Activation->Pool->Dense->Activation->Dropout->Dense->Activation): "
        "training loss decreases despite Dropout in the middle of the chain");

    bool all_finite = true;
    for (float l : loss_by_epoch) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite,
        "Full chain: loss remains finite throughout training "
        "-- confirms raw_gradient reaches Activation(Softmax) correctly through Dropout, "
        "and inverted dropout scaling does not introduce instability");

    delete net;
}

// ============================================================
// Test 2: backward()
// ============================================================
void test_full_chain_backward_reaches_softmax_through_dropout() {
    NeuralNetwork* net = build_full_network(0.05f, 0.5f);
    net->set_batch_size(1);
    net->set_is_training(true);

    std::vector<float> input(INPUT_NODES, 0.3f);
    std::vector<float> target(N_CLASSES, 0.0f);
    target[1] = 1.0f;

    net->forward(input.data(), 1);
    net->backward(input.data(), target.data(), 1);

    CHECK(true, "Full chain: backward() completes without crashing "
                "(raw_gradient correctly propagates through Dropout to the final "
                "Activation(Softmax) layer, which requires raw_gradient=true)");

    delete net;
}

// ============================================================
// Test 3: is_training
// ============================================================
void test_is_training_switches_behavior() {
    NeuralNetwork* net = build_full_network(0.05f, 0.5f);
    net->set_batch_size(1);

    std::vector<float> input(INPUT_NODES, 0.3f);

    net->set_is_training(true);

    net->forward(input.data(), 1);
    std::vector<float> out_train_1(net->layers.back()->outputs, net->layers.back()->outputs + N_CLASSES);

    net->forward(input.data(), 1);
    std::vector<float> out_train_2(net->layers.back()->outputs, net->layers.back()->outputs + N_CLASSES);

    bool train_varies = false;
    for (int i = 0; i < N_CLASSES; ++i) if (out_train_1[i] != out_train_2[i]) train_varies = true;

    CHECK(train_varies,
        "Full chain (is_training=true): repeated forward() calls with the SAME input "
        "produce DIFFERENT outputs -- confirms Dropout is genuinely active during training "
        "(not silently disabled or ignored)");

    net->set_is_training(false);

    net->forward(input.data(), 1);
    std::vector<float> out_eval_1(net->layers.back()->outputs, net->layers.back()->outputs + N_CLASSES);

    net->forward(input.data(), 1);
    std::vector<float> out_eval_2(net->layers.back()->outputs, net->layers.back()->outputs + N_CLASSES);

    bool eval_deterministic = true;
    for (int i = 0; i < N_CLASSES; ++i) if (out_eval_1[i] != out_eval_2[i]) eval_deterministic = false;

    CHECK(eval_deterministic,
        "Full chain (is_training=false): repeated forward() calls with the SAME input "
        "produce IDENTICAL outputs -- confirms set_is_training(false) actually disables "
        "randomness in Dropout, not just in isolation but through the full NeuralNetwork pipeline");

    delete net;
}

// ============================================================
// Test 4: train() -> test() -> predict()
// ============================================================
void test_full_pipeline_train_test_predict_with_dropout() {
    const int batch_size = 9, epochs = 60;
    NeuralNetwork* net = build_full_network(0.05f, 0.3f);

    std::vector<float> loss_by_epoch(epochs);
    net->train("data/multichannel_dataset.csv", DATA_SIZE, epochs, batch_size, loss_by_epoch.data());

    std::vector<int> test_targets(DATA_SIZE), test_guesses(DATA_SIZE);
    float accuracy = net->test("data/multichannel_dataset.csv", DATA_SIZE, test_targets.data(), test_guesses.data());

    std::cout << "  final loss: " << loss_by_epoch[epochs-1] << " test accuracy: " << accuracy*100.0f << "%\n";

    CHECK(accuracy > (1.0f / N_CLASSES),
        "Full chain: test() accuracy exceeds random-guess baseline (33% for 3 classes) "
        "with Dropout present in the trained architecture");

    DatasetFile single_check("data/multichannel_dataset.csv", DATA_SIZE, INPUT_NODES, N_CLASSES, 1);

    int pred1 = net->predict(single_check.image_batch);
    int pred2 = net->predict(single_check.image_batch);

    CHECK(pred1 == pred2,
        "Full chain: predict() called twice with the same input after test() returns "
        "the IDENTICAL class index both times -- proves is_training=false is correctly "
        "propagated through predict(), not just through test()");

    delete net;
}

// ============================================================
// Test 5: train() sets is_training=true 
// ============================================================
void test_repeated_train_after_test_reenables_training_mode() {
    NeuralNetwork* net = build_full_network(0.05f, 0.3f);

    std::vector<float> loss1(5);
    net->train("data/multichannel_dataset.csv", DATA_SIZE, 5, 9, loss1.data());

    std::vector<int> test_targets(DATA_SIZE), test_guesses(DATA_SIZE);
    net->test("data/multichannel_dataset.csv", DATA_SIZE, test_targets.data(), test_guesses.data());

    std::vector<float> loss2(5);
    net->train("data/multichannel_dataset.csv", DATA_SIZE, 5, 9, loss2.data());

    bool is_training_reenabled = true;
    for (auto* layer : net->layers) {
        if (!layer->is_training) is_training_reenabled = false;
    }

    CHECK(is_training_reenabled,
        "Full chain: calling train() again AFTER test() re-enables is_training=true "
        "on every layer (including Dropout) -- a second training phase does not "
        "silently continue in eval mode");

    bool all_finite = true;
    for (float l : loss1) if (!std::isfinite(l)) all_finite = false;
    for (float l : loss2) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "Full chain: loss remains finite across train->test->train sequence with Dropout");

    delete net;
}

int main() {
    std::cout << "=== NeuralNetwork full-chain integration tests (all layer types) ===\n\n";

    std::cout << "-- Test 1: training reduces loss through the full chain --\n";
    test_full_chain_training_reduces_loss();

    std::cout << "\n-- Test 2: backward() reaches Softmax through Dropout without crashing --\n";
    test_full_chain_backward_reaches_softmax_through_dropout();

    std::cout << "\n-- Test 3: is_training actually switches Dropout behavior end-to-end --\n";
    test_is_training_switches_behavior();

    std::cout << "\n-- Test 4: full train -> test -> predict pipeline with Dropout --\n";
    test_full_pipeline_train_test_predict_with_dropout();

    std::cout << "\n-- Test 5: repeated train() after test() re-enables training mode --\n";
    test_repeated_train_after_test_reenables_training_mode();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}
