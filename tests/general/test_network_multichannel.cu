#include"../../network/network.cu"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
    g_tests_run++; \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; g_tests_failed++; } \
    else { std::cout << "[ OK ] " << msg << "\n"; } \
} while(0)

static const int C = 3;
static const int H = 8;
static const int W = 8;
static const int INPUT_NODES = C * H * W; // 192
static const int N_CLASSES = 3;
static const int DATA_SIZE = 45;

NeuralNetwork* build_conv_network(float lr) {
    NeuralNetwork* net = new NeuralNetwork(lr);
    Conv* conv = new Conv(/*input_height=*/H, /*input_width=*/W, /*channels=*/C,
                           /*kernel_size=*/3, /*n_kernels=*/4, /*stride=*/1, /*padding=*/1);
    conv->set_learning_rate(lr);
    net->add_layer(conv);

    MaxPooling* pool = new MaxPooling(/*in_w=*/8, /*in_h=*/8, /*channels=*/4,
                                       /*pool=*/2, /*s=*/2);
    net->add_layer(pool);

    // Dense: 64 -> 3 (N_CLASSES)
    Dense* dense = new Dense(/*layer_size=*/4*4*4, /*next_size=*/N_CLASSES);
    dense->set_learning_rate(lr);
    net->add_layer(dense);

    return net;
}

// ============================================================
// Test 1: Conv+MaxPooling+Dense 
// ============================================================
void test_multichannel_training_reduces_loss() {
    const int batch_size = 9; // 45 % 9 == 0, все батчи полные
    const int epochs = 40;

    NeuralNetwork* net = build_conv_network(0.05f);
    std::vector<float> loss_by_epoch(epochs);

    net->train("data/multichannel_dataset.csv", DATA_SIZE, epochs, batch_size, loss_by_epoch.data());

    std::cout << "  loss[0]=" << loss_by_epoch[0] << " loss[" << epochs-1 << "]=" << loss_by_epoch[epochs-1] << "\n";

    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0],
          "Conv+MaxPooling+Dense: training loss decreases on 8x8x3 multichannel dataset (batch_size=9)");

    bool all_finite = true;
    for (float l : loss_by_epoch) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "Conv+MaxPooling+Dense: loss remains finite (no NaN/Inf) throughout training");

    delete net;
}

// ============================================================
// Test 2: incomplete batch
// ============================================================
void test_multichannel_uneven_batch() {
    const int batch_size = 7;
    const int epochs = 30;

    NeuralNetwork* net = build_conv_network(0.05f);
    std::vector<float> loss_by_epoch(epochs);

    net->train("data/multichannel_dataset.csv", DATA_SIZE, epochs, batch_size, loss_by_epoch.data());

    std::cout << "  (uneven, bs=7) loss[0]=" << loss_by_epoch[0]
              << " loss[" << epochs-1 << "]=" << loss_by_epoch[epochs-1] << "\n";

    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0],
          "Conv+MaxPooling+Dense: training loss decreases with uneven batch_size=7 (45%7=3)");

    bool all_finite = true;
    for (float l : loss_by_epoch) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "Conv+MaxPooling+Dense: loss remains finite with uneven last batch");

    delete net;
}

// ============================================================
// Test 3: batch_size=1
// ============================================================
void test_multichannel_batch_size_1() {
    const int batch_size = 1;
    const int epochs = 40;

    NeuralNetwork* net = build_conv_network(0.05f);
    std::vector<float> loss_by_epoch(epochs);

    net->train("data/multichannel_dataset.csv", DATA_SIZE, epochs, batch_size, loss_by_epoch.data());

    std::cout << "  (bs=1) loss[0]=" << loss_by_epoch[0]
              << " loss[" << epochs-1 << "]=" << loss_by_epoch[epochs-1] << "\n";

    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0],
          "Conv+MaxPooling+Dense: training loss decreases with batch_size=1 (legacy-equivalent)");

    delete net;
}

// ============================================================
// Test 4: train -> test -> predict
// ============================================================
void test_multichannel_full_pipeline() {
    const int batch_size = 9;
    const int epochs = 60;

    NeuralNetwork* net = build_conv_network(0.05f);
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/multichannel_dataset.csv", DATA_SIZE, epochs, batch_size, loss_by_epoch.data());

    std::vector<int> test_targets(DATA_SIZE), test_guesses(DATA_SIZE);
    float accuracy = net->test("data/multichannel_dataset.csv", DATA_SIZE, test_targets.data(), test_guesses.data());

    std::cout << "  final loss: " << loss_by_epoch[epochs-1] << "  test accuracy: " << accuracy*100.0f << "%\n";

    CHECK(accuracy > (1.0f / N_CLASSES),
          "Conv+MaxPooling+Dense: test() accuracy exceeds random-guess baseline (33% for 3 classes) "
          "-- confirms Conv layer extracts per-channel signal through full batch pipeline");

    // predict() напрямую после test() -- проверка перехода batch_size=9 -> 1
    // (test() сам вызывает set_batch_size(1) на всех слоях, включая Conv/MaxPooling)
    DatasetFile single_check("data/multichannel_dataset.csv", DATA_SIZE, INPUT_NODES, N_CLASSES, 1);
    int direct_prediction = net->predict(single_check.image_batch);
    CHECK(direct_prediction >= 0 && direct_prediction < N_CLASSES,
          "Conv+MaxPooling+Dense: predict() after test() returns valid class index "
          "(batch_size=1 state correctly propagated through Conv and MaxPooling)");

    delete net;
}

// ============================================================
// Test 5: change batch_size 
// ============================================================
void test_multichannel_switching_batch_sizes() {
    NeuralNetwork* net = build_conv_network(0.03f);

    std::vector<float> loss1(5), loss2(5), loss3(5);
    net->train("data/multichannel_dataset.csv", DATA_SIZE, 5, 3, loss1.data());
    net->train("data/multichannel_dataset.csv", DATA_SIZE, 5, 9, loss2.data());
    net->train("data/multichannel_dataset.csv", DATA_SIZE, 5, 1, loss3.data());

    bool all_finite = true;
    for (float l : loss1) if (!std::isfinite(l)) all_finite = false;
    for (float l : loss2) if (!std::isfinite(l)) all_finite = false;
    for (float l : loss3) if (!std::isfinite(l)) all_finite = false;

    CHECK(all_finite,
          "Conv+MaxPooling+Dense: switching batch_size across consecutive train() calls "
          "(3 -> 9 -> 1) works without errors for full Conv architecture");

    delete net;
}

int main() {
    std::cout << "=== NeuralNetwork multichannel (8x8x3) Conv+MaxPooling+Dense integration tests ===\n\n";

    std::cout << "-- Test 1: training reduces loss on multichannel data --\n";
    test_multichannel_training_reduces_loss();

    std::cout << "\n-- Test 2: uneven last batch (Conv+MaxPooling+Dense) --\n";
    test_multichannel_uneven_batch();

    std::cout << "\n-- Test 3: batch_size=1 (legacy-equivalent, Conv architecture) --\n";
    test_multichannel_batch_size_1();

    std::cout << "\n-- Test 4: full train -> test -> predict on multichannel data --\n";
    test_multichannel_full_pipeline();

    std::cout << "\n-- Test 5: switching batch_size across train() calls (Conv architecture) --\n";
    test_multichannel_switching_batch_sizes();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}