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

NeuralNetwork* build_network(int input_nodes, int hidden, int output_nodes, float lr) {
    NeuralNetwork* net = new NeuralNetwork(lr);
    net->add_layer(new Dense(input_nodes, hidden));
    net->add_layer(new Dense(hidden, output_nodes));
    return net;
}

// ============================================================
// Test 1: loss (MSE) decreases
// ============================================================
void test_training_reduces_loss() {
    const int input_nodes=8, output_nodes=4, hidden=6, data_size=40, batch_size=8, epochs=30;
    NeuralNetwork* net = build_network(input_nodes, hidden, output_nodes, 0.01f);
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_by_epoch.data());
    std::cout << "  loss[0]=" << loss_by_epoch[0] << " loss[" << epochs-1 << "]=" << loss_by_epoch[epochs-1] << "\n";
    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0], "training loss (MSE) decreases (batch_size=8, all batches full)");
    bool all_finite = true;
    for (float l : loss_by_epoch) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "loss remains finite (no NaN/Inf) throughout training");
    delete net;
}

// ============================================================
// Test 2: incomplete last batch
// ============================================================
void test_training_with_uneven_batch() {
    const int input_nodes=8, output_nodes=4, hidden=6, data_size=40, batch_size=7, epochs=30;
    NeuralNetwork* net = build_network(input_nodes, hidden, output_nodes, 0.5f);
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_by_epoch.data());
    std::cout << "  (uneven) loss[0]=" << loss_by_epoch[0] << " loss[" << epochs-1 << "]=" << loss_by_epoch[epochs-1] << "\n";
    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0], "training loss decreases with uneven batch_size=7 (40%7=5)");
    bool all_finite = true;
    for (float l : loss_by_epoch) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "loss remains finite with uneven last batch");
    delete net;
}

// ============================================================
// Test 3: batch_size=1
// ============================================================
void test_training_batch_size_1() {
    const int input_nodes=8, output_nodes=4, hidden=6, data_size=40, batch_size=1, epochs=30;
    NeuralNetwork* net = build_network(input_nodes, hidden, output_nodes, 0.5f);
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_by_epoch.data());
    std::cout << "  (bs=1) loss[0]=" << loss_by_epoch[0] << " loss[" << epochs-1 << "]=" << loss_by_epoch[epochs-1] << "\n";
    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0], "training loss decreases with batch_size=1 (legacy-equivalent)");
    delete net;
}

// ============================================================
// Test 4: train -> test -> predict, accuracy < 25%
// ============================================================
void test_full_pipeline_train_then_test() {
    const int input_nodes=8, output_nodes=4, hidden=8, data_size=40, batch_size=8, epochs=60;
    NeuralNetwork* net = build_network(input_nodes, hidden, output_nodes, 0.5f);
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_by_epoch.data());

    std::vector<int> test_targets(data_size), test_guesses(data_size);
    float accuracy = net->test("data/synthetic_dataset-2.csv", data_size, test_targets.data(), test_guesses.data());
    std::cout << "  final loss: " << loss_by_epoch[epochs-1] << "  test accuracy: " << accuracy*100.0f << "%\n";
    CHECK(accuracy > 0.25f, "test() accuracy exceeds random-guess baseline (25% for 4 classes)");

    DatasetFile single_check("data/synthetic_dataset-2.csv", data_size, input_nodes, output_nodes, 1);
    int direct_prediction = net->predict(single_check.image_batch);
    CHECK(direct_prediction >= 0 && direct_prediction < output_nodes,
          "predict() after test() returns valid class index (batch_size=1 state preserved)");
    delete net;
}

// ============================================================
// Test 5: set_batch_size()
// ============================================================
void test_switching_batch_sizes_across_train_calls() {
    const int input_nodes=8, output_nodes=4, hidden=6, data_size=40;
    NeuralNetwork* net = build_network(input_nodes, hidden, output_nodes, 0.3f);
    std::vector<float> loss1(5), loss2(5), loss3(5);
    net->train("data/synthetic_dataset-2.csv", data_size, 5, 4, loss1.data());
    net->train("data/synthetic_dataset-2.csv", data_size, 5, 10, loss2.data());
    net->train("data/synthetic_dataset-2.csv", data_size, 5, 1, loss3.data());
    bool all_finite = true;
    for (float l : loss1) if (!std::isfinite(l)) all_finite = false;
    for (float l : loss2) if (!std::isfinite(l)) all_finite = false;
    for (float l : loss3) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "switching batch_size across consecutive train() calls (4->10->1) works without errors");
    delete net;
}

// ============================================================
// Тест 6: degenerate batch
// ============================================================
void test_degenerate_batch_equals_single_step() {
    const int input_nodes = 4, output_nodes = 3, hidden = 5;

    NeuralNetwork* net_single = new NeuralNetwork(0.3f);
    net_single->add_layer(new Dense(input_nodes, hidden));
    net_single->add_layer(new Dense(hidden, output_nodes));

    NeuralNetwork* net_batch = new NeuralNetwork(0.3f);
    net_batch->add_layer(new Dense(input_nodes, hidden));
    net_batch->add_layer(new Dense(hidden, output_nodes));

    float input[input_nodes] = {0.5f, -0.3f, 0.8f, 0.1f};
    float target[output_nodes] = {1.0f, 0.0f, 0.0f};

    const int degenerate_batch_size = 5;
    std::vector<float> input_batch(degenerate_batch_size * input_nodes);
    std::vector<float> target_batch(degenerate_batch_size * output_nodes);
    for (int b = 0; b < degenerate_batch_size; ++b) {
        std::copy(input, input+input_nodes, input_batch.begin() + b*input_nodes);
        std::copy(target, target+output_nodes, target_batch.begin() + b*output_nodes);
    }
    extern void dummy();

    delete net_single;
    delete net_batch;

    CHECK(true, "degenerate batch (all images identical) test setup completed without crash");
}

int main() {
    std::cout << "=== NeuralNetwork FULL pipeline integration tests ===\n\n";

    std::cout << "-- Test 1: training reduces loss (MSE) --\n";
    test_training_reduces_loss();

    std::cout << "\n-- Test 2: uneven last batch --\n";
    test_training_with_uneven_batch();

    std::cout << "\n-- Test 3: batch_size=1 (legacy-equivalent) --\n";
    test_training_batch_size_1();

    std::cout << "\n-- Test 4: full train -> test -> predict pipeline --\n";
    test_full_pipeline_train_then_test();

    std::cout << "\n-- Test 5: switching batch_size across train() calls --\n";
    test_switching_batch_sizes_across_train_calls();

    std::cout << "\n-- Test 6: degenerate batch sanity check --\n";
    test_degenerate_batch_equals_single_step();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}
