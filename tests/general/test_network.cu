#include"../../network/network.cu"
#include <vector>
#include <cmath>
#include <iostream>

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
g_tests_run++; \
if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; g_tests_failed++; } \
else { std::cout << "[ OK ] " << msg << "\n"; } \
} while(0)

NeuralNetwork* build_network(int input_nodes, int hidden, int output_nodes, float lr, ActivationType hidden_activation = ActivationType::Sigmoid, ActivationType output_activation = ActivationType::Sigmoid, LossType loss_type = LossType::MSE) {
    NetworkStructure structure(lr, loss_type);
    structure.add_dense(input_nodes, hidden);
    structure.add_activation(hidden, hidden_activation);
    structure.add_dense(hidden, output_nodes);
    structure.add_activation(output_nodes, output_activation);
    return new NeuralNetwork(&structure);
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
    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0], "Sigmoid+MSE: training loss decreases (batch_size=8, all batches full)");
    bool all_finite = true;
    for (float l : loss_by_epoch) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "Sigmoid+MSE: loss remains finite (no NaN/Inf) throughout training");
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
    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0], "Sigmoid+MSE: training loss decreases with uneven batch_size=7 (40%7=5)");
    bool all_finite = true;
    for (float l : loss_by_epoch) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "Sigmoid+MSE: loss remains finite with uneven last batch");
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
    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0], "Sigmoid+MSE: training loss decreases with batch_size=1 (legacy-equivalent)");
    delete net;
}

// ============================================================
// Test 4: train -> test -> predict
// ============================================================
void test_full_pipeline_train_then_test() {
    const int input_nodes=8, output_nodes=4, hidden=8, data_size=40, batch_size=8, epochs=60;
    NeuralNetwork* net = build_network(input_nodes, hidden, output_nodes, 0.5f);
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_by_epoch.data());

    std::vector<int> test_targets(data_size), test_guesses(data_size);
    float accuracy = net->test("data/synthetic_dataset-2.csv", data_size, test_targets.data(), test_guesses.data());
    std::cout << "  final loss: " << loss_by_epoch[epochs-1] << " test accuracy: " << accuracy*100.0f << "%\n";
    CHECK(accuracy > 0.25f, "Sigmoid+MSE: test() accuracy exceeds random-guess baseline (25% for 4 classes)");

    DatasetFile single_check("data/synthetic_dataset-2.csv", data_size, input_nodes, output_nodes, 1);
    int direct_prediction = net->predict(single_check.image_batch);
    CHECK(direct_prediction >= 0 && direct_prediction < output_nodes, "Sigmoid+MSE: predict() after test() returns valid class index (batch_size=1 state preserved)");
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
    CHECK(all_finite, "Sigmoid+MSE: switching batch_size across consecutive train() calls (4->10->1) works without errors");
    delete net;
}

// ============================================================
// Test 6: Softmax + CategoricalCrossEntropy -- loss decreases
// ============================================================
void test_softmax_cce_training_reduces_loss() {
    const int input_nodes=8, output_nodes=4, hidden=6, data_size=40, batch_size=8, epochs=30;
    NeuralNetwork* net = build_network(input_nodes, hidden, output_nodes, 0.01f, ActivationType::ReLU, ActivationType::Softmax, LossType::CategoricalCrossEntropy);
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_by_epoch.data());
    std::cout << "  (ReLU+Softmax+CCE) loss[0]=" << loss_by_epoch[0] << " loss[" << epochs-1 << "]=" << loss_by_epoch[epochs-1] << "\n";
    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0], "ReLU hidden + Softmax output + CategoricalCrossEntropy: training loss decreases");
    bool all_finite = true;
    for (float l : loss_by_epoch) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "ReLU+Softmax+CCE: loss remains finite (no NaN/Inf) ");
    delete net;
}

// ============================================================
// Test 7: Softmax output sums to 1 after training
// ============================================================
void test_softmax_output_sums_to_one_after_training() {
    const int input_nodes=8, output_nodes=4, hidden=6, data_size=40, batch_size=8, epochs=20;
    NeuralNetwork* net = build_network(input_nodes, hidden, output_nodes, 0.05f, ActivationType::ReLU, ActivationType::Softmax, LossType::CategoricalCrossEntropy);
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_by_epoch.data());

    DatasetFile single_check("data/synthetic_dataset-2.csv", data_size, input_nodes, output_nodes, 1);
    for (auto &layer : net->layers) layer->set_batch_size(1);
    net->forward(single_check.image_batch, 1);

    float* final_output = net->layers.back()->outputs;
    float sum = 0.0f;
    for (int j = 0; j < output_nodes; ++j) sum += final_output[j];

    CHECK(std::fabs(sum - 1.0f) < 1e-3f,
        "Softmax Activation layer: output sums to 1.0 after training "
        "(structural invariant holds through full train/forward pipeline with separated Activation layer)");
    delete net;
}

// ============================================================
// Test 8: Softmax+CCE test() accuracy exceeds random baseline
// ============================================================
void test_softmax_cce_full_pipeline_accuracy() {
    const int input_nodes=8, output_nodes=4, hidden=8, data_size=40, batch_size=8, epochs=60;
    NeuralNetwork* net = build_network(input_nodes, hidden, output_nodes, 0.05f, ActivationType::ReLU, ActivationType::Softmax, LossType::CategoricalCrossEntropy);
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_by_epoch.data());

    std::vector<int> test_targets(data_size), test_guesses(data_size);
    float accuracy = net->test("data/synthetic_dataset-2.csv", data_size, test_targets.data(), test_guesses.data());
    std::cout << "  (Softmax) final loss: " << loss_by_epoch[epochs-1] << " test accuracy: " << accuracy*100.0f << "%\n";
    CHECK(accuracy > 0.25f,
        "ReLU+Softmax+CCE: test() accuracy exceeds random-guess baseline (25% for 4 classes)");
    delete net;
}

int main() {
    std::cout << "=== NeuralNetwork FULL pipeline integration tests (separated Activation layer) ===\n\n";

    std::cout << "-- Test 1: training reduces loss (Sigmoid+MSE) --\n";
    test_training_reduces_loss();

    std::cout << "\n-- Test 2: uneven last batch (Sigmoid+MSE) --\n";
    test_training_with_uneven_batch();

    std::cout << "\n-- Test 3: batch_size=1 (Sigmoid+MSE, legacy-equivalent) --\n";
    test_training_batch_size_1();

    std::cout << "\n-- Test 4: full train -> test -> predict pipeline (Sigmoid+MSE) --\n";
    test_full_pipeline_train_then_test();

    std::cout << "\n-- Test 5: switching batch_size across train() calls (Sigmoid+MSE) --\n";
    test_switching_batch_sizes_across_train_calls();

    std::cout << "\n-- Test 6: training reduces loss (ReLU hidden + Softmax output + CCE) --\n";
    test_softmax_cce_training_reduces_loss();

    std::cout << "\n-- Test 7: Softmax Activation layer sums to 1 after full training pipeline --\n";
    test_softmax_output_sums_to_one_after_training();

    std::cout << "\n-- Test 8: full train -> test pipeline accuracy (ReLU + Softmax + CCE) --\n";
    test_softmax_cce_full_pipeline_accuracy();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}
