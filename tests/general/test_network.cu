#include "../../network/network.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

static int g_tests_run = 0, g_tests_failed = 0;
#define CHECK(cond, msg) do { g_tests_run++; \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; g_tests_failed++; } \
    else { std::cout << "[ OK ] " << msg << "\n"; } } while (0)

static const int IMG_C = 1, IMG_H = 12, IMG_W = 12;
static const int INPUT_NODES = IMG_C * IMG_H * IMG_W;
static const int N_CLASSES = 4;
static const int TRAIN_N = 200, TEST_N = 80;
std::string TRAIN_CSV = "data/synthetic_train.csv";
std::string TEST_CSV  = "data/synthetic_test.csv";

static const float MIN_LOSS_DROP_RATIO = 0.7f;
static const float MIN_TEST_ACCURACY   = 0.70f;

// ---------------- network setup ----------------
static NeuralNetwork* build_net(float lr, OptimizerType opt, float dropout_p) {
    NetworkStructure structure(lr, opt, LossType::CategoricalCrossEntropy);
    structure.opt_type = opt;

    structure.add_conv(IMG_W, IMG_H, IMG_C, 3, 4, 1, 1);               // 12x12x4
    structure.add_activation(4 * IMG_H * IMG_W, ActivationType::ReLU);
    structure.add_pool(12, 12, 4, 2, 2);                               // 6x6x4
    structure.add_conv(6, 6, 4, 3, 6, 1, 1);                           // 6x6x6
    structure.add_activation(6 * 6 * 6, ActivationType::ReLU);
    structure.add_pool(6, 6, 6, 2, 2);                                 // 3x3x6 = 54
    structure.add_dense(54, 32);
    structure.add_activation(32, ActivationType::ReLU);
    structure.add_dropout(32, dropout_p);
    structure.add_dense(32, N_CLASSES);
    structure.add_activation(N_CLASSES, ActivationType::Softmax);
    return new NeuralNetwork(&structure);
}

// ---------------- reference based on host-layer methods ----------------
void synth_reference_forward(NeuralNetwork* net, float* input) {
    float* cur = input;
    for (auto* layer : net->layers) {
        layer->forward(cur);
        cur = layer->outputs;
    }
}

void synth_reference_backward(NeuralNetwork* net, float* inputs, float* targets, int bs) {
    int L = (int)net->layers.size();
    size_t n = (size_t)bs * net->output_nodes;
    float* err = (float*)malloc(n * sizeof(float));
    const float* out = net->layers.back()->outputs;
    for (size_t i = 0; i < n; ++i) err[i] = targets[i] - out[i];

    bool raw =
        (net->layers.back()->activation_type == ActivationType::Sigmoid && net->loss_function == LossType::BinaryCrossEntropy) ||
        (net->layers.back()->activation_type == ActivationType::Softmax && net->loss_function == LossType::CategoricalCrossEntropy);

    for (int i = L - 1; i >= 0; --i) {
        float* in = (i == 0) ? inputs : net->layers[i - 1]->outputs;
        float* prev = (i == L - 1) ? net->layers[i]->backward(in, err, raw)
                                   : net->layers[i]->backward(in, err);
        free(err);
        err = prev;
    }
    net->optimizer->step();
    free(err);
}

static float max_rel_diff(const float* a, const float* b, size_t n) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]) / (1.0f + fabsf(b[i]));
        if (!std::isfinite(d)) return INFINITY;
        if (d > worst) worst = d;
    }
    return worst;
}

static std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    size_t bytes = (size_t)f.tellg();
    std::vector<float> v(bytes / sizeof(float));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

// ======================================================================
// Test 1: GPU chain == layer-by-layer host reference
// ======================================================================
void test_synth_forward_chain_equivalence() {
    const int bs = 8;
    NeuralNetwork* net = build_net(0.01f, OptimizerType::SGD, 0.3f);
    net->set_batch_size(bs);
    net->set_is_training(false);
    DatasetFile ds(TRAIN_CSV, TRAIN_N, INPUT_NODES, N_CLASSES, bs);
    net->forward(ds.image_batch, bs);

    int L = (int)net->layers.size();
    std::vector<std::vector<float>> dev(L);
    bool wired = true;
    for (int i = 0; i < L; ++i) {
        size_t n = (size_t)bs * net->layers[i]->output_size;
        dev[i].resize(n);
        if (!net->layers[i]->d_outputs ||
            cudaMemcpy(dev[i].data(), net->layers[i]->d_outputs, n * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cerr << "  layer " << i << ": d_outputs is null or unreadable\n";
            wired = false;
        }
    }
    CHECK(wired, "d_outputs ok");

    synth_reference_forward(net, ds.image_batch);
    int first_bad = -1; float worst = 0.0f;
    for (int i = 0; i < L && wired; ++i) {
        float d = max_rel_diff(dev[i].data(), net->layers[i]->outputs, dev[i].size());
        if (d > worst) worst = d;
        if (d > 1e-4f && first_bad < 0) first_bad = i;
    }
    if (first_bad >= 0) std::cerr << "  first mismatching layer: " << first_bad << "\n";
    CHECK(wired && first_bad < 0, "forward: GPU layers outout match");
    delete net;
}

// ======================================================================
// Test 2: one training step (forward + backward) == reference path
// ======================================================================
void synth_one_step_equivalence(OptimizerType opt, float lr) {
    const int bs = 8;
    const char* w0 = "synthetic_w0.bin";
    const char* wa = "synthetic_wa.bin";
    const char* wb = "synthetic_wb.bin";
    std::remove(w0); std::remove(wa); std::remove(wb);

    NeuralNetwork* a = build_net(lr, opt, 0.3f);
    a->save_weights(w0);
    NeuralNetwork* b = build_net(lr, opt, 0.3f);
    b->load_weights(w0);
    for (NeuralNetwork* n : {a, b}) { n->set_batch_size(bs); n->set_is_training(false); }

    DatasetFile ds(TRAIN_CSV, TRAIN_N, INPUT_NODES, N_CLASSES, bs);
    a->forward(ds.image_batch, bs);
    a->backward(ds.image_batch, ds.target_batch, bs);

    synth_reference_forward(b, ds.image_batch);
    synth_reference_backward(b, ds.image_batch, ds.target_batch, bs);

    a->save_weights(wa);
    b->save_weights(wb);
    std::vector<float> init = read_floats(w0), va = read_floats(wa), vb = read_floats(wb);

    bool same_size = (va.size() == vb.size() && va.size() == init.size() && !va.empty());
    float diff = same_size ? max_rel_diff(va.data(), vb.data(), va.size()) : INFINITY;
    float moved = same_size ? max_rel_diff(va.data(), init.data(), va.size()) : 0.0f;
    std::string name = (opt == OptimizerType::SGD) ? "SGD" : "Momentum";
    std::cout << "  " << name << ": max diff GPU-chain vs reference = " << diff << ", weights moved by " << moved << "\n";

    CHECK(same_size, name + ": weight files have the same non-zero size");
    CHECK(moved > 1e-6f, name + ": step actually changed the weights");
    CHECK(diff < 1e-4f, name + ": weights after one step match the reference path");

    delete a; delete b;
    std::remove(w0); std::remove(wa); std::remove(wb);
}
void test_synth_one_step_sgd()      { synth_one_step_equivalence(OptimizerType::SGD, 0.05f); }
void test_synth_one_step_momentum() { synth_one_step_equivalence(OptimizerType::Momentum, 0.02f); }

// ======================================================================
// Test 3: the result for a sample does not depend on the batch size.
// ======================================================================
void test_synth_batch_independence() {
    const int bs = 8;
    NeuralNetwork* net = build_net(0.01f, OptimizerType::SGD, 0.3f);
    net->set_is_training(false);
    DatasetFile ds(TRAIN_CSV, TRAIN_N, INPUT_NODES, N_CLASSES, bs);

    net->set_batch_size(bs);
    net->forward(ds.image_batch, bs);
    std::vector<float> batched(net->layers.back()->outputs, net->layers.back()->outputs + bs * N_CLASSES);

    net->set_batch_size(1);
    float worst = 0.0f;
    for (int j = 0; j < bs; ++j) {
        net->forward(ds.image_batch + (size_t)j * INPUT_NODES, 1);
        float d = max_rel_diff(batched.data() + j * N_CLASSES, net->layers.back()->outputs, N_CLASSES);
        if (d > worst) worst = d;
    }
    std::cout << "  max diff batch(8) vs single: " << worst << "\n";
    CHECK(worst < 1e-4f, "output for a sample in a batch of 8 matches the output with batch_size=1");
    delete net;
}

// ======================================================================
// Тест 4: режимы Dropout внутри полной цепочки
// ======================================================================
void test_synth_dropout_modes() {
    NeuralNetwork* net = build_net(0.01f, OptimizerType::SGD, 0.5f);
    net->set_batch_size(1);
    DatasetFile ds(TRAIN_CSV, TRAIN_N, INPUT_NODES, N_CLASSES, 1);
    float* x = ds.image_batch;

    net->set_is_training(true);
    net->forward(x, 1);
    std::vector<float> t1(net->layers.back()->outputs, net->layers.back()->outputs + N_CLASSES);
    net->forward(x, 1);
    std::vector<float> t2(net->layers.back()->outputs, net->layers.back()->outputs + N_CLASSES);
    bool differ = false;
    for (int i = 0; i < N_CLASSES; ++i) if (t1[i] != t2[i]) differ = true;
    CHECK(differ, "is_training=true: repeated passes with the same input yield different outputs");

    net->set_is_training(false);
    net->forward(x, 1);
    std::vector<float> e1(net->layers.back()->outputs, net->layers.back()->outputs + N_CLASSES);
    net->forward(x, 1);
    bool same = true;
    for (int i = 0; i < N_CLASSES; ++i) if (e1[i] != net->layers.back()->outputs[i]) same = false;
    CHECK(same, "is_training=false: repeated passes yield identical outputs");
    delete net;
}

// ======================================================================
// Test 5: training and quality on a separate test file
// ======================================================================
static void run_training_case(OptimizerType opt, float lr, const std::string& name) {
    const int epochs = 40, bs = 20;
    NeuralNetwork* net = build_net(lr, opt, 0.2f);
    std::vector<float> loss(epochs);
    net->train(TRAIN_CSV, TRAIN_N, epochs, bs, loss.data());

    bool finite = true;
    for (float l : loss) if (!std::isfinite(l)) finite = false;
    std::vector<int> targets(TEST_N), guesses(TEST_N);
    float acc = net->test(TEST_CSV, TEST_N, targets.data(), guesses.data());
    std::set<int> distinct(guesses.begin(), guesses.end());
    std::cout << "  " << name << ": loss " << loss.front() << " -> " << loss.back()
              << ", test accuracy " << acc * 100.0f << "%, distinct classes predicted " << distinct.size() << "\n";

    CHECK(finite, name + ": loss is finite across all epochs");
    CHECK(loss.back() < MIN_LOSS_DROP_RATIO * loss.front(), name + ": loss decreases significantly");
    CHECK(acc > MIN_TEST_ACCURACY, name + ": accuracy on a separate test file exceeds the threshold");
    CHECK(distinct.size() >= 3, name + ": predictions did not collapse into 1–2 classes");
    delete net;
}
void test_synth_training_quality() {
    run_training_case(OptimizerType::SGD, 0.05f, "SGD");
    run_training_case(OptimizerType::Momentum, 0.02f, "Momentum");
    run_training_case(OptimizerType::Adam, 0.003f, "Adam");
}

// ======================================================================
// Test 6: changing batch sizes, incomplete batch, validation, test() between training sessions
// ======================================================================
void test_synth_transitions() {
    NeuralNetwork* net = build_net(0.03f, OptimizerType::SGD, 0.2f);
    std::vector<float> l1(4), l2(4), l3(2), l4(3);
    std::vector<int> targets(TEST_N), guesses(TEST_N);

    net->train(TRAIN_CSV, TRAIN_N, 4, 20, l1.data());
    net->test(TEST_CSV, TEST_N, targets.data(), guesses.data());
    net->train(TRAIN_CSV, TRAIN_N, 4, 13, l2.data());
    net->train(TRAIN_CSV, TRAIN_N, 2, 1, l3.data());
    net->train(TRAIN_CSV, TRAIN_N, 3, 16, l4.data(), 0.2f);

    bool finite = true;
    for (auto* v : {&l1, &l2, &l3, &l4}) for (float l : *v) if (!std::isfinite(l)) finite = false;
    CHECK(finite, "loss is finite for the sequence bs=20 -> test -> bs=13 -> bs=1 -> bs=16 with validation");

    bool training_on = true;
    for (auto* layer : net->layers) if (!layer->is_training) training_on = false;
    CHECK(training_on, "after train(), is_training=true for all layers (including Dropout)");

    DatasetFile one(TEST_CSV, TEST_N, INPUT_NODES, N_CLASSES, 1);
    net->set_batch_size(1);
    net->set_is_training(false);
    net->forward(one.image_batch, 1);
    float sum = 0.0f;
    for (int j = 0; j < N_CLASSES; ++j) sum += net->layers.back()->outputs[j];
    CHECK(std::fabs(sum - 1.0f) < 1e-3f, "Softmax output sums to 1 after all batch reshapes");

    int p1 = net->predict(one.image_batch), p2 = net->predict(one.image_batch);
    CHECK(p1 == p2 && p1 >= 0 && p1 < N_CLASSES, "predict() is deterministic in evaluation mode and returns a valid class");
    delete net;
}

int main() {
    std::cout << "=== Synthetic dataset: full-pipeline tests (Conv/ReLU/Pool x2, Dense, Dropout, Softmax) ===\n\n";
    std::cout << "-- Test 1: GPU-chain forward vs host reference --\n";   test_synth_forward_chain_equivalence();
    std::cout << "\n-- Test 2a: one SGD step vs reference --\n";            test_synth_one_step_sgd();
    std::cout << "\n-- Test 2b: one Momentum step vs reference --\n";       test_synth_one_step_momentum();
    std::cout << "\n-- Test 3: batch independence --\n";                    test_synth_batch_independence();
    std::cout << "\n-- Test 4: Dropout modes in the chain --\n";            test_synth_dropout_modes();
    std::cout << "\n-- Test 5: training quality (SGD, Momentum, Adam) --\n"; test_synth_training_quality();
    std::cout << "\n-- Test 6: batch-size transitions, uneven batch, validation --\n"; test_synth_transitions();
    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}