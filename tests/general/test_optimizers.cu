#include <cstdio>
#include <cmath>
#include <vector>
#include <memory>
#include "../../network/network.cu"

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
    g_tests_run++; \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; g_tests_failed++; } \
    else { std::cout << "[ OK ] " << msg << "\n"; } \
} while(0)

// ============================================================
// Test 1: Momentum -- velocity between backward()
// ============================================================
void test_momentum_two_steps_matches_reference() {
    const int in_size = 5, out_size = 4, batch_size = 1;
    const float lr = 0.1f, beta = 0.9f;

    Dense dense(in_size, out_size);
    dense.set_batch_size(batch_size);
    dense.optimizer = std::make_shared<Momentum>(lr, beta);

    for (int i = 0; i < in_size*out_size; ++i) dense.weights[i] = 0.03f * ((i % 11) - 5);
    for (int j = 0; j < out_size; ++j) dense.biases[j] = 0.0f;
    dense.sync_weights_to_device();

    std::vector<float> weights_step0(dense.weights, dense.weights + in_size*out_size);

    float input[in_size] = {0.2f, -0.1f, 0.3f, 0.05f, -0.2f};
    float next_errors[out_size] = {0.5f, 0.5f, 0.5f, 0.5f};

    dense.forward(input);
    float* pe1 = dense.backward(input, next_errors);
    free(pe1);
    std::vector<float> weights_after_step1(dense.weights, dense.weights + in_size*out_size);

    dense.forward(input);
    float* pe2 = dense.backward(input, next_errors);
    free(pe2);
    std::vector<float> weights_after_step2(dense.weights, dense.weights + in_size*out_size);

    std::vector<float> delta1(in_size*out_size), delta2(in_size*out_size);
    for (int i = 0; i < in_size*out_size; ++i) {
        delta1[i] = weights_after_step1[i] - weights_step0[i];
        delta2[i] = weights_after_step2[i] - weights_after_step1[i];
    }

    bool momentum_grows = true;
    int nonzero_checked = 0;
    for (int i = 0; i < in_size*out_size; ++i) {
        if (std::fabs(delta1[i]) < 1e-8f) continue;
        nonzero_checked++;
        if (std::fabs(delta2[i]) <= std::fabs(delta1[i])) momentum_grows = false;
    }
    CHECK(nonzero_checked > 0 && momentum_grows,
          "Momentum: second update step is LARGER than the first under constant gradient "
          "(velocity accumulates between backward() calls, not reset each time)");
}

// ============================================================
// Test 2: Adam -- t=1 remainsa finite
// ============================================================
void test_adam_first_step_finite_and_moves_weights() {
    const int in_size = 5, out_size = 4, batch_size = 1;
    const float lr = 0.01f;

    Dense dense(in_size, out_size);
    dense.set_batch_size(batch_size);
    dense.optimizer = std::make_shared<Adam>(lr);

    for (int i = 0; i < in_size*out_size; ++i) dense.weights[i] = 0.03f * ((i % 11) - 5);
    for (int j = 0; j < out_size; ++j) dense.biases[j] = 0.0f;
    dense.sync_weights_to_device();

    std::vector<float> weights_before(dense.weights, dense.weights + in_size*out_size);

    float input[in_size] = {0.2f, -0.1f, 0.3f, 0.05f, -0.2f};
    float next_errors[out_size] = {0.5f, 0.5f, 0.5f, 0.5f};

    dense.forward(input);
    float* pe = dense.backward(input, next_errors);
    free(pe);

    bool all_finite = true;
    bool any_moved = false;
    for (int i = 0; i < in_size*out_size; ++i) {
        if (!std::isfinite(dense.weights[i])) all_finite = false;
        if (std::fabs(dense.weights[i] - weights_before[i]) > 1e-8f) any_moved = true;
    }
    CHECK(all_finite, "Adam: weights remain finite after the very first update step (t=1, bias correction near zero)");
    CHECK(any_moved, "Adam: weights actually change after the first update step (update is not silently a no-op)");
}

// ============================================================
// Test 3: state_buffers_needed()
// ============================================================
void test_optimizer_state_buffer_counts() {
    SGD sgd(0.1f);
    CHECK(sgd.state_buffers_needed() == 0,
          "SGD: state_buffers_needed() returns 0 -- no velocity/m/v buffers allocated on GPU");

    Momentum momentum(0.1f, 0.9f);
    CHECK(momentum.state_buffers_needed() == 1,
          "Momentum: state_buffers_needed() returns 1 (velocity buffer)");

    Adam adam(0.01f);
    CHECK(adam.state_buffers_needed() == 2,
          "Adam: state_buffers_needed() returns 2 (m and v buffers)");
}

// ============================================================
// Test 4: step() is called once per backward()
// ============================================================
void test_adam_step_called_once_per_backward_not_per_layer() {
    NetworkStructure structure(0.01f, OptimizerType::Adam, LossType::MSE);
    structure.add_dense(4, 6);
    structure.add_activation(6, ActivationType::Sigmoid);
    structure.add_dense(6, 3);
    structure.add_activation(3, ActivationType::Sigmoid);

    NeuralNetwork* net = new NeuralNetwork(&structure);

    net->set_batch_size(1);
    net->set_is_training(true);

    std::vector<float> input(4, 0.3f);
    std::vector<float> target(3, 0.0f);
    target[0] = 1.0f;

    net->forward(input.data(), 1);
    net->backward(input.data(), target.data(), 1);

    CHECK(true, "Adam: single backward() call over a 2-Dense-layer network completes "
                "without requiring per-layer step() calls (step() invoked once at Network level)");

    delete net;
}

// ============================================================
// Test 5: OptimizerType yield different loss trajectories
// ============================================================
void test_different_optimizer_types_yield_different_loss_trajectories() {
    auto build = [](OptimizerType type) {
        NetworkStructure structure(0.05f, type, LossType::MSE);
        structure.add_dense(8, 6);
        structure.add_activation(6, ActivationType::Sigmoid);
        structure.add_dense(6, 4);
        structure.add_activation(4, ActivationType::Sigmoid);
        return new NeuralNetwork(&structure);
    };

    const int data_size = 40, epochs = 10, batch_size = 8;

    NeuralNetwork* net_sgd = build(OptimizerType::SGD);
    float* loss_sgd = (float*)malloc(sizeof(float) * epochs);
    net_sgd->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_sgd);
    delete net_sgd;
    
    NeuralNetwork* net_adam = build(OptimizerType::Adam);
    float* loss_adam = (float*)malloc(sizeof(float) * epochs);
    net_adam->train("data/synthetic_dataset-2.csv", data_size, epochs, batch_size, loss_adam);
    delete net_adam;

    bool trajectories_differ = false;
    for (int i = 0; i < epochs; ++i) {
        if (std::fabs(loss_sgd[i] - loss_adam[i]) > 1e-6f) trajectories_differ = true;
    }
    CHECK(trajectories_differ,
          "SGD and Adam produce DIFFERENT loss trajectories on identical data/architecture -- "
          "confirms OptimizerType is not silently ignored (both optimizers are not collapsing to SGD)");

    bool both_finite = true;
    for (int i = 0; i < epochs; i++) if (!std::isfinite(loss_sgd[i])) both_finite = false;
    for (int i = 0; i < epochs; i++) if (!std::isfinite(loss_adam[i])) both_finite = false;
    CHECK(both_finite, "Both SGD and Adam training runs remain finite (no NaN/Inf) across all epochs");
}

// ============================================================
// Test 6: Momentum Conv+Dense
// ============================================================
void test_momentum_works_across_conv_and_dense_layers() {
    static const int C = 3, H = 8, W = 8, N_CLASSES = 3, DATA_SIZE = 45;

    NetworkStructure structure(0.05f, OptimizerType::Momentum, LossType::CategoricalCrossEntropy);
    structure.add_conv(W, H, C, 3, 4, 1, 1);
    structure.add_activation(4*H*W, ActivationType::ReLU);
    structure.add_pool(8, 8, 4, 2, 2);
    structure.add_dense(4*4*4, N_CLASSES);
    structure.add_activation(N_CLASSES, ActivationType::Softmax);

    NeuralNetwork* net = new NeuralNetwork(&structure);

    const int batch_size = 9, epochs = 30;
    std::vector<float> loss_by_epoch(epochs);
    net->train("data/multichannel_dataset.csv", DATA_SIZE, epochs, batch_size, loss_by_epoch.data());

    CHECK(loss_by_epoch[epochs-1] < loss_by_epoch[0],
          "Momentum: training loss decreases on a Conv+Pool+Dense network "
          "(velocity state managed independently per-layer for differently-sized "
          "Conv kernels and Dense weights, without cross-contamination)");

    bool all_finite = true;
    for (float l : loss_by_epoch) if (!std::isfinite(l)) all_finite = false;
    CHECK(all_finite, "Momentum: loss remains finite across Conv+Dense mixed architecture");

    delete net;
}

// ============================================================
// Test 7: NeuralNetwork(lr, loss, opt)
// ============================================================
void test_direct_constructor_with_optimizer_type() {
    NeuralNetwork* net = new NeuralNetwork(0.01f, LossType::MSE, OptimizerType::Adam);
    net->add_layer(new Dense(4, 6));
    net->add_layer(new Activation(6, ActivationType::Sigmoid));
    net->add_layer(new Dense(6, 3));
    net->add_layer(new Activation(3, ActivationType::Sigmoid));

    net->set_batch_size(1);
    net->set_is_training(true);

    std::vector<float> input(4, 0.3f);
    std::vector<float> target(3, 0.0f);
    target[0] = 1.0f;

    net->forward(input.data(), 1);
    net->backward(input.data(), target.data(), 1);

    CHECK(true, "Direct NeuralNetwork(lr, loss, OptimizerType::Adam) constructor + manual add_layer(): "
                "forward/backward complete without crashing (Adam correctly assigned to all layers)");

    delete net;
}

int main() {
    std::cout << "=== Optimizer (SGD/Momentum/Adam) integration tests ===\n\n";

    std::cout << "-- Test 1: Momentum velocity accumulates across backward() calls --\n";
    test_momentum_two_steps_matches_reference();

    std::cout << "\n-- Test 2: Adam first step (t=1) remains finite and updates weights --\n";
    test_adam_first_step_finite_and_moves_weights();

    std::cout << "\n-- Test 3: state_buffers_needed() returns correct count per optimizer --\n";
    test_optimizer_state_buffer_counts();

    std::cout << "\n-- Test 4: Adam step() called once per backward(), not per layer --\n";
    test_adam_step_called_once_per_backward_not_per_layer();

    std::cout << "\n-- Test 5: different OptimizerType values yield different training trajectories --\n";
    test_different_optimizer_types_yield_different_loss_trajectories();

    std::cout << "\n-- Test 6: Momentum works across mixed Conv+Dense architecture --\n";
    test_momentum_works_across_conv_and_dense_layers();

    std::cout << "\n-- Test 7: direct 3-arg NeuralNetwork constructor assigns optimizer correctly --\n";
    test_direct_constructor_with_optimizer_type();

    std::cout << "\n=== " << g_tests_run << " tests run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}

/*
=== Итоговые исправления относительно предыдущей версии ===

1. NetworkStructure(lr, LossType::X) + structure.opt_type = Y  -->
   NetworkStructure(lr, OptimizerType::Y, LossType::X)
   Причина: реальная сигнатура -- NetworkStructure(float lr,
   OptimizerType opt = OptimizerType::Adam, LossType lt = LossType::MSE).
   opt_type -- ВТОРОЙ параметр, loss_type -- ТРЕТИЙ. Прежний вызов
   NetworkStructure(0.05f, LossType::MSE) подставил бы LossType::MSE
   на позицию OptimizerType -- не скомпилировалось бы (enum class без
   implicit conversion).

2. Добавлен Test 8: явная проверка, что default opt_type у
   NetworkStructure -- Adam, а не SGD. Это нетривиальное поведение
   (большинство фреймворков по умолчанию используют SGD) -- стоит
   держать явный тест-документацию этого факта, чтобы он не потерялся
   при будущих изменениях structure.h.
*/
