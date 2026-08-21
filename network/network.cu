#pragma once
#include<iostream>
#include<string>
#include<sstream>
#include<fstream>
#include<vector>
#include <cassert>

#include"matrix/matrix.cuh"
#include"utils/progressbar.h"
#include"utils/utils.h"
#include"utils/file.h"
#include"utils/structure.h"
#include"utils/logging.h"

#include"loss/loss.cuh"
#include"dense.cu"
#include"conv.cu"
#include"maxpooling.cu"
#include"activation.cu"
#include"dropout.cu"


class NeuralNetwork 
{
private:
    float learning_rate;
    std::vector<Layer*> layers;
    LossType loss_function;

    int input_nodes;
    int output_nodes;

    float calculateLoss(const float* target_batch, int current_batch_size);
    int getMaxActivationIndex(float *target);

    public:
    NeuralNetwork(float lr, LossType loss = LossType::MSE);
    NeuralNetwork(const NetworkStructure* structure);
    void add_layer(Layer* Layer);
    void set_learning_rate(float lr);
    void set_batch_size(int bs);
    void set_is_training(bool training);

    void forward(float *inputs, int current_batch_size);
    void backward(float *inputs, float *targets, int current_batch_size);
    int predict(float *input);
    void train(std::string data, int data_size, int epochs, int batch_size, float* loss_by_epoch);
    float test(std::string filePath, int data_size, int* test_targets, int* test_guesses);

    void log_structure();

    void save_weights(std::string path);
    void load_weights(std::string path);

    friend void test_multichannel_softmax_full_pipeline();
    friend void test_softmax_cce_training_reduces_loss();
    friend void test_softmax_output_sums_to_one_after_training();
    friend void test_is_training_switches_behavior();
    friend void test_repeated_train_after_test_reenables_training_mode();
};

NeuralNetwork::NeuralNetwork(float lr, LossType loss) {
    learning_rate = lr;
    loss_function = loss;
    input_nodes = 0;
    output_nodes = 0;
}

NeuralNetwork::NeuralNetwork(const NetworkStructure* structure) {
    learning_rate = structure->learning_rate;
    loss_function = structure->loss_type;

    for (LayerStructure* layer : structure->layers) {
        if (layer->layer_type == LayerType::Conv) {
            auto* c = static_cast<ConvStructure*>(layer);
            Conv* conv = new Conv(c->input_height, c->input_width, c->channels, c->kernel_size, c->num_kernels, c->stride, c->padding);
            add_layer(conv);

        } else if (layer->layer_type == LayerType::Pool) {
            auto* p = static_cast<PoolStructure*>(layer);
            MaxPooling* pool = new MaxPooling(p->input_width, p->input_height, p->channels, p->pool, p->stride);
            add_layer(pool);

        } else if (layer->layer_type == LayerType::Dense) {
            auto* d = static_cast<DenseStructure*>(layer);
            Dense* dense = new Dense(d->input_nodes, d->output_nodes);
            add_layer(dense);

        } else if (layer->layer_type == LayerType::Activation) {
            auto* a = static_cast<ActivationStructure*>(layer);
            int inferred_size = a->size;
            if (inferred_size == 0) {
                if (layers.empty()) {
                    std::cerr << "Activation layer cannot be the first layer in the network." << std::endl;
                    std::exit(1);
                }
                inferred_size = layers.back()->output_size;
            }

            Activation* activation = new Activation(inferred_size, a->activation_type);
            add_layer(activation);

        } else if (layer->layer_type == LayerType::Dropout) {
            auto* dr = static_cast<DropoutStructure*>(layer);

            int inferred_size = dr->size;
            if (inferred_size == 0) {
                if (layers.empty()) {
                    std::cerr << "Dropout layer cannot be the first layer in the network." << std::endl;
                    std::exit(1);
                }
                inferred_size = layers.back()->output_size;
            }

            Dropout* dropout = new Dropout(inferred_size, dr->drop_prob);
            add_layer(dropout);

        } else {
            std::cerr << "Unknown layer type in NetworkStructure" << std::endl;
            std::exit(1);
        }
    }

    set_learning_rate(learning_rate);
}


void NeuralNetwork::log_structure() {
    std::cout << "Layer count: " << layers.size() << std::endl;
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << "Layer " << i + 1 << ": Size = " << layers[i]->size << ", Output Size = " << layers[i]->output_size << ", Learning Rate = " << layers[i]->learning_rate << std::endl;
    }
    std::cout << "Learning Rate: " << learning_rate << std::endl;
}

void NeuralNetwork::add_layer(Layer* layer){
    layer->set_learning_rate(learning_rate);
    if (layers.size() == 0) {
        layers.push_back(layer);
        input_nodes = layer->size;
        output_nodes = layer->output_size;
        return;
    }
    if (layer->size != layers.back()->output_size) {
        std::cerr << "Unmatched layer " << layers.size() + 1 << " dim: " << layer->size << ' ' << "(expected " << layers.back()->output_size << ")" << '\n';
        exit(1);
    }

    layers.push_back(layer);
    output_nodes = layer->output_size;
}

void NeuralNetwork::set_learning_rate(float lr){
    for (auto &layer : layers) layer->set_learning_rate(lr);
}

void NeuralNetwork::set_is_training(bool training) {
    for (auto* layer : layers) layer->set_is_training(training);
}

void NeuralNetwork::set_batch_size(int bs) {
    for (auto* layer : layers) layer->set_batch_size(bs);
}

float NeuralNetwork::calculateLoss(const float* target_batch, int current_batch_size) {
    const float* output_batch = layers.back()->outputs;
    float loss = 0.0f;

    for (int b = 0; b < current_batch_size; ++b) {
        const float* output = output_batch + (size_t)b * output_nodes;
        const float* target = target_batch + (size_t)b * output_nodes;

        for (int i = 0; i < output_nodes; ++i) {
            loss += loss_value(output[i], target[i], loss_function);
        }
    }

    return loss;
}

void NeuralNetwork::forward(float *inputs, int current_batch_size){
    float *layer_inputs;
    size_t first_layer_total = (size_t)current_batch_size * layers[0]->size;

    layer_inputs = (float*)malloc(sizeof(float) * first_layer_total);
    for (size_t i = 0; i < first_layer_total; i++) layer_inputs[i] = inputs[i];

    for (auto &layer : layers) {
        layer->forward(layer_inputs);
        free(layer_inputs);

        size_t layer_total_output = (size_t)current_batch_size * layer->output_size;
        layer_inputs = (float*)malloc(sizeof(float) * layer_total_output);
        for (size_t i = 0; i < layer_total_output; i++) layer_inputs[i] = layer->outputs[i];
    }

    free(layer_inputs);
}


void NeuralNetwork::backward(float *inputs, float *targets, int current_batch_size){
    int num_layers = layers.size();

    float *output = layers.back()->outputs;

    size_t total_output_errors = (size_t)current_batch_size * output_nodes;
    float *output_errors = (float*)malloc(sizeof(float) * total_output_errors);
    for (size_t i = 0; i < total_output_errors; ++i) {
        output_errors[i] = targets[i] - output[i];
    }

    bool raw_gradient_for_last_layer =
        (layers.back()->activation_type == ActivationType::Sigmoid && loss_function == LossType::BinaryCrossEntropy) ||
        (layers.back()->activation_type == ActivationType::Softmax && loss_function == LossType::CategoricalCrossEntropy);

    float *current_errors = output_errors;

    for (int i = num_layers - 1; i >= 0; --i) {
        float *layer_input = (i == 0) ? inputs : layers[i - 1]->outputs;

        float *prev_errors = (i == num_layers - 1)
            ? layers[i]->backward(layer_input, current_errors, raw_gradient_for_last_layer)
            : layers[i]->backward(layer_input, current_errors);

        if (i != num_layers - 1) {
            free(current_errors);
        }
        current_errors = prev_errors;
    }

    free(current_errors);
    free(output_errors);
}


void NeuralNetwork::train(std::string data, int data_size, int epochs, int batch_size, float* loss_by_epoch){
    DatasetFile train_file(data, data_size, input_nodes, output_nodes, batch_size);

    int num_batches = train_file.get_num_batches();
    int progress_tick = num_batches / 100;
    if (num_batches < 100) progress_tick = 1;

    float epoch_loss = 0.0f;
    int last_batch_size_set = -1;

    set_is_training(true);

    for (int epoch = 1; epoch <= epochs; epoch++) {

        float total_loss = 0.0f;

        train_file.shuffle();

        for (int b = 0; b < num_batches; b++) {
            if (b > 0) train_file.next_batch();

            int current_batch_size = train_file.current_batch_size;

            if (current_batch_size != last_batch_size_set) {
                for (auto &layer : layers) layer->set_batch_size(current_batch_size);
                last_batch_size_set = current_batch_size;
            }

            forward(train_file.image_batch, current_batch_size);

            float batch_loss = calculateLoss(train_file.target_batch, current_batch_size);
            total_loss += batch_loss;

            backward(train_file.image_batch, train_file.target_batch, current_batch_size);

            if (b % progress_tick == 0 && b != 0){
                log_train_process(epochs, epoch, num_batches, b, epoch_loss);
            }
        }

        epoch_loss = total_loss / (float)(data_size);
        loss_by_epoch[epoch - 1] = epoch_loss;
    }
}

float NeuralNetwork::test(std::string filePath, int data_size, int* test_targets, int* test_guesses) {
    DatasetFile test_file(filePath, data_size, input_nodes, output_nodes, 1);

    for (auto &layer : layers) layer->set_batch_size(1);
    set_is_training(false);

    int correctGuesses = 0;
    int progress_tick = data_size / 100;
    if (data_size < 100) progress_tick = 1;

    for (int i = 0; i < data_size; i++) {
        int result = predict(test_file.image_batch);
        int test_target = getMaxActivationIndex(test_file.target_batch);

        if (result == test_target) correctGuesses++;

        if (i % progress_tick == 0 && i != 0){
            log_test_process(data_size, i);
        }

        test_targets[i] = test_target;
        test_guesses[i] = result;

        if (i != data_size - 1) test_file.next_batch();
    }

    return (float)correctGuesses / data_size;
}


int NeuralNetwork::getMaxActivationIndex(float *target){
	int maxIndex = -1;
	float maxVal = -1000000;
	for (unsigned i = 0; i < output_nodes; i++){
		if (target[i] > maxVal) {
			maxVal = target[i];
			maxIndex = i;
		}
	}
	if (maxIndex == -1) throw std::runtime_error("Incorrent output values.");
	return maxIndex;
}

int NeuralNetwork::predict(float *input){
    forward(input, 1);
    int prediction = getMaxActivationIndex(layers.back()->outputs);
    return prediction;
}


void NeuralNetwork::save_weights(std::string path) {
    for (auto &layer : layers) {
        layer->save_weights(path);
    }
}

void NeuralNetwork::load_weights(std::string path) {
    if (!check_file_exists(path)) {
        std::cerr << "Failed to load weights " << path << ". File does not exist." << '\n';
        exit(1);
    }

    int bytes_to_skip = 0;

    for (auto &layer : layers) {
        bytes_to_skip += sizeof(float) * layer->load_weights(path, bytes_to_skip);
    }
}