#pragma once
#include<fstream>
#include<string>
#include"utils.h"


void save_report(
    std::string current_directory,
    int layer_count,
    int* layers,
    float learning_rate,
    int epochs,
    int train_data_size,
    int test_data_size,
    float test_accuracy,
    float* accuracy_history,
    int* test_targets,
    int* test_guesses,
    std::string saved_weights
) {
    if (!check_folder_exists(current_directory + "/runs")){
        std::cout << "Directory " << current_directory + "/runs" << " does not exits. Try creating it first. Press any key to continue...";
        char press_to_continue;
        std::cin >> press_to_continue;
        exit(1);
    }

    std::string json_layers = "[";

    for (int i = 0; i < layer_count; ++i) {
        if (i > 0) json_layers += ", ";
        json_layers += std::to_string(layers[i]);
    }

    json_layers += "]";

    std::string json_test_targets = "[";
    std::string json_test_guesses = "[";

    for (int i = 0; i < test_data_size; ++i) {
        if (i > 0) {
            json_test_targets += ", ";
            json_test_guesses += ", ";
        }
        json_test_targets += std::to_string(test_targets[i]);
        json_test_guesses += std::to_string(test_guesses[i]);
    }
    
    json_test_targets += "]";
    json_test_guesses += "]";

    std::string json_accuracy_history = "[";

    for (int i = 0; i < epochs; ++i) {
        if (i > 0) json_accuracy_history += ", ";
        json_accuracy_history += std::to_string(accuracy_history[i]);
    }

    json_accuracy_history += "]";

    
    std::string json_content = 
        "{\n"
        "    \"layerCount\": " + std::to_string(layer_count) + ",\n"
        "    \"layers\": " + json_layers + ",\n"
        "    \"learningRate\": " + std::to_string(learning_rate) + ",\n"
        "    \"epochs\": " + std::to_string(epochs) + ",\n"
        "    \"trainDataSize\": " + std::to_string(train_data_size) + ",\n"
        "    \"testDataSize\": " + std::to_string(test_data_size) + ",\n"
        "    \"testAccuracy\": " + std::to_string(test_accuracy) + ",\n"
        "    \"accuracyHistory\": " + json_accuracy_history + ",\n"
        "    \"testTargets\": " + json_test_targets + ",\n"
        "    \"testGuesses\": " + json_test_guesses + ",\n"
        "    \"savedWeights\": " + (saved_weights == "" ? "null" : "\"" + saved_weights + "\"") + "\n"
        "}";

    std::string filename;
    filename = current_directory + "/runs/" + get_current_datetime_simple() + "_neuron_report.run.json";

    std::ofstream file(filename);
    if (!file) {
        return;
    }

    file << json_content;

    if (!file.good()) {
        std::cout << "Failed to save " << filename << '\n';
        char press_to_continue;
        std::cin >> press_to_continue;
        exit(1);
    }
}

void save_report(
    std::string current_directory,
    int layer_count,
    int* layers,
    int test_data_size,
    float test_accuracy,
    int* test_targets,
    int* test_guesses
) {
    if (!check_folder_exists(current_directory + "/predict")){
        std::cout << "Directory " << current_directory + "/predict" << " does not exits. Try creating it first. Press any key to continue...";
        char press_to_continue;
        std::cin >> press_to_continue;
        exit(1);
    }

    std::string json_layers = "[";

    for (int i = 0; i < layer_count; ++i) {
        if (i > 0) json_layers += ", ";
        json_layers += std::to_string(layers[i]);
    }

    json_layers += "]";
    
    std::string json_test_targets = "[";
    std::string json_test_guesses = "[";

    for (int i = 0; i < test_data_size; ++i) {
        if (i > 0) {
            json_test_targets += ", ";
            json_test_guesses += ", ";
        }
        json_test_targets += std::to_string(test_targets[i]);
        json_test_guesses += std::to_string(test_guesses[i]);
    }

    json_test_targets += "]";
    json_test_guesses += "]";

    
    std::string json_content = 
        "{\n"
        "    \"layerCount\": " + std::to_string(layer_count) + ",\n"
        "    \"layers\": " + json_layers + ",\n"
        "    \"testDataSize\": " + std::to_string(test_data_size) + ",\n"
        "    \"testAccuracy\": " + std::to_string(test_accuracy) + ",\n"
        "    \"testTargets\": " + json_test_targets + ",\n"
        "    \"testGuesses\": " + json_test_guesses + "\n"
        "}";

    std::string filename;
    filename = current_directory + "/predict/" + get_current_datetime_simple() + "_neuron_report.predict.json";

    std::ofstream file(filename);
    if (!file) {
        return;
    }

    file << json_content;

    if (!file.good()) {
        std::cout << "Failed to save " << filename << '\n';
        char press_to_continue;
        std::cin >> press_to_continue;
        exit(1);
    }
}