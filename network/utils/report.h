#pragma once
#include<fstream>
#include<string>
#include"utils.h"


void save_report(
    std::string current_directory,
    std::string json_network_structure,
    int epochs,
    int train_data_size,
    int test_data_size,
    float test_accuracy,
    float* loss_history,
    int* test_targets,
    int* test_guesses,
    std::string saved_weights,
    std::string report_file_name
) {
    if (!check_folder_exists(current_directory + "/runs")){
        std::cout << "Directory " << current_directory + "/runs" << " does not exits. Try creating it first.\n";
        exit(1);
    }

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

    std::string json_loss_history = "[";

    for (int i = 0; i < epochs; ++i) {
        if (i > 0) json_loss_history += ", ";
        json_loss_history += std::to_string(loss_history[i]);
    }

    json_loss_history += "]";

    
    std::string json_content = 
        "{\n"
        "    \"networkStructure\": " + json_network_structure + ",\n"
        "    \"epochs\": " + std::to_string(epochs) + ",\n"
        "    \"trainDataSize\": " + std::to_string(train_data_size) + ",\n"
        "    \"testDataSize\": " + std::to_string(test_data_size) + ",\n"
        "    \"testAccuracy\": " + std::to_string(test_accuracy) + ",\n"
        "    \"lossHistory\": " + json_loss_history + ",\n"
        "    \"testTargets\": " + json_test_targets + ",\n"
        "    \"testGuesses\": " + json_test_guesses + ",\n"
        "    \"savedWeights\": " + (saved_weights == "" ? "null" : "\"" + saved_weights + "\"") + "\n"
        "}";

    std::string filename;
    filename = current_directory + "/runs/" + (report_file_name == "-" ? ( "Z-" + get_current_datetime_simple() ) : report_file_name) + ".run.json";

    std::ofstream file(filename);
    if (!file) {
        return;
    }

    file << json_content;

    if (!file.good()) {
        std::cout << "Failed to save " << filename << '\n';
        exit(1);
    }
}

void save_report(
    std::string current_directory,
    std::string json_network_structure,
    int test_data_size,
    float test_accuracy,
    int* test_targets,
    int* test_guesses,
    std::string report_file_name
) {
    if (!check_folder_exists(current_directory + "/predict")){
        std::cout << "Directory " << current_directory + "/predict" << " does not exits. Try creating it first.\n";
        exit(1);
    }
    
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
        "    \"networkStructure\": " + json_network_structure + ",\n"
        "    \"testDataSize\": " + std::to_string(test_data_size) + ",\n"
        "    \"testAccuracy\": " + std::to_string(test_accuracy) + ",\n"
        "    \"testTargets\": " + json_test_targets + ",\n"
        "    \"testGuesses\": " + json_test_guesses + "\n"
        "}";

    std::string filename;
    filename = current_directory + "/predict/" + (report_file_name == "-" ? ( "Z-" + get_current_datetime_simple() ) : report_file_name) + ".predict.json";

    std::ofstream file(filename);
    if (!file) {
        return;
    }

    file << json_content;

    if (!file.good()) {
        std::cout << "Failed to save " << filename << '\n';
        exit(1);
    }
}