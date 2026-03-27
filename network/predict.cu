#include"network.cu"
#include"utils/report.h"
#include<string>


void predict(
    NetworkStructure* structure,
    int test_data_size,
    const std::string test_data,
    const std::string weights_path,
    const std::string current_directory,
    const std::string report_file_name
){
    NeuralNetwork model(structure);

    model.load_weights(weights_path);

    int* test_targets = (int*)malloc(sizeof(int) * test_data_size);
    int* test_guesses = (int*)malloc(sizeof(int) * test_data_size);

    float accuracy = model.test(test_data, test_data_size, test_targets, test_guesses);

    std::string json_network_structure = structure->structure_to_json();

    save_report(
        current_directory,
        json_network_structure,
        test_data_size,
        accuracy,
        test_targets,
        test_guesses,
        report_file_name
    );

    std::cout << "Final Test Accuracy: " << accuracy * 100.0f << "%" << std::endl;

    free(test_targets);
    free(test_guesses);
}