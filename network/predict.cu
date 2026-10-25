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

void predict(
    NetworkStructure* structure,
    const std::string test_data,
    const std::string weights_path, 
    ImageShape user_shape = ImageShape(0)
){
    NeuralNetwork model(structure);

    ImageShape shape = model.get_input_shape();
    if (user_shape.is_image()) {
        if (user_shape.numel() != shape.numel())
            throw std::runtime_error("Image size c*h*w does not match the network input size.");
        shape = user_shape;
    } else if (!shape.is_image()) {
        throw std::runtime_error("The first layer is Dense: pass <c> <h> <w> after the weights path.");
    }
    if (!check_file_exists(test_data)) throw std::runtime_error("Image file does not exist: " + test_data);

    model.load_weights(weights_path);
    model.set_batch_size(1);
    model.set_is_training(false);

    std::vector<float> input(shape.numel());
    load_image_chw(test_data, shape, input.data());

    const int cls = model.predict(input.data());

    std::vector<float> out(model.get_output_nodes());
    model.get_outputs(out.data());

    const bool is_softmax = (model.get_output_activation() == ActivationType::Softmax);
    std::cout << "class: " << cls << '\n' ;
    const int k = std::min<int>(out.size(), 5);
    std::cout << "\ntop-" <<  k << (is_softmax ? "probabilities:" : "scores:") << ":\n";
    std::vector<float> tmp = out;
    for (int i = 0; i < k; ++i) {
        int m = std::max_element(tmp.begin(), tmp.end()) - tmp.begin();
        std::cout << m << ' ' << tmp[m] << '\n';
        tmp[m] = -INFINITY;
    }
}
