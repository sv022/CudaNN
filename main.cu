#include"network.cu"
#include"utils/report.h"

int main(){
    const int input_nodes = 784;
    int hidden_nodes = 256;
    const int output_nodes = 10;
    float learning_rate = 0.3;
    NeuralNetwork n(input_nodes, hidden_nodes, output_nodes, learning_rate);

    int epochs = 5;
    int train_data_size = 5000;
    int test_data_size = 50;

    // float input[input_nodes];
    // Matrix::initRandomf_static(input, input_nodes, 1);
    // // for (int i = 0; i < input_nodes; i++) std::cout << input[i] << ' ';
    // std::cout << '\n';
    // float target[output_nodes];
    // Matrix::initRandomf_static(target, output_nodes, 1);
    // for (int i = 0; i < output_nodes; i++) std::cout << target[i] << ' ';
    
    n.train("data/data_fashion_train.txt", train_data_size, epochs);

    int *test_targets = (int*)malloc(sizeof(int) * test_data_size);
    int *test_guesses = (int*)malloc(sizeof(int) * test_data_size);

    float accuracy = n.test("data/data_fashion_test.txt", test_data_size, test_targets, test_guesses);
    
    save_report(
        input_nodes,
        hidden_nodes,
        output_nodes,
        learning_rate,
        epochs,
        train_data_size,
        test_data_size,
        accuracy,
        test_targets,
        test_guesses,
        ""
    );

    return 0;
}