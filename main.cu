#include"network.cu"

int main(){
    const int input_nodes = 784;
    int hidden_nodes = 256;
    const int output_nodes = 10;
    NeuralNetwork n(input_nodes, hidden_nodes, output_nodes, 0.3);

    // float input[input_nodes];
    // Matrix::initRandomf_static(input, input_nodes, 1);
    // // for (int i = 0; i < input_nodes; i++) std::cout << input[i] << ' ';
    // std::cout << '\n';
    // float target[output_nodes];
    // Matrix::initRandomf_static(target, output_nodes, 1);
    // for (int i = 0; i < output_nodes; i++) std::cout << target[i] << ' ';
    
    n.train("data/data_fashion_train.txt", 5000, 20);

    float accuracy = n.test("data/data_fashion_test.txt", 50);
    
    std::cout << "Accuracy: " << accuracy << '\n';

    // for (int i = 0; i < label_size; i++) std::cout << targets[i] << ' ';
    return 0;
}