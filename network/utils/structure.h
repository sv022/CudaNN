#pragma once
#include<vector>
#include<string>
#include"../loss/loss.cuh"
#include"../optimizer/optimizer.cuh"


enum class LayerType
{
    Conv,
    Pool,
    Dense,
    Activation,
    Dropout
};

struct LayerStructure
{
    LayerType layer_type;
    virtual ~LayerStructure() = default;
};

struct PoolStructure : LayerStructure
{
    unsigned int input_width;
    unsigned int input_height;

    unsigned int channels;

    unsigned int pool;
    unsigned int stride;

    PoolStructure() : input_width(0), input_height(0), channels(0), pool(0), stride(0) {
        layer_type = LayerType::Pool;
    }
    PoolStructure(int w, int h, int c, int p, int s) : input_width(w), input_height(h), channels(c), pool(p), stride(s) {
        layer_type = LayerType::Pool;
    }
};


struct ConvStructure : LayerStructure
{
    unsigned int input_width;
    unsigned int input_height;

    unsigned int channels;
    unsigned int num_kernels;
    unsigned int kernel_size;

    unsigned int stride;
    unsigned int padding;

    ConvStructure() : input_width(0), input_height(0), channels(0), stride(0), padding(0) {
        layer_type = LayerType::Conv;
    }
    ConvStructure(int w, int h, int c, int k, int num_k, int s, int p)
        : input_width(w), input_height(h), channels(c), kernel_size(k), num_kernels(num_k), stride(s), padding(p) {
        layer_type = LayerType::Conv;
    }
};


struct DenseStructure : LayerStructure
{
    unsigned int input_nodes;
    unsigned int output_nodes;

    DenseStructure() : input_nodes(0), output_nodes(0) {
        layer_type = LayerType::Dense;
    }
    DenseStructure(int i_nodes, int o_nodes) : input_nodes(i_nodes), output_nodes(o_nodes) {
        layer_type = LayerType::Dense;
    }
};

struct ActivationStructure : LayerStructure
{
    unsigned int size;
    ActivationType activation_type;

    ActivationStructure() : size(0), activation_type(ActivationType::Linear) {
        layer_type = LayerType::Activation;
    }
    ActivationStructure(unsigned int s, ActivationType act) : size(s), activation_type(act) {
        layer_type = LayerType::Activation;
    }
};

struct DropoutStructure : LayerStructure
{
    unsigned int size;
    float drop_prob;

    DropoutStructure() : size(0), drop_prob(0.5f) {
        layer_type = LayerType::Dropout;
    }
    DropoutStructure(unsigned int s, float p) : size(s), drop_prob(p) {
        layer_type = LayerType::Dropout;
    }
};


struct NetworkStructure
{
    std::vector<LayerStructure*> layers;
    float learning_rate;
    OptimizerType opt_type;
    LossType loss_type;

    NetworkStructure(float lr = 0.001f, OptimizerType opt = OptimizerType::SGD, LossType lt = LossType::MSE): learning_rate(lr), opt_type(opt), loss_type(lt) {}

    void add_dropout(unsigned int size, float drop_prob) {
        layers.push_back(new DropoutStructure(size, drop_prob));
    }

    void add_activation(unsigned int size, ActivationType activation) {
        layers.push_back(new ActivationStructure(size, activation));
    }

    void add_dense(unsigned int input_nodes, unsigned int output_nodes) {
        layers.push_back(new DenseStructure(input_nodes, output_nodes));
    }

    void add_pool(unsigned int input_width, unsigned int input_height, unsigned int channels, unsigned int pool, unsigned int stride) {
        layers.push_back(new PoolStructure(input_width, input_height, channels, pool, stride));
    }

    void add_conv(unsigned int input_width, unsigned int input_height, 
        unsigned int channels, unsigned int kernel_size, 
        unsigned int num_kernels, unsigned int stride, unsigned int padding
    ){
        layers.push_back(new ConvStructure(input_width, input_height, channels, kernel_size, num_kernels, stride, padding));
    }

    std::string get_structure_string() {
        std::string structure_str;
        for (auto* layer : layers) {
            if (layer->layer_type == LayerType::Conv) {
                auto* conv_layer = static_cast<ConvStructure*>(layer);
                structure_str += "conv_" + std::to_string(conv_layer->input_width) + "x" + std::to_string(conv_layer->input_height) + "x" + std::to_string(conv_layer->channels) + "_" +
                                 std::to_string(conv_layer->kernel_size) + "_" + std::to_string(conv_layer->num_kernels) + "_" +
                                 std::to_string(conv_layer->stride) + "_" + std::to_string(conv_layer->padding) + "-";
            } else if (layer->layer_type == LayerType::Pool) {
                auto* pool_layer = static_cast<PoolStructure*>(layer);
                structure_str += "pool_" + std::to_string(pool_layer->input_width) + "x" + std::to_string(pool_layer->input_height) + "x" + std::to_string(pool_layer->channels) + "_" +
                                 std::to_string(pool_layer->pool) + "_" + std::to_string(pool_layer->stride) + "-";
            } else if (layer->layer_type == LayerType::Dense) {
                auto* dense_layer = static_cast<DenseStructure*>(layer);
                structure_str += "dense_" + std::to_string(dense_layer->input_nodes) + "_" + std::to_string(dense_layer->output_nodes) + "-";
            }
        }
        return structure_str.empty() ? structure_str : structure_str.substr(0, structure_str.size() - 1); // Remove trailing '-'
    }

    std::string structure_to_json() {
        std::string json = "{\n  \t\t\"learning_rate\": " + std::to_string(learning_rate);
        json += (",\n  \t\t\"optimizer\":" + opt_type_to_str(opt_type));
        json += (",\n  \t\t\"loss_function\":" + loss_type_to_str(loss_type));
        json += ",\n  \t\t\"layers\": [\n";

        std::string trailing_comma;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (i < layers.size() - 1) trailing_comma = ",";
            else trailing_comma = "";
            json += "\t\t{\n";
            if (layers[i]->layer_type == LayerType::Conv) {
                auto* conv_layer = static_cast<ConvStructure*>(layers[i]);
                json += "\t\t\t\"type\": \"conv\",\n";
                json += "\t\t\t\"input_width\": " + std::to_string(conv_layer->input_width) + ",\n";
                json += "\t\t\t\"input_height\": " + std::to_string(conv_layer->input_height) + ",\n";
                json += "\t\t\t\"channels\": " + std::to_string(conv_layer->channels) + ",\n";
                json += "\t\t\t\"kernel_size\": " + std::to_string(conv_layer->kernel_size) + ",\n";
                json += "\t\t\t\"num_kernels\": " + std::to_string(conv_layer->num_kernels) + ",\n";
                json += "\t\t\t\"stride\": " + std::to_string(conv_layer->stride) + ",\n";
                json += "\t\t\t\"padding\": " + std::to_string(conv_layer->padding) + "\n\t\t}" + trailing_comma + "\n";
            } else if (layers[i]->layer_type == LayerType::Pool) {
                auto* pool_layer = static_cast<PoolStructure*>(layers[i]);
                json += "\t\t\t\"type\": \"pool\",\n";
                json += "\t\t\t\"input_width\": " + std::to_string(pool_layer->input_width) + ",\n";
                json += "\t\t\t\"input_height\": " + std::to_string(pool_layer->input_height) + ",\n";
                json += "\t\t\t\"channels\": " + std::to_string(pool_layer->channels) + ",\n";
                json += "\t\t\t\"pool\": " + std::to_string(pool_layer->pool) + ",\n";
                json += "\t\t\t\"stride\": " + std::to_string(pool_layer->stride) + "\n\t\t}" + trailing_comma + "\n";
            } else if (layers[i]->layer_type == LayerType::Dense) {
                auto* dense_layer = static_cast<DenseStructure*>(layers[i]);
                json += "\t\t\t\"type\": \"dense\",\n";
                json += "\t\t\t\"input_nodes\": " + std::to_string(dense_layer->input_nodes) + ",\n";
                json += "\t\t\t\"output_nodes\": " + std::to_string(dense_layer->output_nodes) + "\n\t\t}" + trailing_comma + "\n";
            } else if (layers[i]->layer_type == LayerType::Activation) {
                auto* activation_layer = static_cast<ActivationStructure*>(layers[i]);
                json += "\t\t\t\"type\": \"activation\",\n";
                json += "\t\t\t\"activation_type\": " + activation_type_to_str(activation_layer->activation_type) + "\n\t\t}" + trailing_comma + "\n";
            } else if (layers[i]->layer_type == LayerType::Dropout) {
                auto* dropout_layer = static_cast<DropoutStructure*>(layers[i]);
                json += "\t\t\t\"type\": \"dropout\",\n";
                json += "\t\t\t\"drop_prob\": " + std::to_string(dropout_layer->drop_prob) + "\n\t\t}" + trailing_comma + "\n";
            }
        }
        json += "\t]\n}";
        return json;
    }

    ~NetworkStructure(){
        for (auto* l : layers) delete l;
    }
};


void parse_network_structure(const std::string net_str, OptimizerType opt, float learning_rate, LossType loss, NetworkStructure* net) {
    auto layers = split(net_str, '-');

    net->opt_type = opt;
    net->learning_rate = learning_rate;
    net->loss_type = loss;

    for (const auto& layer : layers) {
        if (layer.rfind("conv_", 0) == 0) {
            auto parts = split(layer.substr(5), '_');
            auto whc = split(parts[0], 'x');
            net->add_conv(std::stoi(whc[0]), std::stoi(whc[1]),  std::stoi(whc[2]), std::stoi(parts[1]), std::stoi(parts[2]), std::stoi(parts[3]), std::stoi(parts[4]));
        } else if (layer.rfind("pool_", 0) == 0) {
            auto parts = split(layer.substr(5), '_');
            auto whc = split(parts[0], 'x');
            net->add_pool(std::stoi(whc[0]), std::stoi(whc[1]), std::stoi(whc[2]), std::stoi(parts[1]), std::stoi(parts[2]));
        } else if (layer.rfind("dense_", 0) == 0) {
            auto parts = split(layer.substr(6), '_');
            net->add_dense(std::stoi(parts[0]), std::stoi(parts[1]));
        } else if (layer.rfind("activation_", 0) == 0) {
            auto parts = split(layer.substr(11), '_');
            net->add_activation(0, map_activation_type(parts[0]));
        } else if (layer.rfind("dropout_", 0) == 0) {
            auto parts = split(layer.substr(8), '_');
            net->add_dropout(0, std::stof(parts[0]));
        } else {
            throw std::runtime_error("Unknown layer type: " + layer);
        }
    }
}
