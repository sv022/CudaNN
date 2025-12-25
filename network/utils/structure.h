#include<vector>


enum class LayerType
{
    Conv,
    Pool,
    Dense
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
    ConvStructure(int w, int h, int c, int k, int num_k, int s, int p) : input_width(w), input_height(h), channels(c), kernel_size(k), num_kernels(num_k), stride(s), padding(p) {
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


struct NetworkStructure
{
    std::vector<LayerStructure*> layers;
    float learning_rate;

    NetworkStructure(float lr = 0.001f): learning_rate(lr) {} 

    void add_dense(unsigned int input_nodes, unsigned int output_nodes) {
        layers.push_back(new DenseStructure(input_nodes, output_nodes));
    }

    void add_pool(unsigned int input_width, unsigned int input_height, unsigned int channels, unsigned int pool, unsigned int stride) {
        layers.push_back(new PoolStructure(input_width, input_height, channels, pool, stride));
    }

    void add_conv(unsigned int input_width, unsigned int input_height, unsigned int channels, unsigned int kernel_size, unsigned int num_kernels, unsigned int stride, unsigned int padding){
        layers.push_back(new ConvStructure(input_width, input_height, channels, kernel_size, num_kernels, stride, padding));
    }

    ~NetworkStructure(){
        for (auto* l : layers) delete l;
    }
};

