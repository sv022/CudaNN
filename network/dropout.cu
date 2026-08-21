#pragma once
#include"layer.cuh"

class Dropout : public Layer 
{
    private:
    float drop_prob;
    float* mask;
    
    public:
    Dropout(int size_, float p);

    void set_batch_size(int bs) override;

    void forward(float* inputs) override;

    float* backward(float* /*inputs*/, float* next_errors, bool /*raw_gradient*/) override;

    ~Dropout() { free(mask); }
};

Dropout::Dropout(int size_, float p) {
    size = size_;
    output_size = size_;
    batch_size = 1;
    drop_prob = p;
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * size_);
    mask = (float*)malloc(sizeof(float) * (size_t)batch_size * size_);
}

void Dropout::set_batch_size(int bs) {
    batch_size = bs;
    free(outputs);
    free(mask);
    outputs = (float*)malloc(sizeof(float) * (size_t)batch_size * size);
    mask = (float*)malloc(sizeof(float) * (size_t)batch_size * size);
}

void Dropout::forward(float* inputs)  {
    size_t total = (size_t)batch_size * size;

    if (!is_training) {
        memcpy(outputs, inputs, sizeof(float) * total);
        return;
    }

    float scale = 1.0f / (1.0f - drop_prob);
    for (size_t i = 0; i < total; ++i) {
        float r = (float)rand() / (float)RAND_MAX;
        mask[i] = (r > drop_prob) ? 1.0f : 0.0f;
        outputs[i] = inputs[i] * mask[i] * scale;
    }
}

float* Dropout::backward(float* /*inputs*/, float* next_errors, bool /*raw_gradient*/) {
    size_t total = (size_t)batch_size * size;
    float* local_grad = (float*)malloc(sizeof(float) * total);

    if (!is_training) {
        memcpy(local_grad, next_errors, sizeof(float) * total);
        return local_grad;
    }

    float scale = 1.0f / (1.0f - drop_prob);
    for (size_t i = 0; i < total; ++i) {
        local_grad[i] = next_errors[i] * mask[i] * scale;
    }
    return local_grad;
}