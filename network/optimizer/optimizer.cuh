#pragma once
#include <cmath>
#include"opt_kernels.cu"

enum class OptimizerType { SGD, Momentum, Adam };

class Optimizer {
    protected:
    float lr;

    public:
    virtual int state_buffers_needed() const = 0;
    virtual void update(float* d_weights, const float* d_gradients, float** d_state, int total, int batch_size) = 0;

    virtual void set_lr(float new_lr) { lr = new_lr; }
    virtual float get_lr() const { return lr; }
    virtual void step() {};
};

class SGD : public Optimizer 
{
    public:
    SGD(float lr_) {lr = lr_;}

    int state_buffers_needed() const override { return 0; }

    void update(float* d_weights, const float* d_gradients, float** /*d_state*/, int total, int batch_size) override {
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        sgd_update_kernel<<<blocks, threads>>>(d_weights, d_gradients, lr, total, batch_size);
    }
};

class Momentum : public Optimizer 
{
    private:
    float beta;
    public:
    Momentum(float lr_, float beta_ = 0.9f) {lr = lr_; beta = beta_;}

    int state_buffers_needed() const override { return 1; }

    void update(float* d_weights, const float* d_gradients, float** d_state, int total, int batch_size) override {
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        momentum_update_kernel<<<blocks, threads>>>(d_weights, d_gradients, d_state[0], lr, beta, total, batch_size);
    }
};

class Adam : public Optimizer 
{
    private:
    float beta1, beta2, eps;
    int t = 0;

    public:
    Adam(float lr_, float beta1_ = 0.9f, float beta2_ = 0.999f, float eps_ = 1e-8f) {lr = lr_; beta1 = beta1_; beta2 = beta2_; eps = eps_;}

    int state_buffers_needed() const override { return 2; }

    void update(float* d_weights, const float* d_gradients, float** d_state, int total, int batch_size) override {
        t++;
        float bc1 = 1.0f - std::pow(beta1, (float)t);
        float bc2 = 1.0f - std::pow(beta2, (float)t);

        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        adam_update_kernel<<<blocks, threads>>>(
            d_weights, d_gradients, d_state[0], d_state[1],
            lr, beta1, beta2, eps, bc1, bc2, total, batch_size);
    }

    void step() override { }
    float get_lr() const { return lr; }
};

OptimizerType parse_opt_type(std::string opt_str) {
    if (opt_str == "sgd") return OptimizerType::SGD;
    if (opt_str == "momentum") return OptimizerType::Momentum;
    if (opt_str == "adam") return OptimizerType::Adam;
    throw std::runtime_error("Unknown optimizer type: " + opt_str);
}

std::string opt_type_to_str(OptimizerType opt) {
    if (opt == OptimizerType::SGD) return "\"sgd\"";
    if (opt == OptimizerType::Momentum) return "\"momentum\"";
    if (opt == OptimizerType::Adam) return "\"adam\"";
    return "null";
}