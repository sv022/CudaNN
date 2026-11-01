#include<iostream>
#include<vector>
#include<string>
#include<cmath>
#include<cstdio>
#include<cstdlib>
#include<random>
#include<memory>
#include<algorithm>
#include<fstream>
#include"../../network/batchnorm.cu"

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg) do { \
    g_tests_run++; \
    if (!(cond)) { std::cerr << "[FAIL] " << msg << "\n"; g_tests_failed++; } \
    else { std::cout << "[ OK ] " << msg << "\n"; } \
} while(0)

const float EPS = 1e-5f;
const float MOMENTUM = 0.1f;
const float FWD_TOL = 1e-4f;
const float GRAD_TOL = 5e-3f;
const float GRAD_TOL_LARGE = 2e-2f;

#define IDX(n, c, s) (((size_t)(n) * C + (c)) * S + (s))

struct BatchNormTestAccess {
    static float* gamma(BatchNorm& b)    { return b.d_gamma; }
    static float* beta(BatchNorm& b)     { return b.d_beta; }
    static float* run_mean(BatchNorm& b) { return b.d_run_mean; }
    static float* run_var(BatchNorm& b)  { return b.d_run_var; }
    static float* g_gamma(BatchNorm& b)  { return b.d_g_gamma; }
    static float* g_beta(BatchNorm& b)   { return b.d_g_beta; }
};
typedef BatchNormTestAccess Acc;

// ============================================================
// Helpers
// ============================================================
struct DevBuf {
    float* p = nullptr;
    size_t n;
    explicit DevBuf(size_t n_) : n(n_) { cudaMalloc(&p, n * sizeof(float)); }
    ~DevBuf() { cudaFree(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    void set(const std::vector<float>& h) {
        cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    }
};

static std::vector<float> from_device(const float* d, size_t n) {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

static void to_device(float* d, const std::vector<float>& h) {
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
}

static std::vector<double> to_d(const std::vector<float>& v) {
    return std::vector<double>(v.begin(), v.end());
}

static std::vector<float> random_vec(size_t n, std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> U(lo, hi);
    std::vector<float> v(n);
    for (auto& a : v) a = U(rng);
    return v;
}

static void attach_sgd(BatchNorm& bn, float lr) {
    bn.optimizer = std::make_shared<SGD>(lr);
}

static void set_affine(BatchNorm& bn, const std::vector<float>& gamma, const std::vector<float>& beta) {
    to_device(Acc::gamma(bn), gamma);
    to_device(Acc::beta(bn), beta);
}

static std::vector<float> forward_host(BatchNorm& bn, DevBuf& d_x, const std::vector<float>& x) {
    d_x.set(x);
    bn.forward(d_x.p);
    return from_device(bn.d_outputs, x.size());
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static bool all_finite(const std::vector<float>& v) {
    for (float a : v) if (!std::isfinite(a)) return false;
    return true;
}

static void cpu_stats(const std::vector<float>& x, int N, int C, int S,
                      std::vector<double>& mean, std::vector<double>& var) {
    mean.assign(C, 0.0);
    var.assign(C, 0.0);
    const double M = (double)N * S;
    for (int c = 0; c < C; ++c) {
        for (int n = 0; n < N; ++n) for (int s = 0; s < S; ++s) mean[c] += x[IDX(n, c, s)];
        mean[c] /= M;
        for (int n = 0; n < N; ++n) for (int s = 0; s < S; ++s) {
            double d = x[IDX(n, c, s)] - mean[c];
            var[c] += d * d;
        }
        var[c] /= M;
    }
}

static std::vector<float> cpu_forward(const std::vector<float>& x, int N, int C, int S,
                                      const std::vector<float>& gamma, const std::vector<float>& beta,
                                      const std::vector<double>& mean, const std::vector<double>& var) {
    std::vector<float> y(x.size());
    for (int n = 0; n < N; ++n) for (int c = 0; c < C; ++c) for (int s = 0; s < S; ++s) {
        size_t i = IDX(n, c, s);
        y[i] = (float)(gamma[c] * (x[i] - mean[c]) / std::sqrt(var[c] + EPS) + beta[c]);
    }
    return y;
}

struct GradErr { float dx, dgamma, dbeta; };

static GradErr gradient_check(int N, int C, int S, unsigned seed, int samples = 40) {
    const int size = C * S, total = N * size;
    const float h = 1e-2f;
    std::mt19937 rng(seed);
    auto x = random_vec(total, rng, -1.f, 1.f);
    auto t = random_vec(total, rng, -1.f, 1.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f);
    auto beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    attach_sgd(bn, 0.0f);
    bn.set_batch_size(N);
    bn.set_is_training(true);
    set_affine(bn, gamma, beta);

    DevBuf d_x(total), d_e(total);
    auto loss = [&](const std::vector<float>& y) {
        double L = 0.0;
        for (int i = 0; i < total; ++i) L += 0.5 * (y[i] - t[i]) * (y[i] - t[i]);
        return L;
    };

    auto y = forward_host(bn, d_x, x);
    std::vector<float> e(total);
    for (int i = 0; i < total; ++i) e[i] = t[i] - y[i];
    d_e.set(e);
    float* d_dx = bn.backward(d_x.p, d_e.p);
    auto dx = from_device(d_dx, total);
    auto gg = from_device(Acc::g_gamma(bn), C);
    auto gb = from_device(Acc::g_beta(bn), C);

    GradErr err{0.f, 0.f, 0.f};
    const int step = std::max(1, total / samples);
    for (int i = 0; i < total; i += step) {
        auto xp = x, xm = x;
        xp[i] += h; xm[i] -= h;
        double num = (loss(forward_host(bn, d_x, xp)) - loss(forward_host(bn, d_x, xm))) / (2.0 * h);
        err.dx = std::max(err.dx, (float)(std::fabs(num + dx[i]) / (1.0 + std::fabs(dx[i]))));
    }
    for (int c = 0; c < C; ++c) {
        auto gp = gamma, gm = gamma, bp = beta, bm = beta;
        gp[c] += h; gm[c] -= h; bp[c] += h; bm[c] -= h;
        set_affine(bn, gp, beta); double lgp = loss(forward_host(bn, d_x, x));
        set_affine(bn, gm, beta); double lgm = loss(forward_host(bn, d_x, x));
        set_affine(bn, gamma, bp); double lbp = loss(forward_host(bn, d_x, x));
        set_affine(bn, gamma, bm); double lbm = loss(forward_host(bn, d_x, x));
        double ng = (lgp - lgm) / (2.0 * h), nb = (lbp - lbm) / (2.0 * h);
        err.dgamma = std::max(err.dgamma, (float)(std::fabs(ng + gg[c]) / (1.0 + std::fabs(gg[c]))));
        err.dbeta  = std::max(err.dbeta,  (float)(std::fabs(nb + gb[c]) / (1.0 + std::fabs(gb[c]))));
    }
    return err;
}

// ============================================================
// Test 1: Train mode -- per-channel normalization
// ============================================================
void test_bn_train_output_has_zero_mean_unit_variance() {
    const int N = 16, C = 4, S = 9, size = C * S, total = N * size;
    std::mt19937 rng(1);
    auto x = random_vec(total, rng, 2.0f, 7.0f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    bn.set_is_training(true);
    DevBuf d_x(total);
    auto y = forward_host(bn, d_x, x);

    double worst_mean = 0.0, worst_var = 0.0;
    for (int c = 0; c < C; ++c) {
        double m = 0.0, v = 0.0;
        for (int n = 0; n < N; ++n) for (int s = 0; s < S; ++s) m += y[IDX(n, c, s)];
        m /= (double)N * S;
        for (int n = 0; n < N; ++n) for (int s = 0; s < S; ++s) {
            double d = y[IDX(n, c, s)] - m; v += d * d;
        }
        v /= (double)N * S;
        worst_mean = std::max(worst_mean, std::fabs(m));
        worst_var = std::max(worst_var, std::fabs(v - 1.0));
    }
    CHECK(worst_mean < 1e-4, "BatchNorm(train): per-channel output mean is 0 (worst=" + std::to_string(worst_mean) + ")");
    CHECK(worst_var < 1e-3, "BatchNorm(train): per-channel output variance is 1 (worst deviation=" + std::to_string(worst_var) + ")");
}

// ============================================================
// Test 2: Train mode -- gamma and beta
// ============================================================
void test_bn_train_applies_gamma_and_beta() {
    const int N = 16, C = 3, S = 6, size = C * S, total = N * size;
    std::mt19937 rng(2);
    auto x = random_vec(total, rng, -3.0f, 5.0f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    set_affine(bn, std::vector<float>(C, 2.0f), std::vector<float>(C, 3.0f));
    DevBuf d_x(total);
    auto y = forward_host(bn, d_x, x);

    double worst_mean = 0.0, worst_std = 0.0;
    for (int c = 0; c < C; ++c) {
        double m = 0.0, v = 0.0;
        for (int n = 0; n < N; ++n) for (int s = 0; s < S; ++s) m += y[IDX(n, c, s)];
        m /= (double)N * S;
        for (int n = 0; n < N; ++n) for (int s = 0; s < S; ++s) {
            double d = y[IDX(n, c, s)] - m; v += d * d;
        }
        v /= (double)N * S;
        worst_mean = std::max(worst_mean, std::fabs(m - 3.0));
        worst_std = std::max(worst_std, std::fabs(std::sqrt(v) - 2.0));
    }
    CHECK(worst_mean < 1e-4, "BatchNorm(train): output mean equals beta=3");
    CHECK(worst_std < 1e-3, "BatchNorm(train): output std equals gamma=2");
}

// ============================================================
// Test 3: Train mode -- CPU reference
// ============================================================
void test_bn_train_forward_matches_cpu_reference() {
    const int N = 4, C = 3, S = 5, size = C * S, total = N * size;
    std::mt19937 rng(3);
    auto x = random_vec(total, rng, -1.f, 1.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    set_affine(bn, gamma, beta);
    DevBuf d_x(total);
    auto y = forward_host(bn, d_x, x);

    std::vector<double> mean, var;
    cpu_stats(x, N, C, S, mean, var);
    float diff = max_abs_diff(y, cpu_forward(x, N, C, S, gamma, beta, mean, var));
    CHECK(diff < FWD_TOL, "BatchNorm(train): forward matches CPU reference (max diff=" + std::to_string(diff) + ")");
}

// ============================================================
// Test 4: Train mode -- large reduction (M > blockDim)
// ============================================================
void test_bn_train_forward_large_batch_matches_cpu_reference() {
    const int N = 32, C = 16, S = 196, size = C * S, total = N * size;
    std::mt19937 rng(4);
    auto x = random_vec(total, rng, -3.f, 3.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    set_affine(bn, gamma, beta);
    DevBuf d_x(total);
    auto y = forward_host(bn, d_x, x);

    std::vector<double> mean, var;
    cpu_stats(x, N, C, S, mean, var);
    float diff = max_abs_diff(y, cpu_forward(x, N, C, S, gamma, beta, mean, var));
    CHECK(diff < 2e-4f, "BatchNorm(train, M=6272 per channel): forward matches CPU reference (max diff=" + std::to_string(diff) + ")");
}

// ============================================================
// Test 5: Train mode -- Dense-like layout (S = 1)
// ============================================================
void test_bn_train_forward_dense_layout() {
    const int N = 8, C = 10, S = 1, size = C * S, total = N * size;
    std::mt19937 rng(5);
    auto x = random_vec(total, rng, -2.f, 2.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    set_affine(bn, gamma, beta);
    DevBuf d_x(total);
    auto y = forward_host(bn, d_x, x);

    std::vector<double> mean, var;
    cpu_stats(x, N, C, S, mean, var);
    float diff = max_abs_diff(y, cpu_forward(x, N, C, S, gamma, beta, mean, var));
    CHECK(diff < FWD_TOL, "BatchNorm(train, S=1): per-feature normalization matches CPU reference");
}

// ============================================================
// Test 6: Running statistics -- update rule
// ============================================================
void test_bn_running_stats_update_rule() {
    const int N = 8, C = 3, S = 4, size = C * S, total = N * size;
    const double M = (double)N * S;
    std::mt19937 rng(6);
    auto x = random_vec(total, rng, -2.f, 3.f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    DevBuf d_x(total);
    forward_host(bn, d_x, x);

    std::vector<double> mean, var;
    cpu_stats(x, N, C, S, mean, var);
    auto rm = from_device(Acc::run_mean(bn), C);
    auto rv = from_device(Acc::run_var(bn), C);

    double worst = 0.0;
    for (int c = 0; c < C; ++c) {
        double exp_m = MOMENTUM * mean[c];
        double exp_v = (1.0 - MOMENTUM) * 1.0 + MOMENTUM * var[c] * M / (M - 1.0);
        worst = std::max(worst, std::max(std::fabs(rm[c] - exp_m), std::fabs(rv[c] - exp_v)));
    }
    CHECK(worst < 1e-5, "BatchNorm(train): running mean/var follow momentum=0.1 and unbiased variance (worst=" + std::to_string(worst) + ")");
}

// ============================================================
// Test 7: Running statistics -- convergence
// ============================================================
void test_bn_running_stats_converge_to_batch_stats() {
    const int N = 8, C = 3, S = 4, size = C * S, total = N * size;
    const double M = (double)N * S;
    std::mt19937 rng(7);
    auto x = random_vec(total, rng, -2.f, 3.f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    DevBuf d_x(total);
    for (int k = 0; k < 100; ++k) forward_host(bn, d_x, x);

    std::vector<double> mean, var;
    cpu_stats(x, N, C, S, mean, var);
    auto rm = from_device(Acc::run_mean(bn), C);
    auto rv = from_device(Acc::run_var(bn), C);
    double worst = 0.0;
    for (int c = 0; c < C; ++c) {
        worst = std::max(worst, std::fabs(rm[c] - mean[c]));
        worst = std::max(worst, std::fabs(rv[c] - var[c] * M / (M - 1.0)));
    }
    CHECK(worst < 1e-3, "BatchNorm(train): after 100 steps running stats converge to batch stats (worst=" + std::to_string(worst) + ")");
}

// ============================================================
// Test 8: Eval mode -- initial running statistics
// ============================================================
void test_bn_eval_with_initial_running_stats() {
    const int N = 4, C = 3, S = 5, size = C * S, total = N * size;
    std::mt19937 rng(8);
    auto x = random_vec(total, rng, -2.f, 2.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    bn.set_is_training(false);
    set_affine(bn, gamma, beta);
    DevBuf d_x(total);
    auto y = forward_host(bn, d_x, x);

    auto ref = cpu_forward(x, N, C, S, gamma, beta, std::vector<double>(C, 0.0), std::vector<double>(C, 1.0));
    float diff = max_abs_diff(y, ref);
    CHECK(diff < FWD_TOL, "BatchNorm(eval): with initial stats (mean=0, var=1) output = gamma*x/sqrt(1+eps)+beta");
}

// ============================================================
// Test 9: Eval mode -- trained running statistics
// ============================================================
void test_bn_eval_uses_running_stats_after_training() {
    const int N = 8, C = 3, S = 4, size = C * S, total = N * size;
    std::mt19937 rng(9);
    auto x = random_vec(total, rng, -2.f, 3.f);
    auto x_new = random_vec(total, rng, -2.f, 3.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    set_affine(bn, gamma, beta);
    DevBuf d_x(total);
    for (int k = 0; k < 50; ++k) forward_host(bn, d_x, x);

    bn.set_is_training(false);
    auto y = forward_host(bn, d_x, x_new);
    auto rm = to_d(from_device(Acc::run_mean(bn), C));
    auto rv = to_d(from_device(Acc::run_var(bn), C));
    float diff = max_abs_diff(y, cpu_forward(x_new, N, C, S, gamma, beta, rm, rv));
    CHECK(diff < FWD_TOL, "BatchNorm(eval): output on a new batch is computed from running stats, not batch stats");
}

// ============================================================
// Test 10: Eval mode -- running statistics are frozen
// ============================================================
void test_bn_eval_does_not_modify_running_stats() {
    const int N = 4, C = 3, S = 5, size = C * S, total = N * size;
    std::mt19937 rng(10);
    auto x = random_vec(total, rng, -2.f, 2.f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    DevBuf d_x(total);
    forward_host(bn, d_x, x);
    auto rm_before = from_device(Acc::run_mean(bn), C);
    auto rv_before = from_device(Acc::run_var(bn), C);

    bn.set_is_training(false);
    for (int k = 0; k < 5; ++k) forward_host(bn, d_x, x);
    auto rm_after = from_device(Acc::run_mean(bn), C);
    auto rv_after = from_device(Acc::run_var(bn), C);

    CHECK(rm_before == rm_after && rv_before == rv_after, "BatchNorm(eval): running mean/var are not modified");
}

// ============================================================
// Test 11: Eval mode -- batch independence
// ============================================================
void test_bn_eval_batch_independence() {
    const int N = 4, C = 3, S = 5, size = C * S, total = N * size;
    std::mt19937 rng(11);
    auto x = random_vec(total, rng, -2.f, 2.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    bn.set_batch_size(N);
    set_affine(bn, gamma, beta);
    DevBuf d_x(total);
    for (int k = 0; k < 20; ++k) forward_host(bn, d_x, x);

    bn.set_is_training(false);
    auto y_batch = forward_host(bn, d_x, x);

    bn.set_batch_size(1);
    DevBuf d_one(size);
    float worst = 0.0f;
    for (int n = 0; n < N; ++n) {
        std::vector<float> one(x.begin() + n * size, x.begin() + (n + 1) * size);
        auto y1 = forward_host(bn, d_one, one);
        std::vector<float> row(y_batch.begin() + n * size, y_batch.begin() + (n + 1) * size);
        worst = std::max(worst, max_abs_diff(y1, row));
    }
    CHECK(worst < 1e-6f, "BatchNorm(eval): sample output does not depend on batch composition (batch=4 vs batch=1)");
}

// ============================================================
// Test 12: Backward -- gradient check
// ============================================================
void test_bn_gradient_check_conv_layout() {
    GradErr e = gradient_check(4, 3, 5, 12);
    CHECK(e.dx < GRAD_TOL, "BatchNorm(backward, N=4 C=3 S=5): dx matches finite differences (err=" + std::to_string(e.dx) + ")");
    CHECK(e.dgamma < GRAD_TOL, "BatchNorm(backward, N=4 C=3 S=5): dgamma matches finite differences (err=" + std::to_string(e.dgamma) + ")");
    CHECK(e.dbeta < GRAD_TOL, "BatchNorm(backward, N=4 C=3 S=5): dbeta matches finite differences (err=" + std::to_string(e.dbeta) + ")");
}

void test_bn_gradient_check_dense_layout() {
    GradErr e = gradient_check(6, 5, 1, 13);
    CHECK(e.dx < GRAD_TOL && e.dgamma < GRAD_TOL && e.dbeta < GRAD_TOL,
          "BatchNorm(backward, S=1): dx/dgamma/dbeta match finite differences (dx=" + std::to_string(e.dx) +
          ", dgamma=" + std::to_string(e.dgamma) + ", dbeta=" + std::to_string(e.dbeta) + ")");
}

// ============================================================
// Test: Backward -- large tensor vs double-precision CPU reference
// ============================================================
void test_bn_gradient_check_large() {
    const int N = 32, C = 16, S = 196;
    const int size = C * S, total = N * size;
    const double M = static_cast<double>(N) * S;

    std::mt19937 rng(14);
    auto x = random_vec(total, rng, -1.f, 1.f);
    auto e = random_vec(total, rng, -1.f, 1.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f);
    auto beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    attach_sgd(bn, 0.0f);
    bn.set_batch_size(N);
    bn.set_is_training(true);
    set_affine(bn, gamma, beta);

    DevBuf d_x(total), d_e(total);
    forward_host(bn, d_x, x);
    d_e.set(e);

    auto dx = from_device(bn.backward(d_x.p, d_e.p), total);
    auto gg = from_device(Acc::g_gamma(bn), C);
    auto gb = from_device(Acc::g_beta(bn), C);

    std::vector<double> mean, var;
    cpu_stats(x, N, C, S, mean, var);

    std::vector<double> ref_gg(C, 0.0);
    std::vector<double> ref_gb(C, 0.0);
    std::vector<double> invstd(C);

    for (int c = 0; c < C; ++c) {
        invstd[c] = 1.0 / std::sqrt(var[c] + static_cast<double>(EPS));

        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < S; ++s) {
                size_t i = IDX(n, c, s);
                double xhat = (static_cast<double>(x[i]) - mean[c])
                            * invstd[c];

                ref_gb[c] += static_cast<double>(e[i]);
                ref_gg[c] += static_cast<double>(e[i]) * xhat;
            }
        }
    }

    double err_dx = 0.0, err_gg = 0.0, err_gb = 0.0;
    bool finite_dx = all_finite(dx);
    bool finite_gg = all_finite(gg);
    bool finite_gb = all_finite(gb);

    for (int c = 0; c < C; ++c) {
        err_gg = std::max(
            err_gg,
            std::fabs(static_cast<double>(gg[c]) - ref_gg[c])
                / (1.0 + std::fabs(ref_gg[c]))
        );
        err_gb = std::max(
            err_gb,
            std::fabs(static_cast<double>(gb[c]) - ref_gb[c])
                / (1.0 + std::fabs(ref_gb[c]))
        );

        for (int n = 0; n < N; ++n) {
            for (int s = 0; s < S; ++s) {
                size_t i = IDX(n, c, s);
                double xhat = (static_cast<double>(x[i]) - mean[c])
                            * invstd[c];

                double ref_dx = static_cast<double>(gamma[c]) * invstd[c]
                              * (static_cast<double>(e[i])
                                 - ref_gb[c] / M
                                 - xhat * ref_gg[c] / M);

                err_dx = std::max(
                    err_dx,
                    std::fabs(static_cast<double>(dx[i]) - ref_dx)
                        / (1.0 + std::fabs(ref_dx))
                );
            }
        }
    }

    CHECK(finite_dx && err_dx < 1e-4,
          "BatchNorm(backward, large): dx matches CPU double reference "
          "(err=" + std::to_string(err_dx) + ")");

    CHECK(finite_gg && err_gg < 1e-4,
          "BatchNorm(backward, large): dgamma matches CPU double reference "
          "(err=" + std::to_string(err_gg) + ")");

    CHECK(finite_gb && err_gb < 1e-4,
          "BatchNorm(backward, large): dbeta matches CPU double reference "
          "(err=" + std::to_string(err_gb) + ")");
}

// ============================================================
// Test 13: Backward -- identities that dx must satisfy
// ============================================================
void test_bn_backward_dx_is_orthogonal_to_constant_and_xhat() {
    const int N = 8, C = 4, S = 6, size = C * S, total = N * size;
    std::mt19937 rng(15);
    auto x = random_vec(total, rng, -2.f, 4.f);
    auto e = random_vec(total, rng, -1.f, 1.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    attach_sgd(bn, 0.0f);
    bn.set_batch_size(N);
    set_affine(bn, gamma, beta);
    DevBuf d_x(total), d_e(total);
    forward_host(bn, d_x, x);
    d_e.set(e);
    auto dx = from_device(bn.backward(d_x.p, d_e.p), total);

    std::vector<double> mean, var;
    cpu_stats(x, N, C, S, mean, var);
    bool ok_sum = true, ok_xhat = true;
    for (int c = 0; c < C; ++c) {
        double s1 = 0.0, s2 = 0.0, sa = 0.0;
        for (int n = 0; n < N; ++n) for (int s = 0; s < S; ++s) {
            size_t i = IDX(n, c, s);
            double xhat = (x[i] - mean[c]) / std::sqrt(var[c] + EPS);
            s1 += dx[i]; s2 += dx[i] * xhat; sa += std::fabs(dx[i]);
        }
        if (std::fabs(s1) > 1e-3 * sa + 1e-4) ok_sum = false;
        if (std::fabs(s2) > 1e-3 * sa + 1e-4) ok_xhat = false;
    }
    CHECK(ok_sum, "BatchNorm(backward): sum of dx over batch and pixels is 0 for every channel");
    CHECK(ok_xhat, "BatchNorm(backward): dx is orthogonal to normalized input for every channel");
}

// ============================================================
// Test 14: Backward -- first layer (compute_input_grad = false)
// ============================================================
void test_bn_backward_without_input_grad() {
    const int N = 4, C = 3, S = 5, size = C * S, total = N * size;
    std::mt19937 rng(16);
    auto x = random_vec(total, rng, -1.f, 1.f);
    auto e = random_vec(total, rng, -1.f, 1.f);

    BatchNorm bn(size, C);
    attach_sgd(bn, 0.0f);
    bn.set_batch_size(N);
    bn.compute_input_grad = false;
    DevBuf d_x(total), d_e(total);
    forward_host(bn, d_x, x);
    d_e.set(e);
    float* res = bn.backward(d_x.p, d_e.p);

    auto gb = from_device(Acc::g_beta(bn), C);
    double worst = 0.0;
    for (int c = 0; c < C; ++c) {
        double sum = 0.0;
        for (int n = 0; n < N; ++n) for (int s = 0; s < S; ++s) sum += e[IDX(n, c, s)];
        worst = std::max(worst, std::fabs(gb[c] - sum));
    }
    CHECK(res == nullptr, "BatchNorm(backward, compute_input_grad=false): returns nullptr");
    CHECK(worst < 1e-5, "BatchNorm(backward, compute_input_grad=false): dbeta is still computed");
}

// ============================================================
// Test 15: Optimizer -- exact SGD step
// ============================================================
void test_bn_sgd_step_is_exact() {
    const int N = 4, C = 3, S = 5, size = C * S, total = N * size;
    const float lr = 0.1f;
    std::mt19937 rng(17);
    auto x = random_vec(total, rng, -1.f, 1.f);
    auto e = random_vec(total, rng, -1.f, 1.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    attach_sgd(bn, lr);
    bn.set_batch_size(N);
    set_affine(bn, gamma, beta);
    DevBuf d_x(total), d_e(total);
    forward_host(bn, d_x, x);
    d_e.set(e);
    bn.backward(d_x.p, d_e.p);

    auto gg = from_device(Acc::g_gamma(bn), C), gb = from_device(Acc::g_beta(bn), C);
    auto g_new = from_device(Acc::gamma(bn), C), b_new = from_device(Acc::beta(bn), C);
    float worst = 0.0f;
    for (int c = 0; c < C; ++c) {
        worst = std::max(worst, std::fabs(g_new[c] - (gamma[c] + lr * gg[c] / N)));
        worst = std::max(worst, std::fabs(b_new[c] - (beta[c] + lr * gb[c] / N)));
    }
    CHECK(worst < 1e-5f, "BatchNorm(SGD): gamma/beta change by lr*grad/batch_size (worst=" + std::to_string(worst) + ")");
}

// ============================================================
// Test 16: Optimizer -- learning an affine target
// ============================================================
void test_bn_learns_target_affine_parameters() {
    const int N = 8, C = 2, S = 4, size = C * S, total = N * size;
    std::mt19937 rng(18);
    auto x = random_vec(total, rng, -2.f, 2.f);
    std::vector<float> g_target = {2.0f, 0.5f}, b_target = {1.0f, -1.0f};

    std::vector<double> mean, var;
    cpu_stats(x, N, C, S, mean, var);
    auto t = cpu_forward(x, N, C, S, g_target, b_target, mean, var);

    BatchNorm bn(size, C);
    attach_sgd(bn, 0.1f);
    bn.set_batch_size(N);
    DevBuf d_x(total), d_e(total);
    for (int it = 0; it < 100; ++it) {
        auto y = forward_host(bn, d_x, x);
        std::vector<float> e(total);
        for (int i = 0; i < total; ++i) e[i] = t[i] - y[i];
        d_e.set(e);
        bn.backward(d_x.p, d_e.p);
    }
    auto g = from_device(Acc::gamma(bn), C), b = from_device(Acc::beta(bn), C);
    float worst = 0.0f;
    for (int c = 0; c < C; ++c)
        worst = std::max(worst, std::max(std::fabs(g[c] - g_target[c]), std::fabs(b[c] - b_target[c])));
    CHECK(worst < 1e-3f, "BatchNorm(SGD): 100 steps recover target gamma/beta (worst error=" + std::to_string(worst) + ")");
}

// ============================================================
// Test 17: Optimizer -- stateful optimizers
// ============================================================
void test_bn_momentum_and_adam_keep_parameters_finite() {
    const int N = 8, C = 3, S = 4, size = C * S, total = N * size;
    for (int kind = 0; kind < 2; ++kind) {
        std::mt19937 rng(19 + kind);
        BatchNorm bn(size, C);
        if (kind == 0) bn.optimizer = std::make_shared<Momentum>(0.01f, 0.9f);
        else bn.optimizer = std::make_shared<Adam>(0.01f, 0.9f, 0.999f, 1e-8f);
        bn.set_batch_size(N);
        DevBuf d_x(total), d_e(total);
        for (int it = 0; it < 20; ++it) {
            forward_host(bn, d_x, random_vec(total, rng, -2.f, 2.f));
            d_e.set(random_vec(total, rng, -1.f, 1.f));
            bn.backward(d_x.p, d_e.p);
        }
        auto g = from_device(Acc::gamma(bn), C), b = from_device(Acc::beta(bn), C);
        bool changed = false;
        for (int c = 0; c < C; ++c) if (g[c] != 1.0f || b[c] != 0.0f) changed = true;
        std::string name = kind == 0 ? "Momentum" : "Adam";
        CHECK(all_finite(g) && all_finite(b) && changed, "BatchNorm(" + name + "): parameters stay finite and are updated");
    }
}

// ============================================================
// Test 18: Edge case -- one sample, one pixel (M = 1)
// ============================================================
void test_bn_single_sample_training_is_safe() {
    const int C = 6, S = 1, N = 1, size = C * S;
    std::mt19937 rng(20);
    auto x = random_vec(size, rng, -2.f, 2.f);
    auto e = random_vec(size, rng, -1.f, 1.f);
    auto beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    attach_sgd(bn, 0.0f);
    set_affine(bn, std::vector<float>(C, 1.0f), beta);
    DevBuf d_x(size), d_e(size);
    auto y = forward_host(bn, d_x, x);

    auto rm = from_device(Acc::run_mean(bn), C), rv = from_device(Acc::run_var(bn), C);
    bool stats_untouched = true;
    for (int c = 0; c < C; ++c) if (rm[c] != 0.0f || rv[c] != 1.0f) stats_untouched = false;

    d_e.set(e);
    auto dx = from_device(bn.backward(d_x.p, d_e.p), size);
    auto gg = from_device(Acc::g_gamma(bn), C), gb = from_device(Acc::g_beta(bn), C);

    float dev_y = max_abs_diff(y, beta), dev_dx = 0.0f, dev_gb = 0.0f, dev_gg = 0.0f;
    for (int c = 0; c < C; ++c) {
        dev_dx = std::max(dev_dx, std::fabs(dx[c]));
        dev_gb = std::max(dev_gb, std::fabs(gb[c] - e[c]));
        dev_gg = std::max(dev_gg, std::fabs(gg[c]));
    }
    CHECK(all_finite(y) && dev_y < 1e-5f, "BatchNorm(train, M=1): output equals beta and is finite");
    CHECK(stats_untouched, "BatchNorm(train, M=1): running stats are not updated (no division by M-1=0)");
    CHECK(all_finite(dx) && dev_dx < 1e-6f && dev_gb < 1e-6f && dev_gg < 1e-6f,
          "BatchNorm(train, M=1): dx=0, dbeta=error, dgamma=0");
}

// ============================================================
// Test 19: Batch size changes
// ============================================================
void test_bn_set_batch_size_reallocates_buffers() {
    const int C = 4, S = 5, size = C * S;
    std::mt19937 rng(21);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm bn(size, C);
    set_affine(bn, gamma, beta);

    const int batches[] = {1, 8, 3};
    for (int N : batches) {
        bn.set_batch_size(N);
        auto x = random_vec((size_t)N * size, rng, -2.f, 2.f);
        DevBuf d_x((size_t)N * size);
        auto y = forward_host(bn, d_x, x);
        std::vector<double> mean, var;
        cpu_stats(x, N, C, S, mean, var);
        float diff = max_abs_diff(y, cpu_forward(x, N, C, S, gamma, beta, mean, var));
        CHECK(diff < FWD_TOL, "BatchNorm: set_batch_size(" + std::to_string(N) + ") reallocates buffers, forward matches CPU reference");
    }
}

// ============================================================
// Test 20: Serialization
// ============================================================
void test_bn_save_load_roundtrip() {
    const int N = 4, C = 3, S = 5, size = C * S, total = N * size;
    const std::string path = "bn_test_weights.bin";
    const int HEADER = 8;
    std::mt19937 rng(22);
    auto x = random_vec(total, rng, -2.f, 2.f);
    auto gamma = random_vec(C, rng, 0.5f, 1.5f), beta = random_vec(C, rng, -0.5f, 0.5f);

    BatchNorm src(size, C);
    src.set_batch_size(N);
    set_affine(src, gamma, beta);
    DevBuf d_x(total);
    for (int k = 0; k < 10; ++k) forward_host(src, d_x, random_vec(total, rng, -2.f, 3.f));

    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        const char junk[HEADER] = {1, 2, 3, 4, 5, 6, 7, 8};
        f.write(junk, HEADER);
    }
    src.save_weights(path);

    BatchNorm dst(size, C);
    dst.set_batch_size(N);
    int count = dst.load_weights(path, HEADER);

    bool same = from_device(Acc::gamma(src), C) == from_device(Acc::gamma(dst), C)
             && from_device(Acc::beta(src), C) == from_device(Acc::beta(dst), C)
             && from_device(Acc::run_mean(src), C) == from_device(Acc::run_mean(dst), C)
             && from_device(Acc::run_var(src), C) == from_device(Acc::run_var(dst), C);

    src.set_is_training(false);
    dst.set_is_training(false);
    auto y_src = forward_host(src, d_x, x);
    auto y_dst = forward_host(dst, d_x, x);
    std::remove(path.c_str());

    CHECK(count == 4 * C, "BatchNorm: load_weights returns 4*channels floats");
    CHECK(same, "BatchNorm: gamma, beta, running mean and variance survive save/load (with nonzero file offset)");
    CHECK(y_src == y_dst, "BatchNorm(eval): outputs before and after save/load are identical");
}

int main() {
    srand(123);
    std::cout << "=== BatchNorm layer tests ===\n\n";

    test_bn_train_output_has_zero_mean_unit_variance();
    test_bn_train_applies_gamma_and_beta();
    test_bn_train_forward_matches_cpu_reference();
    test_bn_train_forward_large_batch_matches_cpu_reference();
    test_bn_train_forward_dense_layout();
    test_bn_running_stats_update_rule();
    test_bn_running_stats_converge_to_batch_stats();
    test_bn_eval_with_initial_running_stats();
    test_bn_eval_uses_running_stats_after_training();
    test_bn_eval_does_not_modify_running_stats();
    test_bn_eval_batch_independence();
    test_bn_gradient_check_conv_layout();
    test_bn_gradient_check_dense_layout();
    test_bn_gradient_check_large();
    test_bn_backward_dx_is_orthogonal_to_constant_and_xhat();
    test_bn_backward_without_input_grad();
    test_bn_sgd_step_is_exact();
    test_bn_learns_target_affine_parameters();
    test_bn_momentum_and_adam_keep_parameters_finite();
    test_bn_single_sample_training_is_safe();
    test_bn_set_batch_size_reallocates_buffers();
    test_bn_save_load_roundtrip();

    cudaError_t err = cudaGetLastError();
    CHECK(err == cudaSuccess, std::string("No pending CUDA errors (") + cudaGetErrorString(err) + ")");

    std::cout << "\n=== " << g_tests_run << " checks run, " << g_tests_failed << " failed ===\n";
    return g_tests_failed == 0 ? 0 : 1;
}