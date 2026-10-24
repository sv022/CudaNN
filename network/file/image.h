#pragma once
#include<vector>
#include<algorithm>
#include<cmath>

struct ImageShape {
    int c, h, w;
    ImageShape(int flat) : c(flat), h(0), w(0) {}
    ImageShape(int c_, int h_, int w_) : c(c_), h(h_), w(w_) {}
    int numel() const { return c * (h > 0 ? h : 1) * (w > 0 ? w : 1); }
    bool is_image() const { return h > 0 && w > 0; }
};

struct AxisWeights {
    std::vector<int> start, count;
    std::vector<size_t> offset;
    std::vector<float> w;
};

static AxisWeights make_axis_weights(int src, int dst) {
    AxisWeights a;
    a.start.resize(dst); a.count.resize(dst); a.offset.resize(dst);
    const double scale = (double)src / dst;

    for (int i = 0; i < dst; ++i) {
        a.offset[i] = a.w.size();
        if (scale > 1.0) {
            double lo = i * scale, hi = (i + 1) * scale;
            int s0 = std::min((int)std::floor(lo), src - 1);
            int s1 = std::min((int)std::ceil(hi), src);
            double total = 0.0;
            size_t first = a.w.size();
            for (int s = s0; s < s1; ++s) {
                double ov = std::min(hi, (double)s + 1.0) - std::max(lo, (double)s);
                if (ov < 0.0) ov = 0.0;
                a.w.push_back((float)ov);
                total += ov;
            }
            for (size_t k = first; k < a.w.size(); ++k) a.w[k] = (float)(a.w[k] / total);
            a.start[i] = s0;
            a.count[i] = s1 - s0;
        } else {
            double f = std::max(0.0, (i + 0.5) * scale - 0.5);
            int s0 = std::min((int)f, src - 1);
            int s1 = std::min(s0 + 1, src - 1);
            float t = (float)(f - s0);
            a.start[i] = s0;
            if (s1 == s0) { a.w.push_back(1.0f); a.count[i] = 1; }
            else { a.w.push_back(1.0f - t); a.w.push_back(t); a.count[i] = 2; }
        }
    }
    return a;
}

static void resize_to_chw(const unsigned char* src, int sw, int sh, int C,
                          float* dst, int W, int H)
{
    const AxisWeights ax = make_axis_weights(sw, W);
    const AxisWeights ay = make_axis_weights(sh, H);

    std::vector<float> tmp((size_t)sh * W * C);
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c) {
                float acc = 0.0f;
                const float* w = &ax.w[ax.offset[x]];
                for (int k = 0; k < ax.count[x]; ++k)
                    acc += w[k] * src[((size_t)y * sw + ax.start[x] + k) * C + c];
                tmp[((size_t)y * W + x) * C + c] = acc;
            }

    for (int c = 0; c < C; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                float acc = 0.0f;
                const float* w = &ay.w[ay.offset[y]];
                for (int k = 0; k < ay.count[y]; ++k)
                    acc += w[k] * tmp[((size_t)(ay.start[y] + k) * W + x) * C + c];
                dst[((size_t)c * H + y) * W + x] = acc / 255.0f;
            }
}