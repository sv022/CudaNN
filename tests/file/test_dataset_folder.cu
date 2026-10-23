#include "../../network/file/file.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Tests on a real image directory (labels.txt + class folders).
//
// Usage:  test_dataset_real <dataset_dir> [C H W [max_samples [batch_size]]]
//   dataset_dir  directory that contains labels.txt and the class folders
//   C H W        network input shape (default 3 64 64) - set it to your network input
//   max_samples  use only the first N valid rows (default 0 = all) - to keep the run short
//   batch_size   batch size used by the tests (default 32)
//
// The tests do not assume a particular number of images: the expected values are derived from
// labels.txt by an independent reference parser, and image content is compared with a direct
// decode through stb_image. Only the public interface of DatasetFile is used.

struct Config {
    std::string dir;
    int C = 3, H = 64, W = 64;
    int max_n = 0;
    int bs = 32;
};

struct Entry {
    std::string path;
    std::string rel;
    int label = 0;
};

static Config cfg;
static std::vector<Entry> ref_entries;
static int n_classes = 0;
static bool data_ok = false;

static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << (msg) << " (line " << __LINE__ << ")\n"; \
        tests_failed++; \
    } else { \
        std::cout << "[ OK ] " << (msg) << "\n"; \
    } \
} while(0)

#define CHECK_EQ(a, b, msg) do { \
    tests_run++; \
    auto _a = (a); \
    auto _b = (b); \
    if (!(_a == _b)) { \
        std::cerr << "[FAIL] " << (msg) << ": got " << _a << ", expected " << _b \
                  << " (line " << __LINE__ << ")\n"; \
        tests_failed++; \
    } else { \
        std::cout << "[ OK ] " << (msg) << "\n"; \
    } \
} while(0)

#define CHECK_THROWS(expr, ExcT, substr, msg) do { \
    bool _ok = false; \
    std::string _what = "no exception"; \
    try { expr; } \
    catch (const ExcT& e) { _what = e.what(); _ok = (_what.find(substr) != std::string::npos); } \
    catch (...) { _what = "wrong exception type"; } \
    CHECK(_ok, std::string(msg) + " [" + _what + "]"); \
} while(0)

using Clock = std::chrono::steady_clock;

static double secs_since(Clock::time_point t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static bool near_f(float a, float b, float eps = 1e-6f) { return std::fabs(a - b) < eps; }

// ---------------- reference parser (independent of DatasetFile) ----------------

static std::string trim_right(std::string s)
{
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    return s;
}

static bool parse_entry(const std::string& raw, std::string& rel, int& label)
{
    std::string line = trim_right(raw);
    if (line.empty()) return false;
    size_t sep = line.find_last_of(" \t");
    if (sep == std::string::npos) return false;
    const char* num = line.c_str() + sep + 1;
    char* end;
    long v = std::strtol(num, &end, 10);
    if (end == num || *end != '\0' || v < 0) return false;
    rel = trim_right(line.substr(0, sep));
    std::replace(rel.begin(), rel.end(), '\\', '/');
    label = (int)v;
    return true;
}

struct RefImage {
    std::vector<float> chw;
    std::vector<float> direct;
    std::vector<double> src_mean;
    int sw = 0, sh = 0;
    bool same_size = false;
};

static bool decode_ref(const std::string& path, RefImage& r)
{
    int w = 0, h = 0, n = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, cfg.C);
    if (!data) return false;

    r.sw = w; r.sh = h;
    r.chw.assign((size_t)cfg.C * cfg.H * cfg.W, 0.f);
    resize_to_chw(data, w, h, cfg.C, r.chw.data(), cfg.W, cfg.H);

    r.src_mean.assign(cfg.C, 0.0);
    for (size_t p = 0; p < (size_t)w * h; p++)
        for (int c = 0; c < cfg.C; c++) r.src_mean[c] += data[p * cfg.C + c];
    for (int c = 0; c < cfg.C; c++) r.src_mean[c] /= 255.0 * w * h;

    r.same_size = (w == cfg.W && h == cfg.H);
    r.direct.clear();
    if (r.same_size) {
        r.direct.resize((size_t)cfg.C * cfg.H * cfg.W);
        for (int c = 0; c < cfg.C; c++)
            for (int y = 0; y < cfg.H; y++)
                for (int x = 0; x < cfg.W; x++)
                    r.direct[((size_t)c * cfg.H + y) * cfg.W + x] = data[((size_t)y * cfg.W + x) * cfg.C + c] / 255.0f;
    }
    stbi_image_free(data);
    return true;
}

// ---------------- helpers ----------------

static int input_nodes() { return cfg.C * cfg.H * cfg.W; }

static std::unique_ptr<DatasetFile> make_real(int bs, float vf = 0.f, unsigned seed = 12345u)
{
    return std::unique_ptr<DatasetFile>(new DatasetFile(cfg.dir, (int)ref_entries.size(),
        ImageShape(cfg.C, cfg.H, cfg.W), n_classes, bs, vf, seed));
}

static uint64_t fnv(const float* p, size_t n, uint64_t h = 1469598103934665603ULL)
{
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n * sizeof(float); i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static uint64_t row_hash(const DatasetFile& d, int j)
{
    uint64_t h = fnv(d.image_batch + (size_t)j * input_nodes(), input_nodes());
    return fnv(d.target_batch + (size_t)j * n_classes, n_classes, h);
}

// hashes of (image, target) of every sample of the current mode, in batch order
static std::vector<uint64_t> epoch_hashes(DatasetFile& d)
{
    std::vector<uint64_t> hs;
    for (int b = 0; b < d.get_num_batches(); b++) {
        d.get_batch(b);
        for (int j = 0; j < d.current_batch_size; j++) hs.push_back(row_hash(d, j));
    }
    return hs;
}

static std::vector<uint64_t> sorted_hashes(std::vector<uint64_t> v)
{
    std::sort(v.begin(), v.end());
    return v;
}

static bool skip_if_no_data(const char* name)
{
    if (data_ok) return false;
    std::cout << "[SKIP] " << name << ": the data set was not read successfully\n";
    return true;
}

// ---------------- test groups ----------------

static void test_labels_integrity()
{
    std::cout << "\n--- real data: labels.txt integrity ---\n";

    const std::string list_path = cfg.dir + "/labels.txt";
    std::ifstream f(list_path, std::ios::binary);
    CHECK(f.is_open(), "labels.txt can be opened in " + cfg.dir);
    if (!f.is_open()) return;

    std::map<std::string, std::set<int>> folder_labels;
    std::map<int, int> label_count;
    std::vector<std::string> unreadable;
    int lines = 0, unparsable = 0, parsed = 0, max_label = -1;
    std::string raw;
    bool first = true;
    std::vector<Entry> all;

    while (std::getline(f, raw)) {
        if (first && raw.compare(0, 3, "\xEF\xBB\xBF") == 0) raw.erase(0, 3);
        first = false;
        ++lines;
        std::string rel;
        int label = 0;
        if (!parse_entry(raw, rel, label)) { if (!trim_right(raw).empty()) ++unparsable; continue; }
        ++parsed;
        max_label = std::max(max_label, label);
        Entry e;
        e.rel = rel;
        e.label = label;
        e.path = cfg.dir + "/" + rel;
        all.push_back(e);
        size_t slash = rel.find('/');
        folder_labels[slash == std::string::npos ? "" : rel.substr(0, slash)].insert(label);
        ++label_count[label];
    }

    std::cout << "[INFO] labels.txt: " << lines << " lines, " << parsed << " parsable rows, "
              << unparsable << " unparsable non-empty lines\n";
    for (auto& kv : folder_labels) {
        std::cout << "[INFO] folder '" << kv.first << "' -> label(s):";
        for (int l : kv.second) std::cout << " " << l;
        int cnt = 0;
        for (auto& e : all) if (e.rel.compare(0, kv.first.size(), kv.first) == 0) ++cnt;
        std::cout << ", " << cnt << " rows\n";
    }

    CHECK(parsed > 0, "labels.txt has at least one parsable row");
    CHECK_EQ(unparsable, 0, "labels.txt has no malformed non-empty lines");

    bool one_label_per_folder = true, distinct = true;
    std::set<int> seen;
    for (auto& kv : folder_labels) {
        if (kv.second.size() != 1) one_label_per_folder = false;
        else if (!seen.insert(*kv.second.begin()).second) distinct = false;
    }
    CHECK(one_label_per_folder, "every class folder maps to exactly one label");
    CHECK(distinct, "different class folders have different labels");

    int bad_ext = 0;
    for (auto& e : all) {
        int w = 0, h = 0, n = 0;
        if (!stbi_info(e.path.c_str(), &w, &h, &n)) {
            if (unreadable.size() < 5) unreadable.push_back(e.rel);
            ++bad_ext;
        }
    }
    for (auto& u : unreadable) std::cout << "[INFO] unreadable: " << u << "\n";
    CHECK_EQ(bad_ext, 0, "every file listed in labels.txt exists and has a readable image header");

    std::set<std::string> uniq;
    int dup = 0;
    for (auto& e : all) if (!uniq.insert(e.rel).second) ++dup;
    std::cout << "[INFO] duplicate paths in labels.txt: " << dup << "\n";

    n_classes = max_label + 1;
    for (auto& e : all) {
        int w = 0, h = 0, n = 0;
        if (stbi_info(e.path.c_str(), &w, &h, &n)) ref_entries.push_back(e);
    }
    
    if (cfg.max_n > 0 && (int)ref_entries.size() > cfg.max_n) ref_entries.resize(cfg.max_n);

    std::cout << "[INFO] classes (max label + 1): " << n_classes << ", samples used: " << ref_entries.size() << "\n";
    data_ok = !ref_entries.empty() && n_classes > 0 && n_classes <= 1000;
    CHECK(data_ok, "the reference list of valid samples is not empty");
}

static void test_real_construct_and_stats()
{
    std::cout << "\n--- real data: constructor / getters ---\n";
    if (skip_if_no_data("constructor")) return;

    const int total = (int)ref_entries.size();
    const int bs = cfg.bs;

    auto t0 = Clock::now();
    auto d = make_real(bs);
    std::cout << "[INFO] construction (scan + first batch): " << secs_since(t0) << " s\n";

    CHECK_EQ(d->get_train_size(), total, "train_size == number of valid rows");
    CHECK_EQ(d->get_val_size(), 0, "no split: val_size == 0");
    CHECK_EQ(d->get_train_num_batches(), (total + bs - 1) / bs, "train_num_batches == ceil(size / batch_size)");
    CHECK_EQ(d->get_batch_size(), bs, "get_batch_size()");
    CHECK_EQ(d->get_input_nodes(), input_nodes(), "get_input_nodes() == C*H*W");
    CHECK_EQ(d->get_output_nodes(), n_classes, "get_output_nodes() == number of classes");
    CHECK_EQ(d->current_batch_size, std::min(bs, total), "constructor loads batch 0");

    CHECK_THROWS(make_real(0), std::invalid_argument, "batch_size must be positive",
        "batch_size 0 is rejected for the real directory");
    CHECK_THROWS(std::unique_ptr<DatasetFile>(new DatasetFile(cfg.dir, total + 1, ImageShape(cfg.C, cfg.H, cfg.W),
        n_classes, bs)), std::runtime_error, "expected " + std::to_string(total + 1),
        "size larger than the number of valid rows throws runtime_error");
}

static void test_real_content()
{
    std::cout << "\n--- real data: content of batches ---\n";
    if (skip_if_no_data("content")) return;

    const int total = (int)ref_entries.size();
    const int bs = cfg.bs;
    const int in = input_nodes();
    const int plane = cfg.H * cfg.W;
    const int stride = std::max(1, total / 200);

    auto d = make_real(bs);
    int label_bad = 0, image_bad = 0, mean_bad = 0, direct_bad = 0, decode_fail = 0, verified = 0, same_size_cnt = 0;

    for (int b = 0; b < d->get_num_batches(); b++) {
        d->get_batch(b);
        for (int j = 0; j < d->current_batch_size; j++) {
            const int i = b * bs + j;
            const Entry& e = ref_entries[i];

            for (int k = 0; k < n_classes; k++) {
                float expected = (k == e.label) ? 1.f : 0.f;
                if (d->target_batch[(size_t)j * n_classes + k] != expected) { ++label_bad; break; }
            }

            if (i % stride != 0) continue;

            RefImage r;
            if (!decode_ref(e.path, r)) { ++decode_fail; continue; }
            ++verified;

            const float* got = d->image_batch + (size_t)j * in;
            bool same = true;
            for (int k = 0; k < in && same; k++) same = near_f(got[k], r.chw[k]);
            if (!same) ++image_bad;

            bool mean_ok = true;
            for (int c = 0; c < cfg.C; c++) {
                double s = 0.0;
                for (int k = 0; k < plane; k++) s += got[(size_t)c * plane + k];
                if (std::fabs(s / plane - r.src_mean[c]) > 0.02) mean_ok = false;
            }
            if (!mean_ok) ++mean_bad;

            if (r.same_size) {
                ++same_size_cnt;
                bool exact = true;
                for (int k = 0; k < in && exact; k++) exact = near_f(got[k], r.direct[k]);
                if (!exact) ++direct_bad;
            }
        }
    }

    std::cout << "[INFO] images verified against a direct decode: " << verified << " of " << total
              << " (every " << stride << "th), " << same_size_cnt << " of them already have the network input size\n";

    CHECK_EQ(label_bad, 0, "targets of every sample match the label from labels.txt (file order)");
    CHECK_EQ(decode_fail, 0, "all verified images can be decoded by the reference decoder");
    CHECK_EQ(image_bad, 0, "verified images match the reference decode + resize (CHW, /255)");
    CHECK_EQ(mean_bad, 0, "per-channel mean of the resized image matches the source mean (tolerance 0.02)");
    CHECK_EQ(direct_bad, 0, "images that already have the input size equal the raw pixels / 255 in CHW order");
}

static void test_real_epoch_statistics()
{
    std::cout << "\n--- real data: epoch statistics ---\n";
    if (skip_if_no_data("epoch statistics")) return;

    const int total = (int)ref_entries.size();
    const int in = input_nodes();
    auto d = make_real(cfg.bs);

    std::map<int, int> expected_hist, got_hist;
    for (auto& e : ref_entries) ++expected_hist[e.label];

    long bad_values = 0, bad_onehot = 0, samples = 0;
    double sum = 0.0, vmin = 1e30, vmax = -1e30;

    for (int b = 0; b < d->get_num_batches(); b++) {
        d->get_batch(b);
        for (int j = 0; j < d->current_batch_size; j++) {
            ++samples;
            const float* img = d->image_batch + (size_t)j * in;
            for (int k = 0; k < in; k++) {
                float v = img[k];
                if (!std::isfinite(v) || v < 0.f || v > 1.f) ++bad_values;
                sum += v; vmin = std::min<double>(vmin, v); vmax = std::max<double>(vmax, v);
            }
            const float* t = d->target_batch + (size_t)j * n_classes;
            int ones = 0, hot = -1;
            for (int k = 0; k < n_classes; k++) {
                if (t[k] == 1.f) { ++ones; hot = k; }
                else if (t[k] != 0.f) ones = 100;
            }
            if (ones != 1) ++bad_onehot; else ++got_hist[hot];
        }
    }

    std::cout << "[INFO] pixel values: min " << vmin << ", max " << vmax << ", mean " << sum / ((double)samples * in) << "\n";
    for (auto& kv : got_hist) std::cout << "[INFO] class " << kv.first << ": " << kv.second << " samples\n";

    CHECK_EQ((int)samples, total, "a full epoch yields every sample exactly once (count)");
    CHECK_EQ(bad_values, 0L, "all pixel values are finite and in [0, 1]");
    CHECK_EQ(bad_onehot, 0L, "all targets are one-hot");
    CHECK(got_hist == expected_hist, "class histogram of the epoch equals the histogram of labels.txt");
    CHECK(vmax > vmin, "pixel values are not constant over the data set");
}

static void test_real_batch_size_independence()
{
    std::cout << "\n--- real data: batch size independence ---\n";
    if (skip_if_no_data("batch size independence")) return;

    const int total = (int)ref_entries.size();
    auto a = make_real(cfg.bs);
    auto b = make_real(7);
    auto c = make_real(total);

    std::vector<uint64_t> ha = epoch_hashes(*a), hb = epoch_hashes(*b), hc = epoch_hashes(*c);
    CHECK(ha == hb, "batch size 7 and the configured batch size give the same sample sequence");
    CHECK(ha == hc, "one batch of the whole data set gives the same sample sequence");
    CHECK_EQ(c->get_train_num_batches(), 1, "batch_size == data set size: one batch");
}

static void test_real_split_and_shuffle()
{
    std::cout << "\n--- real data: split / shuffle ---\n";
    if (skip_if_no_data("split / shuffle")) return;

    const int total = (int)ref_entries.size();
    if (total < 10) { std::cout << "[SKIP] split / shuffle: at least 10 samples are needed\n"; return; }

    auto plain = make_real(cfg.bs);
    const std::vector<uint64_t> full = epoch_hashes(*plain);

    auto d = make_real(cfg.bs, 0.2f, 1u);
    CHECK_EQ(d->get_val_size(), (int)(total * 0.2f), "split 0.2: val_size == (int)(size * 0.2)");
    CHECK_EQ(d->get_train_size() + d->get_val_size(), total, "split 0.2: train_size + val_size == size");

    std::vector<uint64_t> tr = epoch_hashes(*d);
    d->use_val();
    std::vector<uint64_t> va = epoch_hashes(*d);
    d->use_train();
    CHECK_EQ((int)tr.size(), d->get_train_size(), "train epoch has train_size samples");
    CHECK_EQ((int)va.size(), d->get_val_size(), "val epoch has val_size samples");

    std::vector<uint64_t> joined = tr;
    joined.insert(joined.end(), va.begin(), va.end());
    CHECK(sorted_hashes(joined) == sorted_hashes(full), "train + val together contain every sample exactly once");

    auto d2 = make_real(cfg.bs, 0.2f, 1u);
    auto d3 = make_real(cfg.bs, 0.2f, 2u);
    CHECK(epoch_hashes(*d2) == tr, "same seed gives the same split");
    CHECK(epoch_hashes(*d3) != tr, "different seed gives a different split");

    auto s1 = make_real(cfg.bs, 0.f, 5u);
    s1->shuffle();
    std::vector<uint64_t> shuffled = epoch_hashes(*s1);
    CHECK(sorted_hashes(shuffled) == sorted_hashes(full), "shuffle() keeps the same (image, target) pairs");
    CHECK(shuffled != full, "shuffle() changes the order");

    auto s2 = make_real(cfg.bs, 0.f, 5u);
    s2->shuffle();
    CHECK(epoch_hashes(*s2) == shuffled, "same seed gives the same shuffled order");
    s1->shuffle(); s2->shuffle();
    CHECK(epoch_hashes(*s1) == epoch_hashes(*s2), "same seed gives the same order after the second shuffle");

    std::vector<uint64_t> val_before = va;
    auto sv = make_real(cfg.bs, 0.2f, 1u);
    std::vector<uint64_t> train_before = epoch_hashes(*sv);
    sv->use_val();
    sv->shuffle();
    std::vector<uint64_t> val_after = epoch_hashes(*sv);
    sv->use_train();
    CHECK(sorted_hashes(val_after) == sorted_hashes(val_before), "shuffle() in val mode keeps the same val samples");
    CHECK(epoch_hashes(*sv) == train_before, "shuffle() in val mode does not touch the train region");
}

static void test_real_throughput()
{
    std::cout << "\n--- real data: throughput (informational) ---\n";
    if (skip_if_no_data("throughput")) return;

    const int total = (int)ref_entries.size();
    auto d = make_real(cfg.bs);

    auto t0 = Clock::now();
    for (int b = 0; b < d->get_num_batches(); b++) d->get_batch(b);
    double sec = secs_since(t0);

    std::cout << "[INFO] one epoch (single thread): " << sec << " s, " << (sec > 0 ? total / sec : 0.0)
              << " images/s, " << (d->get_num_batches() > 0 ? 1000.0 * sec / d->get_num_batches() : 0.0)
              << " ms per batch of " << cfg.bs << "\n";

    t0 = Clock::now();
    d->shuffle();
    for (int b = 0; b < d->get_num_batches(); b++) d->get_batch(b);
    double sec2 = secs_since(t0);
    std::cout << "[INFO] one shuffled epoch: " << sec2 << " s (compare with the sequential epoch for the disk cache effect)\n";
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <dataset_dir> [C H W [max_samples [batch_size]]]\n";
        return 64;
    }

    cfg.dir = argv[1];
    while (cfg.dir.size() > 1 && (cfg.dir.back() == '/' || cfg.dir.back() == '\\')) cfg.dir.pop_back();
    if (argc >= 5) { cfg.C = std::atoi(argv[2]); cfg.H = std::atoi(argv[3]); cfg.W = std::atoi(argv[4]); }
    if (argc >= 6) cfg.max_n = std::atoi(argv[5]);
    if (argc >= 7) cfg.bs = std::atoi(argv[6]);

    if (cfg.C < 1 || cfg.C > 4 || cfg.H < 1 || cfg.W < 1 || cfg.max_n < 0 || cfg.bs < 1) {
        std::cerr << "invalid shape, max_samples or batch_size\n";
        return 64;
    }
    std::cout << "data set: " << cfg.dir << ", shape " << cfg.C << "x" << cfg.H << "x" << cfg.W
              << ", max_samples " << cfg.max_n << ", batch_size " << cfg.bs << "\n";

    try {
        test_labels_integrity();
        test_real_construct_and_stats();
        test_real_content();
        test_real_epoch_statistics();
        test_real_batch_size_independence();
        test_real_split_and_shuffle();
        test_real_throughput();
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] unexpected exception: " << e.what() << "\n";
        tests_failed++;
    }

    std::cout << "\n=== " << tests_run << " tests run, " << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}
