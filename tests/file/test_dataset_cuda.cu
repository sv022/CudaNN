#include "../../network/file/file.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/wait.h>
#endif

static const int IMG_C = 1, IMG_H = 12, IMG_W = 12;
static const int INPUT_NODES = IMG_C * IMG_H * IMG_W;
static const int N_CLASSES = 4;
static const int TRAIN_N = 200, TEST_N = 80;
std::string TRAIN_CSV = "data/synthetic_train.csv";
std::string TEST_CSV  = "data/synthetic_test.csv";

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

using Rows = std::vector<std::vector<float>>;

struct Ref {
    std::vector<int> label;
    std::vector<std::vector<float>> feat;
};

struct Snap {
    std::vector<float> img, tgt;
    int n = 0;
    bool operator==(const Snap& o) const { return n == o.n && img == o.img && tgt == o.tgt; }
    bool operator!=(const Snap& o) const { return !(*this == o); }
};

static std::vector<std::string> tmp_files;
static std::string exe_path;

static Ref load_ref(const std::string& path, int in, int out)
{
    Ref r;
    std::ifstream f(path);
    std::string line;
    if (!std::getline(f, line)) return r;

    while (std::getline(f, line)) {
        std::vector<std::string> tok;
        std::string t;
        std::istringstream ss(line);
        while (std::getline(ss, t, ',')) tok.push_back(t);
        if ((int)tok.size() != in + 1) continue;

        int lab = std::atoi(tok[0].c_str());
        if (lab < 0 || lab >= out) continue;

        std::vector<float> v(in);
        for (int i = 0; i < in; i++) v[i] = std::strtof(tok[i + 1].c_str(), nullptr);
        r.label.push_back(lab);
        r.feat.push_back(v);
    }
    return r;
}

static Snap snap(const DatasetFile& d, int in = INPUT_NODES, int out = N_CLASSES)
{
    Snap s;
    s.n = d.current_batch_size;
    if (s.n > 0) {
        s.img.assign(d.image_batch, d.image_batch + (size_t)s.n * in);
        s.tgt.assign(d.target_batch, d.target_batch + (size_t)s.n * out);
    }
    return s;
}

static std::vector<float> sample_row(const DatasetFile& d, int j, int in = INPUT_NODES, int out = N_CLASSES)
{
    std::vector<float> r(d.image_batch + (size_t)j * in, d.image_batch + (size_t)(j + 1) * in);
    r.insert(r.end(), d.target_batch + (size_t)j * out, d.target_batch + (size_t)(j + 1) * out);
    return r;
}

static std::vector<float> ref_row(const Ref& r, int i, int out = N_CLASSES)
{
    std::vector<float> v = r.feat[i];
    std::vector<float> t(out, 0.f);
    t[r.label[i]] = 1.f;
    v.insert(v.end(), t.begin(), t.end());
    return v;
}

static std::unique_ptr<DatasetFile> make(int bs, float vf = 0.f, unsigned seed = 12345u,
                                         const std::string& path = TRAIN_CSV, int n = TRAIN_N)
{
    return std::unique_ptr<DatasetFile>(new DatasetFile(path, n, INPUT_NODES, N_CLASSES, bs, vf, seed));
}

static std::unique_ptr<DatasetFile> make_small(const std::string& path, int size, int bs)
{
    return std::unique_ptr<DatasetFile>(new DatasetFile(path, size, 3, 4, bs));
}

static Rows epoch_rows(DatasetFile& d)
{
    Rows rows;
    for (int b = 0; b < d.get_num_batches(); b++) {
        d.get_batch(b);
        for (int j = 0; j < d.current_batch_size; j++) rows.push_back(sample_row(d, j));
    }
    return rows;
}

static std::string write_tmp(const std::string& name, const std::string& content)
{
    std::string p = "_tmp_" + name + ".csv";
    std::ofstream f(p, std::ios::binary);
    f << content;
    tmp_files.push_back(p);
    return p;
}

static std::string capture_cerr(const std::function<void()>& fn)
{
    std::ostringstream ss;
    std::streambuf* old = std::cerr.rdbuf(ss.rdbuf());
    try { fn(); } catch (...) { std::cerr.rdbuf(old); throw; }
    std::cerr.rdbuf(old);
    return ss.str();
}

static int count_substr(const std::string& s, const std::string& sub)
{
    int n = 0;
    for (size_t p = s.find(sub); p != std::string::npos; p = s.find(sub, p + 1)) ++n;
    return n;
}

static bool has(const std::string& s, const std::string& sub)
{
    return s.find(sub) != std::string::npos;
}

// ---------------- isolated cases (may crash) ----------------

static std::map<std::string, std::function<int()>>& isolated_cases()
{
    static std::map<std::string, std::function<int()>> m;
    return m;
}

static void register_isolated_cases()
{
    isolated_cases()["next_batch_uninit_num_batches"] = [] {
        auto d = make(32);
        d->next_batch();
        return (d->current_batch == 1 && d->current_batch_size == 32) ? 0 : 3;
    };
    isolated_cases()["get_batch_negative_index"] = [] {
        auto d = make(32);
        d->get_batch(-1);
        return 0;
    };
    isolated_cases()["batch_size_zero"] = [] {
        auto d = make(0);
        return 0;
    };
}

static int run_isolated(const std::string& name)
{
#ifdef _WIN32
    std::string cmd = "\"" + exe_path + "\" --case " + name + " > NUL 2>&1";
#else
    std::string cmd = "\"" + exe_path + "\" --case " + name + " > /dev/null 2>&1";
#endif
    int rc = std::system(cmd.c_str());
#ifndef _WIN32
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    if (WIFSIGNALED(rc)) return 128 + WTERMSIG(rc);
#endif
    return rc;
}

static std::string describe_exit(int code)
{
    if (code == 0) return "completed normally, result as intended";
    if (code == 3) return "completed, but result is WRONG (silent corruption)";
    if (code == 2) return "terminated with C++ exception";
    return "CRASHED / abnormal termination (code " + std::to_string(code) + ")";
}

// ---------------- test groups ----------------

static void test_split_and_parse_val_fraction()
{
    std::cout << "\n--- split / parse_val_fraction ---\n";

    CHECK((split("a,b,c", ',') == std::vector<std::string>{"a", "b", "c"}), "split: basic tokens");
    CHECK((split("a,,b,", ',') == std::vector<std::string>{"a", "b"}), "split: empty tokens are dropped");
    CHECK(split("", ',').empty(), "split: empty string gives empty result");
    CHECK(split(",,,", ',').empty(), "split: only separators gives empty result");

    auto a = parse_val_fraction("100/0.2");
    CHECK(a.first == 100 && a.second == 0.2f, "parse_val_fraction: \"100/0.2\" -> {100, 0.2}");
    auto b = parse_val_fraction("100");
    CHECK(b.first == 100 && b.second == 0.0f, "parse_val_fraction: \"100\" -> {100, 0}");
    auto c = parse_val_fraction("");
    CHECK(c.first == 0 && c.second == 0.0f, "parse_val_fraction: empty string -> {0, 0}");

    CHECK_THROWS(parse_val_fraction("5/1"), std::out_of_range, "greater than 1",
                 "parse_val_fraction: fraction == 1 throws out_of_range");
    CHECK_THROWS(parse_val_fraction("5/-0.1"), std::out_of_range, "smaller than 0",
                 "parse_val_fraction: negative fraction throws out_of_range");
    CHECK_THROWS(parse_val_fraction("abc"), std::invalid_argument, "",
                 "parse_val_fraction: non-numeric input throws invalid_argument");
}

static void test_constructor_and_getters()
{
    std::cout << "\n--- constructor / getters ---\n";

    auto d = make(32);
    CHECK_EQ(d->get_train_size(), 200, "no split: train_size == 200");
    CHECK_EQ(d->get_val_size(), 0, "no split: val_size == 0");
    CHECK_EQ(d->get_train_num_batches(), 7, "no split: train_num_batches == ceil(200/32) == 7");
    CHECK_EQ(d->get_val_num_batches(), 0, "no split: val_num_batches == 0");
    CHECK_EQ(d->get_num_batches(), 7, "no split: get_num_batches() == 7");
    CHECK(!d->has_val(), "no split: has_val() == false");
    CHECK_EQ(d->get_batch_size(), 32, "get_batch_size() == 32");
    CHECK_EQ(d->get_input_nodes(), INPUT_NODES, "get_input_nodes() == 144");
    CHECK_EQ(d->get_output_nodes(), N_CLASSES, "get_output_nodes() == 4");
    CHECK_EQ(d->current_batch_size, 32, "constructor loads batch 0 (reset)");

    auto v = make(32, 0.2f);
    CHECK_EQ(v->get_val_size(), (int)(TRAIN_N * 0.2f), "split 0.2: val_size == (int)(size * 0.2)");
    CHECK_EQ(v->get_train_size(), TRAIN_N - v->get_val_size(), "split 0.2: train_size == size - val_size");
    CHECK(v->has_val(), "split 0.2: has_val() == true");
    CHECK_EQ(v->get_train_num_batches(), (v->get_train_size() + 31) / 32, "split 0.2: train_num_batches");
    CHECK_EQ(v->get_val_num_batches(), (v->get_val_size() + 31) / 32, "split 0.2: val_num_batches");
    CHECK_EQ(v->get_num_batches(), v->get_train_num_batches(), "split 0.2: initial mode is train");

    float vf = 0.333f;
    auto t = make(32, vf);
    CHECK_EQ(t->get_val_size(), static_cast<int>(TRAIN_N * vf), "val_size uses int truncation");

    auto e = make(32, 1.0f);
    CHECK(e->get_train_size() == 0 && e->get_val_size() == TRAIN_N,
          "PIN: val_fraction == 1.0 is not rejected (train_size 0, val_size 200)");
    CHECK(e->get_train_num_batches() == 0 && e->get_val_num_batches() == 7,
          "PIN: val_fraction == 1.0 gives 0 train batches and 7 val batches");

    auto n = make(32, -0.5f);
    CHECK(n->get_val_size() == 0 && n->get_train_size() == TRAIN_N && !n->has_val(),
          "negative val_fraction behaves as 0");

    auto ts = make(16, 0.f, 12345u, TEST_CSV, TEST_N);
    Ref r = load_ref(TEST_CSV, INPUT_NODES, N_CLASSES);
    CHECK_EQ((int)r.label.size(), TEST_N, "reference parser finds 80 valid rows in test csv");
    CHECK(ts->get_train_size() == 80 && ts->get_train_num_batches() == 5, "test csv: size 80, 5 batches of 16");
    bool ok = true;
    for (int j = 0; j < 16; j++) ok = ok && (sample_row(*ts, j) == ref_row(r, j));
    CHECK(ok, "test csv: first batch matches reference rows");
}

static void test_constructor_errors()
{
    std::cout << "\n--- constructor errors ---\n";

    CHECK_THROWS(make(32, 0.f, 1u, "data/file.txt"), std::runtime_error, "Unsupported file format",
                 "unsupported extension .txt throws runtime_error");
    CHECK_THROWS(make(32, 0.f, 1u, "data/file.CSV"), std::runtime_error, "Unsupported file format",
                 "extension check is case-sensitive (.CSV rejected)");
    CHECK_THROWS(make(32, 0.f, 1u, "noextension"), std::runtime_error, "Unsupported file format",
                 "path without extension throws runtime_error");

    std::string err = capture_cerr([] {
        try { make(32, 0.f, 1u, "data/does_not_exist.csv"); } catch (const std::runtime_error&) {}
    });
    CHECK(has(err, "Could not open file"), "missing file: message written to cerr");
    CHECK_THROWS(make(32, 0.f, 1u, "data/does_not_exist.csv"), std::runtime_error, "Error opening file.",
                 "missing file throws runtime_error \"Error opening file.\"");

    CHECK_THROWS(make(32, 0.f, 1u, TRAIN_CSV, TRAIN_N + 1), std::runtime_error,
                "contains 200 valid rows, expected 201",
                "size larger than the train csv row count throws runtime_error");

    std::string empty = write_tmp("empty", "");
    CHECK_THROWS(make_small(empty, 3, 2), std::runtime_error, "CSV file is empty",
                 "empty file throws runtime_error \"CSV file is empty\"");

    std::string hdr0 = write_tmp("hdr_only0", "h\n");
    try {
        auto d = make_small(hdr0, 0, 2);
        std::cout << "[INFO] size=0: constructed, train_num_batches=" << d->get_train_num_batches() << "\n";
    } catch (const std::exception& e) {
        std::cout << "[INFO] size=0: threw: " << e.what() << "\n";
    }
}

static void test_batches_content()
{
    std::cout << "\n--- batches: content and order ---\n";

    Ref r = load_ref(TRAIN_CSV, INPUT_NODES, N_CLASSES);
    CHECK_EQ((int)r.label.size(), TRAIN_N, "reference parser finds 200 valid rows in train csv");

    auto d = make(32);
    bool order_ok = true;
    for (int b = 0; b < d->get_num_batches(); b++) {
        d->get_batch(b);
        for (int j = 0; j < d->current_batch_size; j++)
            order_ok = order_ok && (sample_row(*d, j) == ref_row(r, b * 32 + j));
    }
    CHECK(order_ok, "without shuffle, batches follow file order and match reference rows");

    bool one_hot_ok = true;
    for (auto& row : epoch_rows(*d)) {
        float sum = 0.f;
        int ones = 0;
        for (int i = 0; i < N_CLASSES; i++) {
            float t = row[INPUT_NODES + i];
            sum += t;
            ones += (t == 1.f);
        }
        one_hot_ok = one_hot_ok && (sum == 1.0f) && (ones == 1);
    }
    CHECK(one_hot_ok, "all targets are one-hot");

    d->get_batch(6);
    CHECK(d->current_batch_size == 8 && d->current_batch == 6, "last batch is partial (8 of 32)");
    d->get_batch(5);
    CHECK_EQ(d->current_batch_size, 32, "batch 5 has full size");

    Snap s0 = snap(*d);
    d->get_batch(0);
    s0 = snap(*d);
    d->get_batch(3);
    d->reset();
    CHECK(d->current_batch == 0 && snap(*d) == s0, "reset() returns to batch 0 with identical data");

    d->get_batch(7);
    CHECK_EQ(d->current_batch_size, 200 - 7 * 32, "PIN: get_batch(7) past the end gives negative size (-24)");
    CHECK_EQ(d->current_batch, 7, "PIN: get_batch(7) stores out-of-range current_batch");
    d->get_batch(100);
    CHECK(d->current_batch_size < 0, "PIN: get_batch(100) gives negative size");
    d->reset();

    Rows rows = epoch_rows(*d);
    std::multiset<std::vector<float>> got(rows.begin(), rows.end()), expected;
    for (int i = 0; i < TRAIN_N; i++) expected.insert(ref_row(r, i));
    CHECK_EQ((int)rows.size(), TRAIN_N, "full epoch yields 200 samples");
    CHECK(got == expected, "full epoch covers every reference sample exactly once");

    std::string three = write_tmp("three", "h\n0,1,2,3\n1,4,5,6\n2,7,8,9\n");
    auto small = make_small(three, 3, 8);
    CHECK(small->get_train_num_batches() == 1 && small->current_batch_size == 3,
          "batch_size > dataset size: 1 batch with 3 samples");
}

static void test_shuffle_and_seed()
{
    std::cout << "\n--- shuffle / seed ---\n";

    auto d = make(32);
    Rows before = epoch_rows(*d);
    d->shuffle();
    CHECK_EQ(d->current_batch, 0, "shuffle() resets current_batch to 0");
    Rows after = epoch_rows(*d);
    CHECK(std::multiset<std::vector<float>>(after.begin(), after.end()) ==
          std::multiset<std::vector<float>>(before.begin(), before.end()),
          "shuffle() keeps the same set of samples");
    CHECK(after != before, "shuffle() changes the order");

    auto a = make(32, 0.f, 777u);
    auto b = make(32, 0.f, 777u);
    a->shuffle(); b->shuffle();
    bool first = (epoch_rows(*a) == epoch_rows(*b));
    a->shuffle(); b->shuffle();
    bool second = (epoch_rows(*a) == epoch_rows(*b));
    CHECK(first && second, "same seed gives identical order across consecutive shuffles");

    auto s1 = make(32, 0.f, 1u);
    auto s2 = make(32, 0.f, 2u);
    s1->shuffle(); s2->shuffle();
    CHECK(epoch_rows(*s1) != epoch_rows(*s2), "different seeds give different order");

    auto n1 = make(32, 0.f, 1u);
    auto n2 = make(32, 0.f, 999u);
    CHECK(snap(*n1) == snap(*n2), "val_fraction == 0: constructor does not shuffle (seed has no effect)");

    auto v1 = make(32, 0.2f, 1u);
    auto v2 = make(32, 0.2f, 2u);
    auto v3 = make(32, 0.2f, 1u);
    CHECK(snap(*v1) != snap(*v2), "val_fraction > 0: different seed gives different split");
    CHECK(snap(*v1) == snap(*v3), "val_fraction > 0: same seed gives same split");

    Ref r = load_ref(TRAIN_CSV, INPUT_NODES, N_CLASSES);
    bool in_file_order = true;
    for (int j = 0; j < 32; j++) in_file_order = in_file_order && (sample_row(*v1, j) == ref_row(r, j));
    CHECK(!in_file_order, "val_fraction > 0: constructor shuffles indices");
}

static void test_train_val_modes()
{
    std::cout << "\n--- train / val modes ---\n";

    auto plain = make(32);
    CHECK_THROWS(plain->use_val(), std::runtime_error, "constructed without a validation split",
        "use_val() without split throws runtime_error");
    CHECK_EQ(plain->get_num_batches(), 7, "failed use_val() keeps train mode");

    auto d = make(32, 0.2f);
    Snap train0 = snap(*d);
    CHECK_EQ(d->get_train_size(), 160, "split 0.2: train_size == 160");
    CHECK_EQ(d->get_val_size(), 40, "split 0.2: val_size == 40");

    d->use_val();
    CHECK(d->get_num_batches() == d->get_val_num_batches() && d->current_batch == 0,
        "use_val() switches num_batches to val and resets");
    CHECK(snap(*d) != train0, "val batch 0 differs from train batch 0");
    CHECK_EQ(d->current_batch_size, 32, "val batch 0 has size 32");
    d->get_batch(1);
    CHECK_EQ(d->current_batch_size, 8, "val batch 1 has size 8 (40 - 32)");
    CHECK_EQ(d->current_batch, 1, "val get_batch(1) stores current_batch == 1");

    d->get_batch(1);
    d->use_train();
    CHECK(d->get_num_batches() == d->get_train_num_batches() && d->current_batch == 0 && snap(*d) == train0,
        "use_train() switches back and resets to train batch 0");
    {
        Ref r = load_ref(TRAIN_CSV, INPUT_NODES, N_CLASSES);
        Rows tr = epoch_rows(*d);
        d->use_val();
        Rows va = epoch_rows(*d);
        d->use_train();

        CHECK_EQ((int)tr.size(), 160, "full train epoch yields 160 samples");
        CHECK_EQ((int)va.size(), 40, "full val epoch yields 40 samples");

        std::multiset<std::vector<float>> got(tr.begin(), tr.end()), expected;
        got.insert(va.begin(), va.end());
        for (int i = 0; i < TRAIN_N; i++) expected.insert(ref_row(r, i));
        CHECK(got == expected, "train + val epochs cover every reference sample exactly once");
    }

    auto big = make(64, 0.2f);
    CHECK_EQ(big->get_train_num_batches(), 3, "batch 64, split 0.2: 3 train batches");
    big->get_batch(2);
    CHECK_EQ(big->current_batch_size, 160 - 2 * 64, "last train batch has 32 samples (no leak from val)");
    big->use_val();
    CHECK_EQ(big->get_num_batches(), 1, "batch 64, split 0.2: 1 val batch");
    CHECK_EQ(big->current_batch_size, 40, "batch 64: single val batch has 40 samples");

    auto sh = make(32, 0.2f);
    Rows trainBefore = epoch_rows(*sh);
    sh->use_val();
    Rows valBefore = epoch_rows(*sh);
    sh->shuffle();
    CHECK_EQ(sh->current_batch, 0, "shuffle() in val mode resets current_batch to 0");
    Rows valAfter = epoch_rows(*sh);
    sh->use_train();
    Rows trainAfter = epoch_rows(*sh);
    CHECK(trainAfter == trainBefore, "shuffle() in val mode does not touch the train region");
    CHECK(valAfter != valBefore, "shuffle() in val mode permutes the val region");
    CHECK(std::multiset<std::vector<float>>(valAfter.begin(), valAfter.end()) ==
          std::multiset<std::vector<float>>(valBefore.begin(), valBefore.end()),
        "shuffle() in val mode keeps the same set of val samples");

    auto sh2 = make(32, 0.2f);
    sh2->use_val();
    Rows valBefore2 = epoch_rows(*sh2);
    sh2->use_train();
    Rows trainBefore2 = epoch_rows(*sh2);
    sh2->shuffle();
    Rows trainAfter2 = epoch_rows(*sh2);
    sh2->use_val();
    Rows valAfter2 = epoch_rows(*sh2);
    CHECK(valAfter2 == valBefore2, "shuffle() in train mode does not touch the val region");
    CHECK(trainAfter2 != trainBefore2, "shuffle() in train mode permutes the train region");
}

static void test_csv_parsing()
{
    std::cout << "\n--- csv parsing edge cases ---\n";

    {
        std::string p = write_tmp("badlabel", "h\n0,1,2,3\n9,1,2,3\n-1,1,2,3\n1,4,5,6\n");
        std::unique_ptr<DatasetFile> d;
        std::string err = capture_cerr([&] { d = make_small(p, 2, 2); });
        CHECK(!has(err, "Warning"), "out-of-range labels are skipped silently");
        CHECK(d->image_batch[0] == 1.f && d->target_batch[0] == 1.f &&
              d->image_batch[3] == 4.f && d->target_batch[4 + 1] == 1.f,
              "rows around skipped labels are loaded correctly");
    }
    {
        std::string p = write_tmp("badfeat", "h\n0,1,2,3\n1,1,2\n1,1\n2,7,8,9\n");
        std::unique_ptr<DatasetFile> d;
        std::string err = capture_cerr([&] { d = make_small(p, 2, 2); });
        CHECK_EQ(count_substr(err, "Warning: line 3 has wrong number of features"), 2,
                 "PIN: both bad rows report \"line 3\" (line number sticks to accepted count + 2)");
        CHECK(!has(err, "line 4"), "no warning mentions line 4");
        CHECK(d->target_batch[4 + 2] == 1.f && d->image_batch[3] == 7.f, "valid rows after bad ones are loaded");
    }
    {
        std::string p = write_tmp("extra", "h\n0,1,2,3,4,5\n");
        std::unique_ptr<DatasetFile> d;
        std::string err = capture_cerr([&] { d = make_small(p, 1, 1); });
        CHECK(!has(err, "Warning") && d->image_batch[2] == 3.f, "extra features are ignored without warning");
    }
    {
        std::string p = write_tmp("dcomma", "h\n0,1,,3\n1,4,5,6\n");
        std::unique_ptr<DatasetFile> d;
        std::string err = capture_cerr([&] { d = make_small(p, 1, 1); });
        CHECK_EQ(count_substr(err, "Warning: line 2"), 1, "double comma row triggers a warning");
        CHECK(d->image_batch[0] == 4.f && d->target_batch[1] == 1.f, "double comma row is skipped");
    }
    {
        std::string p = write_tmp("ws", "h\n0, 1.5, -2e-1,  3\n");
        std::unique_ptr<DatasetFile> d;
        std::string err = capture_cerr([&] { d = make_small(p, 1, 1); });
        CHECK(!has(err, "Warning"), "spaces after commas are accepted");
        CHECK(d->image_batch[0] == 1.5f && d->image_batch[1] == -0.2f && d->image_batch[2] == 3.f,
              "decimal and exponent notation parsed correctly");
    }
    {
        std::string p = write_tmp("crlf", "h\r\n0,1,2,3\r\n1,4,5,6\r\n");
        std::unique_ptr<DatasetFile> d;
        std::string err = capture_cerr([&] { d = make_small(p, 2, 2); });
        CHECK(!has(err, "Warning"), "CRLF line endings produce no warnings");
        CHECK(d->image_batch[5] == 6.f && d->target_batch[4 + 1] == 1.f, "CRLF rows parsed correctly");
    }
    {
        std::string p = write_tmp("blank", "h\n\n0,1,2,3\n\n1,4,5,6\n");
        std::unique_ptr<DatasetFile> d;
        std::string err = capture_cerr([&] { d = make_small(p, 2, 2); });
        CHECK(!has(err, "Warning") && d->image_batch[3] == 4.f, "blank lines are skipped silently");
    }
    {
        std::string p = write_tmp("nohdr", "0,1,2,3\n1,4,5,6\n");
        auto d = make_small(p, 1, 1);
        CHECK(d->image_batch[0] == 4.f && d->target_batch[1] == 1.f,
              "first line is always treated as header (first data row is lost)");
    }
    {
        std::string p = write_tmp("manyrows", "h\n0,1,1,1\n1,2,2,2\n2,3,3,3\n3,4,4,4\n");
        auto d = make_small(p, 2, 2);
        CHECK(d->get_train_size() == 2 && d->get_train_num_batches() == 1 &&
              d->current_batch_size == 2 && d->image_batch[3] == 2.f,
              "loading stops after `size` valid rows");
    }
        {
        std::string p = write_tmp("fewrows", "h\n0,1,2,3\n1,4,5,6\n");
        CHECK_THROWS(make_small(p, 5, 2), std::runtime_error, "contains 2 valid rows, expected 5",
                     "fewer valid rows than size throws runtime_error with found/expected counts");
    }
    {
        std::string p = write_tmp("hdronly", "h\n");
        CHECK_THROWS(make_small(p, 3, 2), std::runtime_error, "contains 0 valid rows, expected 3",
                     "header-only file with size > 0 throws runtime_error");
    }
    {
        std::string p = write_tmp("skipped_count", "h\n0,1,2,3\n9,1,2,3\n1,1,2\n");
        std::string err = capture_cerr([&] {
            try { make_small(p, 2, 2); } catch (const std::runtime_error&) {}
        });
        CHECK(has(err, "Warning: line 3"), "skipped rows still produce warnings before the exception");
        CHECK_THROWS(make_small(p, 2, 2), std::runtime_error, "contains 1 valid rows, expected 2",
                     "skipped (invalid) rows are not counted as valid");
    }
    {
        std::string p = write_tmp("exact", "h\n0,1,2,3\n1,4,5,6\n");
        auto d = make_small(p, 2, 2);
        CHECK_EQ(d->get_train_size(), 2, "exact row count constructs normally");
    }
}

// ---------------- .bin (IMGS) helpers ----------------

struct BinRec { uint16_t label; std::vector<uint8_t> px; };

static float norm255(uint8_t v) { return v * (1.0f / 255.0f); }

static bool near_vec(const std::vector<float>& a, const std::vector<float>& b, float eps)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) if (!(std::fabs(a[i] - b[i]) <= eps)) return false;
    return true;
}

static std::vector<BinRec> make_recs(int n, int pix, int nl)
{
    std::vector<BinRec> r(n);
    for (int i = 0; i < n; i++) {
        r[i].label = (uint16_t)(i % nl);
        r[i].px.resize(pix);
        for (int k = 0; k < pix; k++) r[i].px[k] = (uint8_t)((i * 37 + k * 11 + 3) % 256);
    }
    if (n > 0 && pix >= 5) {                 // bytes that text mode would damage: \n, \r, 0x1A, 0, 255
        const uint8_t sp[5] = {10, 13, 26, 0, 255};
        for (int k = 0; k < 5; k++) r[0].px[k] = sp[k];
    }
    return r;
}

static std::string write_bin_file(const std::string& name, int W, int H, int C,
                                  const std::vector<std::string>& label_names,
                                  const std::vector<BinRec>& recs,
                                  long long declared_count = -1, size_t tail_garbage = 0)
{
    std::string p = "_tmp_" + name + ".bin";
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write("IMGS", 4);
    uint8_t ver = 1, c = (uint8_t)C;
    uint16_t w = (uint16_t)W, h = (uint16_t)H, nl = (uint16_t)label_names.size();
    f.write((const char*)&ver, 1); f.write((const char*)&w, 2); f.write((const char*)&h, 2); f.write((const char*)&c, 1);
    f.write((const char*)&nl, 2);
    for (auto& s : label_names) {
        uint16_t len = (uint16_t)s.size();
        f.write((const char*)&len, 2);
        f.write(s.data(), len);
    }
    uint32_t cnt = declared_count < 0 ? (uint32_t)recs.size() : (uint32_t)declared_count;
    f.write((const char*)&cnt, 4);
    for (auto& r : recs) {
        f.write((const char*)&r.label, 2);
        f.write((const char*)r.px.data(), (std::streamsize)r.px.size());
    }
    for (size_t i = 0; i < tail_garbage; i++) f.put((char)0x7F);
    tmp_files.push_back(p);
    return p;
}

static std::string write_raw_bin(const std::string& name, const std::string& bytes)
{
    std::string p = "_tmp_" + name + ".bin";
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(bytes.data(), (std::streamsize)bytes.size());
    tmp_files.push_back(p);
    return p;
}

static void patch_byte(const std::string& path, std::streamoff off, uint8_t v)
{
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(off);
    f.put((char)v);
}

// expected sample row: channel-planar normalized pixels followed by the one-hot target
static std::vector<float> bin_expected_row(const BinRec& r, int W, int H, int C, int NL)
{
    std::vector<float> v((size_t)W * H * C + NL, 0.f);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            for (int ch = 0; ch < C; ch++)
                v[(size_t)ch * H * W + y * W + x] = norm255(r.px[((size_t)y * W + x) * C + ch]);
    v[(size_t)W * H * C + r.label] = 1.f;
    return v;
}

static std::unique_ptr<DatasetFile> make_bin(const std::string& path, int size, int in, int out, int bs,
                                             float vf = 0.f, unsigned seed = 12345u)
{
    return std::unique_ptr<DatasetFile>(new DatasetFile(path, size, in, out, bs, vf, seed));
}

static Rows rows_of(DatasetFile& d, int in, int out)
{
    Rows rows;
    for (int b = 0; b < d.get_num_batches(); b++) {
        d.get_batch(b);
        for (int j = 0; j < d.current_batch_size; j++) rows.push_back(sample_row(d, j, in, out));
    }
    return rows;
}

// ---------------- .bin (IMGS) tests ----------------

static void test_bin_loading()
{
    std::cout << "\n--- .bin (IMGS): content, layout, normalization ---\n";

    {   // single channel, partial last batch
        const int W = 4, H = 3, C = 1, NL = 3, N = 6, BS = 4, PIX = W * H * C;
        auto recs = make_recs(N, PIX, NL);
        std::string p = write_bin_file("gray", W, H, C, {"a", "b", "c"}, recs);
        auto d = make_bin(p, N, PIX, NL, BS);

        CHECK(d->get_train_size() == N && d->get_train_num_batches() == 2, "bin gray: size 6, 2 batches (4 + 2)");
        bool ok = true;
        for (int j = 0; j < 4; j++)
            ok = ok && near_vec(sample_row(*d, j, PIX, NL), bin_expected_row(recs[j], W, H, C, NL), 1e-6f);
        CHECK(ok, "bin gray: batch 0 equals pixels / 255 with one-hot targets, in file order");

        CHECK(std::fabs(d->image_batch[0] - 10.f / 255.f) < 1e-6f && std::fabs(d->image_batch[1] - 13.f / 255.f) < 1e-6f &&
              std::fabs(d->image_batch[2] - 26.f / 255.f) < 1e-6f && d->image_batch[3] == 0.f &&
              std::fabs(d->image_batch[4] - 1.0f) < 1e-6f,
              "bin gray: bytes 10, 13, 26, 0, 255 survive intact (binary mode)");

        d->get_batch(1);
        CHECK_EQ(d->current_batch_size, 2, "bin gray: last batch is partial (2 of 4)");
        ok = true;
        for (int j = 0; j < 2; j++)
            ok = ok && near_vec(sample_row(*d, j, PIX, NL), bin_expected_row(recs[4 + j], W, H, C, NL), 1e-6f);
        CHECK(ok, "bin gray: batch 1 matches records 4 and 5");
    }

    {   // three channels: interleaved HWC in the file -> channel-planar CHW in the batch
        const int W = 3, H = 2, C = 3, NL = 2, N = 3, PIX = W * H * C;
        std::vector<BinRec> recs(N);
        for (int i = 0; i < N; i++) {
            recs[i].label = (uint16_t)(i % NL);
            recs[i].px.resize(PIX);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int ch = 0; ch < C; ch++)
                        recs[i].px[((size_t)y * W + x) * C + ch] = (uint8_t)(100 * ch + 10 * y + x + i);
        }
        std::string p = write_bin_file("color", W, H, C, {"a", "b"}, recs);
        auto d = make_bin(p, N, PIX, NL, N);
        bool ok = true;
        for (int j = 0; j < N; j++)
            ok = ok && near_vec(sample_row(*d, j, PIX, NL), bin_expected_row(recs[j], W, H, C, NL), 1e-6f);
        CHECK(ok, "bin color: interleaved HWC bytes become channel-planar CHW");
    }
}

static void test_bin_size_and_records()
{
    std::cout << "\n--- .bin: size, header count, skipped records ---\n";
    const int W = 4, H = 3, C = 1, NL = 3, PIX = W * H * C;

    {
        auto recs = make_recs(6, PIX, NL);
        std::string p = write_bin_file("prefix", W, H, C, {"a", "b", "c"}, recs);
        auto d = make_bin(p, 4, PIX, NL, 4);
        bool ok = (d->get_train_size() == 4);
        for (int j = 0; j < 4; j++)
            ok = ok && near_vec(sample_row(*d, j, PIX, NL), bin_expected_row(recs[j], W, H, C, NL), 1e-6f);
        CHECK(ok, "bin: size smaller than the file loads the first `size` records");
        CHECK_THROWS(make_bin(p, 7, PIX, NL, 4), std::runtime_error, "file contains only 6 valid rows, expected 7",
                     "bin: size larger than the record count throws with found/expected counts");
    }
    {
        auto recs = make_recs(6, PIX, NL);
        std::string p0 = write_bin_file("cnt0", W, H, C, {"a", "b", "c"}, recs, 0);
        auto d0 = make_bin(p0, 6, PIX, NL, 6);
        CHECK(near_vec(sample_row(*d0, 5, PIX, NL), bin_expected_row(recs[5], W, H, C, NL), 1e-6f),
              "PIN: header count == 0 (unfinished export) is ignored, records are read until EOF");
        std::string p9 = write_bin_file("cnt999", W, H, C, {"a", "b", "c"}, recs, 999);
        auto d9 = make_bin(p9, 6, PIX, NL, 6);
        CHECK_EQ(d9->get_train_size(), 6, "PIN: header count larger than the real one is ignored");
    }
    {
        auto recs = make_recs(6, PIX, NL);
        recs[1].label = 9;                   // label index outside [0, NL)
        recs[4].label = 7;
        std::string p = write_bin_file("badlab", W, H, C, {"a", "b", "c"}, recs);
        std::unique_ptr<DatasetFile> d;
        std::string err = capture_cerr([&] { d = make_bin(p, 4, PIX, NL, 4); });
        CHECK(!has(err, "Warning"), "bin: out-of-range label indices are skipped silently");
        const int keep[4] = {0, 2, 3, 5};
        bool ok = true;
        for (int j = 0; j < 4; j++)
            ok = ok && near_vec(sample_row(*d, j, PIX, NL), bin_expected_row(recs[keep[j]], W, H, C, NL), 1e-6f);
        CHECK(ok, "bin: rows around skipped records are loaded in file order");
        CHECK_THROWS(make_bin(p, 5, PIX, NL, 4), std::runtime_error, "file contains only 4 valid rows, expected 5",
                     "bin: skipped records are not counted as valid");
    }
    {
        auto recs = make_recs(3, PIX, NL);
        std::string p = write_bin_file("tail", W, H, C, {"a", "b", "c"}, recs, -1, 5);   // trailing bytes shorter than a record
        auto d = make_bin(p, 3, PIX, NL, 3);
        CHECK_EQ(d->get_train_size(), 3, "bin: trailing partial record does not affect complete records");
        CHECK_THROWS(make_bin(p, 4, PIX, NL, 3), std::runtime_error, "file contains only 3 valid rows, expected 4",
                     "bin: trailing partial record is not counted as a row");
    }
}

static void test_bin_errors()
{
    std::cout << "\n--- .bin: header and file errors ---\n";
    const int W = 4, H = 3, C = 1, NL = 3, N = 4, PIX = W * H * C;
    auto good = [&] { return write_bin_file("e_good", W, H, C, {"a", "b", "c"}, make_recs(N, PIX, NL)); };

    {
        std::string p = good();
        auto d = make_bin(p, N, PIX, NL, 2);
        CHECK_EQ(d->get_train_size(), N, "control: the untouched file loads without errors");
    }
    {
        std::string p = good(); patch_byte(p, 0, 'X');
        CHECK_THROWS(make_bin(p, N, PIX, NL, 2), std::runtime_error, "bad .bin header", "bin: wrong signature");
    }
    {
        std::string p = good(); patch_byte(p, 4, 2);
        CHECK_THROWS(make_bin(p, N, PIX, NL, 2), std::runtime_error, "bad .bin header", "bin: unsupported version");
    }
    {
        std::string p = write_raw_bin("e_empty", "");
        CHECK_THROWS(make_bin(p, N, PIX, NL, 2), std::runtime_error, "bad .bin header", "bin: empty file");
    }
    {   // header without the sample count
        std::string b = std::string("IMGS") + std::string("\x01", 1) +
                        std::string("\x04\x00\x03\x00\x01\x01\x00\x01\x00" "a", 10);
        std::string p = write_raw_bin("e_nocount", b);
        CHECK_THROWS(make_bin(p, 1, PIX, 1, 1), std::runtime_error, "truncated .bin header",
                     "bin: header cut before the sample count");
    }
    {
        std::string p = good();
        CHECK_THROWS(make_bin(p, N, PIX + 1, NL, 2), std::runtime_error, "does not match the network",
                     "bin: w*h*c differs from the network input size");
        CHECK_THROWS(make_bin(p, N, PIX, NL + 1, 2), std::runtime_error, "does not match the network",
                     "bin: label count differs from output_nodes");
    }
    {
        std::string err = capture_cerr([] {
            try { make_bin("data/does_not_exist.bin", 1, 12, 3, 1); } catch (const std::runtime_error&) {}
        });
        CHECK(has(err, "Could not open file"), "bin: missing file message is written to cerr");
        CHECK_THROWS(make_bin("data/does_not_exist.bin", 1, 12, 3, 1), std::runtime_error, "Error opening file.",
                     "bin: missing file throws runtime_error");
    }
    {
        std::string p = good();
        std::string up = "_tmp_e_good.BIN";
        { std::ifstream in(p, std::ios::binary); std::ofstream out(up, std::ios::binary | std::ios::trunc); out << in.rdbuf(); }
        tmp_files.push_back(up);
        CHECK_THROWS(make_bin(up, N, PIX, NL, 2), std::runtime_error, "Unsupported file format",
                     "bin: extension check is case-sensitive (.BIN rejected)");
    }
}

static void test_bin_split_and_csv()
{
    std::cout << "\n--- .bin: train/val split and agreement with csv ---\n";
    const int W = 4, H = 3, C = 1, NL = 3, PIX = W * H * C;

    {
        const int N = 20;
        auto recs = make_recs(N, PIX, NL);
        std::string p = write_bin_file("split", W, H, C, {"a", "b", "c"}, recs);
        auto d = make_bin(p, N, PIX, NL, 4, 0.25f, 7u);
        CHECK(d->get_val_size() == 5 && d->get_train_size() == 15, "bin split 0.25: 15 train + 5 val");
        Rows tr = rows_of(*d, PIX, NL);
        d->use_val();
        Rows va = rows_of(*d, PIX, NL);
        d->use_train();
        std::multiset<std::vector<float>> got(tr.begin(), tr.end()), expected;
        got.insert(va.begin(), va.end());
        for (int i = 0; i < N; i++) expected.insert(bin_expected_row(recs[i], W, H, C, NL));
        CHECK(got == expected, "bin split: train + val cover every record exactly once");
        auto d2 = make_bin(p, N, PIX, NL, 4, 0.25f, 7u);
        CHECK(snap(*d, PIX, NL) == snap(*d2, PIX, NL), "bin split: same seed gives the same split");
    }
    {   // the same samples stored as csv and as bin give the same batches
        const int N = 5;
        auto recs = make_recs(N, PIX, NL);
        std::string p = write_bin_file("xfmt", W, H, C, {"a", "b", "c"}, recs);
        std::string text = "h\n";
        for (auto& r : recs) {
            text += std::to_string(r.label);
            for (int k = 0; k < PIX; k++) {
                char buf[32];
                std::snprintf(buf, sizeof buf, ",%.9g", r.px[k] / 255.0f);
                text += buf;
            }
            text += "\n";
        }
        std::string csv = write_tmp("xfmt", text);
        auto dc = make_bin(csv, N, PIX, NL, N);
        auto db = make_bin(p, N, PIX, NL, N);
        bool same = true;
        for (int j = 0; j < N; j++)
            same = same && near_vec(sample_row(*dc, j, PIX, NL), sample_row(*db, j, PIX, NL), 1e-6f);
        CHECK(same, "csv and bin with the same samples give the same batches (single channel)");
    }
}

static void test_crash_cases()
{
    std::cout << "\n--- crash / UB cases (isolated processes) ---\n";

    int c1 = run_isolated("next_batch_uninit_num_batches");
    std::cout << "[INFO] next_batch() with uninitialized num_batches: " << describe_exit(c1) << "\n";

    int c2 = run_isolated("get_batch_negative_index");
    std::cout << "[INFO] get_batch(-1): " << describe_exit(c2) << "\n";

    int c3 = run_isolated("batch_size_zero");
    std::cout << "[INFO] batch_size == 0: " << describe_exit(c3) << "\n";
    CHECK(c3 != 0, "batch_size == 0 does not terminate normally (division by zero or CUDA error)");

    int c4 = run_isolated("copy_double_free");
    std::cout << "[INFO] copy of DatasetFile: " << describe_exit(c4) << "\n";
}

int main(int argc, char** argv)
{
    exe_path = argv[0];
    register_isolated_cases();

    if (argc >= 3 && std::string(argv[1]) == "--case") {
        auto it = isolated_cases().find(argv[2]);
        if (it == isolated_cases().end()) return 64;
        try { return it->second(); }
        catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 2; }
    }

    try {
        test_split_and_parse_val_fraction();
        test_constructor_and_getters();
        test_constructor_errors();
        test_batches_content();
        test_shuffle_and_seed();
        test_train_val_modes();
        test_csv_parsing();
        test_bin_loading();
        test_bin_size_and_records();
        test_bin_errors();
        test_bin_split_and_csv();
        test_crash_cases();
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] unexpected exception: " << e.what() << "\n";
        tests_failed++;
    }

    for (auto& p : tmp_files) std::remove(p.c_str());

    std::cout << "\n=== " << tests_run << " tests run, " << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}