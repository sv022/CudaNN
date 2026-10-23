#include "../../network/file/file.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

static const int N_CLASSES = 4;
static const int IMG_N = 10;
static const int SRC_W = 4, SRC_H = 4;
static const int IN_RGB4 = 3 * SRC_H * SRC_W;
static const ImageShape SHAPE_RGB4(3, SRC_H, SRC_W);
static const char* const CLASS_DIRS[N_CLASSES] = { "helicopter", "plane", "ship", "tank" };

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

struct Snap {
    std::vector<float> img, tgt;
    int n = 0;
    bool operator==(const Snap& o) const { return n == o.n && img == o.img && tgt == o.tgt; }
    bool operator!=(const Snap& o) const { return !(*this == o); }
};

static std::vector<std::string> tmp_files;
static std::vector<std::string> tmp_dirs;

// ---------------- filesystem helpers (no std::filesystem) ----------------

static bool make_dir(const std::string& p)
{
#ifdef _WIN32
    return _mkdir(p.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(p.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

static void remove_dir(const std::string& p)
{
#ifdef _WIN32
    _rmdir(p.c_str());
#else
    rmdir(p.c_str());
#endif
}

static std::string make_img_dir(const std::string& name)
{
    std::string d = "_tmp_img_" + name;
    make_dir(d);
    tmp_dirs.push_back(d);
    return d;
}

static std::string write_file(const std::string& path, const std::string& content)
{
    std::ofstream f(path, std::ios::binary);
    f << content;
    tmp_files.push_back(path);
    return path;
}

static void write_list(const std::string& dir, const std::string& content)
{
    write_file(dir + "/labels.txt", content);
}

static std::vector<unsigned char> solid(int w, int h, int r, int g, int b)
{
    std::vector<unsigned char> px((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; i++) {
        px[3 * i + 0] = (unsigned char)r;
        px[3 * i + 1] = (unsigned char)g;
        px[3 * i + 2] = (unsigned char)b;
    }
    return px;
}

static std::string ppm_bytes(int w, int h, const std::vector<unsigned char>& rgb)
{
    std::ostringstream hdr;
    hdr << "P6\n" << w << " " << h << "\n255\n";
    std::string body = hdr.str();
    body.append((const char*)rgb.data(), rgb.size());
    return body;
}

static std::string add_ppm(const std::string& dir, const std::string& sub, const std::string& name,
                           int w, int h, const std::vector<unsigned char>& rgb)
{
    std::string folder = dir;
    if (!sub.empty()) {
        folder += "/" + sub;
        make_dir(folder);
        tmp_dirs.push_back(folder);
    }
    return write_file(folder + "/" + name, ppm_bytes(w, h, rgb));
}

// n images; image i is solid (v, v+1, v+2) with v = 10 * (i + 1); label = i % N_CLASSES.
// labels.txt uses backslashes and CRLF, as in the original Windows data set.
static std::string build_main_dir(const std::string& name, int n = IMG_N, int w = SRC_W, int h = SRC_H)
{
    std::string dir = make_img_dir(name);
    std::string list;
    for (int i = 0; i < n; i++) {
        int cls = i % N_CLASSES;
        int v = 10 * (i + 1);
        std::string file = "img_" + std::to_string(i) + ".ppm";
        add_ppm(dir, CLASS_DIRS[cls], file, w, h, solid(w, h, v, v + 1, v + 2));
        list += std::string(CLASS_DIRS[cls]) + "\\" + file + " " + std::to_string(cls) + "\r\n";
    }
    write_list(dir, list);
    return dir;
}

static std::unique_ptr<DatasetFile> make_img(const std::string& dir, int n, const ImageShape& shape,
                                             int bs, float vf = 0.f, unsigned seed = 12345u)
{
    return std::unique_ptr<DatasetFile>(new DatasetFile(dir, n, shape, N_CLASSES, bs, vf, seed));
}

// ---------------- sample helpers ----------------

static bool near_f(float a, float b, float eps = 1e-6f) { return std::fabs(a - b) < eps; }

static bool vec_near(const std::vector<float>& a, const std::vector<float>& b, float eps = 1e-6f)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) if (!near_f(a[i], b[i], eps)) return false;
    return true;
}

static Snap snap(const DatasetFile& d, int in, int out = N_CLASSES)
{
    Snap s;
    s.n = d.current_batch_size;
    if (s.n > 0) {
        s.img.assign(d.image_batch, d.image_batch + (size_t)s.n * in);
        s.tgt.assign(d.target_batch, d.target_batch + (size_t)s.n * out);
    }
    return s;
}

static std::vector<float> sample_row(const DatasetFile& d, int j, int in, int out = N_CLASSES)
{
    std::vector<float> r(d.image_batch + (size_t)j * in, d.image_batch + (size_t)(j + 1) * in);
    r.insert(r.end(), d.target_batch + (size_t)j * out, d.target_batch + (size_t)(j + 1) * out);
    return r;
}

static Rows epoch_rows(DatasetFile& d, int in)
{
    Rows rows;
    for (int b = 0; b < d.get_num_batches(); b++) {
        d.get_batch(b);
        for (int j = 0; j < d.current_batch_size; j++) rows.push_back(sample_row(d, j, in));
    }
    return rows;
}

// reference row of image i from build_main_dir (3 x SRC_H x SRC_W, CHW, /255, one-hot)
static std::vector<float> ref_img_row(int i)
{
    int v = 10 * (i + 1);
    std::vector<float> r;
    for (int ch = 0; ch < 3; ch++) r.insert(r.end(), (size_t)SRC_H * SRC_W, (float)(v + ch) / 255.0f);
    std::vector<float> t(N_CLASSES, 0.f);
    t[i % N_CLASSES] = 1.f;
    r.insert(r.end(), t.begin(), t.end());
    return r;
}

static int id_of(const std::vector<float>& row) { return (int)std::lround(row[0] * 255.0f) / 10 - 1; }

static std::vector<int> ids_of(const Rows& rows)
{
    std::vector<int> ids;
    for (auto& r : rows) ids.push_back(id_of(r));
    return ids;
}

static std::vector<int> sorted_ids(std::vector<int> v)
{
    std::sort(v.begin(), v.end());
    return v;
}

static std::vector<int> range_ids(int n)
{
    std::vector<int> v(n);
    for (int i = 0; i < n; i++) v[i] = i;
    return v;
}

static bool onehot_matches_id(const Rows& rows, int in)
{
    for (auto& row : rows) {
        int hot = -1, ones = 0;
        for (int k = 0; k < N_CLASSES; k++) {
            float t = row[in + k];
            if (t == 1.f) { hot = k; ones++; }
            else if (t != 0.f) return false;
        }
        if (ones != 1 || hot != id_of(row) % N_CLASSES) return false;
    }
    return true;
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

// ---------------- test groups ----------------

static void test_image_shape()
{
    std::cout << "\n--- ImageShape ---\n";

    ImageShape flat(144);
    CHECK(!flat.is_image() && flat.numel() == 144, "flat shape: numel == value, is_image() == false");

    ImageShape img(3, 12, 12);
    CHECK(img.is_image() && img.numel() == 432, "image shape 3x12x12: numel == 432, is_image() == true");

    ImageShape implicit = 7;
    CHECK(implicit.numel() == 7 && !implicit.is_image(), "implicit conversion from int gives a flat shape");

    ImageShape gray(1, 28, 28);
    CHECK_EQ(gray.numel(), 784, "image shape 1x28x28: numel == 784");
}

static void test_resize_to_chw()
{
    std::cout << "\n--- resize_to_chw ---\n";

    {
        std::vector<unsigned char> src(5 * 3, 100);
        std::vector<float> dst(2 * 2);
        resize_to_chw(src.data(), 5, 3, 1, dst.data(), 2, 2);
        bool ok = true;
        for (float v : dst) ok = ok && near_f(v, 100.0f / 255.0f);
        CHECK(ok, "uniform image stays uniform after downscale");
    }
    {
        unsigned char src[4] = { 0, 100, 200, 60 };
        float dst[1];
        resize_to_chw(src, 2, 2, 1, dst, 1, 1);
        CHECK(near_f(dst[0], 90.0f / 255.0f), "2x2 -> 1x1 gives the mean of four pixels");
    }
    {
        unsigned char src[8] = { 0, 100, 200, 40, 20, 120, 220, 60 };
        float dst[2];
        resize_to_chw(src, 4, 2, 1, dst, 2, 1);
        CHECK(near_f(dst[0], 60.0f / 255.0f) && near_f(dst[1], 130.0f / 255.0f),
              "4x2 -> 2x1 averages 2x2 blocks");
    }
    {
        unsigned char src[3] = { 30, 60, 90 };
        float dst[2];
        resize_to_chw(src, 3, 1, 1, dst, 2, 1);
        CHECK(near_f(dst[0], 40.0f / 255.0f) && near_f(dst[1], 80.0f / 255.0f),
              "3 -> 2 uses fractional box weights (2/3, 1/3) and (1/3, 2/3)");
    }
    {
        unsigned char src[2] = { 0, 100 };
        float dst[4];
        resize_to_chw(src, 2, 1, 1, dst, 4, 1);
        CHECK(near_f(dst[0], 0.0f) && near_f(dst[1], 25.0f / 255.0f) &&
              near_f(dst[2], 75.0f / 255.0f) && near_f(dst[3], 100.0f / 255.0f),
              "2 -> 4 upscale is linear with pixel-centre alignment (0, 25, 75, 100)");
    }
    {
        unsigned char src[1] = { 77 };
        std::vector<float> dst(3 * 3);
        resize_to_chw(src, 1, 1, 1, dst.data(), 3, 3);
        bool ok = true;
        for (float v : dst) ok = ok && near_f(v, 77.0f / 255.0f);
        CHECK(ok, "1x1 -> 3x3 upscale replicates the pixel");
    }
    {
        unsigned char src[6] = { 10, 20, 30, 40, 50, 60 };
        float dst[6];
        resize_to_chw(src, 2, 1, 3, dst, 2, 1);
        const float expected[6] = { 10, 40, 20, 50, 30, 60 };
        bool ok = true;
        for (int i = 0; i < 6; i++) ok = ok && near_f(dst[i], expected[i] / 255.0f);
        CHECK(ok, "same size: HWC -> CHW layout and division by 255");
    }
    {
        unsigned char src[12] = { 0, 100, 200,  40, 140, 240,  80, 180, 20,  120, 220, 60 };
        float dst[3];
        resize_to_chw(src, 2, 2, 3, dst, 1, 1);
        CHECK(near_f(dst[0], 60.0f / 255.0f) && near_f(dst[1], 160.0f / 255.0f) && near_f(dst[2], 130.0f / 255.0f),
              "2x2 -> 1x1 with 3 channels: channels are averaged independently");
    }
}

static void test_image_constructor_and_getters()
{
    std::cout << "\n--- image dataset: constructor / getters ---\n";

    std::string dir = build_main_dir("ctor");

    auto d = make_img(dir, IMG_N, SHAPE_RGB4, 4);
    CHECK_EQ(d->get_train_size(), 10, "no split: train_size == 10");
    CHECK_EQ(d->get_val_size(), 0, "no split: val_size == 0");
    CHECK_EQ(d->get_train_num_batches(), 3, "no split: train_num_batches == ceil(10/4) == 3");
    CHECK_EQ(d->get_val_num_batches(), 0, "no split: val_num_batches == 0");
    CHECK_EQ(d->get_num_batches(), 3, "no split: get_num_batches() == 3");
    CHECK(!d->has_val(), "no split: has_val() == false");
    CHECK_EQ(d->get_batch_size(), 4, "get_batch_size() == 4");
    CHECK_EQ(d->get_input_nodes(), IN_RGB4, "get_input_nodes() == C*H*W == 48");
    CHECK_EQ(d->get_output_nodes(), N_CLASSES, "get_output_nodes() == 4");
    CHECK_EQ(d->current_batch_size, 4, "constructor loads batch 0 (reset)");

    auto v = make_img(dir, IMG_N, SHAPE_RGB4, 4, 0.2f);
    CHECK(v->get_val_size() == 2 && v->get_train_size() == 8 && v->has_val(), "split 0.2: train 8, val 2");
    CHECK(v->get_train_num_batches() == 2 && v->get_val_num_batches() == 1, "split 0.2: 2 train batches, 1 val batch");
    CHECK_EQ(v->get_num_batches(), v->get_train_num_batches(), "split 0.2: initial mode is train");

    auto part = make_img(dir, 6, SHAPE_RGB4, 4);
    CHECK_EQ(part->get_train_size(), 6, "size smaller than the list: only the first 6 valid rows are used");

    CHECK_THROWS(make_img(dir, IMG_N + 1, SHAPE_RGB4, 4), std::runtime_error,
        "contains 10 valid rows, expected 11",
        "size larger than the number of valid rows throws runtime_error");

    CHECK_THROWS(std::unique_ptr<DatasetFile>(new DatasetFile(dir, 1, 48, N_CLASSES, 4)), std::invalid_argument,
        "requires ImageShape",
        "directory with a flat shape throws invalid_argument");
    CHECK_THROWS(make_img(dir, 1, ImageShape(0, 4, 4), 4), std::invalid_argument,
        "channels must be in [1, 4]", "0 channels throws invalid_argument");
    CHECK_THROWS(make_img(dir, 1, ImageShape(5, 4, 4), 4), std::invalid_argument,
        "channels must be in [1, 4]", "5 channels throws invalid_argument");
    CHECK_THROWS(make_img(dir, 1, SHAPE_RGB4, 0), std::invalid_argument,
        "batch_size must be positive", "batch_size 0 throws invalid_argument in directory mode");
    CHECK_THROWS(make_img(dir, -1, SHAPE_RGB4, 4), std::invalid_argument,
        "size can not be negative", "negative size throws invalid_argument in directory mode");

    std::string nolist = make_img_dir("nolist");
    std::string err = capture_cerr([&] {
        try { make_img(nolist, 1, SHAPE_RGB4, 4); } catch (const std::runtime_error&) {}
    });
    CHECK(has(err, "Could not open file"), "directory without labels.txt: message written to cerr");
    CHECK_THROWS(make_img(nolist, 1, SHAPE_RGB4, 4), std::runtime_error, "Error opening file.",
        "directory without labels.txt throws runtime_error");

    CHECK_THROWS(make_img("_tmp_img_does_not_exist", 1, SHAPE_RGB4, 4), std::runtime_error,
        "Unsupported file format", "non-existent path is still reported as unsupported format");
    CHECK_THROWS(make_img("data/file.txt", 1, SHAPE_RGB4, 4), std::runtime_error,
        "Unsupported file format", ".txt file is still reported as unsupported format");
}

static void test_image_labels_scan()
{
    std::cout << "\n--- image dataset: labels.txt scan ---\n";

    std::string dir = make_img_dir("scan");
    add_ppm(dir, "plane", "a.ppm", 2, 2, solid(2, 2, 10, 10, 10));
    add_ppm(dir, "ship", "b b.ppm", 2, 2, solid(2, 2, 20, 20, 20));
    write_file(dir + "/ship/bad.ppm", "this is not an image");

    ImageShape shape(3, 2, 2);
    auto mk = [&](int n) { return std::unique_ptr<DatasetFile>(new DatasetFile(dir, n, shape, N_CLASSES, 2)); };

    std::unique_ptr<DatasetFile> d;
    std::string err;

    write_list(dir, "plane\\a.ppm 0\r\nship\\b b.ppm 1\r\n");
    err = capture_cerr([&] { d = mk(2); });
    CHECK(d->get_train_size() == 2 && !has(err, "Warning"),
        "backslashes, CRLF and a space in the file name are accepted");
    CHECK(d->target_batch[0] == 1.f && d->target_batch[N_CLASSES + 1] == 1.f,
        "labels of both rows are read (0 and 1)");
    CHECK(near_f(d->image_batch[0], 10.0f / 255.0f) && near_f(d->image_batch[12], 20.0f / 255.0f),
        "images are matched to their rows (including the name with a space)");

    write_list(dir, "plane/a.ppm 0\nship/b b.ppm 1\n");
    err = capture_cerr([&] { d = mk(2); });
    CHECK(d->get_train_size() == 2 && !has(err, "Warning"), "forward slashes are accepted");

    write_list(dir, "plane\\a.ppm\t0\nship\\b b.ppm 1   \n");
    err = capture_cerr([&] { d = mk(2); });
    CHECK(d->get_train_size() == 2 && !has(err, "Warning"), "tab separator and trailing spaces are accepted");

    write_list(dir, "\xEF\xBB\xBF" "plane\\a.ppm 0\n");
    err = capture_cerr([&] { d = mk(1); });
    CHECK(!has(err, "Warning") && d->get_train_size() == 1, "UTF-8 BOM in the first line is ignored");

    write_list(dir, "plane\\a.ppm 9\nplane\\a.ppm -1\nplane\\a.ppm 4\nplane\\a.ppm 3.5\nplane\\a.ppm abc\n\nbroken\nplane\\a.ppm 3\n");
    err = capture_cerr([&] { d = mk(1); });
    CHECK(!has(err, "Warning"), "bad labels, blank and malformed lines are skipped silently");
    CHECK(d->get_train_size() == 1 && d->target_batch[3] == 1.f, "the first valid row (label 3 == output_nodes - 1) is loaded");

    write_list(dir, "\nplane\\nope.ppm 0\nplane\\a.ppm 0\n");
    err = capture_cerr([&] { d = mk(1); });
    CHECK_EQ(count_substr(err, "Warning: line 2 image can not be read"), 1,
        "missing image: warning uses the real line number of labels.txt");
    CHECK(d->get_train_size() == 1 && near_f(d->image_batch[0], 10.0f / 255.0f), "row after the missing image is loaded");

    write_list(dir, "ship\\bad.ppm 0\nplane\\a.ppm 0\n");
    err = capture_cerr([&] { d = mk(1); });
    CHECK_EQ(count_substr(err, "Warning: line 1 image can not be read"), 1, "file that is not an image gives a warning");

    write_list(dir, "plane\\a.ppm 0\nship\\b b.ppm 1\nplane\\a.ppm 2\n");
    d = mk(2);
    CHECK_EQ(d->get_train_size(), 2, "loading stops after size valid rows");
    CHECK_THROWS(mk(4), std::runtime_error, "contains 3 valid rows, expected 4",
        "fewer valid rows than size throws runtime_error with found/expected counts");

    write_list(dir, "plane\\a.ppm 9\nplane\\nope.ppm 0\nplane\\a.ppm 0\n");
    CHECK_THROWS(mk(2), std::runtime_error, "contains 1 valid rows, expected 2",
        "skipped rows are not counted as valid");

    write_list(dir, "");
    CHECK_THROWS(mk(1), std::runtime_error, "contains 0 valid rows, expected 1",
        "empty labels.txt with size > 0 throws runtime_error");

    write_list(dir, "plane\\a.ppm 0\r\n");
    d = mk(1);
    CHECK_EQ(d->get_train_size(), 1, "exact row count constructs normally");
}

static void test_image_batches_content()
{
    std::cout << "\n--- image dataset: batches content and order ---\n";

    std::string dir = build_main_dir("batches");
    auto d = make_img(dir, IMG_N, SHAPE_RGB4, 4);

    bool order_ok = true;
    for (int b = 0; b < d->get_num_batches(); b++) {
        d->get_batch(b);
        for (int j = 0; j < d->current_batch_size; j++)
            order_ok = order_ok && vec_near(sample_row(*d, j, IN_RGB4), ref_img_row(b * 4 + j));
    }
    CHECK(order_ok, "without shuffle, batches follow labels.txt order and match reference rows (CHW, /255, one-hot)");

    d->reset();
    CHECK(near_f(d->image_batch[0], 10.0f / 255.0f) && near_f(d->image_batch[SRC_H * SRC_W - 1], 10.0f / 255.0f),
        "channel 0 plane (R) occupies the first H*W values");
    CHECK(near_f(d->image_batch[SRC_H * SRC_W], 11.0f / 255.0f) && near_f(d->image_batch[2 * SRC_H * SRC_W - 1], 11.0f / 255.0f),
        "channel 1 plane (G) follows the R plane");
    CHECK(near_f(d->image_batch[2 * SRC_H * SRC_W], 12.0f / 255.0f) && near_f(d->image_batch[IN_RGB4 - 1], 12.0f / 255.0f),
        "channel 2 plane (B) is the last one");

    bool range_ok = true;
    for (auto& row : epoch_rows(*d, IN_RGB4))
        for (int i = 0; i < IN_RGB4; i++) range_ok = range_ok && row[i] >= 0.f && row[i] <= 1.f;
    CHECK(range_ok, "all pixel values are in [0, 1]");

    CHECK(onehot_matches_id(epoch_rows(*d, IN_RGB4), IN_RGB4), "all targets are one-hot and match the labels.txt label");

    d->get_batch(2);
    CHECK_EQ(d->current_batch_size, 2, "last batch is partial (2 of 4)");
    d->get_batch(1);
    CHECK_EQ(d->current_batch_size, 4, "batch 1 has full size");

    d->get_batch(0);
    Snap s0 = snap(*d, IN_RGB4);
    d->get_batch(2);
    d->reset();
    CHECK(snap(*d, IN_RGB4) == s0, "reset() returns to batch 0 with identical data");

    d->next_batch();
    Snap s1 = snap(*d, IN_RGB4);
    d->next_batch();
    d->next_batch();
    CHECK(snap(*d, IN_RGB4) == s0 && s1 != s0, "next_batch() wraps around to batch 0 after the last batch");

    Rows rows = epoch_rows(*d, IN_RGB4);
    CHECK_EQ((int)rows.size(), IMG_N, "full epoch yields 10 samples");
    CHECK(sorted_ids(ids_of(rows)) == range_ids(IMG_N), "full epoch covers every image exactly once");

    auto big = make_img(dir, IMG_N, SHAPE_RGB4, 16);
    CHECK(big->get_train_num_batches() == 1 && big->current_batch_size == IMG_N,
        "batch_size > dataset size: 1 batch with 10 samples");
}

static void test_image_resize_and_channels()
{
    std::cout << "\n--- image dataset: resize and channels ---\n";

    {
        std::string dir = build_main_dir("big", 3, 8, 8);
        auto d = make_img(dir, 3, SHAPE_RGB4, 3);
        bool ok = true;
        for (int j = 0; j < 3; j++) ok = ok && vec_near(sample_row(*d, j, IN_RGB4), ref_img_row(j));
        CHECK(ok, "8x8 sources are downscaled to 4x4 (uniform images keep their values)");
    }
    {
        std::string dir = build_main_dir("small", 3, 2, 2);
        auto d = make_img(dir, 3, SHAPE_RGB4, 3);
        bool ok = true;
        for (int j = 0; j < 3; j++) ok = ok && vec_near(sample_row(*d, j, IN_RGB4), ref_img_row(j));
        CHECK(ok, "2x2 sources are upscaled to 4x4 (uniform images keep their values)");
    }
    {
        std::string dir = build_main_dir("wide", 3, 8, 2);
        auto d = make_img(dir, 3, SHAPE_RGB4, 3);
        bool ok = true;
        for (int j = 0; j < 3; j++) ok = ok && vec_near(sample_row(*d, j, IN_RGB4), ref_img_row(j));
        CHECK(ok, "non-square 8x2 sources are stretched to 4x4");
    }
    {
        std::string dir = make_img_dir("mixed");
        std::string list;
        const int sizes[3][2] = { { 2, 2 }, { 8, 8 }, { 6, 3 } };
        for (int i = 0; i < 3; i++) {
            int cls = i % N_CLASSES, v = 10 * (i + 1);
            std::string file = "img_" + std::to_string(i) + ".ppm";
            add_ppm(dir, CLASS_DIRS[cls], file, sizes[i][0], sizes[i][1],
                    solid(sizes[i][0], sizes[i][1], v, v + 1, v + 2));
            list += std::string(CLASS_DIRS[cls]) + "\\" + file + " " + std::to_string(cls) + "\n";
        }
        write_list(dir, list);
        auto d = make_img(dir, 3, SHAPE_RGB4, 3);
        bool ok = true;
        for (int j = 0; j < 3; j++) ok = ok && vec_near(sample_row(*d, j, IN_RGB4), ref_img_row(j));
        CHECK(ok, "images of different sizes in one data set all get the network input size");
    }
    {
        std::string dir = make_img_dir("grad");
        std::vector<unsigned char> px = { 0, 0, 0, 200, 200, 200 };
        add_ppm(dir, "", "g.ppm", 2, 1, px);
        write_list(dir, "g.ppm 0\n");
        auto d1 = make_img(dir, 1, ImageShape(1, 1, 1), 1);
        CHECK(near_f(d1->image_batch[0], 100.0f / 255.0f), "gradient 0..200 -> 1x1 gray gives 100/255");
        auto d3 = make_img(dir, 1, ImageShape(3, 1, 1), 1);
        CHECK(near_f(d3->image_batch[0], 100.0f / 255.0f) && near_f(d3->image_batch[1], 100.0f / 255.0f) &&
              near_f(d3->image_batch[2], 100.0f / 255.0f), "gradient 0..200 -> 1x1 RGB gives 100/255 in every channel");
        auto dw = make_img(dir, 1, ImageShape(1, 1, 4), 1);
        CHECK(near_f(dw->image_batch[0], 0.0f) && near_f(dw->image_batch[1], 50.0f / 255.0f) &&
              near_f(dw->image_batch[2], 150.0f / 255.0f) && near_f(dw->image_batch[3], 200.0f / 255.0f),
            "gradient 0..200 upscaled 2 -> 4 gives 0, 50, 150, 200 (/255)");
    }
    {
        std::string dir = make_img_dir("gray");
        std::string list;
        for (int i = 0; i < 2; i++) {
            int v = 40 * (i + 1);
            add_ppm(dir, "", "g" + std::to_string(i) + ".ppm", SRC_W, SRC_H, solid(SRC_W, SRC_H, v, v, v));
            list += "g" + std::to_string(i) + ".ppm " + std::to_string(i) + "\n";
        }
        write_list(dir, list);
        auto d = make_img(dir, 2, ImageShape(1, SRC_H, SRC_W), 2);
        CHECK_EQ(d->get_input_nodes(), SRC_H * SRC_W, "1 channel: input_nodes == H*W");
        bool ok = true;
        for (int i = 0; i < SRC_H * SRC_W; i++)
            ok = ok && near_f(d->image_batch[i], 40.0f / 255.0f) && near_f(d->image_batch[SRC_H * SRC_W + i], 80.0f / 255.0f);
        CHECK(ok, "1 channel: a gray RGB image keeps its value after conversion");
    }
    {
        std::string dir = build_main_dir("rgba", 1);
        auto d = make_img(dir, 1, ImageShape(4, SRC_H, SRC_W), 1);
        bool ok = true;
        for (int i = 0; i < SRC_H * SRC_W; i++) ok = ok && near_f(d->image_batch[3 * SRC_H * SRC_W + i], 1.0f);
        CHECK(ok, "4 channels: the alpha plane of an opaque image is 1.0");
        CHECK(near_f(d->image_batch[0], 10.0f / 255.0f) && near_f(d->image_batch[SRC_H * SRC_W], 11.0f / 255.0f),
            "4 channels: RGB planes keep their values");
    }
}

static void test_image_lazy_loading()
{
    std::cout << "\n--- image dataset: lazy loading and decode errors ---\n";

    {
        std::string dir = make_img_dir("lazy");
        std::string path = add_ppm(dir, "", "a.ppm", 2, 2, solid(2, 2, 10, 10, 10));
        write_list(dir, "a.ppm 0\n");
        auto d = make_img(dir, 1, ImageShape(3, 2, 2), 1);
        CHECK(near_f(d->image_batch[0], 10.0f / 255.0f), "image is decoded on construction (batch 0)");

        add_ppm(dir, "", "a.ppm", 2, 2, solid(2, 2, 50, 50, 50));
        d->reset();
        CHECK(near_f(d->image_batch[0], 50.0f / 255.0f), "files are decoded on every get_batch, not cached at construction");

        std::remove(path.c_str());
        CHECK_THROWS(d->reset(), std::runtime_error, "failed to decode image",
            "file removed after construction: get_batch throws runtime_error");
    }
    {
        std::string dir = make_img_dir("trunc");
        add_ppm(dir, "", "ok.ppm", 4, 4, solid(4, 4, 10, 10, 10));
        write_file(dir + "/t.ppm", std::string("P6\n4 4\n255\n") + "abc");
        write_list(dir, "ok.ppm 0\nt.ppm 1\n");

        std::string err;
        std::unique_ptr<DatasetFile> d;
        err = capture_cerr([&] { d = make_img(dir, 2, ImageShape(3, 4, 4), 1); });
        CHECK(d != nullptr && !has(err, "Warning"),
            "truncated image passes the header-only scan (no warning at construction)");

        d->reset();
        Snap s0 = snap(*d, 48);
        CHECK_THROWS(d->get_batch(1), std::runtime_error, "t.ppm",
            "truncated image: get_batch throws runtime_error that names the file");
        d->reset();
        CHECK(snap(*d, 48) == s0, "data set stays usable after a failed get_batch");

        CHECK_THROWS(make_img(dir, 2, ImageShape(3, 4, 4), 2), std::runtime_error, "failed to decode image",
            "decode error inside the constructor (batch 0 contains the bad file) throws runtime_error");
    }
}

static void test_image_train_val_shuffle()
{
    std::cout << "\n--- image dataset: train / val / shuffle ---\n";

    std::string dir = build_main_dir("split");
    const std::vector<int> all = range_ids(IMG_N);

    auto plain = make_img(dir, IMG_N, SHAPE_RGB4, 4);
    CHECK_THROWS(plain->use_val(), std::runtime_error, "constructed without a validation split",
        "use_val() without split throws runtime_error");
    CHECK(ids_of(epoch_rows(*plain, IN_RGB4)) == all, "val_fraction == 0: images are in labels.txt order");

    auto d = make_img(dir, IMG_N, SHAPE_RGB4, 4, 0.2f);
    Rows tr = epoch_rows(*d, IN_RGB4);
    d->use_val();
    CHECK(d->get_num_batches() == d->get_val_num_batches(), "use_val() switches num_batches to val");
    CHECK_EQ(d->current_batch_size, 2, "val batch 0 has size 2");
    Rows va = epoch_rows(*d, IN_RGB4);
    d->use_train();
    CHECK_EQ(d->get_num_batches(), d->get_train_num_batches(), "use_train() switches back");
    CHECK(tr.size() == 8 && va.size() == 2, "train epoch has 8 samples, val epoch has 2");

    std::vector<int> joined = ids_of(tr);
    std::vector<int> val_ids = ids_of(va);
    joined.insert(joined.end(), val_ids.begin(), val_ids.end());
    CHECK(sorted_ids(joined) == all, "train and val are disjoint and cover every image exactly once");
    CHECK(onehot_matches_id(tr, IN_RGB4) && onehot_matches_id(va, IN_RGB4), "targets match images in both train and val");

    d->get_batch(1);
    CHECK_EQ(d->current_batch_size, 4, "last train batch has 4 samples (no leak from val)");

    auto s1 = make_img(dir, IMG_N, SHAPE_RGB4, 4, 0.2f, 1u);
    auto s2 = make_img(dir, IMG_N, SHAPE_RGB4, 4, 0.2f, 2u);
    auto s3 = make_img(dir, IMG_N, SHAPE_RGB4, 4, 0.2f, 1u);
    CHECK(ids_of(epoch_rows(*s1, IN_RGB4)) == ids_of(epoch_rows(*s3, IN_RGB4)), "same seed gives the same split");
    CHECK(ids_of(epoch_rows(*s1, IN_RGB4)) != ids_of(epoch_rows(*s2, IN_RGB4)), "different seed gives a different split");

    auto sh = make_img(dir, IMG_N, SHAPE_RGB4, 4);
    std::vector<int> before = ids_of(epoch_rows(*sh, IN_RGB4));
    sh->shuffle();
    Rows after_rows = epoch_rows(*sh, IN_RGB4);
    CHECK(sorted_ids(ids_of(after_rows)) == all, "shuffle() keeps the same set of images");
    CHECK(ids_of(after_rows) != before, "shuffle() changes the order");
    CHECK(onehot_matches_id(after_rows, IN_RGB4), "after shuffle(), every target still matches its image");

    auto a = make_img(dir, IMG_N, SHAPE_RGB4, 4, 0.f, 777u);
    auto b = make_img(dir, IMG_N, SHAPE_RGB4, 4, 0.f, 777u);
    a->shuffle(); b->shuffle();
    bool first = ids_of(epoch_rows(*a, IN_RGB4)) == ids_of(epoch_rows(*b, IN_RGB4));
    a->shuffle(); b->shuffle();
    bool second = ids_of(epoch_rows(*a, IN_RGB4)) == ids_of(epoch_rows(*b, IN_RGB4));
    CHECK(first && second, "same seed gives identical order across consecutive shuffles");

    auto sv = make_img(dir, IMG_N, SHAPE_RGB4, 4, 0.2f);
    std::vector<int> train_before = ids_of(epoch_rows(*sv, IN_RGB4));
    sv->use_val();
    std::vector<int> val_before = ids_of(epoch_rows(*sv, IN_RGB4));
    sv->shuffle();
    std::vector<int> val_after = ids_of(epoch_rows(*sv, IN_RGB4));
    sv->use_train();
    CHECK(sorted_ids(val_after) == sorted_ids(val_before), "shuffle() in val mode keeps the same val images");
    CHECK(ids_of(epoch_rows(*sv, IN_RGB4)) == train_before, "shuffle() in val mode does not touch the train region");

    auto st = make_img(dir, IMG_N, SHAPE_RGB4, 4, 0.2f);
    st->use_val();
    std::vector<int> val_ref = ids_of(epoch_rows(*st, IN_RGB4));
    st->use_train();
    st->shuffle();
    st->use_val();
    CHECK(ids_of(epoch_rows(*st, IN_RGB4)) == val_ref, "shuffle() in train mode does not touch the val region");
}

static void test_csv_regression()
{
    std::cout << "\n--- csv mode is not affected ---\n";

    std::string p = write_file("_tmp_img_regress.csv", "h\n0,1,2,3\n1,4,5,6\n");
    std::unique_ptr<DatasetFile> d(new DatasetFile(p, 2, 3, 4, 2));
    CHECK_EQ(d->get_input_nodes(), 3, "CSV: an int is still accepted as the input size (implicit flat ImageShape)");
    CHECK(d->image_batch[3] == 4.f && d->target_batch[N_CLASSES + 1] == 1.f, "CSV: rows are loaded as before");

    std::unique_ptr<DatasetFile> d2(new DatasetFile(p, 2, ImageShape(1, 1, 3), 4, 2));
    CHECK(d2->get_input_nodes() == 3 && d2->image_batch[3] == 4.f,
        "CSV: a file (not a directory) with an ImageShape argument still goes through the CSV branch");
}

int main()
{
    try {
        test_image_shape();
        test_resize_to_chw();
        test_image_constructor_and_getters();
        test_image_labels_scan();
        test_image_batches_content();
        test_image_resize_and_channels();
        test_image_lazy_loading();
        test_image_train_val_shuffle();
        test_csv_regression();
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] unexpected exception: " << e.what() << "\n";
        tests_failed++;
    }

    for (auto& p : tmp_files) std::remove(p.c_str());
    for (auto it = tmp_dirs.rbegin(); it != tmp_dirs.rend(); ++it) remove_dir(*it);

    std::cout << "\n=== " << tests_run << " tests run, " << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}
