#include"include.h"


struct BinMeta {
    uint8_t  version = 0;
    uint16_t w = 0, h = 0;
    uint8_t  c = 0;
    std::vector<std::string> labels;
    uint64_t count = 0;
    uint64_t data_offset = 0;
    size_t   pix_bytes = 0, rec_bytes = 0;
};

template <class T> static void rd(std::ifstream &f, T &v) {
    if (!f.read(reinterpret_cast<char*>(&v), sizeof(T))) throw std::runtime_error("IMGS: truncated header");
}

static BinMeta read_bin_meta(std::ifstream &f, uint64_t file_size) {
    char sig[4];
    if (!f.read(sig, 4) || memcmp(sig, "IMGS", 4) != 0) throw std::runtime_error("IMGS: bad signature");
    BinMeta m;
    rd(f, m.version); rd(f, m.w); rd(f, m.h); rd(f, m.c);
    if (m.version != 1) throw std::runtime_error("IMGS: unsupported version");
    uint16_t n_labels; rd(f, n_labels);
    for (int i = 0; i < n_labels; ++i) {
        uint16_t len; rd(f, len);
        std::string s(len, '\0');
        if (len && !f.read(&s[0], len)) throw std::runtime_error("IMGS: truncated label");
        m.labels.push_back(s);
    }
    uint32_t cnt; rd(f, cnt);
    m.count = cnt;
    m.data_offset = (uint64_t)f.tellg();
    m.pix_bytes = (size_t)m.w * m.h * m.c;
    m.rec_bytes = 2 + m.pix_bytes;
    uint64_t avail = (file_size - m.data_offset) / m.rec_bytes;
    if (m.count == 0 || m.count > avail) {
        std::cerr << "Warning: IMGS header count=" << m.count << ", records in file=" << avail << "\n";
        m.count = avail;
    }
    return m;
}
