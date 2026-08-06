#include"../../network/utils/file.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <set>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                   << " -> " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

__global__ void batch_scale_kernel(const float* input_batch, float* output_batch,
                                    int input_nodes, int batch_size_actual)
{
    int b = blockIdx.x; 
    int p = threadIdx.x;

    if (b >= batch_size_actual || p >= input_nodes) return;

    int idx = b * input_nodes + p;
    float scale = (float)(b + 1);
    output_batch[idx] = input_batch[idx] * scale;
}

static int tests_run = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << msg << " (line " << __LINE__ << ")\n"; \
        tests_failed++; \
    } else { \
        std::cout << "[ OK ] " << msg << "\n"; \
    } \
} while(0)

int main()
{
    const int size = 37;
    const int input_nodes = 8;
    const int output_nodes = 4;
    const int batch_size = 8;

    DatasetFile ds("data/synthetic_dataset.csv", size, input_nodes, output_nodes, batch_size);

    CHECK(ds.get_num_batches() == 5, "num_batches == ceil(37/8) == 5");

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, (size_t)batch_size * input_nodes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, (size_t)batch_size * input_nodes * sizeof(float)));

    std::vector<float> h_output(batch_size * input_nodes);

    std::set<int> all_seen_indices;
    int total_batches = ds.get_num_batches();

    for (int bi = 0; bi < total_batches; bi++) {
        if (bi > 0) ds.next_batch();

        int cur_bs = ds.current_batch_size;

        CUDA_CHECK(cudaMemcpy(d_input, ds.image_batch, (size_t)cur_bs * input_nodes * sizeof(float), cudaMemcpyHostToDevice));

        batch_scale_kernel<<<cur_bs, input_nodes>>>(d_input, d_output, input_nodes, cur_bs);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, (size_t)cur_bs * input_nodes * sizeof(float), cudaMemcpyDeviceToHost));

        if (bi < total_batches - 1) {
            CHECK(cur_bs == batch_size, ("batch " + std::to_string(bi) + " has full size " + std::to_string(batch_size)).c_str());
        } else {
            CHECK(cur_bs == (size % batch_size), ("last batch " + std::to_string(bi) + " has partial size " + std::to_string(size % batch_size)).c_str());
        }

        bool batch_values_ok = true;
        for (int b = 0; b < cur_bs; b++) {
            float first_feature = ds.image_batch[b * input_nodes + 0];
            int orig_idx = (int)std::round(first_feature / 100.0f);
            all_seen_indices.insert(orig_idx);

            for (int p = 0; p < input_nodes; p++) {
                float expected_input = orig_idx * 100.0f + p * 0.5f;
                float actual_input = ds.image_batch[b * input_nodes + p];
                if (std::fabs(expected_input - actual_input) > 1e-3) batch_values_ok = false;

                float expected_output = actual_input * (float)(b + 1);
                float actual_output = h_output[b * input_nodes + p];
                if (std::fabs(expected_output - actual_output) > 1e-3) batch_values_ok = false;
            }
        }
        CHECK(batch_values_ok, ("batch " + std::to_string(bi) + ": GPU kernel output matches expected scale per image-in-batch").c_str());
    }

    CHECK((int)all_seen_indices.size() == size, "full pass over dataset visits all 37 unique original images");
    bool all_present = true;
    for (int i = 0; i < size; i++) if (all_seen_indices.find(i) == all_seen_indices.end()) all_present = false;
    CHECK(all_present, "all original indices 0..36 present after full epoch traversal");

    ds.shuffle();
    std::set<int> shuffled_seen;
    ds.get_batch(0);
    for (int bi = 0; bi < total_batches; bi++) {
        if (bi > 0) ds.next_batch();
        for (int b = 0; b < ds.current_batch_size; b++) {
            float first_feature = ds.image_batch[b * input_nodes + 0];
            int orig_idx = (int)std::round(first_feature / 100.0f);
            shuffled_seen.insert(orig_idx);
        }
    }
    CHECK((int)shuffled_seen.size() == size, "after shuffle, full epoch pass still visits all 37 unique images exactly once");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "\n=== " << tests_run << " tests run, " << tests_failed << " failed ===\n";
    return tests_failed == 0 ? 0 : 1;
}
