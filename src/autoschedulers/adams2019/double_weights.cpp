#include "Halide.h"
#include <string>
#include <fstream>
#include <iostream>
#include "NetworkSize.h"
#include "Featurization.h"

using Halide::Buffer;
using namespace Halide;

constexpr uint32_t kSignature = 0x68776631;

int main(int argc, char *argv[]) {
    const std::string infile = argv[1];
    const std::string outfile = argv[2];
    std::cout << "infile: " << infile << " outfile: " << outfile << '\n';
    std::ifstream in(infile, std::ios_base::binary);
    std::ofstream out(outfile, std::ios_base::trunc | std::ios_base::binary);

    uint32_t pipeline_features_version = Halide::Internal::PipelineFeatures::version();
    uint32_t schedule_features_version = Halide::Internal::ScheduleFeatures::version();

    // old buffers, only half of current channel sizes
    Halide::Runtime::Buffer<float> old_head1_filter{head1_channels / 2, head1_w, head1_h};
    Halide::Runtime::Buffer<float> old_head1_bias{head1_channels / 2};

    Halide::Runtime::Buffer<float> old_head2_filter{head2_channels / 2, head2_w};
    Halide::Runtime::Buffer<float> old_head2_bias{head2_channels / 2};

    Halide::Runtime::Buffer<float> old_conv1_filter{conv1_channels / 2, (head1_channels + head2_channels) / 2};
    Halide::Runtime::Buffer<float> old_conv1_bias{conv1_channels / 2};

    // new buffers
    Halide::Runtime::Buffer<float> head1_filter{head1_channels, head1_w, head1_h};
    Halide::Runtime::Buffer<float> head1_bias{head1_channels};

    Halide::Runtime::Buffer<float> head2_filter{head2_channels, head2_w};
    Halide::Runtime::Buffer<float> head2_bias{head2_channels};

    Halide::Runtime::Buffer<float> conv1_filter{conv1_channels, head1_channels + head2_channels};
    Halide::Runtime::Buffer<float> conv1_bias{conv1_channels};

    uint32_t signature;
    in.read((char *)&signature, sizeof(signature));
    if (in.fail() || signature != kSignature) {
        return 1;
    }

    in.read((char *)&pipeline_features_version, sizeof(pipeline_features_version));
    if (in.fail()) {
        return 1;
    }

    in.read((char *)&schedule_features_version, sizeof(schedule_features_version));
    if (in.fail()) {
        return 1;
    }

    uint32_t buffer_count;
    in.read((char *)&buffer_count, sizeof(buffer_count));
    if (in.fail() || buffer_count != 6) {
        return 1;
    }

    const auto load_one = [&in](Halide::Runtime::Buffer<float> &buf) -> bool {
        uint32_t dimension_count;
        in.read((char *)&dimension_count, sizeof(dimension_count));
        if (in.fail() || dimension_count != (uint32_t)buf.dimensions()) {
            return false;
        }
        for (uint32_t d = 0; d < dimension_count; d++) {
            uint32_t extent;
            in.read((char *)&extent, sizeof(extent));
            if (in.fail() || (int)extent != (int)buf.extent(d)) {
                return false;
            }
        }
        in.read((char *)(buf.data()), buf.size_in_bytes());
        if (in.fail()) {
            return false;
        }
        return true;
    };

    if (!load_one(old_head1_filter)) {
        return 1;
    }
    if (!load_one(old_head1_bias)) {
        return 1;
    }
    if (!load_one(old_head2_filter)) {
        return 1;
    }
    if (!load_one(old_head2_bias)) {
        return 1;
    }
    if (!load_one(old_conv1_filter)) {
        return 1;
    }
    if (!load_one(old_conv1_bias)) {
        
        return 1;
    }

    std::cout << "Load " << infile << " completed\n";
    
    const auto save_one = [&out](const Halide::Runtime::Buffer<float> &buf) -> bool {
        const uint32_t dimension_count = buf.dimensions();
        out.write((const char *)&dimension_count, sizeof(dimension_count));
        if (out.fail()) {
            return false;
        }
        for (uint32_t d = 0; d < dimension_count; d++) {
            uint32_t extent = buf.extent(d);
            out.write((const char *)&extent, sizeof(extent));
            if (out.fail()) {
                return false;
            }
        }
        out.write((const char *)(buf.data()), buf.size_in_bytes());
        if (out.fail()) {
            return false;
        }
        return true;
    };

    // copy-paste the old weights into the new weights
    // head1_filter
    for (uint32_t i = 0; i < head1_channels / 2; ++i) {
        for (uint32_t j = 0; j < head1_w; ++j) {
            for (uint32_t k = 0; k < head1_h; ++k) {
                head1_filter(i, j, k) = old_head1_filter(i, j, k);
                head1_filter(i + head1_channels / 2, j, k) = old_head1_filter(i, j, k);
            }
        }
    }
    // head1_bias
    for (uint32_t i = 0; i < head1_channels / 2; ++i) {
        head1_bias(i) = old_head1_bias(i);
        head1_bias(i + head1_channels / 2) = old_head1_bias(i);
    }

    // head2_filter
    for (uint32_t i = 0; i < head2_channels / 2; ++i) {
        for (uint32_t j = 0; j < head2_w; ++j) {
            head2_filter(i, j) = old_head2_filter(i, j);
            head2_filter(i + head2_channels / 2, j) = old_head2_filter(i, j);
        }
    }
    // head2_bias
    for (uint32_t i = 0; i < head2_channels / 2; ++i) {
        head2_bias(i) = old_head2_bias(i);
        head2_bias(i + head2_channels / 2) = old_head2_bias(i);
    }

    // conv1_filter
    for (uint32_t i = 0; i < conv1_channels / 2; ++i) {
        for (uint32_t j = 0; j < head1_channels / 2; ++j) {
            conv1_filter(i, j) = old_conv1_filter(i, j);
            conv1_filter(i + conv1_channels / 2, j) = old_conv1_filter(i, j);
            conv1_filter(i, j + head1_channels / 2) = old_conv1_filter(i, j);
            conv1_filter(i + conv1_channels / 2, j + head1_channels / 2) = old_conv1_filter(i, j);
        }
        for (uint32_t j = 0; j < head2_channels / 2; ++j) {
            conv1_filter(i, j + head1_channels) = old_conv1_filter(i, j);
            conv1_filter(i + conv1_channels / 2, j + head1_channels) = old_conv1_filter(i, j);
            conv1_filter(i, j + head1_channels + head2_channels / 2) = old_conv1_filter(i, j);
            conv1_filter(i + conv1_channels / 2, j + head1_channels + head2_channels / 2) = old_conv1_filter(i, j);   
        }
    }

    std::cout << "Weights copy completed\n";
    // save the new weights
    const uint32_t output_signature = kSignature;
    out.write((const char *)&output_signature, sizeof(output_signature));
    if (out.fail()) {
        return false;
    }

    out.write((const char *)&pipeline_features_version, sizeof(pipeline_features_version));
    if (out.fail()) {
        return false;
    }

    out.write((const char *)&schedule_features_version, sizeof(schedule_features_version));
    if (out.fail()) {
        return false;
    }

    const uint32_t output_buffer_count = 6;
    out.write((const char *)&output_buffer_count, sizeof(output_buffer_count));
    if (out.fail()) {
        return false;
    }

    if (!save_one(head1_filter)) {
        return false;
    }
    if (!save_one(head1_bias)) {
        return false;
    }
    if (!save_one(head2_filter)) {
        return false;
    }
    if (!save_one(head2_bias)) {
        return false;
    }
    if (!save_one(conv1_filter)) {
        return false;
    }
    if (!save_one(conv1_bias)) {
        return false;
    }
    std::cout << "Save " << outfile << " completed\n";

    return 0;
}