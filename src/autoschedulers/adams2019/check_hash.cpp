#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>


#include "DefaultCostModel.h"
#include "HalideBuffer.h"
#include "NetworkSize.h"

namespace {

using namespace Halide;

using Halide::Runtime::Buffer;
using std::map;
using std::string;
using std::vector;


    uint64_t hash_floats(uint64_t h, const float *begin, const float *end) {
        while (begin != end) {
            uint32_t bits = *((const uint32_t *)begin);
            // From boost
            h ^= (bits + 0x9e3779b9 + (h << 6) + (h >> 2));
            begin++;
        }
        return h;
    }

    bool ends_with(const string &str, const string &suffix) {
        if (str.size() < suffix.size()) {
            return false;
        }
        size_t off = str.size() - suffix.size();
        for (size_t i = 0; i < suffix.size(); i++) {
            if (str[off + i] != suffix[i]) {
                return false;
            }
        }
        return true;
    }
}


int main(int argc, char ** argv) {
    if(argc != 4) {
        std::cout << "usage: check_hash [hash_list_binary_filepath] [featurization_filepath] [output_filepath]\n";
        return 0;
    }

    std::string hash_list_binary_filepath(argv[1]);
    std::string feature_filepath(argv[2]);
    std::string output_filepath(argv[3]);

    vector<float> scratch(10 * 1024 * 1024);
    if (!ends_with(feature_filepath, ".featurization")) {
        std::cout << "Skipping file: " << feature_filepath << "\n";
        return 0;
    }
    std::ifstream feature_file(feature_filepath);
    feature_file.read((char *)(scratch.data()), scratch.size() * sizeof(float));
    const size_t floats_read = feature_file.gcount() / sizeof(float);
    const size_t num_features = floats_read;
    const size_t features_per_stage = head2_w + (head1_w + 1) * head1_h;
    feature_file.close();
    // Note we do not check file.fail(). The various failure cases
    // are handled below by checking the number of floats read. We
    // expect truncated files if the benchmarking or
    // autoscheduling procedure crashes and want to filter them
    // out with a warning.

    if (floats_read == scratch.size()) {
        std::cout << "Too-large feature: " << feature_filepath << " " << floats_read << "\n";
    }
    if (num_features % features_per_stage != 0) {
        std::cout << "Truncated feature: " << feature_filepath << " " << floats_read << "\n";
    }
    const size_t num_stages = num_features / features_per_stage;

    uint64_t schedule_hash = 0;
    for (size_t i = 0; i < num_stages; i++) {
        schedule_hash =
            hash_floats(schedule_hash,
                &scratch[i * features_per_stage],
                &scratch[i * features_per_stage + head2_w]);
    }


    
    // read all generated hashes
    std::fstream hash_file(hash_list_binary_filepath, std::ios::in | std::ios::out | std::ios::app | std::ios::binary);
    std::ofstream output_file(output_filepath, std::ios::out);

    std::streampos fileSize;
    hash_file.seekg(0, std::ios::end);
    fileSize = hash_file.tellg();
    hash_file.seekg(0, std::ios::beg);

    uint64_t* hashes = new uint64_t [fileSize / sizeof(uint64_t)];
    hash_file.read((char*)hashes, fileSize);

    bool found = false;
    for(size_t i = 0; i < fileSize / sizeof(uint64_t); i ++) {
        found = hashes[i] == schedule_hash;
        if (found) {
            std::cout << "seen this before: " << schedule_hash << "\n";
            output_file << 1;
            return 0;
        }
    }

    
    // write this hash to file and continue
    std::cout << "new hash: " << schedule_hash << "\n";
    output_file << 0;
    hash_file.write((char*)&schedule_hash, sizeof(schedule_hash));
}