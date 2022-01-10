#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

// A sample is a featurization + a vector of per-stage runtimes + some ids, all together in one file.
// This utility concats the runtime and ids onto a featurization to produce a sample.
int main(int argc, char **argv) {
    if (argc != 6) {
        std::cout << "Usage: featurization_to_sample in.featurization runtime pipeline_id schedule_id out.sample\n";
        return -1;
    }

    std::ifstream src(argv[1], std::ios::binary);
    if (!src) {
        std::cerr << "Unable to open input file: " << argv[1] << "\n";
        return -1;
    }

    std::ofstream dst(argv[5], std::ios::binary);
    if (!dst) {
        std::cerr << "Unable to open output file: " << argv[5] << "\n";
        return -1;
    }

    dst << src.rdbuf();

    std::ifstream runtime_file(argv[2]);
    if (!runtime_file) {
        std::cerr << "Unable to open runtime file: " << argv[2] << '\n';
    }
    std::vector<float> per_stage_runtimes;
    std::string name;
    float runtime;
    while (runtime_file >> name >> runtime) {
        per_stage_runtimes.push_back(runtime);
    }
    int size = per_stage_runtimes.size();
    int32_t pid = atoi(argv[3]);
    int32_t sid = atoi(argv[4]);

    dst.write((const char *)&size, 4);
    for (float r: runtimes) {
        dst.write((const char *)&r, 4);
    }
    dst.write((const char *)&pid, 4);
    dst.write((const char *)&sid, 4);

    src.close();
    dst.close();

    return 0;
}
