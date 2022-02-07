#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <optional>
#include <cassert>

// A sample is a featurization + a vector of per-stage runtimes + some ids, all together in one file.
// This utility concats the runtime and ids onto a featurization to produce a sample.

// key: producer value: consumer
using DAG = std::map<std::string, std::string>;

using ProfilerRuntimes = std::map<std::string, float>;

std::optional<DAG> parse_DAG(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Unable to open DAG file: " << filename << '\n';
        return std::nullopt;
    }
    DAG dag;
    std::string producer_name, consumer_name;
    while (file >> producer_name >> consumer_name) {
        assert(dag.find(producer_name) == dag.end());
        dag[producer_name] = consumer_name;
    }
    return dag;
}

std::optional<ProfilerRuntimes> parse_profiler_runtimes(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Unable to open profiler runtime file: " << filename << '\n';
        return std::nullopt;
    }
    ProfilerRuntimes runtimes;
    std::string name;
    float runtime;
    while (file >> name >> runtime) {
        assert(runtimes.find(name) == runtimes.end());
        runtimes[name] = runtime;
    }
    return runtimes;
}

std::optional<std::vector<std::string>> parse_feature_ordering(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Unable to open profiler ordering file: " << filename << '\n';
        return std::nullopt;
    }
    std::vector<std::string> ordering;
    std::string name;
    while (file >> name) {
        ordering.push_back(name);
    }
    return ordering;
}

std::vector<float> sort_runtimes(const ProfilerRuntimes& runtimes, const std::vector<std::string>& ordering) {
    std::vector<float> sorted;
    for (const auto& n: ordering) {
        auto it = runtimes.find(n);
        if (it != runtimes.end()) {
            sorted.push_back(it->second);
        }
    }
    return sorted;
}

std::vector<std::vector<int>> construct_transfrom_matrix(DAG& dag, const ProfilerRuntimes& runtimes, const std::vector<std::string>& ordering) {
    std::vector<std::vector<int>> mat(ordering.size(), std::vector<int>(ordering.size()));
    std::map<std::string, int> sorted_runtime_names;
    std::map<std::string, int> sorted_ordering_names;
    int cnt = 0;
    for (const auto& n: ordering) {
        auto it = runtimes.find(n);
        if (it != runtimes.end()) {
            sorted_runtime_names[it->first] = cnt++;
        }
    }
    cnt = 0;
    for (const auto& n: ordering) {
        sorted_ordering_names[n] = cnt++;
    }
    for (size_t i = 0; i < ordering.size(); ++i) {
        // this function is not inlined
        std::string name = ordering[i];
        if (runtimes.find(name) != runtimes.end()) {
            mat[sorted_runtime_names[name]][sorted_ordering_names[name]] = 1;
        } else {
            // otherwise, the function is inlined, we need to find its consumer using DAG
            std::string consumer_name;
            std::vector<int> indices;
            while (true) {
                indices.push_back(sorted_ordering_names[name]);
                consumer_name = dag[name];
                if (runtimes.find(consumer_name) != runtimes.end()) {
                    break;
                }
                // if nested inlining happens, we continue
                name = consumer_name;
            }
            for (auto& idx: indices) {
                mat[sorted_runtime_names[consumer_name]][idx] = 1;
            }
        }
    }
    std::vector<std::vector<int>> transposed(ordering.size(), std::vector<int>(ordering.size()));
    for (size_t i = 0; i < ordering.size(); ++i) {
        for (size_t j = 0; j < ordering.size(); ++j) {
            transposed[i][j] = mat[j][i];
        }
    }
    return transposed;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        std::cout << "Usage: featurization_to_sample in.featurization runtime DAG ordering pipeline_id schedule_id out.sample out.metadata\n";
        return -1;
    }

    std::ifstream src(argv[1], std::ios::binary);
    if (!src) {
        std::cerr << "Unable to open input file: " << argv[1] << "\n";
        return -1;
    }

    std::ofstream sample_dst(argv[7], std::ios::binary);
    if (!sample_dst) {
        std::cerr << "Unable to open sample output file: " << argv[7] << "\n";
        return -1;
    }

    sample_dst << src.rdbuf();

    int32_t pid = atoi(argv[5]);
    int32_t sid = atoi(argv[6]);
    sample_dst.write((const char *)&pid, 4);
    sample_dst.write((const char *)&sid, 4);

    std::ofstream metadata_dst(argv[8], std::ios::binary);
    if (!metadata_dst) {
        std::cerr << "Unable to open metadata output file: " << argv[8] << "\n";
        return -1;
    }

    auto profiler_runtimes = parse_profiler_runtimes(argv[2]);
    if (!profiler_runtimes) {
        return -1;
    }
    auto dag = parse_DAG(argv[3]);
    if (!dag) {
        return -1;
    }
    auto ordering = parse_feature_ordering(argv[4]);
    if (!ordering) {
        return -1;
    }
    auto sorted_runtimes = sort_runtimes(*profiler_runtimes, *ordering);

    auto mat = construct_transfrom_matrix(*dag, *profiler_runtimes, *ordering);

    int32_t runtime_size = sorted_runtimes.size();
    int32_t row = mat.size(), column = mat[0].size();
    int32_t ordering_size = ordering->size();
    assert(row == ordering_size && column == ordering_size);
    metadata_dst.write((const char*)&runtime_size, 4);
    metadata_dst.write((const char*)&ordering_size, 4);
    // per-stage runtimes are already milliseconds
    for (float r: sorted_runtimes) {
        metadata_dst.write((const char *)&r, 4);
    }
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            metadata_dst.write((const char*)&(mat[i][j]), 4);
        }
    }
    src.close();
    sample_dst.close();
    metadata_dst.close();

    return 0;
}
