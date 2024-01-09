#include "Halide.h"
#include <fstream>
#include <set>

std::set<std::string> load_unique_pipelines(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        exit(1);
    }
    std::set<std::string> unique_pipelines;
    std::string line;
    std::set<std::string> hashes;
    while(std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string md5hash, token, path;
        int count = 0;
        while (std::getline(ss, token, ' ')) {
            if (count == 0) {
                md5hash = token;
            // OK very strange hard-coding I love it
            } else if (count == 2) {
                path = token;
            }
            count++;
        }
        if (hashes.find(md5hash) == hashes.end()) {
            hashes.insert(md5hash);
            unique_pipelines.insert(path);
        }
    }
    std::cout << "unique pipeline size " << unique_pipelines.size() << std::endl;
    return unique_pipelines;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./load_unique <md5sum result>\n";
        return 1;
    }

    auto unique_pipelines = load_unique_pipelines(argv[1]);
    std::vector<std::string> unique_pipelines_vec;
    unique_pipelines_vec.reserve(unique_pipelines.size());
    for (auto& pipeline : unique_pipelines) {
        unique_pipelines_vec.push_back(pipeline);
    }
    // split the set into multiple files, with each containing 10000 pipelines.
    int count = (unique_pipelines_vec.size() / 10000) + 1;
    for (int i = 0; i < count; i++) {
        std::ofstream ofs("unique_pipelines_md5_" + std::to_string(i) + ".txt");
        for (int j = i * 10000; j < (i + 1) * 10000 && j < unique_pipelines_vec.size(); j++) {
            ofs << unique_pipelines_vec[j] << std::endl;
        }
        ofs.close();
    }
    return 0;
}