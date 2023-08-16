#include "Halide.h"
#include <fstream>
#include <set>

bool error_flag = false;
void my_error(Halide::JITUserContext *ucon, const char *msg) {
    error_flag = true;
    std::cout << "custom error: " << msg << std::endl;
}


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
        std::cerr << "Usage: ./driver <md5sum result>\n";
        return 1;
    }
    const int width = 128;
    const int height = 128;
    Halide::Func input("input");
    Halide::Func blur_x("blur_x");
    Halide::Func blur_y("blur_y");
    Halide::Var x("x"), y("y");
    input(x, y) = x + y;
    blur_x(x, y) = (input(x - 1, y) + input(x, y) + input(x + 1, y)) / 3;
    blur_y(x, y) = (blur_x(x, y - 1) + blur_x(x, y) + blur_x(x, y + 1)) / 3;
    Halide::Pipeline p({blur_y});
    Halide::Buffer<int> reference = p.realize({width, height});

    auto unique_pipelines = load_unique_pipelines(argv[1]);
    std::vector<std::string> wrong_pipelines;
    for (const auto &path : unique_pipelines) {
        std::cout << "Running pipeline " << path << std::endl;
        Halide::Pipeline dp = Halide::deserialize_pipeline(path, {});
        error_flag = false;
        dp.jit_handlers().custom_error = my_error;
        Halide::Buffer<int> result = dp.realize({width, height});
        bool wrong_result = false;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; ++x) {
                if (result(x, y) != reference(x, y) && !error_flag) {
                    std::cout << "Wrong result: result(" << x << ", " << y << ") = " << result(x, y)
                              << " instead of " << reference(x, y) << std::endl;
                    wrong_pipelines.push_back(path);
                    wrong_result = true;
                }
                if (wrong_result) {
                    break;
                }
            }
            if (wrong_result) {
                break;
            }
        }
    }

    std::cout << "Wrong pipelines: " << wrong_pipelines.size() << std::endl;
    for (const auto &path : wrong_pipelines) {
        std::cout << path << std::endl;
    }
    return 0;
}
