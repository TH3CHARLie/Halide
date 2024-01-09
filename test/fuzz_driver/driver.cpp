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


std::vector<std::string> load_pipelines(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        exit(1);
    }
    std::vector<std::string> unique_pipelines;
    std::string line;
    while(std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::string token, path;
        int count = 0;
        while (std::getline(ss, token, ' ')) {
            if (count == 0) {
                path = token;
            }
            count++;
        }
        unique_pipelines.push_back(path);
    }
    std::cout << "unique pipeline size " << unique_pipelines.size() << std::endl;
    return unique_pipelines;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./driver <md5sum result>\n";
        return 1;
    }
    using namespace Halide;
    const int width = 128;
    const int height = 128;
    ImageParam input(Float(32), 2, "input");
    const float r_sigma = 0.1;
    const int s_sigma = 8;
    Func bilateral_grid{"bilateral_grid"};

    Var x("x"), y("y"), z("z"), c("c");

    // Add a boundary condition
    Func clamped = Halide::BoundaryConditions::repeat_edge(input);

    // Construct the bilateral grid
    RDom r(0, s_sigma, 0, s_sigma);
    Expr val = clamped(x * s_sigma + r.x - s_sigma / 2, y * s_sigma + r.y - s_sigma / 2);
    val = clamp(val, 0.0f, 1.0f);

    Expr zi = cast<int>(val * (1.0f / r_sigma) + 0.5f);

    Func histogram("histogram");
    histogram(x, y, z, c) = 0.0f;
    histogram(x, y, zi, c) += mux(c, {val, 1.0f});

    // Blur the grid using a five-tap filter
    Func blurx("blurx"), blury("blury"), blurz("blurz");
    blurz(x, y, z, c) = (histogram(x, y, z - 2, c) +
                            histogram(x, y, z - 1, c) * 4 +
                            histogram(x, y, z, c) * 6 +
                            histogram(x, y, z + 1, c) * 4 +
                            histogram(x, y, z + 2, c));
    blurx(x, y, z, c) = (blurz(x - 2, y, z, c) +
                            blurz(x - 1, y, z, c) * 4 +
                            blurz(x, y, z, c) * 6 +
                            blurz(x + 1, y, z, c) * 4 +
                            blurz(x + 2, y, z, c));
    blury(x, y, z, c) = (blurx(x, y - 2, z, c) +
                            blurx(x, y - 1, z, c) * 4 +
                            blurx(x, y, z, c) * 6 +
                            blurx(x, y + 1, z, c) * 4 +
                            blurx(x, y + 2, z, c));

    // Take trilinear samples to compute the output
    val = clamp(input(x, y), 0.0f, 1.0f);
    Expr zv = val * (1.0f / r_sigma);
    zi = cast<int>(zv);
    Expr zf = zv - zi;
    Expr xf = cast<float>(x % s_sigma) / s_sigma;
    Expr yf = cast<float>(y % s_sigma) / s_sigma;
    Expr xi = x / s_sigma;
    Expr yi = y / s_sigma;
    Func interpolated("interpolated");
    interpolated(x, y, c) =
        lerp(lerp(lerp(blury(xi, yi, zi, c), blury(xi + 1, yi, zi, c), xf),
                    lerp(blury(xi, yi + 1, zi, c), blury(xi + 1, yi + 1, zi, c), xf), yf),
                lerp(lerp(blury(xi, yi, zi + 1, c), blury(xi + 1, yi, zi + 1, c), xf),
                    lerp(blury(xi, yi + 1, zi + 1, c), blury(xi + 1, yi + 1, zi + 1, c), xf), yf),
                zf);

    // Normalize
    bilateral_grid(x, y) = interpolated(x, y, 0) / interpolated(x, y, 1);
    Pipeline p({bilateral_grid});

    // create input buffer
    Buffer<float, 2> input_buf(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float val = (float)(x + y * width) / (width * height);
            input_buf(x, y) = val;
        }
    }
    input.set(input_buf);

    Halide::Buffer<int> reference = p.realize({width, height});

    auto unique_pipelines = load_pipelines(argv[1]);
    std::vector<std::string> wrong_pipelines;
    std::vector<std::string> internal_error_pipelines;
    for (const auto &path : unique_pipelines) {
        std::cout << "Running pipeline " << path << std::endl;
        std::map<std::string, Parameter> params;
        params.insert({"input", input.parameter()});
        Halide::Pipeline dp = Halide::deserialize_pipeline(path, params);
        error_flag = false;
        dp.jit_handlers().custom_error = my_error;
        Halide::Buffer<float, 2> result;
        try {
            result = dp.realize({width, height});
        } catch (const Halide::InternalError &e) {
            std::cout << "Internal error: " << e.what() << std::endl;
            internal_error_pipelines.push_back(path);
            continue;
        }
        bool wrong_result = false;
        float tolerance = 1e-5f;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; ++x) {
                if (std::abs(result(x, y) - reference(x, y)) > tolerance && !error_flag) {
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
    std::cout << "Internal error pipelines: " << internal_error_pipelines.size() << std::endl;
    for (const auto &path : internal_error_pipelines) {
        std::cout << path << std::endl;
    }
    return 0;
}
