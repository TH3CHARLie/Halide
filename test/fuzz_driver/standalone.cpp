#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./blur_driver <input_hlpipe_file>\n";
        return 1;
    }
    const int width = 128;
    const int height = 128;
    std::string filename = argv[1];
    Func input("input");
    Func blur_x("blur_x");
    Func blur_y("blur_y");
    Var x("x"), y("y");
    input(x, y) = x + y;
    blur_x(x, y) = (input(x - 1, y) + input(x, y) + input(x + 1, y)) / 3;
    blur_y(x, y) = (blur_x(x, y - 1) + blur_x(x, y) + blur_x(x, y + 1)) / 3;
    Pipeline p({blur_y});
    Buffer<int> buf = p.realize({width, height});

    Var n, s, l;
    Func input_1("input_1");
    Func blur_x_1("blur_x_1");
    Func blur_y_1("blur_y_1");
    input_1(x, y) = x + y;
    blur_x_1(x, y) = (input_1(x - 1, y) + input_1(x, y) + input_1(x + 1, y)) / 3;
    blur_y_1(x, y) = (blur_x_1(x, y - 1) + blur_x_1(x, y) + blur_x_1(x, y + 1)) / 3;
    Pipeline p_1({blur_y_1});
    Buffer<int> buf2 = p_1.realize({width, height});

    // compare buffer values
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            if (buf(i, j) != buf2(i, j)) {
                std::cerr << "Error: buf(" << i << ", " << j << ") = " << buf(i, j) << ", buf2(" << i << ", " << j << ") = " << buf2(i, j) << "\n";
                return 1;
            }
        }
    }
    return 0;
}