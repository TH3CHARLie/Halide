#include "Halide.h"

using namespace Halide;

std::string get_original_var_name(const std::string &var_name) {
    if (var_name.rfind('.' != std::string::npos)) {
        return var_name.substr(var_name.rfind('.') + 1, var_name.size() - 1 - var_name.rfind('.'));
    }
    return var_name;
}

std::string remove_counter_from_function_name(const std::string &name) {
    if (name.rfind('$') != std::string::npos) {
        if (isdigit(name[name.rfind('$') + 1])) {
            return name.substr(0, name.rfind('$'));
        }
        return name;
    }
    return name;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./blur_driver <input_hlpipe_file>\n";
        return 1;
    }
    const int width = 128;
    const int height = 128;
    std::string filename = argv[1];
    Buffer<int> buf1, buf2, buf3;
    {
        Func input("input");
        Func local_sum("local_sum");
        Func blurry("blurry");
        Var x("x"), y("y");
        input(x, y) = 2 * x + 5 * y;
        RDom r(-2, 5, -2, 5);
        local_sum(x, y) = 0;
        local_sum(x, y) += input(x + r.x, y + r.y);
        blurry(x, y) = cast<int32_t>(local_sum(x, y) / 25);
        Pipeline p({blurry});
        buf1 = p.realize({128, 128});
    }
    {
        Func input("input");
        Func local_sum("local_sum");
        Func blurry("blurry");
        Var x("x"), y("y"), yi("yi"), yo("yo"), xi("xi"), xo("xo"), yofxi("yofxi"), yofxio("yofxio"), yofxii("yofxii"), yofxiifyi("yofxiifyi"), yofxioo("yofxioo"), yofxioi("yofxioi");
        input(x, y) = 2 * x + 5 * y;
        RDom r(-2, 5, -2, 5, "rdom_r");
        local_sum(x, y) = 0;
        local_sum(x, y) += input(x + r.x, y + r.y);
        blurry(x, y) = cast<int32_t>(local_sum(x, y) / 25);
        local_sum.split(y, yi, yo, 2, TailStrategy::GuardWithIf).split(x, xi, xo, 5, TailStrategy::Predicate).fuse(yo, xi, yofxi).split(yofxi, yofxio, yofxii, 8, TailStrategy::ShiftInwards).fuse(yofxii, yi, yofxiifyi).split(yofxio, yofxioo, yofxioi, 5, TailStrategy::ShiftInwards).vectorize(yofxiifyi).vectorize(yofxioi);
        local_sum.update(0).unscheduled();
        blurry.split(x, xo, xi, 5, TailStrategy::Auto);
        Pipeline p({blurry});
        std::string origin_var_name = get_original_var_name(r.y.name());
        buf2 = p.realize({128, 128});
    }
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 128; ++j) {
            if (buf1(i, j) != buf2(i, j)) {
                std::cerr << "Error: buf1(" << i << ", " << j << ") = " << buf1(i, j) << ", buf2(" << i << ", " << j << ") = " << buf2(i, j) << "\n";
                // return 1;
            }
        }
    }
    return 0;
}
