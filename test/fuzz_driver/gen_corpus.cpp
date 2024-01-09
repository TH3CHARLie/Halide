
#include "Halide.h"

using namespace Halide;

int main(int argc, char **argv) {
    const int width = 128;
    const int height = 128;
    Func input("input");
    Func local_sum("local_sum");
    Func blurry("blurry");
    Var x("x"), y("y"), yi("yi"), yo("yo");
    input(x, y) = 2 * x + 5 * y;
    RDom r(-2, 5, -2, 5);
    local_sum(x, y) = 0;
    local_sum(x, y) += input(x + r.x, y + r.y);
    blurry(x, y) = cast<int32_t>(local_sum(x, y) / 25);
    Pipeline p({blurry});
    local_sum.split(y, yi, yo, 8);
    serialize_pipeline(p, "blurry.hlpipe");
    return 0;
}
