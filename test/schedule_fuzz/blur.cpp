#include "Halide.h"
#include <fuzzer/FuzzedDataProvider.h>

using namespace Halide;

enum ScheduleOption {
    ComputeRoot,
    ComputeAt,
    ComputeWith,
    Serial,
    Split,
    Fuse,
    Parallel,
    Vectorize,
    Unroll,
    Tile,
    Reorder,
    StoreAt,
    // ...
};

void generate_random_schedule(Internal::Function & function, FuzzedDataProvider &fdp) {
}

Pipeline mutate_schedule(const Pipeline &p, FuzzedDataProvider &fdp) {
    std::vector<Internal::Function> outputs;
    for (const auto &f : p.outputs()) {
        outputs.push_back(f.function());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    std::vector<std::string> order = topological_order(outputs, env);

    // TODO: assume no update stages
    for (size_t i = 0; i < order.size(); i++) {
        Function f = env[order[order.size() - i - 1]];
        generate_random_schedule(f, fdp);
    }

    return p;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    ImageParam input(Int(32), 2);
    Func blur_x("blur_x");
    Func blur_y("blur_y");
    Var x("x"), y("y"), xi("xi"), yi("yi");
    blur_x(x, y) = (input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3;
    Pipeline p({blur_y});
    FuzzedDataProvider fdp(data, size);
    Pipeline mutate_p = mutate_schedule(p, fdp);
    std::string ll_file = "blur.ll";
    p.compile_to_llvm_assembly(ll_file, {input}, "blur", get_host_target());
    return 0;
}


