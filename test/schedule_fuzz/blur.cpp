#include "Halide.h"
#include <cstdio>
#include <fuzzer/FuzzedDataProvider.h>
#include <iostream>
#include <unordered_set>
#include <functional>
#include <exception>

using namespace Halide;

// 1. reuse existing vars
// 2. generate new vars (not necessarily from vars)
// 3. generate extended vars from existing vars
Var generate_random_var(FuzzedDataProvider &fdp, std::vector<Var> &vars) {
    std::function<Var()> operations[] = {
        [&]() {
            int index = fdp.ConsumeIntegralInRange<int>(0, vars.size() - 1);
            return vars[index];
        },
        [&]() {
            int index = fdp.ConsumeIntegralInRange<int>(0, vars.size() - 1);
            int next_index = fdp.ConsumeIntegralInRange<int>(0, 100);
            return Var(vars[index].name() + "_i_" + std::to_string(next_index));
        },
        [&]() {
            int index = fdp.ConsumeIntegralInRange<int>(0, 25);
            return Var(std::string(1, 'a' + index));
        }
    };
    return fdp.PickValueInArray(operations)();
}

void merge_vars(std::vector<Var> &vars, const std::vector<Var> &new_vars) {
    std::unordered_set<std::string> var_names;
    for (const auto &v : vars) {
        var_names.insert(v.name());
    }
    for (const auto &v : new_vars) {
        if (var_names.find(v.name()) == var_names.end()) {
            vars.push_back(v);
        }
    }
}

static const int factors[] = {1, 2, 3, 4, 6, 8, 16, 32, 64, 128};

void generate_random_schedule(FuzzedDataProvider &fdp, Internal::Function &function, std::vector<Var> &vars) {
    std::function<void()> operations[] = {
        [&]() {
            Func(function).compute_root();
            std::cout << ".compute_root()";
        },
        [&]() {
            Var old = generate_random_var(fdp, vars);
            Var outer = generate_random_var(fdp, vars);
            Var inner = generate_random_var(fdp, vars);
            merge_vars(vars, {old, outer, inner});
            int factor = fdp.PickValueInArray(factors);
            Func(function).split(old, outer, inner, factor);
            std::cout << ".split(" << old.name() << ", " << outer.name() << ", " << inner.name() << ", " << factor << ")";
        },
        [&]() {
            Var var = generate_random_var(fdp, vars);
            merge_vars(vars, {var});
            Func(function).vectorize(var);
            std::cout << ".vectorize(" << var.name() << ")";
        },
        [&]() {
            Var var = generate_random_var(fdp, vars);
            merge_vars(vars, {var});
            Func(function).unroll(var);
            std::cout << ".unroll(" << var.name() << ")";
        },
        [&]() {
            int size = fdp.ConsumeIntegralInRange<int>(1, vars.size() - 1);
            std::vector<VarOrRVar> reorder_vars;
            for (int i = 0; i < size; ++i) {
                int index = fdp.ConsumeIntegralInRange<int>(0, vars.size() - 1);
                reorder_vars.push_back(VarOrRVar(vars[index]));
            }
            Func(function).reorder(reorder_vars);
            std::cout << ".reorder(";
            for (int i = 0; i < size; ++i) {
                std::cout << reorder_vars[i].name();
                if (i != size - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << ")";
        }
    };
    // depth here means how many schedules we are applying to the function
    int depth = fdp.ConsumeIntegralInRange<int>(1, 5);
    std::cout << "schedule for function " << function.name() << " depth " << depth << ": " << function.name();
    for (int i = 0; i < depth; ++i) {
        fdp.PickValueInArray(operations)();
    }
    std::cout << "\n";
}

Pipeline mutate_schedule(FuzzedDataProvider &fdp, const Pipeline &p, const std::vector<Var> &original_vars) {
    std::vector<Internal::Function> outputs;
    for (const auto &f : p.outputs()) {
        outputs.push_back(f.function());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    std::vector<std::string> order = topological_order(outputs, env);
    std::vector<Var> vars = original_vars;
    // TODO: assume no update stages
    for (size_t i = 0; i < order.size(); i++) {
        generate_random_schedule(fdp, env[order[order.size() - i - 1]], vars);
    }

    return p;
}

static int counter = 0;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!Halide::exceptions_enabled()) {
        std::cout << "[SKIP] Halide was compiled without exceptions.\n";
        return 0;
    }
    Func input("input");
    Func blur_x("blur_x");
    Func blur_y("blur_y");
    Var x("x"), y("y");
    input(x, y) = x + y;
    blur_x(x, y) = (input(x - 1, y) + input(x, y) + input(x + 1, y)) / 3;
    blur_y(x, y) = (blur_x(x, y - 1) + blur_x(x, y) + blur_x(x, y + 1)) / 3;
    Pipeline p({blur_y});
    FuzzedDataProvider fdp(data, size);
    try {
        Pipeline mutate_p = mutate_schedule(fdp, p, {x, y});
        std::string ll_file = "blur" + std::to_string(counter) + ".ll";
        p.compile_to_module({}, "blur", get_host_target());
        std::map<std::string, Internal::Parameter> params;
        std::string hlpipe_file = "serdes_res/blur_" + std::to_string(counter) + ".hlpipe";
        serialize_pipeline(mutate_p, hlpipe_file, params);
    } catch (const Halide::CompileError &e) {
        std::cout << "\nexception: " << e.what() << "\n";
    }
    std::cout << "counter: " << counter << "\n";
    counter++;
    return 0;
}