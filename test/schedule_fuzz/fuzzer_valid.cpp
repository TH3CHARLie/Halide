#include "Halide.h"
#include <cstdio>
#include <fuzzer/FuzzedDataProvider.h>
#include <iostream>
#include <unordered_set>
#include <functional>
#include <exception>
#include <fstream>

using namespace Halide;

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

std::string get_original_var_name(const std::string &var_name) {
    if (var_name.rfind('.' != std::string::npos)) {
        return var_name.substr(var_name.rfind('.') + 1, var_name.size() - 1 - var_name.rfind('.'));
    }
    return var_name;
}

std::vector<std::string> get_function_var_names(const Internal::Function &function) {
    std::vector<std::string> var_names;
    const auto &dims = function.definition().schedule().dims();
    for (const auto &d : dims) {
        if (d.var != Var::outermost().name()) {
            var_names.push_back(get_original_var_name(d.var));
        }
    }
    return var_names;
}

int common_split_factors[] = {1, 2, 4, 8, 16, 32, 64, 128};
TailStrategy tail_strategies[] = {TailStrategy::RoundUp, TailStrategy::GuardWithIf, TailStrategy::Predicate, TailStrategy::PredicateLoads, TailStrategy::PredicateStores, TailStrategy::ShiftInwards, TailStrategy::Auto};

std::string tail_strategy_to_string(TailStrategy tail_strategy) {
    switch (tail_strategy) {
        case TailStrategy::RoundUp:
            return "TailStrategy::RoundUp";
        case TailStrategy::GuardWithIf:
            return "TailStrategy::GuardWithIf";
        case TailStrategy::Predicate:
            return "TailStrategy::Predicate";
        case TailStrategy::PredicateLoads:
            return "TailStrategy::PredicateLoads";
        case TailStrategy::PredicateStores:
            return "TailStrategy::PredicateStores";
        case TailStrategy::ShiftInwards:
            return "TailStrategy::ShiftInwards";
        case TailStrategy::Auto:
            return "TailStrategy::Auto";
        default:
            return "TailStrategy::Other";
    }
}

void generate_loop_schedule(FuzzedDataProvider &fdp, Internal::Function &function, const std::vector<Func> &before_funcs, const std::vector<Func> &after_funcs, std::ostream &out) {
    // available choices: split reorder fuse unroll vectorize parallel serial
    // we only call each primitive once
    bool set_split = false;
    bool set_reorder = false;
    bool set_fuse = false;
    bool set_unroll = false;
    bool set_vectorize = false;
    bool set_parallel = false;
    bool set_serial = false;
    std::function<void()> operations[] = {
        [&]() {
            if (set_split) {
                return;
            }
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            int split_factor = fdp.PickValueInArray(common_split_factors);
            TailStrategy tail_strategy = fdp.PickValueInArray(tail_strategies);
            out << ".split(" << var_name_picked << ", " << var_name_picked << "_outer, " << var_name_picked << "_inner, " << split_factor << ", " << tail_strategy_to_string(tail_strategy) << ")";
            Func(function).split(Var(var_name_picked), Var(var_name_picked + "_outer"), Var(var_name_picked + "_inner"), split_factor, tail_strategy);
            set_split = true;
        },
        [&]() {
            if (fdp.remaining_bytes() <= 0) {
                return;
            }
            if (set_reorder) {
                return;
            }
            std::vector<std::string> var_names = get_function_var_names(function);
            // for reorder we always reorder everything we have in this function
            // but we need a random premutation, fdp is really bad at that, so we do rejection sampling.....
            std::vector<VarOrRVar> reorder_vars;
            std::vector<std::string> var_names_premutated;
            while (reorder_vars.size() != var_names.size() && fdp.remaining_bytes() > 0) {
                std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
                if (std::find(var_names_premutated.begin(), var_names_premutated.end(), var_name_picked) == var_names_premutated.end()) {
                    var_names_premutated.push_back(var_name_picked);
                    reorder_vars.push_back(Var(var_name_picked));
                }
            }
            out << ".reorder(";
            for (int i = 0; i < reorder_vars.size(); ++i) {
                out << reorder_vars[i].name();
                if (i != reorder_vars.size() - 1) {
                    out << ", ";
                }
            }
            out << ")";
            Func(function).reorder(reorder_vars);
            set_reorder = true;
        },
        [&]() {
            if (set_fuse) {
                return;
            }
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string inner_var_name = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            std::string outer_var_name = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            while (outer_var_name == inner_var_name && fdp.remaining_bytes() > 0) {
                outer_var_name = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            }
            std::string fused_var_name = inner_var_name + "_" + outer_var_name + "_fused";
            out << ".fuse(" << inner_var_name << ", " << outer_var_name << ", " << fused_var_name << ")";
            Func(function).fuse(Var(inner_var_name), Var(outer_var_name), Var(fused_var_name));
            set_fuse = true;
        },
        [&]() {
            if (set_unroll) {
                return;
            }
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            out << ".unroll(" << var_name_picked << ")";
            Func(function).unroll(Var(var_name_picked));
            set_unroll = true;
        },
        [&]() {
            if (set_vectorize) {
                return;
            }
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            out << ".vectorize(" << var_name_picked << ")";
            Func(function).vectorize(Var(var_name_picked));
            set_vectorize = true;
        },
        [&]() {
            if (set_parallel) {
                return;
            }
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            out << ".parallel(" << var_name_picked << ")";
            Func(function).parallel(Var(var_name_picked));
            set_parallel = true;
        },
        [&]() {
            if (set_serial) {
                return;
            }
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            out << ".serial(" << var_name_picked << ")";
            Func(function).serial(Var(var_name_picked));
            set_serial = true;
        }
    };
    int depth = fdp.ConsumeIntegralInRange<int>(0, 5);
    for (int i = 0; i < depth; ++i) {
        fdp.PickValueInArray(operations)();
    }
}

void generate_compute_schedule(FuzzedDataProvider &fdp, Internal::Function &function, const std::vector<Func> &before_funcs, const std::vector<Func> &after_funcs, std::ostream &out) {
    bool set_compute_root = false;
    bool set_compute_at = false;
    std::function<void()> operations[] = {
        [&]() {
            if (set_compute_root) {
                return;
            }
            out << ".compute_root()";
            Func(function).compute_root();
            set_compute_root = true;
        },
        [&]() {
            if (set_compute_at || before_funcs.empty()) {
                return;
            }
            Func compute_at_func = before_funcs[fdp.ConsumeIntegralInRange<int>(0, before_funcs.size() - 1)];
            auto compute_at_func_var_names = get_function_var_names(compute_at_func.function());
            std::string var_name_picked = compute_at_func_var_names[fdp.ConsumeIntegralInRange<int>(0, compute_at_func_var_names.size() - 1)];
            out << ".compute_at(" << compute_at_func.name() << ", " << var_name_picked << ")";
            Func(function).compute_at(compute_at_func, Var(var_name_picked));
            set_compute_at = true;
        }
    };
    fdp.PickValueInArray(operations)();
}

void generate_store_schedule(FuzzedDataProvider &fdp, Internal::Function &function, const std::vector<Func> &before_funcs, const std::vector<Func> &after_funcs, std::ostream &out) {
    bool set_store_root = false;
    bool set_store_at = false;
    std::function<void()> operations[] = {
        [&]() {
            if (set_store_root) {
                return;
            }
            out << ".store_root()";
            Func(function).store_root();
            set_store_root = true;
        },
        [&]() {
            if (set_store_at || after_funcs.empty()) {
                return;
            }
            Func store_at_func = after_funcs[fdp.ConsumeIntegralInRange<int>(0, after_funcs.size() - 1)];
            auto store_at_func_var_names = get_function_var_names(store_at_func.function());
            std::string var_name_picked = store_at_func_var_names[fdp.ConsumeIntegralInRange<int>(0, store_at_func_var_names.size() - 1)];
            out << ".store_at(" << store_at_func.name() << ", " << var_name_picked << ")";
            Func(function).store_at(store_at_func, Var(var_name_picked));
            set_store_at = true;
        }
    };
    fdp.PickValueInArray(operations)();
}

void generate_function_schedule(FuzzedDataProvider &fdp, std::map<std::string, Internal::Function> &env, const std::vector<std::string> &order, int index, std::ostream &out) {
    Internal::Function &function = env[order[index]];
    out << "planned schedule for function " << function.name();
    // assemble the Funcs before the current Func
    std::vector<Func> before_funcs;
    for (int i = 0; i < index; ++i) {
        before_funcs.push_back(Func(env[order[i]]));
    }

    // assemble the Funcs after the current Func
    std::vector<Func> after_funcs;
    for (int i = index + 1; i < order.size(); ++i) {
        after_funcs.push_back(Func(env[order[i]]));
    }
    // Following Alex Reinking's work on Halide formal semantics
    // TODO: generate a condition stmt so we can use f.specialize(cond)
    if (fdp.ConsumeBool()) {
        generate_loop_schedule(fdp, function, before_funcs, after_funcs, out);
    }
    if (fdp.ConsumeBool()) {
        generate_compute_schedule(fdp, function, before_funcs, after_funcs, out);
    }
    if (fdp.ConsumeBool()) {
        generate_store_schedule(fdp, function, before_funcs, after_funcs, out);
    }

    // depth here means how many schedules we are applying to the function

    out << "\n";
}

void generate_schedule(FuzzedDataProvider &fdp, Pipeline &p, std::ostream& out) {
    std::vector<Internal::Function> outputs;
    for (const auto &f : p.outputs()) {
        outputs.push_back(f.function());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    // TODO: should we preturb the order?
    std::vector<std::string> order = topological_order(outputs, env);
    // TODO: assume no update stage
    for (int i = order.size() - 1; i >= 0; --i) {
        generate_function_schedule(fdp, env, order, i, out);
    }
}

static int counter = 0;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    if (!Halide::exceptions_enabled()) {
        std::cout << "[SKIP] Halide was compiled without exceptions.\n";
        return 0;
    }
    std::ofstream out("fuzz_res/blur_" + std::to_string(counter) + ".txt");
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
        generate_schedule(fdp, p, out);
        p.compile_to_module({}, "blur", get_host_target());
        std::map<std::string, Internal::Parameter> params;
        std::string hlpipe_file = "fuzz_res/blur_" + std::to_string(counter) + ".hlpipe";
        serialize_pipeline(p, hlpipe_file, params);
    } catch (const Halide::CompileError &e) {
        out << "\nexception: " << e.what() << "\n";
    }
    counter++;
    return 0;
}
