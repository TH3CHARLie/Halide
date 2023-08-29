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

std::vector<std::string> get_stage_var_names(const Stage &stage) {
    std::vector<std::string> var_names;
    const auto &dims = stage.get_schedule().dims();
    for (const auto &d : dims) {
        if (d.var != Var::outermost().name()) {
            var_names.push_back(get_original_var_name(d.var));
        }
    }
    return var_names;
}

int common_split_factors[] = {1, 2, 4, 8};
TailStrategy tail_strategies[] = {TailStrategy::RoundUp, TailStrategy::GuardWithIf, TailStrategy::Predicate, /* temporarily disable TailStrategy::PredicateLoads, */ TailStrategy::PredicateStores, TailStrategy::ShiftInwards, TailStrategy::Auto};

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

void generate_loop_schedule(FuzzedDataProvider &fdp, Internal::Function &function, Stage stage, std::ostream &out) {
    // available choices: split reorder fuse unroll vectorize parallel serial
    // we only call vectorize once (#7779)
    bool is_vectorized = false;
    std::function<void()> operations[] = {
        [&]() {
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            int split_factor = fdp.PickValueInArray(common_split_factors);
            TailStrategy tail_strategy = fdp.PickValueInArray(tail_strategies);
            out << ".split(" << var_name_picked << ", " << var_name_picked << "_outer, " << var_name_picked << "_inner, " << split_factor << ", " << tail_strategy_to_string(tail_strategy) << ")";
            Func(function).split(Var(var_name_picked), Var(var_name_picked + "_outer"), Var(var_name_picked + "_inner"), split_factor, tail_strategy);
        },
        [&]() {
            if (fdp.remaining_bytes() <= 0) {
                return;
            }
            std::vector<std::string> var_names = get_function_var_names(function);
            // we need at least two vars to reorder
            if (var_names.size() < 2) {
                return;
            }
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
        },
        [&]() {
            std::vector<std::string> var_names = get_function_var_names(function);
            // we need at least two vars to fuse
            if (var_names.size() < 2) {
                return;
            }
            std::string inner_var_name = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            std::string outer_var_name = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            while (outer_var_name == inner_var_name && fdp.remaining_bytes() > 0) {
                outer_var_name = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            }
            std::string fused_var_name = inner_var_name + "_" + outer_var_name + "_fused";
            out << ".fuse(" << inner_var_name << ", " << outer_var_name << ", " << fused_var_name << ")";
            Func(function).fuse(Var(inner_var_name), Var(outer_var_name), Var(fused_var_name));
        },
        [&]() {
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            out << ".unroll(" << var_name_picked << ")";
            Func(function).unroll(Var(var_name_picked));
        },
        [&]() {
            if (is_vectorized) {
                return;
            }
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            out << ".vectorize(" << var_name_picked << ")";
            Func(function).vectorize(Var(var_name_picked));
            is_vectorized = true;
        },
        [&]() {
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            out << ".parallel(" << var_name_picked << ")";
            Func(function).parallel(Var(var_name_picked));
        },
        [&]() {
            std::vector<std::string> var_names = get_function_var_names(function);
            std::string var_name_picked = var_names[fdp.ConsumeIntegralInRange<int>(0, var_names.size() - 1)];
            out << ".serial(" << var_name_picked << ")";
            Func(function).serial(Var(var_name_picked));
        }
    };
    int depth = fdp.ConsumeIntegralInRange<int>(0, 5);
    for (int i = 0; i < depth; ++i) {
        fdp.PickValueInArray(operations)();
    }
}

void generate_compute_schedule(FuzzedDataProvider &fdp, Internal::Function &function, const std::vector<Func> &consumers, std::ostream &out) {
    out << "planned compute schedule for function " << function.name();
    std::function<void()> operations[] = {
        [&]() {
            out << ".compute_root()";
            Func(function).compute_root();
        },
        [&]() {
            if (consumers.empty()) {
                return;
            }
            Func compute_at_func = consumers[fdp.ConsumeIntegralInRange<int>(0, consumers.size() - 1)];
            auto compute_at_func_var_names = get_function_var_names(compute_at_func.function());
            std::string var_name_picked = compute_at_func_var_names[fdp.ConsumeIntegralInRange<int>(0, compute_at_func_var_names.size() - 1)];
            out << ".compute_at(" << compute_at_func.name() << ", " << var_name_picked << ")";
            Func(function).compute_at(compute_at_func, Var(var_name_picked));
        }
    };
    fdp.PickValueInArray(operations)();
    out << "\n";
}

void generate_store_schedule(FuzzedDataProvider &fdp, Internal::Function &function, const std::vector<Func> &consumers, std::ostream &out) {
    out << "planned store schedule for function " << function.name();
    std::function<void()> operations[] = {
        [&]() {
            out << ".store_root()";
            Func(function).store_root();
        },
        [&]() {
            if (consumers.empty()) {
                return;
            }
            Func store_at_func = consumers[fdp.ConsumeIntegralInRange<int>(0, consumers.size() - 1)];
            auto store_at_func_var_names = get_function_var_names(store_at_func.function());
            std::string var_name_picked = store_at_func_var_names[fdp.ConsumeIntegralInRange<int>(0, store_at_func_var_names.size() - 1)];
            out << ".store_at(" << store_at_func.name() << ", " << var_name_picked << ")";
            Func(function).store_at(store_at_func, Var(var_name_picked));
        }
    };
    fdp.PickValueInArray(operations)();
    out << "\n";
}

void generate_loop_schedule_per_function(FuzzedDataProvider &fdp, Internal::Function &function, std::ostream &out) {
    for (int s = 0; s <= (int)function.updates().size(); s++) {
        if (s == 0) {
            out << "planned loop schedule for function " << function.name();
            generate_loop_schedule(fdp, function, Stage(function, function.definition(), 0), out);
            out << "\n";
        } else {
            out << "\nplanned loop schedule for function " << function.name() << ".update(" << s - 1 << ")";
            generate_loop_schedule(fdp, function, Stage(function, function.update(s - 1), s), out);
            // but if this call still does not schedule anything, we explict call unscheduled()
            bool any_scheduled = function.has_pure_definition() && function.definition().schedule().touched();
            if (any_scheduled && !function.update(s - 1).schedule().touched()) {
                out << ".unscheduled()";
                Stage(function, function.update(s - 1), s).unscheduled();
            }
            out << "\n";
        }
    }
}

void generate_schedule(FuzzedDataProvider &fdp, Pipeline &p, std::ostream& out) {
    std::vector<Internal::Function> outputs;
    for (const auto &f : p.outputs()) {
        outputs.push_back(f.function());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    // TODO: should we preturb the order?
    std::vector<std::string> order = topological_order(outputs, env);
    for (int i = order.size() - 1; i >= 0; --i) {
        if (fdp.ConsumeBool()) {
            generate_loop_schedule_per_function(fdp, env[order[i]], out);
        }
        out.flush();
    }
    for (int i = order.size() - 1; i >= 0; --i) {
        // TODO: this works for linear relationship within a pipeline
        // we need better mechanism to figure out correct producer/consumer relationship
        std::vector<Func> producers;
        for (int j = 0; j < i; ++j) {
            producers.push_back(Func(env[order[j]]));
        }
        std::vector<Func> consumers;
        for (int j = i + 1; j < order.size(); ++j) {
            consumers.push_back(Func(env[order[j]]));
        }
        if (fdp.ConsumeBool()) {
            generate_compute_schedule(fdp, env[order[i]], consumers, out);
        }
        if (fdp.ConsumeBool()) {
            generate_store_schedule(fdp, env[order[i]], consumers, out);
        }
        out.flush();
    }
}

static int counter = 0;
static int internal_error_count = 0;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    std::string output_dir = Internal::get_env_variable("FUZZER_OUTPUT_DIR");
    if (!Halide::exceptions_enabled()) {
        std::cout << "[SKIP] Halide was compiled without exceptions.\n";
        return 0;
    }
    std::ofstream out(output_dir + "/blur_" + std::to_string(counter) + ".txt");
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
    FuzzedDataProvider fdp(data, size);
    try {
        generate_schedule(fdp, p, out);
        // compute hash of this schedule, read a global hash object
        p.compile_to_module({}, "blur", get_host_target());
        std::map<std::string, Internal::Parameter> params;
        std::string hlpipe_file = output_dir + "/blur_" + std::to_string(counter) + ".hlpipe";
        serialize_pipeline(p, hlpipe_file, params);
    } catch (const Halide::CompileError &e) {
        out << "\nUser Error: " << e.what() << "\n";
    } catch (const Halide::InternalError &e) {
        out << "\nInternal Error: \n" << e.what() << "\n";
        internal_error_count++;
    }
    counter++;
    if (internal_error_count > 0) {
        std::cout << "current fuzzing counter " << counter << " internal error count " << internal_error_count << "\n";
    }
    return 0;
}
