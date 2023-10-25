#include "Halide.h"
#include <cstdio>
#include <random>
#include <iostream>
#include <unordered_set>
#include <functional>
#include <exception>
#include <fstream>

using namespace Halide;


static int counter = 0;
static int internal_error_count = 0;

static std::random_device rnd;
static std::mt19937 rnd_eng;

void init_random_engine() {
    rnd_eng.seed(rnd());
}

int random_in_range(int low, int high) {
    std::uniform_int_distribution<int> dist(low, high);
    return dist(rnd_eng);
}

bool random_bool() {
    std::uniform_int_distribution<int> dist(0, 1);
    return dist(rnd_eng) == 1;
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

std::vector<VarOrRVar> get_function_vars(const Internal::Function &function) {
    std::vector<VarOrRVar> vars;
    const auto &dims = function.definition().schedule().dims();
    for (const auto &d : dims) {
        if (d.var != Var::outermost().name()) {
            if (d.is_rvar()) {
                vars.push_back(RVar(d.var));
            } else {
                vars.push_back(Var(d.var));
            }
        }
    }
    return vars;
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

std::vector<VarOrRVar> get_stage_vars_or_rvars(const Stage &stage) {
    std::vector<VarOrRVar> vars;
    const auto &dims = stage.get_schedule().dims();
    for (const auto &d : dims) {
        if (d.var != Var::outermost().name()) {
            if (d.is_rvar()) {
                vars.push_back(RVar(d.var));
            } else {
                vars.push_back(Var(d.var));
            }
        }
    }
    return vars;
}

std::vector<Var> get_stage_vars(const Stage &stage) {
    std::vector<Var> vars;
    const auto &dims = stage.get_schedule().dims();
    for (const auto &d : dims) {
        if (d.var != Var::outermost().name()) {
            if (!d.is_rvar()) {
                vars.push_back(Var(d.var));
            }
        }
    }
    return vars;
}

std::vector<RVar> get_stage_rvars(const Stage &stage) {
    std::vector<RVar> rvars;
    const auto &dims = stage.get_schedule().dims();
    for (const auto &d : dims) {
        if (d.var != Var::outermost().name()) {
            if (d.is_rvar()) {
                rvars.push_back(RVar(d.var));
            }
        }
    }
    return rvars;
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

void generate_loop_schedule(Internal::Function &function, Stage stage, std::ostream &out, bool is_update = false) {
    // available choices: split reorder fuse unroll vectorize parallel serial
    bool is_rfactored = false;
    std::function<void()> operations[] = {
        [&]() {
            std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
            VarOrRVar var_picked = vars[random_in_range(0, vars.size() - 1)];
            int split_factor = common_split_factors[random_in_range(0, sizeof(common_split_factors) / sizeof(int) - 1)];
            TailStrategy tail_strategy = tail_strategies[random_in_range(0, sizeof(tail_strategies) / sizeof(TailStrategy) - 1)];
            std::string var_name = get_original_var_name(var_picked.name());
            if (var_picked.is_rvar) {
                out << ".split(RVar(" << var_name << "), " << "RVar(" << var_name << "o), " << "RVar(" << var_name << "i), " << split_factor << ", " << tail_strategy_to_string(tail_strategy) << ")";
                stage.split(RVar(var_name), RVar(var_name + "o"), RVar(var_name + "i"), split_factor, tail_strategy);
            } else {
                out << ".split(Var(" << var_name << "), " << "Var(" << var_name << "o), " << "Var(" << var_name << "i), " << split_factor << ", " << tail_strategy_to_string(tail_strategy) << ")";
                stage.split(Var(var_name), Var(var_name + "o"), Var(var_name + "i"), split_factor, tail_strategy);
            }
        },
        [&]() {
            std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
            // we need at least two vars to reorder
            if (vars.size() < 2) {
                return;
            }
            // for reorder we always reorder everything we have in this function
            // but we need a random premutation, fdp is really bad at that, so we do rejection sampling.....
            std::vector<VarOrRVar> reorder_vars;
            std::vector<std::string> var_names_premutated;
            while (reorder_vars.size() != vars.size()) {
                VarOrRVar var_picked = vars[random_in_range(0, vars.size() - 1)];
                if (std::find(var_names_premutated.begin(), var_names_premutated.end(), var_picked.name()) == var_names_premutated.end()) {
                    var_names_premutated.push_back(var_picked.name());
                    if (var_picked.is_rvar) {
                        reorder_vars.push_back(RVar(get_original_var_name(var_picked.name())));
                    } else {
                        reorder_vars.push_back(Var(get_original_var_name(var_picked.name())));
                    }
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
            stage.reorder(reorder_vars);
        },
        [&]() {
            std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
            VarOrRVar var_picked = vars[random_in_range(0, vars.size() - 1)];
            // we need at least two vars to fuse
            if (vars.size() < 2) {
                return;
            }
            VarOrRVar inner_var = vars[random_in_range(0, vars.size() - 1)];
            VarOrRVar outer_var = vars[random_in_range(0, vars.size() - 1)];
            while (outer_var.name() == inner_var.name()) {
                outer_var = vars[random_in_range(0, vars.size() - 1)];
            }
            std::string inner_name = get_original_var_name(inner_var.name());
            std::string outer_name = get_original_var_name(outer_var.name());
            // TODO: we rely on Halide to catch the error here, but we should do better
            if (random_bool()) {
                std::string fused_var_name = "r_" + inner_name + "_" + outer_name + "_f";
                out << ".fuse(RVar(" << inner_name << "), RVar(" << outer_name << "), RVar(" << fused_var_name << "))";
                stage.fuse(VarOrRVar(inner_name, inner_var.is_rvar), VarOrRVar(outer_name, outer_var.is_rvar) , RVar(fused_var_name));
            } else {
                std::string fused_var_name = inner_name + "_" + outer_name + "_f";
                out << ".fuse(Var(" << inner_name << "), Var(" << outer_name << "), Var(" << fused_var_name << "))";
                stage.fuse(VarOrRVar(inner_name, inner_var.is_rvar), VarOrRVar(outer_name, outer_var.is_rvar) , Var(fused_var_name));
            }
        },
        [&]() {
            std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
            VarOrRVar var_picked = vars[random_in_range(0, vars.size() - 1)];
            out << ".unroll(" << var_picked.name() << ")";
            std::string var_name = get_original_var_name(var_picked.name());
            if (var_picked.is_rvar) {
                stage.unroll(RVar(var_name));
            } else {
                stage.unroll(Var(var_name));
            }
        },
        [&]() {
            std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
            VarOrRVar var_picked = vars[random_in_range(0, vars.size() - 1)];
            out << ".vectorize(" << var_picked.name() << ")";
            std::string var_name = get_original_var_name(var_picked.name());
            if (var_picked.is_rvar) {
                stage.vectorize(RVar(var_name));
            } else {
                stage.vectorize(Var(var_name));
            }
        },
        [&]() {
            std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
            VarOrRVar var_picked = vars[random_in_range(0, vars.size() - 1)];
            out << ".parallel(" << var_picked.name() << ")";
            std::string var_name = get_original_var_name(var_picked.name());
            if (var_picked.is_rvar) {
                stage.parallel(RVar(var_name));
            } else {
                stage.parallel(Var(var_name));
            }
        },
        [&]() {
            std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
            VarOrRVar var_picked = vars[random_in_range(0, vars.size() - 1)];
            out << ".serial(" << var_picked.name() << ")";
            std::string var_name = get_original_var_name(var_picked.name());
            if (var_picked.is_rvar) {
                stage.serial(RVar(var_name));
            } else {
                stage.serial(Var(var_name));
            }
        }
    };
    int depth = random_in_range(0, 5);
    for (int i = 0; i < depth; ++i) {
        operations[random_in_range(0, sizeof(operations) / sizeof(std::function<void()>) - 1)]();
    }
}

void generate_compute_schedule(Internal::Function &function, const std::vector<Func> &consumers, std::ostream &out) {
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
            Func compute_at_func = consumers[random_in_range(0, consumers.size() - 1)];
            std::vector<VarOrRVar> compute_at_func_vars = get_function_vars(compute_at_func.function());
            VarOrRVar var_picked = compute_at_func_vars[random_in_range(0, compute_at_func_vars.size() - 1)];
            out << ".compute_at(" << compute_at_func.name() << ", " << var_picked.name() << ")";
            if (var_picked.is_rvar) {
                Func(function).compute_at(compute_at_func, RVar(var_picked.name()));
            } else {
                Func(function).compute_at(compute_at_func, Var(var_picked.name()));
            }
        }
    };
    operations[random_in_range(0, sizeof(operations) / sizeof(std::function<void()>) - 1)]();
    out << "\n";
}

void generate_store_schedule(Internal::Function &function, const std::vector<Func> &consumers, std::ostream &out) {
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
            Func store_at_func = consumers[random_in_range(0, consumers.size() - 1)];
            std::vector<VarOrRVar> store_at_func_vars = get_function_vars(store_at_func.function());
            VarOrRVar var_picked = store_at_func_vars[random_in_range(0, store_at_func_vars.size() - 1)];
            out << ".store_at(" << store_at_func.name() << ", " << var_picked.name() << ")";
            if (var_picked.is_rvar) {
                Func(function).store_at(store_at_func, RVar(var_picked.name()));
            } else {
                Func(function).store_at(store_at_func, Var(var_picked.name()));
            }
        }
    };
    operations[random_in_range(0, sizeof(operations) / sizeof(std::function<void()>) - 1)]();
    out << "\n";
}

void generate_loop_schedule_per_function(Internal::Function &function, std::ostream &out) {
    for (int s = 0; s <= (int)function.updates().size(); s++) {
        if (s == 0) {
            out << "planned loop schedule for function " << function.name();
            generate_loop_schedule(function, Stage(function, function.definition(), 0), out, false);
            out << "\n";
        } else {
            out << "planned loop schedule for function " << function.name() << ".update(" << s - 1 << ")";
            generate_loop_schedule(function, Stage(function, function.update(s - 1), s), out, true);
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

void generate_rfactor_schedule(Internal::Function &function, std::ostream &out) {
    if (function.updates().size() == 0) {
        return;
    }
    int update_stage_index = random_in_range(0, function.updates().size() - 1);
    Stage stage = Stage(function, function.update(update_stage_index), update_stage_index + 1);
    std::vector<Var> vars = {Var("u"), Var("v"), Var("w"), Var("z")};
    std::vector<RVar> rvars = get_stage_rvars(stage);
    Var var_picked = vars[random_in_range(0, vars.size() - 1)];
    RVar rvar_picked = rvars[random_in_range(0, rvars.size() - 1)];
    // NOTE: rfactor won't work on fused rvars
    if (rvar_picked.name().find("f") != std::string::npos) {
        return;
    }
    out << "planned rfactor schedule for function " << function.name() << ".rfactor(" << rvar_picked.name() << ", " << var_picked.name() << ")";
    std::string var_name = get_original_var_name(var_picked.name());
    std::string rvar_name = get_original_var_name(rvar_picked.name());
    stage.rfactor(RVar(rvar_name), Var(var_name));
}

void generate_schedule(Pipeline &p, std::ostream& out) {
    std::vector<Internal::Function> outputs;
    for (const auto &f : p.outputs()) {
        outputs.push_back(f.function());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    // TODO: should we preturb the order?
    std::vector<std::string> order = topological_order(outputs, env);
    for (int i = order.size() - 1; i >= 0; --i) {
        if (random_bool()) {
            generate_loop_schedule_per_function(env[order[i]], out);
        }
        out.flush();
    }
    // once we generate loop schedules in the original DAG, we try to generate rfactor/in schedules to change the DAG
    for (int i = order.size() - 1; i >= 0; --i) {
        if (random_bool()) {
            generate_rfactor_schedule(env[order[i]], out);
        }
        out.flush();
    }
    // rebuild the DAG in case any new functions are created
    std::map<std::string, Internal::Function> new_env = build_environment(outputs);
    std::vector<std::string> new_order = topological_order(outputs, new_env);

    // for new functions, generate loop schedules
    for (int i = new_order.size() - 1; i >= 0; --i) {
        if (env.count(new_order[i]) == 0) {
            if (random_bool()) {
                generate_loop_schedule_per_function(new_env[new_order[i]], out);
            }
        }
        out.flush();
    }

    for (int i = new_order.size() - 1; i >= 0; --i) {
        // TODO: this works for linear relationship within a pipeline
        // we need better mechanism to figure out correct producer/consumer relationship
        std::vector<Func> producers;
        for (int j = 0; j < i; ++j) {
            producers.push_back(Func(new_env[new_order[j]]));
        }
        std::vector<Func> consumers;
        for (int j = i + 1; j < new_order.size(); ++j) {
            consumers.push_back(Func(new_env[new_order[j]]));
        }
        if (random_bool()) {
            generate_compute_schedule(new_env[new_order[i]], consumers, out);
        }
        if (random_bool()) {
            generate_store_schedule(new_env[new_order[i]], consumers, out);
        }
        out.flush();
    }
}

int main(int argc, char *argv[]) {
    std::string output_dir = Internal::get_env_variable("FUZZER_OUTPUT_DIR");
    if (!Halide::exceptions_enabled()) {
        std::cout << "[SKIP] Halide was compiled without exceptions.\n";
        return 0;
    }
    std::string max_runs_string = Internal::get_env_variable("FUZZER_MAX_RUNS");
    int max_runs = std::stoi(max_runs_string);
    for (int i = 0; i < max_runs; ++i) {
        std::ofstream out(output_dir + "/blur_" + std::to_string(counter) + ".txt");
        Func input("input");
        Func local_sum("local_sum");
        Func blurry("blurry");
        Var x("x"), y("y");
        input(x, y) = 2 * x + 5 * y;
        RDom r(-2, 5, -2, 5, "rdom_r");
        local_sum(x, y) = 0;
        local_sum(x, y) += input(x + r.x, y + r.y);
        blurry(x, y) = cast<int32_t>(local_sum(x, y) / 25);
        Pipeline p({blurry});
        try {
            generate_schedule(p, out);
            // compute hash of this schedule, read a global hash object
            p.compile_to_module({}, "blur", get_host_target());
            std::map<std::string, Parameter> params;
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
    }
    return 0;
}

