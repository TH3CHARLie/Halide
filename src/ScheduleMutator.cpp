#include "ScheduleMutator.h"
#include "FindCalls.h"
#include "RealizationOrder.h"
#include "Func.h"
#include "Function.h"
#include <set>

namespace Halide {
int ScheduleMutator::random_split_factor() {
    static int available_split_factors[] = {1, 2, 3, 4, 5, 6, 8, 16};
    std::uniform_int_distribution<int> split_factor_dist(0, 7);
    int split_factor_idx = split_factor_dist(rng);
    return available_split_factors[split_factor_idx];
}

TailStrategy ScheduleMutator::random_tail_strategy() {
    // NOTE: there are 8 tail strategies, but we only take the first 7 since we don't want to mutate to Auto
    std::uniform_int_distribution<int> tail_strategy_dist(0, 7);
    int tail_strategy_idx = tail_strategy_dist(rng);
    return static_cast<TailStrategy>(tail_strategy_idx);
}

// The available for types are: Serial = 0, Parallel = 1, Vectorized = 2, Unrolled = 3
Internal::ForType ScheduleMutator::random_for_type() {
    std::uniform_int_distribution<int> for_type_dist(0, 3);
    int for_type_idx = for_type_dist(rng);
    return static_cast<Internal::ForType>(for_type_idx);
}

bool ScheduleMutator::is_vectorizable(const std::string &var, Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Internal::StageSchedule &schedule = definition.schedule();
    std::vector<Internal::Split> &splits = schedule.splits();
    if (splits.size() == 0) {
        return false;
    }
    std::vector<std::string> const_factor_vars;
    for (int idx = 0; idx < splits.size(); ++idx) {
        auto &split = splits[idx];
        if (split.split_type == Internal::Split::SplitType::SplitVar) {
            if (std::find(const_factor_vars.begin(), const_factor_vars.end(), split.old_var) != const_factor_vars.end()) {
                if (Internal::is_const(split.factor)) {
                    if (split.inner == var || split.outer == var) {
                        return true;
                    } else {
                        const_factor_vars.push_back(split.inner);
                        const_factor_vars.push_back(split.outer);
                    }
                }
            } else {
                if (Internal::is_const(split.factor)) {
                    if (split.inner == var) {
                        return true;
                    } else {
                        const_factor_vars.push_back(split.inner);
                    }
                }
            }
        } else if (split.split_type == Internal::Split::SplitType::FuseVars) {
            if (split.old_var == var) {
                return is_vectorizable(split.inner, function, definition, stage_idx) && is_vectorizable(split.outer, function, definition, stage_idx);
            }
        }
    }
    return false;
}

bool ScheduleMutator::is_function_inlined(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    const Internal::FuncSchedule &func_schedule = function.schedule();
    LoopLevel compute_level = func_schedule.compute_level();
    if (func_schedule.compute_level().is_inlined()) {
        return true;
    }
    return false;
}

bool ScheduleMutator::is_input_func(const Internal::Function &f) const {
    if (const Internal::Call *call = f.is_wrapper()) {
        return call->call_type == Internal::Call::Image;
    }
    return false;
}

std::string get_original_var_name(const std::string &var_name) {
    if (var_name.rfind('.' != std::string::npos)) {
        return var_name.substr(var_name.rfind('.') + 1, var_name.size() - 1 - var_name.rfind('.'));
    }
    return var_name;
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

void ScheduleMutator::mutate_function_schedule(Internal::Function &function, bool is_output_func) {
    // randomize a boolean, we only do one of the two mutations: loop schedule or compute schedule
    std::uniform_int_distribution<int> bool_dist(0, 1);
    bool is_loop_schedule = bool_dist(rng);
    if (is_loop_schedule) {
        for (int s = 0; s <= (int)function.updates().size(); ++s) {
            if (s == 0) {
                // Mutate the pure definition
                mutate_loop_schedule(function, function.definition(), 0);
            } else {
                // Mutate the update definition
                mutate_loop_schedule(function, function.update(s - 1), s);
            }
        }
    } else {
        mutate_compute_schedule(function, is_output_func);
    }
}

enum class LoopScheduleMutationType {
    DoNothing,
    // ChangeExistingSplit,
    AddNewSplit,
    AddNewRename,
    AddNewFuse,
    // RemoveSplit,
    ChangexistingDim,
    ReorderDims,
};

void ScheduleMutator::mutate_loop_schedule(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    // we don't attempt to mutate input functions
    if (is_input_func(function)) {
        return;
    }
    // randomly choose a mutation type
    std::uniform_int_distribution<int> mutation_type_dist(0, 5);
    LoopScheduleMutationType mutation_type = static_cast<LoopScheduleMutationType>(mutation_type_dist(rng));
    switch (mutation_type) {
        case LoopScheduleMutationType::DoNothing:
            // actually do nothing, skip
            break;
        // TODO: disable this for now
        // case LoopScheduleMutationType::ChangeExistingSplit:
        //     mutate_change_existing_split(function, definition, stage_idx);
        //     break;
        case LoopScheduleMutationType::AddNewSplit:
            mutate_add_new_split(function, definition, stage_idx);
            break;
        case LoopScheduleMutationType::AddNewRename:
            mutate_add_new_rename(function, definition, stage_idx);
            break;
        case LoopScheduleMutationType::AddNewFuse:
            mutate_add_new_fuse(function, definition, stage_idx);
            break;
        // case LoopScheduleMutationType::RemoveSplit:
        //     mutate_remove_split(function, definition, stage_idx);
        //     break;
        case LoopScheduleMutationType::ChangexistingDim:
            mutate_change_existing_dim(function, definition, stage_idx);
            break;
        case LoopScheduleMutationType::ReorderDims:
            mutate_reorder_dims(function, definition, stage_idx);
            break;
        default:
            break;
    }
}

void ScheduleMutator::mutate_change_existing_split(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Internal::StageSchedule &schedule = definition.schedule();
    std::vector<Internal::Split> &splits = schedule.splits();
    if (splits.size() == 0) {
        return;
    }
    // randomly choose a split to mutate
    std::uniform_int_distribution<int> split_idx_dist(0, splits.size() - 1);
    int split_idx = split_idx_dist(rng);
    splits[split_idx].factor = random_split_factor();
    // we need to check if the new tail strategy is of type PredicateStores or PredicateLoads
    // if so, we need to make sure that the inner var is not already in the split list
    TailStrategy tail_strategy = random_tail_strategy();
    if (tail_strategy == TailStrategy::PredicateStores || tail_strategy == TailStrategy::PredicateLoads) {
        std::set<std::string> predicated_vars;
        predicated_vars.insert(splits[split_idx].inner);
        for (auto &split : splits) {
            if (predicated_vars.count(split.old_var) != 0) {
                return;
            }
        }
    }
    // we also need to make sure that when we change the tail strategy to PredicateStores or PredicateLoads, the var we are spliting is not some other
    // split's inner var
    if (tail_strategy == TailStrategy::PredicateStores || tail_strategy == TailStrategy::PredicateLoads) {
        std::set<std::string> inner_vars;
        for (int i = 0; i < split_idx; ++i) {
            inner_vars.insert(splits[i].inner);
        }
        if (inner_vars.count(splits[split_idx].old_var) != 0) {
            return;
        }
    }
    splits[split_idx].tail = tail_strategy;
}

void ScheduleMutator::mutate_add_new_split(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Stage stage(function, definition, stage_idx);
    std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
    std::uniform_int_distribution<int> var_dist(0, vars.size() - 1);
    int var_idx = var_dist(rng);
    VarOrRVar var_picked = vars[var_idx];
    // we need to check if the var is from a split of PredicateStores or PredicateLoads
    std::set<std::string> predicated_vars;
    auto splits = definition.schedule().splits();
    for (auto &split : splits) {
        if (split.tail == TailStrategy::PredicateStores || split.tail == TailStrategy::PredicateLoads) {
            predicated_vars.insert(split.inner);
        }
    }
    if (predicated_vars.count(var_picked.name()) != 0) {
        return;
    }
    int split_factor = random_split_factor();
    TailStrategy tail_strategy = random_tail_strategy();
    // similar to the above mutate case, make sure we are not spliting inner var of some other split using PredicateStores or PredicateLoads
    if (tail_strategy == TailStrategy::PredicateStores || tail_strategy == TailStrategy::PredicateLoads) {
        std::set<std::string> inner_vars;
        for (auto &split : splits) {
            inner_vars.insert(split.inner);
        }
        if (inner_vars.count(var_picked.name()) != 0) {
            return;
        }
    }
    std::string var_name = get_original_var_name(var_picked.name());
    if (var_picked.is_rvar) {
        stage.split(RVar(var_name), RVar(var_name + "o"), RVar(var_name + "i"), split_factor, tail_strategy);
    } else {
        stage.split(Var(var_name), Var(var_name + "o"), Var(var_name + "i"), split_factor, tail_strategy);
    }
}

void ScheduleMutator::mutate_add_new_rename(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Stage stage(function, definition, stage_idx);
    std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
    std::uniform_int_distribution<int> var_dist(0, vars.size() - 1);
    int var_idx = var_dist(rng);
    VarOrRVar var_picked = vars[var_idx];
    std::string var_name = get_original_var_name(var_picked.name());
    if (var_picked.is_rvar) {
        stage.rename(RVar(var_name), RVar(var_name + "rn"));
    } else {
        stage.rename(Var(var_name), Var(var_name + "rn"));
    }
}

void ScheduleMutator::mutate_add_new_fuse(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Stage stage(function, definition, stage_idx);
    std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
    if (vars.size() < 2) {
        return;
    }
    std::uniform_int_distribution<int> var_dist(0, vars.size() - 1);
    VarOrRVar inner_var = vars[var_dist(rng)];
    VarOrRVar outer_var = vars[var_dist(rng)];
    while (inner_var.name() == outer_var.name()) {
        outer_var = vars[var_dist(rng)];
    }
    std::string inner_var_name = get_original_var_name(inner_var.name());
    std::string outer_var_name = get_original_var_name(outer_var.name());
    // TODO: how to do with Rvars?
    stage.fuse(Var(inner_var_name), Var(outer_var_name), Var(inner_var_name + "f" + outer_var_name));
    // whenever we create a new fused var, we mark the new var as serial.
    stage.serial(Var(inner_var_name + "f" + outer_var_name));
}

// void ScheduleMutator::mutate_remove_split(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
//     Stage stage(function, definition, stage_idx);
//     std::vector<Internal::Split> splits = definition.schedule().splits();
//     if (splits.size() == 0) {
//         return;
//     }
//     std::uniform_int_distribution<int> split_idx_dist(0, splits.size() - 1);
//     int split_idx = split_idx_dist(rng);
//     Internal::Split split = splits[split_idx];
//     std::set<std::string> removed_vars;

//     auto add_to_removed_vars = [&removed_vars](const Internal::Split &split) {
//         if (split.split_type == Internal::Split::SplitType::SplitVar) {
//             removed_vars.insert(split.inner);
//             removed_vars.insert(split.outer);
//         } else if (split.split_type == Internal::Split::SplitType::FuseVars) {
//             removed_vars.insert(split.old_var);
//         } else if (split.split_type == Internal::Split::SplitType::RenameVar) {
//             removed_vars.insert(split.outer);
//         }
//     };

//     auto need_remove = [&removed_vars](const Internal::Split &split) {
//         if (split.split_type == Internal::Split::SplitType::SplitVar) {
//             return removed_vars.count(split.old_var) != 0;
//         } else if (split.split_type == Internal::Split::SplitType::FuseVars) {
//             return removed_vars.count(split.inner) != 0 || removed_vars.count(split.outer) != 0;
//         } else if (split.split_type == Internal::Split::SplitType::RenameVar) {
//             return removed_vars.count(split.old_var) != 0;
//         }
//         return false;
//     };

//     add_to_removed_vars(split);
//     std::vector<int> removed_split_idxs;
//     removed_split_idxs.push_back(split_idx);
//     for (int idx = 0; idx < splits.size(); ++idx) {
//         if (idx != split_idx && need_remove(splits[idx])) {
//             add_to_removed_vars(splits[idx]);
//             removed_split_idxs.push_back(idx);
//         }
//     }
//     std::vector<int> removed_dim_idxs;
//     std::vector<Internal::Dim> dims = definition.schedule().dims();
//     for (int idx = 0; idx < dims.size(); ++idx) {
//         if (dims[idx].var != Var::outermost().name()) {
//             if (removed_vars.count(dims[idx].var) != 0) {
//                 removed_dim_idxs.push_back(idx);
//             }
//         }
//     }
//     std::vector<Internal::Split> new_splits;
//     for (int idx = 0; idx < splits.size(); ++idx) {
//         if (std::find(removed_split_idxs.begin(), removed_split_idxs.end(), idx) == removed_split_idxs.end()) {
//             new_splits.push_back(splits[idx]);
//         }
//     }
//     definition.schedule().splits() = new_splits;
//     std::vector<Internal::Dim> new_dims;
//     for (int idx = 0; idx < dims.size(); ++idx) {
//         if (std::find(removed_dim_idxs.begin(), removed_dim_idxs.end(), idx) == removed_dim_idxs.end()) {
//             new_dims.push_back(dims[idx]);
//         }
//     }
//     definition.schedule().dims() = new_dims;
//     return;
// }

void ScheduleMutator::mutate_change_existing_dim(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    bool is_inlined = false;
    function.schedule().compute_level().lock();
    is_inlined = function.schedule().compute_level().is_inlined();
    function.schedule().compute_level().unlock();
    // if the function is inlined, we don't mutate the dim
    if (is_inlined) {
        return;
    }
    Stage stage(function, definition, stage_idx);
    Internal::StageSchedule &schedule = definition.schedule();
    std::vector<Internal::Dim> &dims = schedule.dims();
    if (dims.size() == 0) {
        return;
    }
    std::uniform_int_distribution<int> dim_idx_dist(0, dims.size() - 1);
    int dim_idx = dim_idx_dist(rng);
    Internal::Dim &dim = dims[dim_idx];
    // we don't mutate the outermost dim and rvars
    if (dim.var == Var::outermost().name() || dim.is_rvar()) {
        return;
    }
    Internal::ForType for_type = random_for_type();
    // TODO: we want to suppress some Can only vectorize/unroll for loops over a constant extent errors
    if (for_type == Internal::ForType::Vectorized || for_type == Internal::ForType::Unrolled) {
        if (is_vectorizable(dim.var, function, definition, stage_idx)) {
            if (for_type == Internal::ForType::Vectorized) {
                stage.vectorize(Var(dim.var));
            } else {
                stage.unroll(Var(dim.var));
            }
        }
    } else {
        if (for_type == Internal::ForType::Parallel) {
            stage.parallel(Var(dim.var));
        } else {
            stage.serial(Var(dim.var));
        }
    }
}

void ScheduleMutator::mutate_reorder_dims(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Internal::StageSchedule &schedule = definition.schedule();
    std::vector<Internal::Dim> &dims = schedule.dims();
    // if the stage only has 1 dim other than __outermost, we don't shuffle it
    if (dims.size() <= 2) {
        return;
    }
    std::shuffle(dims.begin(), dims.end() - 1, rng);
}


enum class ComputeScheduleMutationType {
    DoNothing,
    ComputeRoot,
    ComputeAt,
    ComputeInline
};

void ScheduleMutator::mutate_compute_schedule(Internal::Function &function, bool is_output_func) {
    // output function should always be compute_root;
    if (is_output_func) {
        return;
    }
    if (is_input_func(function)) {
        return;
    }
    // randomly choose a mutation type
    std::uniform_int_distribution<int> mutation_type_dist(0, 3);
    ComputeScheduleMutationType mutation_type = static_cast<ComputeScheduleMutationType>(mutation_type_dist(rng));
    if (mutation_type == ComputeScheduleMutationType::DoNothing) {
        return;
    } else if (mutation_type == ComputeScheduleMutationType::ComputeRoot) {
        Func(function).compute_root();
    } else if (mutation_type == ComputeScheduleMutationType::ComputeAt) {
        Func(function).compute_root();
        return;
    } else if (mutation_type == ComputeScheduleMutationType::ComputeInline) {
        // we check if any of the dims are parallelized or vectorized, if so, we don't inline
        bool can_inline = true;
        auto dims = function.definition().schedule().dims();
        for (const auto &d: dims) {
            if (d.for_type == Internal::ForType::Parallel || d.for_type == Internal::ForType::Vectorized || d.for_type == Internal::ForType::Unrolled) {
                can_inline = false;
                break;
            }
        }
        if (can_inline) {
            Func(function).compute_inline();
        }
    }
}

// Top-level entry point
Pipeline ScheduleMutator::mutate() {
    std::vector<Internal::Function> outputs;
    std::set<std::string> output_function_names;
    for (const auto &f : p.outputs()) {
        outputs.push_back(f.function());
        output_function_names.insert(f.name());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    std::vector<std::string> order = topological_order(outputs, env);
    for (int i = 0; i < order.size(); i++) {
        bool is_output_func = output_function_names.count(order[i]) != 0;
        mutate_function_schedule(env[order[i]], is_output_func);
    }
    return this->p;
}

}

