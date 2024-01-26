#include "ScheduleMutator.h"
#include "FindCalls.h"
#include "RealizationOrder.h"
#include "Func.h"
#include "Function.h"

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

Pipeline ScheduleMutator::mutate() {
    std::vector<Internal::Function> outputs;
    for (const auto &f : p.outputs()) {
        outputs.push_back(f.function());
    }
    std::map<std::string, Internal::Function> env = build_environment(outputs);
    std::vector<std::string> order = topological_order(outputs, env);
    for (int i = 0; i < order.size(); i++) {
        mutate_function_schedule(env[order[i]]);
    }
    return this->p;
}

void ScheduleMutator::mutate_function_schedule(Internal::Function &function) {
    for (int s = 0; s <= (int)function.updates().size(); ++s) {
        if (s == 0) {
            // Mutate the pure definition
            mutate_loop_schedule(function, function.definition(), 0);
            // mutate_compute_schedule(function.definition().schedule());
        } else {
            // Mutate the update definition
            mutate_loop_schedule(function, function.update(s - 1), s);
        }
    }
}

enum class LoopScheduleMutationType {
    DoNothing,
    ChangeExistingSplit,
    AddNewSplit,
    AddNewRename,
    AddNewFuse,
    // RemoveSplit,
    ChangexistingDim,
    ReorderDims,
};

void ScheduleMutator::mutate_loop_schedule(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    // randomly choose a mutation type
    std::uniform_int_distribution<int> mutation_type_dist(0, 6);
    LoopScheduleMutationType mutation_type = static_cast<LoopScheduleMutationType>(mutation_type_dist(rng));
    switch (mutation_type) {
        case LoopScheduleMutationType::DoNothing:
            // actually do nothing, skip
            break;
        case LoopScheduleMutationType::ChangeExistingSplit:
            mutate_change_existing_split(function, definition, stage_idx);
            break;
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
    splits[split_idx].tail = random_tail_strategy();
}

void ScheduleMutator::mutate_add_new_split(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Stage stage(function, definition, stage_idx);
    std::vector<VarOrRVar> vars = get_stage_vars_or_rvars(stage);
    std::uniform_int_distribution<int> var_dist(0, vars.size() - 1);
    int var_idx = var_dist(rng);
    VarOrRVar var_picked = vars[var_idx];
    int split_factor = random_split_factor();
    TailStrategy tail_strategy = random_tail_strategy();
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
}


// TODO: we probably can just remove the last split from the split list, and update dim list accordingly.
//       but this is essentially equal to not doing a mutate_add_new_split so there's not much point in doing this
//       unless we do deletion in a more random way (something in the middle of a split-chain)


// void ScheduleMutator::mutate_remove_split(Internal::Function &f, Internal::Definition &d, int stage_idx) {
    // Internal::StageSchedule &schedule = d.schedule();
    // std::vector<Internal::Split> &splits = schedule.splits();
    // for simplicity we only remove the last
    // if (splits.size() == 0) {
    //     return;
    // }
    // // randomly choose a split to mutate
    // std::uniform_int_distribution<int> split_idx_dist(0, splits.size() - 1);
    // int split_idx = split_idx_dist(rng);
    // splits.erase(splits.begin() + split_idx);
// }

void ScheduleMutator::mutate_change_existing_dim(Internal::Function &function, Internal::Definition &definition, int stage_idx) {
    Internal::StageSchedule &schedule = definition.schedule();
    std::vector<Internal::Dim> &dims = schedule.dims();
    if (dims.size() == 0) {
        return;
    }
    std::uniform_int_distribution<int> dim_idx_dist(0, dims.size() - 1);
    int dim_idx = dim_idx_dist(rng);
    Internal::Dim &dim = dims[dim_idx];
    if (dim.var == Var::outermost().name()) {
        return;
    }
    // TODO: we want to suppress some Can only vectorize/unroll for loops over a constant extent errors
    Internal::ForType for_type = random_for_type();
    if (for_type == Internal::ForType::Vectorized || for_type == Internal::ForType::Unrolled) {
        if (is_vectorizable(dim.var, function, definition, stage_idx)) {
            dim.for_type = for_type;
        }
    } else {
        dim.for_type = random_for_type();
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

}

