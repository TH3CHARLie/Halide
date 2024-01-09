#include "ScheduleMutator.h"
#include "FindCalls.h"
#include "RealizationOrder.h"
#include "Func.h"
#include "Function.h"

namespace Halide {
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
            mutate_loop_schedule(function.definition().schedule());
        } else {
            // TODO: handle update defintion later
            break;
        }
    }
}

void ScheduleMutator::mutate_loop_schedule(Internal::StageSchedule &schedule) {
    static int available_split_factors[] = {1, 2, 4, 8, 16};
    // flip a coin to decide whether to mutate the loop nest
    std::bernoulli_distribution coin(0.5);
    if (!coin(rng)) {
        // std::cout << "not mutating loop nest\n";
        return;
    } else {
        // std::cout << "mutating loop nest\n";
    }
    std::vector<Internal::Split> &splits = schedule.splits();
    for (auto &split: splits) {
        // std::cout << "split: " << split.old_var << " -> " << split.outer << ", " << split.inner << " old split factor = " << split.factor << "\n";
        // randomly choose a new split factor
        std::uniform_int_distribution<int> split_factor_dist(0, 5);
        int split_factor_idx = split_factor_dist(rng);
        if (split_factor_idx == 5) {
            // don't mutate the split factor
            // std::cout << "not mutating split factor\n";
            continue;
        } else {
            split.factor = available_split_factors[split_factor_idx];
        }
        // std::cout<< "new split factor = " << split.factor << "\n";
    }
}
}