#ifndef HALIDE_INTERNAL_SCHEDULE_MUTATOR_H
#define HALIDE_INTERNAL_SCHEDULE_MUTATOR_H

#include <random>
#include "Pipeline.h"
namespace Halide {

class ScheduleMutator {
public:
    ScheduleMutator(Pipeline p, unsigned int seed) : p(p), rng(seed) {
    }

    Pipeline mutate();

private:

    void mutate_function_schedule(Internal::Function &f);

    void mutate_loop_schedule(Internal::Function &f, Internal::Definition &d, int stage_idx);

    void mutate_change_existing_split(Internal::Function &f, Internal::Definition &d, int stage_idx);

    void mutate_add_new_split(Internal::Function &f, Internal::Definition &d, int stage_idx);

    void mutate_add_new_rename(Internal::Function &f, Internal::Definition &d, int stage_idx);

    void mutate_add_new_fuse(Internal::Function &f, Internal::Definition &d, int stage_idx);

    // void mutate_remove_split(Internal::Function &f, Internal::Definition &d, int stage_idx);

    int random_split_factor();

    TailStrategy random_tail_strategy();

    TailStrategy random_rvar_tail_strategy();

    Pipeline p;

    std::mt19937 rng;
};
}

#endif
